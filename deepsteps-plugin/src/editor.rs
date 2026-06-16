//! Custom egui editor for DeepSteps.
//!
//! Restores the Stage-1 UX: a 16-step grid with a live playhead and click-to-
//! toggle cells, plus sliders for the latent vector, per-step pitches, timing,
//! and tuning. The step on/off pattern is owned by the audio thread via
//! [`SharedState`]; everything else is a nih-plug param edited through the
//! `ParamSetter`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use nih_plug::prelude::{AsyncExecutor, Editor, ParamSetter};
use nih_plug_egui::{create_egui_editor, egui, resizable_window::ResizableWindow, widgets::ParamSlider};

use crate::params::DeepStepsParams;
use crate::shared::{SharedState, NO_STEP};
use crate::training::{Task, TrainShared, TrainStatus};

/// State handed to every egui frame.
struct EditorState {
    params: Arc<DeepStepsParams>,
    shared: Arc<SharedState>,
    train: Arc<TrainShared>,
    exec: AsyncExecutor<crate::DeepSteps>,
    /// Training hyperparameters, editable in the GUI (default to the upstream
    /// `train_export.py` values). Atomics so the panel can mutate them while the
    /// surrounding `EditorState` is only borrowed immutably (the egui frame).
    epochs: AtomicUsize,
    batch: AtomicUsize,
}

pub fn create(
    params: Arc<DeepStepsParams>,
    shared: Arc<SharedState>,
    train: Arc<TrainShared>,
    exec: AsyncExecutor<crate::DeepSteps>,
) -> Option<Box<dyn Editor>> {
    let egui_state = params.editor_state.clone();
    create_egui_editor(
        egui_state,
        EditorState {
            params,
            shared,
            train,
            exec,
            epochs: AtomicUsize::new(200),
            batch: AtomicUsize::new(16),
        },
        |_ctx, _state| {},
        |ctx, setter, state| {
            // Reclaim decoders retired by a hot-swap. This runs on the GUI thread
            // (never the audio thread), so the old model's heap is freed here, not
            // inside `process()` (see `TrainShared::swap_model`).
            state.train.collect_garbage();

            // Keep repainting while the playhead moves or a training run is in
            // progress (so the progress bar advances). Otherwise let egui idle.
            if state.shared.current() != NO_STEP
                || matches!(state.train.status(), TrainStatus::Running | TrainStatus::Ingesting)
            {
                ctx.request_repaint();
            }

            // Scale text with the window so the UI stays readable at any size.
            // Derived from the window's logical width (`EguiState`, independent of
            // the styling so there's no feedback loop), floored at 1.0 so text is
            // never smaller than the baseline. We scale font sizes rather than the
            // egui zoom factor on purpose: zoom changes points-per-pixel, which
            // would desync `ResizableWindow`'s drag-corner math (it measures in
            // points but the wrapper resizes in pixels). Keeping zoom at 1 means
            // points == pixels, so the resize corner stays correct.
            let scale = (state.params.editor_state.size().0 as f32 / 540.0).clamp(1.0, 1.8);
            apply_text_scale(ctx, scale);

            // ResizableWindow adds a drag corner; dragging publishes a new size
            // via `request_resize`, so the host window tracks the editor and the
            // egui surface fills it (no black margins). Min size keeps every panel
            // reachable. Replaces a bare CentralPanel, which couldn't resize.
            ResizableWindow::new("ds-resize")
                .min_size(egui::vec2(360.0, 280.0))
                .show(ctx, &state.params.editor_state, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("DeepSteps");
                    ui.add_space(4.0);

                    step_grid(ui, &state.shared);
                    ui.add_space(8.0);

                    let p = &state.params;

                    egui::CollapsingHeader::new("Latent")
                        .default_open(true)
                        .show(ui, |ui| {
                            labeled(ui, scale, "A", |ui| ui.add(ParamSlider::for_param(&p.latent_a, setter)));
                            labeled(ui, scale, "B", |ui| ui.add(ParamSlider::for_param(&p.latent_b, setter)));
                            labeled(ui, scale, "C", |ui| ui.add(ParamSlider::for_param(&p.latent_c, setter)));
                            labeled(ui, scale, "D", |ui| ui.add(ParamSlider::for_param(&p.latent_d, setter)));
                        });

                    egui::CollapsingHeader::new("Timing")
                        .default_open(true)
                        .show(ui, |ui| {
                            labeled(ui, scale, "Gate", |ui| ui.add(ParamSlider::for_param(&p.gate, setter)));
                            labeled(ui, scale, "Substep", |ui| ui.add(ParamSlider::for_param(&p.substep_scale, setter)));
                            labeled(ui, scale, "Seq Len", |ui| ui.add(ParamSlider::for_param(&p.seq_len, setter)));
                        });

                    egui::CollapsingHeader::new("Tuning")
                        .default_open(true)
                        .show(ui, |ui| {
                            labeled(ui, scale, "Key", |ui| ui.add(ParamSlider::for_param(&p.key, setter)));
                            labeled(ui, scale, "Scale", |ui| ui.add(ParamSlider::for_param(&p.scale, setter)));
                        });

                    training_section(ui, setter, state);

                    egui::CollapsingHeader::new("Pitches")
                        .default_open(true)
                        .show(ui, |ui| {
                            // Reflow the 16 pitch sliders to the window width: each
                            // step is a label + slider pair (~150px), so the column
                            // count grows/shrinks with `available_width`.
                            let avail = ui.available_width();
                            let cols = ((avail / 150.0).floor() as usize).clamp(1, 8);
                            let slider_w = (avail / cols as f32 - 34.0).clamp(60.0, 160.0);
                            egui::Grid::new("pitch-grid").num_columns(cols * 2).show(ui, |ui| {
                                for (i, note) in p.notes.iter().enumerate() {
                                    ui.label(format!("{:>2}", i + 1));
                                    ui.add_sized(
                                        [slider_w, 18.0 * scale],
                                        ParamSlider::for_param(&note.pitch, setter),
                                    );
                                    if i % cols == cols - 1 {
                                        ui.end_row();
                                    }
                                }
                            });
                        });
                });
            });
        },
    )
}

/// The "Training" panel: build a dataset (capture live patterns or ingest audio
/// files), train an autoencoder on a background thread with a live progress bar,
/// and encode the current pattern back into the latent sliders. All heavy work
/// is dispatched via `exec.execute_background`; the audio thread is untouched.
fn training_section(ui: &mut egui::Ui, setter: &ParamSetter, state: &EditorState) {
    // Clone the shared handles out so the closures below only borrow `state` for
    // the editable `epochs`/`batch` fields (avoids overlapping borrows).
    let train = state.train.clone();
    let shared = state.shared.clone();
    let params = state.params.clone();
    let exec = state.exec.clone();

    egui::CollapsingHeader::new("Training")
        .default_open(false)
        .show(ui, |ui| {
            let status = train.status();
            let busy = matches!(status, TrainStatus::Running | TrainStatus::Ingesting);
            let n = train.dataset_len();

            ui.horizontal(|ui| {
                ui.label(format!("Dataset: {n}"));
                if ui.button("Capture pattern").clicked() {
                    let v = crate::dataset::encode_grid(shared.mask(), &shared.substeps());
                    if let Ok(mut d) = train.dataset.lock() {
                        d.push(v);
                    }
                }
                if ui.add_enabled(!busy, egui::Button::new("Add audio…")).clicked() {
                    // `pick_files()` is a blocking, modal native dialog: it stalls
                    // this GUI frame until the user dismisses it. That is fine —
                    // it blocks only the editor thread, never the audio thread —
                    // and the actual decode/onset work is dispatched to the
                    // background thread below.
                    if let Some(files) = rfd::FileDialog::new()
                        .add_filter("audio", &["wav", "flac"])
                        .pick_files()
                    {
                        if let Ok(mut q) = train.pending_paths.lock() {
                            q.extend(files);
                        }
                        exec.execute_background(Task::IngestAudio);
                    }
                }
                if ui.add_enabled(n > 0 && !busy, egui::Button::new("Clear")).clicked() {
                    if let Ok(mut d) = train.dataset.lock() {
                        d.clear();
                    }
                }
            });

            ui.horizontal(|ui| {
                // Atomics edited via a local copy, written back after the widget.
                let mut epochs = state.epochs.load(Relaxed);
                let mut batch = state.batch.load(Relaxed);
                ui.label("Epochs");
                if ui.add(egui::DragValue::new(&mut epochs).range(1..=5000)).changed() {
                    state.epochs.store(epochs, Relaxed);
                }
                ui.label("Batch");
                if ui.add(egui::DragValue::new(&mut batch).range(1..=512)).changed() {
                    state.batch.store(batch, Relaxed);
                }
            });

            ui.horizontal(|ui| {
                if ui
                    .add_enabled(n > 0 && !busy, egui::Button::new("Train"))
                    .clicked()
                {
                    exec.execute_background(Task::Train {
                        epochs: state.epochs.load(Relaxed),
                        batch: state.batch.load(Relaxed),
                        // Fixed seed -> reproducible training for a given dataset.
                        seed: 0x5_1EED,
                    });
                }
                if status == TrainStatus::Running
                    && ui.button("Cancel").clicked()
                {
                    train.cancel.store(true, Relaxed);
                }
            });

            if status == TrainStatus::Running {
                let e = train.epoch.load(Relaxed);
                let t = train.total_epochs.load(Relaxed).max(1);
                ui.add(
                    egui::ProgressBar::new(e as f32 / t as f32)
                        .text(format!("epoch {e}/{t}   loss {:.4}", train.last_loss())),
                );
            }

            ui.label(format!("Status: {}", status_text(status)));

            ui.horizontal(|ui| {
                let trained = train.has_trained_model();
                ui.label(format!(
                    "Model: {}",
                    if trained { "Trained" } else { "Default (baked)" }
                ));
                // The encoder's output is unbounded, but the latent params are
                // `[0,1]`, so `set_latents` clamps. Encoding then re-decoding a
                // pattern is therefore approximate, not a faithful round-trip —
                // flag it on hover so the result isn't surprising.
                if ui
                    .add_enabled(trained, egui::Button::new("Encode pattern → latent"))
                    .on_hover_text(
                        "Sets the latent sliders to this pattern's encoded latent.\n\
                         Values are clamped to 0..1, so re-decoding is approximate.",
                    )
                    .clicked()
                {
                    let grid = crate::dataset::encode_grid(shared.mask(), &shared.substeps());
                    let x: Vec<f64> = grid.iter().map(|&v| v as f64).collect();
                    if let Some(z) = train.encode(&x) {
                        set_latents(setter, &params, z);
                    }
                }
            });
        });
}

fn status_text(s: TrainStatus) -> &'static str {
    match s {
        TrainStatus::Idle => "idle",
        TrainStatus::Ingesting => "ingesting audio…",
        TrainStatus::Running => "training…",
        TrainStatus::Done => "done",
        TrainStatus::Cancelled => "cancelled",
        TrainStatus::Error => "error (empty dataset?)",
    }
}

/// Write a latent vector into the 4 latent params as a single automation gesture
/// each, clamped to the params' `[0,1]` range (the encoder output is unbounded).
fn set_latents(setter: &ParamSetter, p: &DeepStepsParams, z: [f64; 4]) {
    let targets = [&p.latent_a, &p.latent_b, &p.latent_c, &p.latent_d];
    for (param, &v) in targets.iter().zip(z.iter()) {
        let val = (v as f32).clamp(0.0, 1.0);
        setter.begin_set_parameter(*param);
        setter.set_parameter(*param, val);
        setter.end_set_parameter(*param);
    }
}

/// One labelled row: fixed-width label + the widget. Label box and row height
/// scale with `scale` so they keep pace with the scaled font size.
fn labeled(ui: &mut egui::Ui, scale: f32, label: &str, add: impl FnOnce(&mut egui::Ui) -> egui::Response) {
    ui.horizontal(|ui| {
        ui.add_sized([64.0 * scale, 18.0 * scale], egui::Label::new(label));
        add(ui);
    });
}

/// Rewrite the context's text styles to `scale`× a fixed baseline every frame.
/// Sizes are set absolutely (not multiplied in place) so repeated frames don't
/// compound. Font scaling is used instead of `Context::set_zoom_factor` so the
/// point/pixel ratio stays 1:1 and `ResizableWindow`'s drag corner keeps working.
fn apply_text_scale(ctx: &egui::Context, scale: f32) {
    use egui::{FontFamily::{Monospace, Proportional}, FontId, TextStyle};
    let styles = [
        (TextStyle::Small, FontId::new(9.0 * scale, Proportional)),
        (TextStyle::Body, FontId::new(14.0 * scale, Proportional)),
        (TextStyle::Button, FontId::new(14.0 * scale, Proportional)),
        (TextStyle::Heading, FontId::new(20.0 * scale, Proportional)),
        (TextStyle::Monospace, FontId::new(13.0 * scale, Monospace)),
    ];
    // Only restyle when the size actually changed (set_style triggers a relayout).
    let target: f32 = 14.0 * scale;
    let current = ctx
        .style()
        .text_styles
        .get(&TextStyle::Body)
        .map(|f| f.size)
        .unwrap_or(0.0);
    if (current - target).abs() > 0.01 {
        ctx.style_mut(|s| s.text_styles = styles.into_iter().collect());
    }
}

/// Draw the 16-step grid: filled cell = step on, red outline = playhead. A click
/// on a cell toggles it (overriding the decoder until the next regeneration).
fn step_grid(ui: &mut egui::Ui, shared: &SharedState) {
    const N: usize = 16;
    // Use the full available width so the grid widens with the window; cell size
    // is still clamped so cells never get tiny or absurdly large.
    let avail = ui.available_width();
    let gap = 4.0;
    let cell = ((avail - gap * (N as f32 - 1.0)) / N as f32).clamp(12.0, 40.0);

    let (resp, painter) = ui.allocate_painter(
        egui::vec2(N as f32 * cell + (N as f32 - 1.0) * gap, cell),
        egui::Sense::click(),
    );
    let origin = resp.rect.min;
    let current = shared.current();
    let clicked_pos = resp.clicked().then(|| resp.interact_pointer_pos()).flatten();

    for i in 0..N {
        let x = origin.x + i as f32 * (cell + gap);
        let rect = egui::Rect::from_min_size(egui::pos2(x, origin.y), egui::vec2(cell, cell));
        let fill = if shared.get(i) {
            egui::Color32::from_rgb(80, 180, 250)
        } else {
            egui::Color32::from_gray(60)
        };
        painter.rect_filled(rect, 4.0, fill);
        if current == i {
            painter.rect_stroke(
                rect,
                4.0,
                egui::Stroke::new(2.0, egui::Color32::from_rgb(250, 80, 80)),
                egui::StrokeKind::Inside,
            );
        }
        if let Some(pos) = clicked_pos {
            if rect.contains(pos) {
                shared.toggle(i);
            }
        }
    }
}
