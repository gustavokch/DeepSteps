//! Custom egui editor for DeepSteps.
//!
//! Restores the Stage-1 UX: a 16-step grid with a live playhead and click-to-
//! toggle cells, plus sliders for the latent vector, per-step pitches, timing,
//! and tuning. The step on/off pattern is owned by the audio thread via
//! [`SharedState`]; everything else is a nih-plug param edited through the
//! `ParamSetter`.

use std::sync::Arc;

use nih_plug::prelude::Editor;
use nih_plug_egui::{create_egui_editor, egui, resizable_window::ResizableWindow, widgets::ParamSlider};

use crate::params::DeepStepsParams;
use crate::shared::{SharedState, NO_STEP};

/// State handed to every egui frame.
struct EditorState {
    params: Arc<DeepStepsParams>,
    shared: Arc<SharedState>,
}

pub fn create(
    params: Arc<DeepStepsParams>,
    shared: Arc<SharedState>,
) -> Option<Box<dyn Editor>> {
    let egui_state = params.editor_state.clone();
    create_egui_editor(
        egui_state,
        EditorState { params, shared },
        |_ctx, _state| {},
        |ctx, setter, state| {
            // Keep the playhead animating while the transport plays. When stopped
            // (current == NO_STEP) let egui idle instead of spinning at full
            // framerate — a grid click still triggers its own repaint.
            if state.shared.current() != NO_STEP {
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
