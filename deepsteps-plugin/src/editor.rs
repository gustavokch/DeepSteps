//! Custom egui editor for DeepSteps.
//!
//! Restores the Stage-1 UX: a 16-step grid with a live playhead and click-to-
//! toggle cells, plus sliders for the latent vector, per-step pitches, timing,
//! and tuning. The step on/off pattern is owned by the audio thread via
//! [`SharedState`]; everything else is a nih-plug param edited through the
//! `ParamSetter`.

use std::sync::Arc;

use nih_plug::prelude::Editor;
use nih_plug_egui::{create_egui_editor, egui, widgets::ParamSlider};

use crate::params::DeepStepsParams;
use crate::shared::SharedState;

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
            // Keep the playhead animating while the transport plays.
            ctx.request_repaint();

            egui::CentralPanel::default().show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("DeepSteps");
                    ui.add_space(4.0);

                    step_grid(ui, &state.shared);
                    ui.add_space(8.0);

                    let p = &state.params;

                    egui::CollapsingHeader::new("Latent")
                        .default_open(true)
                        .show(ui, |ui| {
                            labeled(ui, "A", |ui| ui.add(ParamSlider::for_param(&p.latent_a, setter)));
                            labeled(ui, "B", |ui| ui.add(ParamSlider::for_param(&p.latent_b, setter)));
                            labeled(ui, "C", |ui| ui.add(ParamSlider::for_param(&p.latent_c, setter)));
                            labeled(ui, "D", |ui| ui.add(ParamSlider::for_param(&p.latent_d, setter)));
                        });

                    egui::CollapsingHeader::new("Timing")
                        .default_open(true)
                        .show(ui, |ui| {
                            labeled(ui, "Gate", |ui| ui.add(ParamSlider::for_param(&p.gate, setter)));
                            labeled(ui, "Substep", |ui| ui.add(ParamSlider::for_param(&p.substep_scale, setter)));
                            labeled(ui, "Seq Len", |ui| ui.add(ParamSlider::for_param(&p.seq_len, setter)));
                        });

                    egui::CollapsingHeader::new("Tuning")
                        .default_open(true)
                        .show(ui, |ui| {
                            labeled(ui, "Key", |ui| ui.add(ParamSlider::for_param(&p.key, setter)));
                            labeled(ui, "Scale", |ui| ui.add(ParamSlider::for_param(&p.scale, setter)));
                        });

                    egui::CollapsingHeader::new("Pitches")
                        .default_open(true)
                        .show(ui, |ui| {
                            // 4 columns of (label + slider) => 4 rows of 4 steps,
                            // keeping the 16 pitches compact.
                            egui::Grid::new("pitch-grid").num_columns(8).show(ui, |ui| {
                                for (i, note) in p.notes.iter().enumerate() {
                                    ui.label(format!("{:>2}", i + 1));
                                    ui.add_sized(
                                        [90.0, 18.0],
                                        ParamSlider::for_param(&note.pitch, setter),
                                    );
                                    if i % 4 == 3 {
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

/// One labelled row: fixed-width label + the widget.
fn labeled(ui: &mut egui::Ui, label: &str, add: impl FnOnce(&mut egui::Ui) -> egui::Response) {
    ui.horizontal(|ui| {
        ui.add_sized([64.0, 18.0], egui::Label::new(label));
        add(ui);
    });
}

/// Draw the 16-step grid: filled cell = step on, red outline = playhead. A click
/// on a cell toggles it (overriding the decoder until the next regeneration).
fn step_grid(ui: &mut egui::Ui, shared: &SharedState) {
    const N: usize = 16;
    let avail = ui.available_width().min(576.0);
    let gap = 4.0;
    let cell = ((avail - gap * (N as f32 - 1.0)) / N as f32).clamp(12.0, 32.0);

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
