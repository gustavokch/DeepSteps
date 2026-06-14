pub mod decoder;
pub mod editor;
pub mod params;
pub mod sequencer;
pub mod shared;

use nih_plug::prelude::*;
use std::sync::Arc;

use decoder::Decoder;
use params::{DeepStepsParams, ScaleParam};
use sequencer::{quantize, schedule_step, steps_in_range, Scale, STEP_BEATS};
use shared::{pack, SharedState, NO_STEP};

/// A NoteOff scheduled to fire in a future process block. `remaining` is the
/// sample count from the *current* block's start until the NoteOff should fire;
/// it is decremented by `nframes` each block until it lands inside a block.
struct PendingOff {
    note: u8,
    remaining: i64,
}

struct DeepSteps {
    params: Arc<DeepStepsParams>,
    decoder: Decoder,
    /// Last latent vector that produced `steps`/`substeps`. Initialised to NaN so
    /// the first `maybe_regen` always regenerates (NaN != anything).
    last_latent: [f64; 4],
    /// Per-step timing offsets from the decoder (not exposed in the GUI). The
    /// on/off pattern itself lives in `shared.steps` so the editor can read and
    /// toggle it.
    substeps: [f64; 16],
    /// Audio<->GUI shared state: step on/off mask + playhead index.
    shared: Arc<SharedState>,
    pending: Vec<PendingOff>,
    sample_rate: f32,
}

impl Default for DeepSteps {
    fn default() -> Self {
        let decoder = match Decoder::from_json_str(include_str!("../weights/decoder.json")) {
            Ok(d) => d,
            Err(e) => {
                nih_log!("DeepSteps: bad weights, using empty decoder: {e}");
                Decoder::empty()
            }
        };
        Self {
            params: Arc::new(DeepStepsParams::default()),
            decoder,
            last_latent: [f64::NAN; 4],
            substeps: [0.0; 16],
            shared: Arc::new(SharedState::default()),
            pending: Vec::new(),
            sample_rate: 44100.0,
        }
    }
}

impl DeepSteps {
    fn map_scale(s: ScaleParam) -> Scale {
        match s {
            ScaleParam::Chromatic => Scale::Chromatic,
            ScaleParam::PentMajor => Scale::PentMajor,
            ScaleParam::PentMinor => Scale::PentMinor,
            ScaleParam::Major => Scale::Major,
            ScaleParam::NatMinor => Scale::NatMinor,
            ScaleParam::HarmMinor => Scale::HarmMinor,
            ScaleParam::MelMinor => Scale::MelMinor,
            ScaleParam::Dorian => Scale::Dorian,
            ScaleParam::Phrygian => Scale::Phrygian,
            ScaleParam::Lydian => Scale::Lydian,
            ScaleParam::Mixolydian => Scale::Mixolydian,
            ScaleParam::Locrian => Scale::Locrian,
            ScaleParam::Blues => Scale::Blues,
            ScaleParam::WholeTone => Scale::WholeTone,
        }
    }

    /// Re-run the decoder iff the latent params changed since last call.
    fn maybe_regen(&mut self) {
        let p = &self.params;
        let z = [
            p.latent_a.value() as f64,
            p.latent_b.value() as f64,
            p.latent_c.value() as f64,
            p.latent_d.value() as f64,
        ];
        if z != self.last_latent {
            let (s, ss) = self.decoder.generate(&z);
            // Publish the freshly-decoded pattern as the playback source of
            // truth. This overwrites any user grid toggles — moving a latent
            // regenerates, which is the intended generative behaviour.
            self.shared.set_mask(pack(&s));
            self.substeps = ss;
            self.last_latent = z;
        }
    }
}

impl Plugin for DeepSteps {
    const NAME: &'static str = "DeepSteps";
    const VENDOR: &'static str = "DeepSteps";
    const URL: &'static str = "https://github.com/gustavokch/DeepSteps";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // MIDI-only plugin: no audio I/O. An empty layout slice is the idiom used by
    // the bundled `midi_inverter` example at this rev.
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[];

    // Accept note input (Basic) as well as output. The process loop currently
    // ignores incoming events, but a coherent note IO config is required: host
    // MIDI CC -> latent mapping is a planned feature, and clap-validator 0.3.2
    // has a bug where its output-note-port query passes `is_input=true`
    // (src/plugin/ext/note_ports.rs:115), so an output-only note config makes
    // the note-ports query fail. Advertising an input port makes the query
    // succeed legitimately and matches the bundled `midi_inverter` example.
    const MIDI_INPUT: MidiConfig = MidiConfig::Basic;
    const MIDI_OUTPUT: MidiConfig = MidiConfig::Basic;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        editor::create(self.params.clone(), self.shared.clone())
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.sample_rate = buffer_config.sample_rate;
        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        self.maybe_regen();

        let nframes = buffer.samples() as i64;
        let transport = context.transport();

        // Not playing: flush any pending NoteOffs so notes don't hang, then bail.
        if !transport.playing {
            self.shared.set_current(NO_STEP);
            for po in self.pending.drain(..) {
                context.send_event(NoteEvent::NoteOff {
                    timing: 0,
                    voice_id: None,
                    // channel: 0 == MIDI channel 1 (nih-plug is 0-indexed);
                    // matches the Pd patch's channel 1 per NOTES-sequencer.md.
                    channel: 0,
                    note: po.note,
                    velocity: 0.0,
                });
            }
            return ProcessStatus::Normal;
        }

        // Need tempo + position to map beats to samples; otherwise stay silent.
        let (tempo, pos_beats) = match (transport.tempo, transport.pos_beats()) {
            (Some(tp), Some(pb)) => (tp, pb),
            _ => return ProcessStatus::Normal,
        };

        // Drain pending NoteOffs (for notes started in earlier blocks) BEFORE
        // stepping, so a same-block NoteOff can never precede its own NoteOn.
        let mut still = Vec::with_capacity(self.pending.len());
        for mut po in self.pending.drain(..) {
            if po.remaining < nframes {
                context.send_event(NoteEvent::NoteOff {
                    timing: po.remaining.max(0) as u32,
                    voice_id: None,
                    // channel: 0 == MIDI channel 1 (nih-plug is 0-indexed).
                    channel: 0,
                    note: po.note,
                    velocity: 0.0,
                });
            } else {
                po.remaining -= nframes;
                still.push(po);
            }
        }
        self.pending = still;

        let beats_per_sample = tempo / 60.0 / self.sample_rate as f64;
        let block_end = pos_beats + nframes as f64 * beats_per_sample;
        let seq_len = self.params.seq_len.value() as usize;
        let gate_ms = self.params.gate.value() as f64;
        let substep_scale = self.params.substep_scale.value() as f64;
        let scale = Self::map_scale(self.params.scale.value());
        let key = self.params.key.value();

        for abs_step in steps_in_range(pos_beats, block_end, seq_len) {
            let idx = abs_step.rem_euclid(seq_len as i64) as usize;
            // Publish the playhead so the editor highlights the active step,
            // whether or not this step is on. Last-write-wins within a block: if a
            // block spans multiple steps the editor only sees the final one. Fine
            // at normal block sizes (<=1 step/block at 16 steps/bar); revisit if a
            // host uses very large blocks.
            self.shared.set_current(idx);
            if !self.shared.get(idx) {
                continue;
            }
            let onset_beat = abs_step as f64 * STEP_BEATS;
            let raw = self.params.notes[idx].pitch.value();
            let note = quantize(raw, scale, key).clamp(0, 127) as u8;
            let ev = schedule_step(
                onset_beat,
                self.substeps[idx],
                substep_scale,
                gate_ms,
                note as i32,
                100,
            );

            // NoteOff duration sourced from the scheduled event's gate_ms.
            let dur_samples = (ev.gate_ms / 1000.0 * self.sample_rate as f64).round() as i64;

            // NoteOn sample offset within this block, clamped into range.
            let on_smp = (((ev.on_beat - pos_beats) / beats_per_sample).round() as i64)
                .clamp(0, nframes - 1);
            context.send_event(NoteEvent::NoteOn {
                timing: on_smp as u32,
                voice_id: None,
                // channel: 0 == MIDI channel 1 (nih-plug is 0-indexed);
                // matches the Pd patch's channel 1 per NOTES-sequencer.md.
                channel: 0,
                note,
                velocity: ev.vel as f32 / 127.0,
            });

            // Matching NoteOff: emit now if it lands in this block, else defer.
            let off_abs = on_smp + dur_samples;
            if off_abs < nframes {
                context.send_event(NoteEvent::NoteOff {
                    timing: off_abs as u32,
                    voice_id: None,
                    // channel: 0 == MIDI channel 1 (nih-plug is 0-indexed).
                    channel: 0,
                    note,
                    velocity: 0.0,
                });
            } else {
                self.pending.push(PendingOff {
                    note,
                    remaining: off_abs - nframes,
                });
            }
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for DeepSteps {
    const CLAP_ID: &'static str = "dev.gruber.deepsteps";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("Autoencoder MIDI step sequencer");
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[ClapFeature::NoteEffect, ClapFeature::Utility];
}

impl Vst3Plugin for DeepSteps {
    // 16 bytes: D e e p S t e p s G e n M i d i.
    const VST3_CLASS_ID: [u8; 16] = *b"DeepStepsGenMidi";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Instrument, Vst3SubCategory::Tools];
}

nih_export_clap!(DeepSteps);
nih_export_vst3!(DeepSteps);
