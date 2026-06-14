//! nih-plug parameter definitions for the DeepSteps plugin.
//!
//! Mirrors the Stage-1 ofxGui widgets (see `Deep_Steps_project/src/ofApp.cpp`)
//! and `NOTES-sequencer.md`. The struct is not wired into a plugin yet
//! (Task 13/14); `dead_code` is therefore expected until then.
#![allow(dead_code)]

use nih_plug::prelude::*;
use std::sync::Arc;

/// Sequencer scale, mirroring `sequencer::Scale` (Task 9). Kept separate so the
/// nih-plug `Enum` derive (host automation IDs/names) stays decoupled from the
/// DSP enum; Task 13 maps between them.
#[derive(Enum, PartialEq, Eq, Clone, Copy, Debug)]
pub enum ScaleParam {
    #[id = "chromatic"]
    #[name = "Chromatic"]
    Chromatic,
    #[id = "pentmaj"]
    #[name = "Pentatonic Major"]
    PentMajor,
    #[id = "pentmin"]
    #[name = "Pentatonic Minor"]
    PentMinor,
    #[id = "major"]
    #[name = "Major"]
    Major,
    #[id = "natmin"]
    #[name = "Natural Minor"]
    NatMinor,
    #[id = "harmmin"]
    #[name = "Harmonic Minor"]
    HarmMinor,
    #[id = "melmin"]
    #[name = "Melodic Minor"]
    MelMinor,
    #[id = "dorian"]
    #[name = "Dorian"]
    Dorian,
    #[id = "phrygian"]
    #[name = "Phrygian"]
    Phrygian,
    #[id = "lydian"]
    #[name = "Lydian"]
    Lydian,
    #[id = "mixolydian"]
    #[name = "Mixolydian"]
    Mixolydian,
    #[id = "locrian"]
    #[name = "Locrian"]
    Locrian,
    #[id = "blues"]
    #[name = "Blues"]
    Blues,
    #[id = "wholetone"]
    #[name = "Whole Tone"]
    WholeTone,
}

/// A single sequencer step's note pitch. Used as an array of 16 nested params so
/// each gets a `_1`..`_16` id suffix from the `#[nested(array, ...)]` derive.
#[derive(Params)]
pub struct NoteParam {
    /// MIDI note number for this step.
    #[id = "pitch"]
    pub pitch: IntParam,
}

impl Default for NoteParam {
    fn default() -> Self {
        Self {
            // Stage-1 GUI default pitch is 50 (ofApp.cpp note sliders, and the
            // Stage-1 MIDI capture emitted pitch 50). Range clamped to 24..96
            // per the Stage-2 plan (GUI allowed 24..127).
            pitch: IntParam::new("Pitch", 50, IntRange::Linear { min: 24, max: 96 }),
        }
    }
}

/// Top-level plugin parameters.
#[derive(Params)]
pub struct DeepStepsParams {
    /// Latent dimension A driving the decoder (later host-MIDI-CC-mappable).
    #[id = "latentA"]
    pub latent_a: FloatParam,
    /// Latent dimension B driving the decoder.
    #[id = "latentB"]
    pub latent_b: FloatParam,
    /// Latent dimension C driving the decoder.
    #[id = "latentC"]
    pub latent_c: FloatParam,
    /// Latent dimension D driving the decoder.
    #[id = "latentD"]
    pub latent_d: FloatParam,

    /// Gate length in milliseconds (NOTES: Gate-Length slider, absolute ms).
    #[id = "gate"]
    pub gate: FloatParam,
    /// Substep scaling in pulses (NOTES: 0..6; 0 = no timing offset).
    #[id = "substep"]
    pub substep_scale: FloatParam,

    /// Active sequence length in steps.
    #[id = "seqlen"]
    pub seq_len: IntParam,
    /// Musical key (0 = C .. 11 = B).
    #[id = "key"]
    pub key: IntParam,
    /// Scale used to quantise decoded pitches.
    #[id = "scale"]
    pub scale: EnumParam<ScaleParam>,

    /// Per-step note pitches (16 steps).
    #[nested(array, group = "Notes")]
    pub notes: [NoteParam; 16],
}

impl Default for DeepStepsParams {
    fn default() -> Self {
        // The latent params share an identical definition. Plan spec: range
        // 0.0..1.0, default 0.5 (the decoder expects normalised latents; the
        // Stage-1 GUI sliders used -10..10/default 0, but the plugin normalises
        // here and Task 13 will rescale as needed).
        let latent = |name: &'static str| {
            FloatParam::new(name, 0.5, FloatRange::Linear { min: 0.0, max: 1.0 })
        };

        Self {
            latent_a: latent("Latent A"),
            latent_b: latent("Latent B"),
            latent_c: latent("Latent C"),
            latent_d: latent("Latent D"),

            // ofApp.cpp: gateLength range 1..1000 ms, default 100.
            gate: FloatParam::new("Gate", 100.0, FloatRange::Linear { min: 1.0, max: 1000.0 })
                .with_unit(" ms"),
            // ofApp.cpp: substepsScale range 0..6, default 0.
            substep_scale: FloatParam::new(
                "Substep Scale",
                0.0,
                FloatRange::Linear { min: 0.0, max: 6.0 },
            ),

            // ofApp.cpp: seqLength range 1..16, default 16.
            seq_len: IntParam::new("Seq Length", 16, IntRange::Linear { min: 1, max: 16 }),
            key: IntParam::new("Key", 0, IntRange::Linear { min: 0, max: 11 }),
            scale: EnumParam::new("Scale", ScaleParam::PentMinor),

            notes: std::array::from_fn(|_| NoteParam::default()),
        }
    }
}

/// Marker to silence the unused-`Arc` import if this module is compiled before
/// the plugin wraps the params in an `Arc` (Task 13).
#[allow(dead_code)]
type _ParamsArc = Arc<DeepStepsParams>;
