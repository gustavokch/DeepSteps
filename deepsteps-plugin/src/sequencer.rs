//! Sequencer: pitch quantization (Task 9), step advance (Task 10), note scheduling (Task 11).
//!
//! Mirrors the Pd `pd quant` subpatch. See `NOTES-sequencer.md` §4.

/// Scale selector for pitch quantization. Order/tables match the Pd `sel` boxes
/// in `pd quant` (NOTES-sequencer.md §4).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Scale {
    Chromatic,
    PentMajor,
    PentMinor,
}

impl Scale {
    fn table(self) -> &'static [i32] {
        match self {
            Scale::Chromatic => &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            Scale::PentMajor => &[0, 2, 4, 7, 9],
            Scale::PentMinor => &[0, 3, 5, 7, 10],
        }
    }
}

/// Snap a raw MIDI note's pitch-class DOWN to the nearest scale member
/// (largest table entry <= pitch-class), preserve octave, then add `key` semitones.
/// Mirrors the Pd patch (NOTES-sequencer.md §4: snap pitch class DOWN, keep octave,
/// then add key 0..11; chromatic = identity + key).
pub fn quantize(note: i32, scale: Scale, key: i32) -> i32 {
    let pc = ((note % 12) + 12) % 12;
    let octave = note - pc;
    let table = scale.table();
    let snapped = table
        .iter()
        .rev()
        .copied()
        .find(|&d| d <= pc)
        .unwrap_or(table[0]);
    octave + snapped + key
}

/// Steps per beat: 16 steps fill a 4-beat bar (NOTES-sequencer.md: `mod 192` = 16*12 pulses).
pub const STEPS_PER_BEAT: f64 = 4.0;
/// One step is 0.25 beat.
pub const STEP_BEATS: f64 = 1.0 / STEPS_PER_BEAT;

/// Wrapped step index (0..seq_len) currently sounding at `beat`.
pub fn step_at_beat(beat: f64, seq_len: usize) -> usize {
    let idx = (beat * STEPS_PER_BEAT).floor() as i64;
    idx.rem_euclid(seq_len as i64) as usize
}

/// Absolute (pre-wrap) step indices whose onset (`index * STEP_BEATS`) lies in the
/// half-open interval `[start_beat, end_beat)`. The caller wraps via `step_at_beat`
/// or `% seq_len`. Onsets are exact 0.25-beat multiples, so plain ceil/compare is exact.
pub fn steps_in_range(start_beat: f64, end_beat: f64, _seq_len: usize) -> Vec<i64> {
    let first = (start_beat / STEP_BEATS).ceil() as i64;
    let mut out = Vec::new();
    let mut s = first;
    while (s as f64) * STEP_BEATS < end_beat {
        out.push(s);
        s += 1;
    }
    out
}

pub const PULSES_PER_BEAT: f64 = 48.0; // 48 PPQN (NOTES-sequencer.md §1 line 82, §6).

/// Substep value in [0,1] -> timing offset in beats.
/// offset = substep_scale * (2*substep - 1) pulses, / 48 PPQN. substep 0.5 -> 0.
/// `substep_scale` is the substep-scale slider in pulses (range 0..6 per NOTES §3
/// lines 133,145–146; `expr $f1*($f2-$f3)+$f3` with ±S bounds).
///
/// NOTE (VERIFY in Task 16, NOTES §3 line 141 / §6 "Needs A/B"): the Pd patch
/// truncates this offset to whole pulses (`int`) and matches a discrete clock pulse.
/// We use a continuous beat offset instead — an intentional approximation that
/// host-transport timing affords; to be A/B-checked later.
pub fn substep_offset_beats(substep: f64, substep_scale: f64) -> f64 {
    substep_scale * (2.0 * substep - 1.0) / PULSES_PER_BEAT
}

pub struct StepEvent {
    pub note: i32,
    pub vel: u8,
    pub on_beat: f64,
    pub gate_ms: f64, // absolute ms; audio layer converts to a NoteOff sample offset (Task 13).
}

/// Schedule one active step: apply the substep timing offset to the step onset,
/// carry the note, fixed velocity (100, NOTES §5 line 207), and absolute gate (ms,
/// NOTES §5 lines 222–224). Mirrors the Pd patch (substep lerp + Gate-Length slider).
pub fn schedule_step(
    step_onset_beat: f64,
    substep: f64,
    substep_scale: f64,
    gate_ms: f64,
    note: i32,
    vel: u8,
) -> StepEvent {
    StepEvent {
        note,
        vel,
        on_beat: step_onset_beat + substep_offset_beats(substep, substep_scale),
        gate_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substep_offset_is_zero_on_grid() {
        assert!((substep_offset_beats(0.5, 6.0)).abs() < 1e-12);
    }
    #[test]
    fn substep_offset_scales_and_signs() {
        // S=6 pulses, substep 1.0 -> +6 pulses = +6/48 beat = +0.125
        assert!((substep_offset_beats(1.0, 6.0) - 0.125).abs() < 1e-12);
        // substep 0.0 -> -6 pulses = -0.125
        assert!((substep_offset_beats(0.0, 6.0) + 0.125).abs() < 1e-12);
        // S=4, substep 0.75 -> 4*(0.5)=2 pulses = 2/48
        assert!((substep_offset_beats(0.75, 4.0) - (2.0 / 48.0)).abs() < 1e-12);
    }
    #[test]
    fn schedule_applies_offset_and_carries_gate_ms() {
        let ev = schedule_step(1.0, 0.5, 6.0, 100.0, 60, 100);
        assert_eq!(ev.note, 60);
        assert_eq!(ev.vel, 100);
        assert!((ev.on_beat - 1.0).abs() < 1e-12); // substep 0.5 -> no offset
        assert_eq!(ev.gate_ms, 100.0);
        // off-grid substep shifts on_beat
        let ev2 = schedule_step(1.0, 1.0, 6.0, 50.0, 64, 100);
        assert!((ev2.on_beat - 1.125).abs() < 1e-12);
    }

    #[test]
    fn snaps_pitch_class_down_to_scale_member() {
        // pent-minor [0,3,5,7,10], key=0, octave base 60
        assert_eq!(quantize(60, Scale::PentMinor, 0), 60); // pc0 in scale
        assert_eq!(quantize(62, Scale::PentMinor, 0), 60); // pc2 -> down to 0
        assert_eq!(quantize(64, Scale::PentMinor, 0), 63); // pc4 -> down to 3
        assert_eq!(quantize(71, Scale::PentMinor, 0), 70); // pc11 -> down to 10
    }

    #[test]
    fn chromatic_is_identity() {
        for n in 48..=72 {
            assert_eq!(quantize(n, Scale::Chromatic, 0), n);
        }
    }

    #[test]
    fn key_offset_adds_semitones() {
        assert_eq!(quantize(60, Scale::PentMinor, 2), 62); // 60 snapped(0) + key2
    }

    #[test]
    fn pent_major_snaps_down() {
        // [0,2,4,7,9]: pc5 -> 4, pc6 -> 4, pc11 -> 9
        assert_eq!(quantize(65, Scale::PentMajor, 0), 64);
        assert_eq!(quantize(66, Scale::PentMajor, 0), 64);
        assert_eq!(quantize(71, Scale::PentMajor, 0), 69);
    }

    #[test]
    fn steps_per_beat_and_boundary_crossing() {
        // 4 steps per beat; step = 0.25 beat. Block [0.0, 0.30) contains step onsets at 0.0 and 0.25.
        assert_eq!(steps_in_range(0.0, 0.30, 16), vec![0, 1]);
    }

    #[test]
    fn step_index_wraps_mod_seq_length() {
        assert_eq!(step_at_beat(4.25, 16), 1); // 4.25 beats = 17th step -> mod 16 = 1
    }

    #[test]
    fn range_is_half_open() {
        // a step onset exactly at end is excluded; exactly at start is included
        assert_eq!(steps_in_range(0.0, 0.25, 16), vec![0]); // 0.25 excluded
        assert_eq!(steps_in_range(0.25, 0.50, 16), vec![1]); // 0.25 included, 0.50 excluded
    }

    #[test]
    fn empty_when_no_boundary_in_range() {
        assert_eq!(steps_in_range(0.05, 0.24, 16), Vec::<i64>::new());
    }
}
