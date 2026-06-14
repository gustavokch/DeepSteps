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

#[cfg(test)]
mod tests {
    use super::*;

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
