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
}
