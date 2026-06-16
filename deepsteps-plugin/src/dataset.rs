//! Dataset construction for in-app training. A dataset is a `Vec<[f32; 32]>`,
//! each row the 32-dim representation the autoencoder trains on: 16 onset
//! one-hots + 16 substep timing offsets.
//!
//! `encode_onsets` is an exact port of `Deep_Steps_project/tools/corpus_encode.py`
//! (the audio-file path), pinned by the ported `test_corpus_encode.py` cases.
//! `encode_grid` captures the plugin's current step pattern (the user-pattern
//! path). Audio decoding + onset detection live in `dataset_audio` (added next).

const PER_QUARTER_NOTE: i64 = 48;
const SIXTEENTHS_DIV: i64 = 16; // bar_length = 1
const PPQN_PER_BAR: i64 = PER_QUARTER_NOTE * 4; // 192

/// Round-half-to-even, matching Python's built-in `round()` / `np.round`, so
/// the encoding agrees with the upstream pipeline on half-way onset positions.
fn banker_round(x: f64) -> f64 {
    let floor = x.floor();
    let diff = x - floor;
    if (diff - 0.5).abs() < 1e-9 {
        if (floor as i64).rem_euclid(2) == 0 {
            floor
        } else {
            floor + 1.0
        }
    } else {
        x.round()
    }
}

/// Encode onset sample positions within one bar to a 32-dim vector.
/// `onsets`: integer sample positions; `dur`: bar length in samples.
/// Port of `corpus_encode.encode_onsets`.
pub fn encode_onsets(onsets: &[i64], dur: i64) -> [f32; 32] {
    let mut out = [0.0f32; 32];
    let ppqn_timebase = banker_round(dur as f64 / PPQN_PER_BAR as f64) as i64;
    let sixteenths = banker_round(dur as f64 / SIXTEENTHS_DIV as f64) as i64;
    // Degenerate bar (dur too small): nothing to encode.
    if ppqn_timebase < 1 || sixteenths < 1 {
        return out;
    }

    // Round onsets to nearest 16th; drop onsets landing on the same step as the
    // immediately preceding (kept) onset.
    let mut onset_points_rounded: Vec<i64> = Vec::new();
    let mut kept: Vec<i64> = Vec::new();
    let mut previous: Option<i64> = None;
    for &onset in onsets {
        let r = banker_round(onset as f64 / sixteenths as f64) as i64;
        if Some(r) != previous {
            onset_points_rounded.push(r);
            previous = Some(r);
            kept.push(onset);
        }
        // else: duplicate step, dropped (not added to `kept`)
    }

    // One-hot over in-range steps.
    for &o in &onset_points_rounded {
        if (0..SIXTEENTHS_DIV).contains(&o) {
            out[o as usize] = 1.0;
        }
    }

    // Substep timing offsets for each kept onset, in the order they appear.
    let mut substeps: Vec<f32> = Vec::with_capacity(kept.len());
    for &o in &kept {
        let ppqn_onset = o.div_euclid(ppqn_timebase) * ppqn_timebase;
        let nearest = banker_round(o as f64 / sixteenths as f64) as i64 * sixteenths;
        let ss = (ppqn_onset - nearest).div_euclid(ppqn_timebase);
        substeps.push(((ss + 6) as f32) / 12.0);
    }

    // Reindex substeps back onto the 16-step grid (only where a step is on).
    let mut j = 0;
    for step in 0..16 {
        if out[step] == 1.0 && j < substeps.len() {
            out[16 + step] = substeps[j];
            j += 1;
        }
    }
    out
}

/// Encode the plugin's current step pattern into a 32-dim training sample.
/// `mask`: per-step on/off bits; `substeps`: the decoder's per-step substep
/// values (already in the `[0,1]` domain the model emits and trains on).
pub fn encode_grid(mask: u16, substeps: &[f64; 16]) -> [f32; 32] {
    let mut out = [0.0f32; 32];
    for i in 0..16 {
        if mask & (1 << i) != 0 {
            out[i] = 1.0;
            out[16 + i] = substeps[i] as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Ported verbatim from tools/test_corpus_encode.py.

    #[test]
    fn single_bar_onsets_to_32dim() {
        let dur = 192;
        let sixteenth = dur / 16; // 12
        let onsets: Vec<i64> = [0, 4, 8, 12].iter().map(|s| s * sixteenth).collect();
        let v = encode_onsets(&onsets, dur);
        for i in [0, 4, 8, 12] {
            assert_eq!(v[i], 1.0);
            assert!((v[16 + i] - 0.5).abs() < 1e-6, "on-grid substep should be 0.5");
        }
        let onhot_sum: f32 = v[..16].iter().sum();
        assert_eq!(onhot_sum, 4.0);
        assert_eq!(v[16 + 1], 0.0); // empty step carries substep 0
    }

    #[test]
    fn off_grid_substep_not_half() {
        let dur = 192;
        let onset = 4 * 12 + 5; // 53
        let v = encode_onsets(&[onset], dur);
        assert_eq!(v[4], 1.0);
        let onhot_sum: f32 = v[..16].iter().sum();
        assert_eq!(onhot_sum, 1.0);
        let expected = (5.0 + 6.0) / 12.0;
        assert!((expected - 0.5f32).abs() > 1e-6);
        assert!((v[16 + 4] - expected).abs() < 1e-6);
    }

    #[test]
    fn duplicate_step_dropped() {
        let dur = 192;
        let v = encode_onsets(&[4 * 12, 4 * 12 + 2], dur); // [48, 50] both round to step 4
        assert_eq!(v[4], 1.0);
        let onhot_sum: f32 = v[..16].iter().sum();
        assert_eq!(onhot_sum, 1.0);
    }

    #[test]
    fn out_of_range_step_excluded() {
        let dur = 192;
        let v = encode_onsets(&[0, 16 * 12], dur); // [0, 192]; 192 rounds to step 16, excluded
        assert_eq!(v[0], 1.0);
        let onhot_sum: f32 = v[..16].iter().sum();
        assert_eq!(onhot_sum, 1.0);
    }

    #[test]
    fn banker_round_matches_python() {
        assert_eq!(banker_round(0.5), 0.0);
        assert_eq!(banker_round(1.5), 2.0);
        assert_eq!(banker_round(2.5), 2.0);
        assert_eq!(banker_round(-2.5), -2.0);
        assert_eq!(banker_round(4.4167), 4.0);
    }

    #[test]
    fn grid_roundtrip() {
        let mask = (1 << 0) | (1 << 3) | (1 << 15);
        let mut substeps = [0.0f64; 16];
        substeps[0] = 0.5;
        substeps[3] = 0.9;
        substeps[15] = 0.25;
        let v = encode_grid(mask, &substeps);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[3], 1.0);
        assert_eq!(v[15], 1.0);
        assert_eq!(v[1], 0.0);
        assert!((v[16] - 0.5).abs() < 1e-6);
        assert!((v[16 + 3] - 0.9).abs() < 1e-6);
        assert!((v[16 + 15] - 0.25).abs() < 1e-6);
        // off steps carry no substep
        assert_eq!(v[16 + 1], 0.0);
    }
}
