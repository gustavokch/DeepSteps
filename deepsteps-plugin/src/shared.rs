//! Lock-free state shared between the audio thread and the egui editor.
//!
//! The step on/off pattern and the playhead index are runtime state (not
//! nih-plug params), so the editor cannot read them through the param system.
//! `SharedState` is wrapped in an `Arc`, held by the plugin, and cloned into the
//! editor; both sides touch it through atomics with `Relaxed` ordering — the
//! values are independent (a display mask and a playhead index) with no
//! cross-variable invariant to protect.

use std::sync::atomic::{AtomicU16, AtomicU64, AtomicUsize, Ordering};

/// Playhead sentinel meaning "no step is currently playing" (transport stopped).
pub const NO_STEP: usize = usize::MAX;

pub struct SharedState {
    /// Effective per-step on/off mask: bit `i` set => step `i` plays. Written by
    /// the decoder on regeneration and flipped by user grid clicks; read by both
    /// playback (`process`) and the editor's grid.
    pub steps: AtomicU16,
    /// Index of the step currently playing, for the editor's playhead highlight,
    /// or [`NO_STEP`] when the transport is stopped.
    pub current_step: AtomicUsize,
    /// Per-step substep timing offsets (the decoder's raw `[0,1]` outputs), as
    /// `f64::to_bits`. Written by the decoder on regeneration; read by the editor
    /// so the capture/encode features can rebuild a full 32-dim sample from the
    /// live pattern. Display/training only — never read by playback.
    pub substeps: [AtomicU64; 16],
}

impl Default for SharedState {
    fn default() -> Self {
        Self {
            steps: AtomicU16::new(0),
            current_step: AtomicUsize::new(NO_STEP),
            substeps: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
}

impl SharedState {
    /// Whether step `i` is currently on. `i` must be in `0..16` — `1 << i` on the
    /// `u16` mask is unsound otherwise.
    pub fn get(&self, i: usize) -> bool {
        debug_assert!(i < 16, "step index {i} out of range 0..16");
        self.steps.load(Ordering::Relaxed) & (1 << i) != 0
    }

    /// Replace the whole on/off mask (used by the decoder on regeneration).
    pub fn set_mask(&self, mask: u16) {
        self.steps.store(mask, Ordering::Relaxed);
    }

    /// Read the whole on/off mask.
    pub fn mask(&self) -> u16 {
        self.steps.load(Ordering::Relaxed)
    }

    /// Flip step `i` — used by a user grid click. `i` must be in `0..16`.
    /// Persists until the next regeneration overwrites the mask.
    pub fn toggle(&self, i: usize) {
        debug_assert!(i < 16, "step index {i} out of range 0..16");
        self.steps.fetch_xor(1 << i, Ordering::Relaxed);
    }

    /// Publish the currently-playing step index for the playhead highlight.
    pub fn set_current(&self, idx: usize) {
        self.current_step.store(idx, Ordering::Relaxed);
    }

    /// Read the currently-playing step index (or [`NO_STEP`]).
    pub fn current(&self) -> usize {
        self.current_step.load(Ordering::Relaxed)
    }

    /// Publish the decoder's per-step substep offsets (called on regeneration).
    pub fn set_substeps(&self, ss: &[f64; 16]) {
        for (a, &v) in self.substeps.iter().zip(ss.iter()) {
            a.store(v.to_bits(), Ordering::Relaxed);
        }
    }

    /// Read the per-step substep offsets for pattern capture / encoding.
    pub fn substeps(&self) -> [f64; 16] {
        std::array::from_fn(|i| f64::from_bits(self.substeps[i].load(Ordering::Relaxed)))
    }
}

/// Pack a 16-element bool pattern into a `u16` mask (bit `i` = `pattern[i]`).
pub fn pack(pattern: &[bool; 16]) -> u16 {
    let mut m = 0u16;
    for (i, &on) in pattern.iter().enumerate() {
        if on {
            m |= 1 << i;
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_roundtrip_and_toggle() {
        let s = SharedState::default();
        assert_eq!(s.mask(), 0);
        assert_eq!(s.current(), NO_STEP);

        let mut pat = [false; 16];
        pat[0] = true;
        pat[3] = true;
        pat[15] = true;
        s.set_mask(pack(&pat));

        assert!(s.get(0));
        assert!(!s.get(1));
        assert!(s.get(3));
        assert!(s.get(15));

        s.toggle(1);
        assert!(s.get(1));
        s.toggle(0);
        assert!(!s.get(0));

        s.set_current(7);
        assert_eq!(s.current(), 7);
    }

    #[test]
    fn pack_matches_bits() {
        let mut pat = [false; 16];
        pat[2] = true;
        pat[5] = true;
        assert_eq!(pack(&pat), (1 << 2) | (1 << 5));
    }
}
