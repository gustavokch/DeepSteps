//! Lock-free state shared between the audio thread and the egui editor.
//!
//! The step on/off pattern and the playhead index are runtime state (not
//! nih-plug params), so the editor cannot read them through the param system.
//! `SharedState` is wrapped in an `Arc`, held by the plugin, and cloned into the
//! editor; both sides touch it through atomics with `Relaxed` ordering — the
//! values are independent (a display mask and a playhead index) with no
//! cross-variable invariant to protect.

use std::sync::atomic::{AtomicU16, AtomicUsize, Ordering};

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
}

impl Default for SharedState {
    fn default() -> Self {
        Self {
            steps: AtomicU16::new(0),
            current_step: AtomicUsize::new(NO_STEP),
        }
    }
}

impl SharedState {
    /// Whether step `i` (0..16) is currently on.
    pub fn get(&self, i: usize) -> bool {
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

    /// Flip step `i` (0..16) — used by a user grid click. Persists until the next
    /// regeneration overwrites the mask.
    pub fn toggle(&self, i: usize) {
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
