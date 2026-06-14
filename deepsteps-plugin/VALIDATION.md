# DeepSteps plugin — validation

Stage 2 Phase E. Automated checks are done in-session; the host/interactive checks
below are run by hand (they need a running DAW, ALSA sequencer wiring, and — for the
A/B — the Stage 1 X11 app).

## 1. Automated (done)

- **cargo test** — `cd deepsteps-plugin && cargo test` → 12 passing.
  - Decoder forward pass matches the Python reference vectors to ~1e-16.
  - Sequencer: scale quantize (snap-down), beat→step mapping, substep offset, gate.
- **clap-validator** — `clap-validator validate target/bundled/deepsteps-plugin.clap`
  → **18 passed, 0 failed, 3 skipped**. The 3 skips are `preset-discovery-*`
  (not implemented; expected for a MIDI plugin).

## 2. Carla smoke test (manual)

Build the bundles first: `cd deepsteps-plugin && cargo xtask bundle deepsteps-plugin --release`
→ `target/bundled/deepsteps-plugin.{clap,vst3}`.

Point Carla at the bundle dir (Settings → Plugin Paths → CLAP), or load the file
directly. Then:

1. Add **DeepSteps** (CLAP) to the rack.
2. Wire its MIDI out to a monitor: in a terminal, `aseqdump -l` to find Carla's MIDI
   out port, then `aseqdump -p <client:port>`. (Or route it into a synth to hear it.)
3. Press **Play** on Carla's transport.
4. **Expect:** a stream of note-on/note-off in `aseqdump`, advancing 4 steps per beat
   (16 per bar), velocity 100, channel 1.
5. Turn the **latent A–D** params — the pattern (which steps fire) should change.
6. Turn **key** / **scale** — pitches should shift / re-quantize (snap-down). The
   **scale** dropdown has 14 options: Chromatic, Pentatonic Major, Pentatonic Minor
   (the original Pd patch's three), plus Stage-2 additions Major, Natural Minor,
   Harmonic Minor, Melodic Minor, Dorian, Phrygian, Lydian, Mixolydian, Locrian,
   Blues, Whole Tone. Every scale snaps each pitch DOWN to the nearest scale member.
7. **seq length** < 16 should shorten the loop; **gate** should change note lengths;
   **substep** > 0 should nudge note timing off the grid.
8. Press **Stop** — no hung notes (all pending note-offs flush).

## 3. A/B vs Stage 1 (manual) — read the caveat

**Caveat:** the Stage 1 standalone constructs the autoencoder with **random-init
weights every launch** (it never loads a trained model), so its *generated pattern*
is not reproducible and does **not** match the Stage 2 plugin's pattern (which uses
the committed, offline-trained `weights/decoder.json`). A note-for-note A/B of the
*decoder output* is therefore not meaningful. What A/B can validate is the
**sequencer behavior** — timing, step advance, scale quantization, gate — given a
pattern.

Practical A/B:

1. **Decoder** is already A/B'd numerically: the Rust forward pass matches the Python
   decoder to ~1e-16 (`reference_vectors.json`), and the Python path is bit-faithful
   to `AE_init.py`. No host needed.
2. **Sequencer timing/quantization** — compare behavior, not exact notes:
   - Stage 1: run `Deep_Steps_project/tools/run_midi_test.sh` (X11 + ALSA) to capture
     its MIDI out for a known clock; note step spacing, velocity (100), channel, and
     how pitches snap to the selected scale.
   - Stage 2: capture the plugin's MIDI out in Carla at the same tempo with the same
     scale/key/notes.
   - **Expect agreement on:** 4 steps/beat spacing, velocity 100, snap-down scale
     quantization, gate length in ms, channel.
3. **Known A/B-flagged approximations** (from `NOTES-sequencer.md`, to check here):
   - Substep timing uses a continuous beat offset, vs the Pd patch's integer-pulse
     (48 PPQN) truncation + discrete clock match. Small timing differences are
     expected; confirm they're sub-perceptual.
   - Block-seam behavior: a step onset exactly on a process-block boundary could in
     principle double-fire or drop if the host's per-block `pos_beats` doesn't line up
     with our computed block end. Watch for any doubled/missing note at loop points or
     tempo changes.

Record findings (pass/fail per expectation) below when run.

### Results

_(fill in when the manual checks are performed)_
