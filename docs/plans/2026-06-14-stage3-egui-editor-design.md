# Stage 3 — egui Editor (GUI Port)

## Context

Stage 2 shipped the plugin params-only (host-generic UI). This stage adds a custom
**egui** editor that restores the Stage-1 openFrameworks UX (`DS-UI.png`): a 16-step
grid with a live playhead, plus controls for the latent vector, per-step pitches,
timing, and tuning. The training UI from the original (GENERATE / Train / Make Dataset
/ Open Corpus / Epochs / Loss) is dropped — there is no runtime training, and
regeneration is reactive on latent change.

Decisions: `nih_plug_egui` backend; clean modern egui look; clickable step toggles
(override the decoder until next regen); full editor in one pass.

## Architecture

The step on/off pattern and the playhead index are audio-thread runtime state, not
nih-plug params, so the editor reaches them through a lock-free bridge.

- **`src/shared.rs`** — `SharedState { steps: AtomicU16, current_step: AtomicUsize }`,
  held as `Arc` by the plugin and cloned into the editor. `Relaxed` ordering (no
  cross-variable invariant). `pack(&[bool;16]) -> u16` helper.
- **`src/lib.rs`** — `maybe_regen` writes the decoded pattern into `shared.steps`
  (now the playback source of truth, replacing the old `self.steps` array). `process`
  reads each step's on/off from the mask, publishes the playing index to
  `shared.current_step`, and writes `NO_STEP` when stopped. `editor()` returns the
  egui editor.
- **`src/editor.rs`** — `create_egui_editor`; `ParamSlider` widgets for latent / gate /
  substep / seq_len / key / scale / 16 pitches; a custom painter-drawn 16-cell grid
  (fill = on, red outline = playhead, click = `shared.toggle(i)`). Repaints each frame
  so the playhead animates.
- **`src/params.rs`** — `#[persist = "editor-state"] editor_state: Arc<EguiState>`
  (default 600×640, the Stage-1 canvas).

## Trade-offs

- Step toggles are runtime overrides, **not** params: not preset-saved, not automatable,
  and a latent move regenerates over them (intended generative semantics). Promoting to
  16 bool params + merge logic is a possible later pass.
- Both `nih_plug` and `nih_plug_egui` are pinned to the same git rev
  (`f36931f7…`); leaving the top-level `nih_plug` unpinned pulled a second, newer copy
  and broke the `PersistentField`/`Params` derive with a trait mismatch.

## Verification

- `cargo test` — 16 pass (existing sequencer/decoder + new `SharedState` bit tests).
- `cargo build --release`, `cargo xtask bundle deepsteps-plugin --release` — clean.
- `clap-validator validate` — 18 passed / 0 failed / 3 skipped (unchanged).
- Manual (Carla): editor opens at persisted size; grid shows the decoder pattern;
  playhead advances during play; clicking a cell flips it and changes MIDI output;
  moving a latent regenerates the grid.
