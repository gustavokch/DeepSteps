# Runtime Autoencoder Training + In-App Dataset Building (Stage 3)

*2026-06-16*

## Context

Upstream DeepSteps shipped a full ML pipeline — **train + infer**. The Stage 2 Rust port
kept only **inference**: a frozen decoder (`weights/decoder.json`) baked into the plugin
via `include_str!`, run in pure f64 (`src/decoder.rs`). Training stayed offline in Python
(`Deep_Steps_project/`), and only the *decoder* shipped.

Stage 3 ports the **training half** into the live plugin: a from-scratch autoencoder
(encoder + decoder) that trains in-session on a dataset the user builds inside the plugin,
then hot-swaps the trained decoder into the audio path. The user can record/load material,
train a model live, and have the latent sliders drive *their* model.

### Decisions
1. **Dataset source = both**: audio files from disk (decode + onset detection) **and** user
   patterns captured from the grid.
2. **Fidelity = fix the bugs**: correct Adam bias correction (per-param step counter `t`),
   per-epoch batch shuffle. Does not match the Python numerically; converges better.
3. **Encoder exposed at runtime**: "encode current grid pattern → set the 4 latent sliders".
4. **Persist trained model** in DAW state (`#[persist]`); baked `decoder.json` stays as the
   default/fallback.

### Constraints (from the codebase)
- MIDI-only plugin, no host audio (`lib.rs` empty `AUDIO_IO_LAYOUTS`) → audio datasets come
  from disk files, not the host stream.
- `process()` is the audio thread; training must never run there, and the audio thread must
  never lock a `Mutex`.
- nih-plug `BackgroundTask` must stay heap-free → data flows through shared `Arc`s.
- The existing op-list JSON format is the serialization target, so a trained model loads
  into `decoder.rs` unchanged.

## Architecture

`decoder.rs` stays the untouched, panic-free inference path. A separate pure training
module produces models; the two communicate only through the JSON op format. Training runs
on nih-plug's background task; the trained decoder hot-swaps via `arc-swap` (wait-free read
on the audio thread).

### New modules
- **`src/autoencoder.rs`** — pure f64 full AE (encoder 32→16→8→4, decoder 4→8→16→32;
  Dense/ReLU/BatchNorm/Sigmoid; manual backprop; Adam). Ported from `AE_init.py` with the
  two bug fixes. Tiny inline PCG PRNG for init + shuffle (no `rand`).
- **`src/model_ops.rs`** — `ExportOp`/`ModelExport`/`TrainedModel` (`Serialize+Deserialize`),
  byte-compatible with `decoder.json` and `train_export.py`. `export_decoder`/`export_encoder`
  + `to_decoder` (round-trips through the same JSON `decoder.rs` consumes).
- **`src/dataset.rs`** — `encode_onsets` (exact port of `corpus_encode.py`, incl. Python
  banker's rounding) and `encode_grid` (capture the live pattern).
- **`src/audio.rs`** — `decode_audio` (symphonia → mono f32) and `detect_onsets`
  (spectral-flux + adaptive peak-pick via rustfft); `file_to_sample` glues them to
  `encode_onsets`. Intentionally not a librosa clone.
- **`src/training.rs`** — `TrainShared` (dataset, pending paths, progress atomics,
  `ArcSwap<Decoder>` model + `ArcSwap<Option<Decoder>>` encoder, persisted-model handle),
  the background `executor`, and `run_training`/`run_ingest`.

### Modified
- **`src/lib.rs`** — `BackgroundTask = training::Task`; `task_executor` wired; owned
  `decoder` replaced by `train: Arc<TrainShared>` whose `ArcSwap<Decoder>` the audio thread
  loads; `model_generation` invalidates `last_latent` on swap; `initialize()` restores a
  persisted model.
- **`src/params.rs`** — `#[persist = "trained-model"] trained_model: Arc<Mutex<Option<TrainedModel>>>`,
  shared (same `Arc`) with `TrainShared`.
- **`src/shared.rs`** — per-step `substeps: [AtomicU64; 16]`, written on regeneration, read by
  the GUI for capture/encode.
- **`src/editor.rs`** — "Training" panel: dataset size, *Capture pattern* / *Add audio…* /
  *Clear*, epochs/batch, *Train* / *Cancel*, progress bar + live loss, model status, and
  *Encode pattern → latent* (writes the 4 latents via begin/set/end gestures, clamped to
  `[0,1]`).

### Threading / safety
The audio thread only does a wait-free `model.load()` (an `ArcSwap`) and `Relaxed` atomic
reads — never a `Mutex`. The dataset and trained-model `Mutex`es are GUI/background only.
On training finish the executor exports the model, parses it back into a `Decoder`/encoder,
`ArcSwap::store`s them, bumps `model_generation`, and fills the persisted slot.

## Verification
- `cargo test --workspace` (34): AE overfit-to-zero (gradient correctness gate), Adam-fix
  lock-in, per-epoch shuffle determinism, export↔decoder parity (1e-9), `corpus_encode`
  parity with the Python `test_corpus_encode.py` cases, onset-detection sanity, and an
  end-to-end train→hot-swap→persist→encode integration test.
- `cargo clippy --all-targets -- -D warnings` clean; CLAP + VST3 bundle builds; both
  headless host scale-tests pass (audio path intact through a real host); `pluginval`
  (VST3, strictness 8) in CI.
- Manual: train in Carla (VST3), watch the loss bar, confirm the pattern swaps, test
  *Encode → latent*, and save/reload the project to verify persistence.
