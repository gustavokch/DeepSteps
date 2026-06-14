# DeepSteps Stage 2 — nih-plug CLAP/VST3 Plugin (Design)

Stage 2 of the Linux port: a full Rust rewrite of DeepSteps as a CLAP + VST3
**MIDI-generator** plugin using nih-plug. Reuses none of the C++/OF/Pd code; the
Stage 1 Linux standalone (PR #1) is the **behavioral reference** to A/B against.

> Stage 1 (standalone) build: `docs/BUILDING-linux.md`.
> Original staging rationale: `docs/plans/2026-06-13-linux-port.md`.

## Locked decisions

- **Role:** MIDI generator only — host makes sound.
- **Model source:** offline-train in Python, **export decoder weights** to a file
  the Rust plugin loads. (The original never ships trained weights or a dataset;
  meaningful output only existed after in-session training on an aubio-built
  corpus — so the weights must be produced offline and frozen.)
- **Clock:** host transport (tempo + PPQ position). The original had no internal
  clock and slaved to external MIDI clock; the plugin syncs to the DAW playhead
  instead.
- **UI:** params-only for v1 (host-generic UI). Custom egui editor is a later pass.
- **Offline corpus:** audio + aubio onset detection (faithful to the original
  corpus path). Bring-your-own audio loop corpus; no corpus ships in-repo.

## Why the model needs offline export

`AE_init.py` constructs `Autoencoder()` with **random-init** weights at every
launch and never loads or saves a model. Meaningful patterns only appear after
the in-session training path (aubio corpus → `Process_corpus.py` → `dataset.csv`
→ `fit()`), which Stage 1 disabled. Consequence:

- With random init + single-sample inference, BatchNorm is numerically
  degenerate (sample var = 0 → centered = 0 → constant output). This only bites
  the *untrained* network.
- After real training, `fit()` populates each BatchNorm's `running_mean` /
  `running_var` (momentum-updated), so inference (`training=False`) is a
  well-defined standard forward pass. **Exported weights must include the running
  statistics**, or the Rust forward pass will not match Python.

## Architecture

Two parts, sharing only the exported weights file and a set of reference vectors.

### Offline pipeline (Python / uv, build-time)

`tools/train_export.py` — reuses the `Autoencoder` / `NeuralNetwork` / `Dense` /
`BatchNormalization` classes from `AE_init.py`:

1. Audio loop corpus → aubio onsets (as `ofApp.cpp`'s disabled corpus path did).
2. Onsets → 32-dim vectors via `Process_corpus.py` logic:
   `[onset_onehot[16], substep_offset[16]]`, where substep =
   PPQN distance from nearest 16th, clamped `[-6, 6]`, normalized `(x+6)/12`.
3. Stack into `dataset.csv`; `ae1.autoencoder.fit(dataset, dataset, epochs, 16)`.
4. **Export decoder weights** to `weights/decoder.json` (versioned, committed):
   per `Dense` layer `{W, b}`; per `BatchNormalization`
   `{gamma, beta, running_mean, running_var, eps=0.01}`; plus `latent_dim=4`,
   `input_dim=32`.
5. Emit `weights/reference_vectors.json`: N `(latent[4] → output[32])` pairs from
   the Python decoder, for Rust unit tests.

### Runtime (Rust / nih-plug)

No Python, no Pd, no OSC, no aubio. Load weights at plugin init; everything else
is native Rust.

## Components

```
deepsteps-plugin/
  Cargo.toml              nih-plug deps; clap + vst3 features
  xtask/                  nih-plug bundler (cargo xtask bundle)
  src/lib.rs              Plugin impl, nih_export_clap! + nih_export_vst3!, process()
  src/params.rs           nih-plug Params
  src/decoder.rs          weights struct + forward pass
  src/sequencer.rs        pulse counter, step->note, scale quantize
  weights/decoder.json            exported, committed
  weights/reference_vectors.json  exported, committed (test fixtures)
tools/train_export.py     uv; corpus -> aubio -> dataset -> train -> export
```

### decoder.rs

Exact replica of `receive_latent` (AE_init.py:383–402):

```
latent[4]
  -> Dense(8)  -> ReLU -> BatchNorm
  -> Dense(16) -> ReLU -> BatchNorm
  -> Dense(32) -> Sigmoid
split -> steps[16] (op > 0.5 ? 1 : 0), substeps[16] (raw 0..1)
```

BatchNorm inference: `out = gamma * (x - running_mean) / sqrt(running_var + eps) + beta`,
`eps = 0.01`. Dense: `out = x·W + b`.

### sequencer.rs

Re-derived from `bin/data/pd/main.pd`, **A/B-verified against the Stage 1
reference capture** (exact PPQN→step divisor extracted from the patch, not
guessed):

- Host `ppq_pos` + `tempo` → 192-PPQN pulse counter → `mod 16` step index.
- Per active step: `NoteOn` at the substep timing offset, `NoteOff` after gate.
  Substep offset = the 16 `expr $f1*($f2-$f3)+$f3` lerps mapping `substep[0..1]`
  to a ± pulse delay scaled by the substep param.
- `makenote` reference defaults: vel 100, dur 200ms (gate is a param).
- Pitch: per-step note → scale-quantize via the patch tables —
  chromatic `0..11`, pentatonic-major `0,2,4,7,9`, pentatonic-minor
  `0,3,5,7,10` — plus key offset and `%12` octave wrap.

### params.rs

nih-plug `Param`s mirroring the ofxGui widgets: 16 note pitches; latent A–D
(host-MIDI-CC-mappable); gate length; substep scale; sequence length (1–16);
key (0–11); scale (enum). Tempo comes from the host, not a param. **The pattern
regenerates reactively whenever a latent param changes** — no Generate button.

### lib.rs

`process()`: read host transport → advance the sequencer for the block →
push `NoteEvent::NoteOn/NoteOff` with in-block sample offsets.

## Data flow

```
host transport (tempo, ppq) ─► pulse counter ─► step boundary
latent params ─► decoder forward ─► steps[16] + substeps[16]  (regen on change)
step boundary + pattern ─► NoteOn (at substep offset) / NoteOff (after gate)
                          ─► scale quantize + key + %12 ─► host MIDI out
```

## Error handling

- Missing / corrupt `decoder.json` → load a bundled fallback (empty or trivial
  pattern), `nih_log` a warning, never crash the host.
- No host transport (no playhead) → emit nothing (silent); optional free-run is a
  later enhancement.
- All params range-clamped by nih-plug.

## Testing

- **Rust unit:** decoder forward pass vs `reference_vectors.json` (ε-match);
  sequencer pulse→step mapping; scale-quantize tables.
- **Validation:** `clap-validator` on the `.clap`.
- **Host:** load in Carla, drive transport, confirm note output in `aseqdump`.
- **A/B:** Rust MIDI output vs the Stage 1 reference capture for identical
  latent + params.

## Build order

1. Offline train + export → `decoder.json` + `reference_vectors.json`.
2. Rust decoder + unit tests vs reference vectors.
3. Rust sequencer + scale quantize + unit tests.
4. nih-plug scaffold: params, `process()`, host transport, MIDI out.
5. `cargo xtask bundle` (clap + vst3), `clap-validator`, host load + A/B.

## Risks

- **Pd → Rust fidelity:** the patch encodes the real timing/quantization; the
  PPQN→step divisor and substep lerp ranges must be extracted exactly and A/B'd.
- **Decoder numeric match:** Rust forward pass must match Python within ε; the
  exported running stats and `eps=0.01` are load-bearing.
- **No shipped corpus:** offline pipeline is bring-your-own audio corpus; the
  weights file is the committed artifact that makes the plugin reproducible.
- **aubio in the offline env:** needs the `aubio` Python package + an audio
  corpus; offline only, never at plugin runtime.
