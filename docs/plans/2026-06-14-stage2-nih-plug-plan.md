# DeepSteps Stage 2 — nih-plug CLAP/VST3 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship DeepSteps as a Linux x86_64 CLAP + VST3 MIDI-generator plugin (nih-plug), driven by host transport, with decoder weights trained offline in Python and frozen into the plugin.

**Architecture:** Two parts sharing only a weights file. (1) Offline Python/uv pipeline: audio corpus → aubio onsets → 32-dim vectors → train the existing `Autoencoder` → export `decoder.json` + `reference_vectors.json`. (2) Rust/nih-plug runtime: load weights → decoder forward pass → 192-PPQN sequencer off host transport → `NoteOn`/`NoteOff`. No Pd, OSC, aubio, or Python at runtime.

**Tech Stack:** Rust (nih-plug, serde, serde_json), Python 3.14 (numpy, aubio) via uv, cargo xtask bundler, clap-validator, Carla.

**Reference:** Design `docs/plans/2026-06-14-stage2-nih-plug-design.md`. Behavioral reference = Stage 1 standalone (`docs/BUILDING-linux.md`, PR #1). Source of truth for algorithms: `Deep_Steps_project/bin/data/AE_init.py` (decoder), `Deep_Steps_project/bin/data/Process_corpus.py` (vector encoding), `Deep_Steps_project/bin/data/pd/main.pd` (sequencer).

**Conventions:** Python deps via `uv add` (never `uv pip`). Run Python via `uv run`. Commit after every passing test. All new Rust lives under `deepsteps-plugin/`.

---

## Phase A — Offline training + export (Python/uv)

Work in `Deep_Steps_project/` (existing uv project). Add offline-only deps there.

### Task 1: Add offline deps

**Files:**
- Modify: `Deep_Steps_project/pyproject.toml`

**Step 1: Add aubio + a dev test runner**

```bash
cd Deep_Steps_project
uv add aubio
uv add --dev pytest
```

**Step 2: Verify import**

Run: `uv run python -c "import aubio, numpy; print(aubio.version)"`
Expected: prints an aubio version, no traceback.

**Step 3: Commit**

```bash
git add Deep_Steps_project/pyproject.toml Deep_Steps_project/uv.lock
git commit -m "Stage 2: add aubio + pytest for offline weight-export pipeline"
```

---

### Task 2: Extract corpus encoding into a tested function

`Process_corpus.py` is an exec-fragment with free globals. Refactor the onset→vector
math into a pure function so it is testable without aubio.

**Files:**
- Create: `Deep_Steps_project/tools/corpus_encode.py`
- Test: `Deep_Steps_project/tools/test_corpus_encode.py`

**Step 1: Write the failing test**

```python
# test_corpus_encode.py
import numpy as np
from corpus_encode import encode_onsets

def test_single_bar_onsets_to_32dim():
    # onsets in samples; dur = total length in samples. One bar.
    # Place onsets exactly on 16th-note boundaries -> substep offset 0.5 (centered).
    dur = 48 * 4 * 1  # per_quarter_note(48) * 4 * bar_length(1) PPQN units as "samples"
    # 4 onsets on steps 0,4,8,12 exactly on the grid
    sixteenth = dur / 16
    onsets = np.array([0, 4, 8, 12]) * sixteenth
    vec = encode_onsets(onsets.astype(int), int(dur))
    assert vec.shape == (32,)
    onehot, substeps = vec[:16], vec[16:]
    assert list(onehot[[0, 4, 8, 12]]) == [1, 1, 1, 1]
    assert onehot.sum() == 4
    # on-grid onset -> substep distance 0 -> normalized (0+6)/12 = 0.5
    for i in (0, 4, 8, 12):
        assert abs(substeps[i] - 0.5) < 1e-6
    # empty steps carry substep 0
    assert substeps[1] == 0
```

**Step 2: Run test to verify it fails**

Run: `cd Deep_Steps_project/tools && uv run pytest test_corpus_encode.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'corpus_encode'`.

**Step 3: Write minimal implementation**

Port the math from `Process_corpus.py` (single-bar path) into a pure function.
`bar_length` is fixed to 1 here (16-step model); the multi-bar split is out of scope.

```python
# corpus_encode.py
import numpy as np

PER_QUARTER_NOTE = 48
SIXTEENTHS_DIV = 16  # bar_length = 1

def encode_onsets(onsets, dur):
    """onsets: int sample positions within one bar. dur: bar length in samples.
    Returns a 32-dim vector [onset_onehot[16], substep_offset[16]]."""
    timebase = PER_QUARTER_NOTE * 4          # 192 PPQN per bar
    num_ppqn = timebase                       # bar_length = 1
    ppqn_timebase = round(dur / num_ppqn)
    sixteenths = round(dur / SIXTEENTHS_DIV)

    onsets = np.round(onsets).astype(int)

    # round onsets to nearest 16th, drop duplicates landing on the same step
    onset_points_rounded = []
    previous = None
    keep = np.ones(len(onsets), dtype=bool)
    for i, onset in enumerate(onsets):
        r = int(round(onset / sixteenths))
        if r != previous:
            onset_points_rounded.append(r)
            previous = r
        else:
            keep[i] = False
    onsets = onsets[keep]

    ppqn_onsets = [int(o // ppqn_timebase) * ppqn_timebase for o in onsets]

    onehot = np.zeros(SIXTEENTHS_DIV)
    for o in onset_points_rounded:
        if o < SIXTEENTHS_DIV:
            onehot[o] = 1

    # substep = signed PPQN distance from nearest 16th, clamped [-6,6], -> (x+6)/12
    substeps = []
    nearest = [int(round(o / sixteenths)) * sixteenths for o in onsets]
    for f, c in zip(ppqn_onsets, nearest):
        ss = (f - c) // ppqn_timebase
        substeps.append((ss + 6) / 12)

    substeps_full = []
    j = 0
    for v in onehot:
        if v == 1 and j < len(substeps):
            substeps_full.append(substeps[j]); j += 1
        else:
            substeps_full.append(0)

    return np.concatenate((onehot, np.array(substeps_full)))
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest test_corpus_encode.py -v`
Expected: PASS.

> Note: if the on-grid substep does not come out to exactly 0.5, read `Process_corpus.py:71-92` again and match its rounding precisely — the encoding must be bit-faithful to what the original trained on. Adjust the test's expected value only if the original math genuinely differs; do not loosen it to hide a port error.

**Step 5: Commit**

```bash
git add Deep_Steps_project/tools/corpus_encode.py Deep_Steps_project/tools/test_corpus_encode.py
git commit -m "Stage 2: extract tested corpus onset->32dim encoder"
```

---

### Task 3: Audio-corpus → dataset builder (aubio wrapper)

Thin I/O layer: for each audio file, aubio onsets → `encode_onsets` → row. Kept
thin because aubio-on-real-audio is not deterministically unit-tested; the math it
calls is already covered by Task 2.

**Files:**
- Create: `Deep_Steps_project/tools/build_dataset.py`
- Test: `Deep_Steps_project/tools/test_build_dataset.py`

**Step 1: Write the failing test** (covers the file-loop assembly, not aubio)

```python
# test_build_dataset.py
import numpy as np
from build_dataset import rows_to_dataset

def test_rows_to_dataset_stacks_and_shapes():
    rows = [np.zeros(32), np.ones(32)]
    ds = rows_to_dataset(rows)
    assert ds.shape == (2, 32)
    assert ds[1].sum() == 32
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest test_build_dataset.py -v`
Expected: FAIL — module/function missing.

**Step 3: Write minimal implementation**

```python
# build_dataset.py
import sys, glob, numpy as np, aubio
from corpus_encode import encode_onsets

def rows_to_dataset(rows):
    return np.vstack(rows)

def onsets_for_file(path, hop=512, method="default"):
    src = aubio.source(path, 0, hop)          # native samplerate, mono-mix
    sr = src.samplerate
    o = aubio.onset(method, 2048, hop, sr)
    times, total = [], 0
    while True:
        samples, read = src()
        if o(samples):
            times.append(o.get_last())
        total += read
        if read < hop:
            break
    return np.array(times), total

def build(corpus_glob):
    rows = []
    for path in sorted(glob.glob(corpus_glob)):
        onsets, dur = onsets_for_file(path)
        if len(onsets) == 0:
            continue
        rows.append(encode_onsets(onsets, dur))
    return rows_to_dataset(rows)

if __name__ == "__main__":
    ds = build(sys.argv[1])                   # e.g. "corpus/*.wav"
    np.savetxt(sys.argv[2], ds, delimiter=",")
    print(f"wrote {ds.shape} -> {sys.argv[2]}")
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest test_build_dataset.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add Deep_Steps_project/tools/build_dataset.py Deep_Steps_project/tools/test_build_dataset.py
git commit -m "Stage 2: aubio audio-corpus -> dataset.csv builder"
```

---

### Task 4: Train + export decoder weights and reference vectors

Reuse the `Autoencoder`/`NeuralNetwork` classes from `AE_init.py` unchanged (import
them), train on the dataset, and serialize the **decoder** layers (including BN
running stats). Also dump N latent→output pairs straight from the trained Python
decoder for the Rust tests. The round-trip test re-implements the forward pass from
the JSON and asserts it matches `ae1.decoder.predict` — this is the contract the
Rust port must satisfy.

**Files:**
- Create: `Deep_Steps_project/tools/train_export.py`
- Test: `Deep_Steps_project/tools/test_train_export.py`

**Step 1: Write the failing test**

```python
# test_train_export.py
import numpy as np
from train_export import build_autoencoder, export_decoder, forward_from_export

def test_export_roundtrip_matches_python_decoder():
    rng = np.random.default_rng(0)
    ds = (rng.random((64, 32)) > 0.5).astype(float)  # dummy bars
    ae = build_autoencoder()
    ae.autoencoder.fit(ds, ds, n_epochs=3, batch_size=16)
    export = export_decoder(ae)
    # three random latents must match the live Python decoder within eps
    for _ in range(3):
        z = rng.random((1, 4))
        py = ae.decoder.predict(z)[0]
        js = forward_from_export(export, z[0])
        assert np.allclose(py, js, atol=1e-6)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest test_train_export.py -v`
Expected: FAIL — module missing.

**Step 3: Write minimal implementation**

`fit` in `AE_init.py` calls `client.send_message('/loss', ...)`. Guard that:
import the module, and if `client` send fails, monkeypatch a no-op. Simplest:
set `AE_init.client` to a dummy before `fit`.

```python
# train_export.py
import sys, json, numpy as np
sys.path.insert(0, "../bin/data")          # AE_init.py lives in bin/data
import AE_init
from AE_init import Autoencoder

class _NullClient:
    def send_message(self, *a, **k): pass

def build_autoencoder():
    AE_init.client = _NullClient()         # silence the OSC /loss calls in fit()
    return Autoencoder()

def _relu(x): return np.where(x >= 0, x, 0)
def _sigmoid(x): return 1 / (1 + np.exp(-x))

def export_decoder(ae):
    """Walk decoder.layers, emit ordered op list mirroring forward_pass(training=False)."""
    ops = []
    for layer in ae.decoder.layers:
        name = type(layer).__name__
        if name == "Dense":
            ops.append({"op": "dense",
                        "W": layer.W.tolist(),
                        "b": layer.w0.tolist()})       # w0 shape (1, n)
        elif name == "Activation":
            ops.append({"op": layer.activation_name})  # 'relu' | 'sigmoid'
        elif name == "BatchNormalization":
            ops.append({"op": "bn",
                        "gamma": layer.gamma.tolist(),
                        "beta": layer.beta.tolist(),
                        "running_mean": layer.running_mean.tolist(),
                        "running_var": layer.running_var.tolist(),
                        "eps": layer.eps})
        else:
            raise ValueError(f"unexpected decoder layer {name}")
    return {"latent_dim": ae.latent_dim, "input_dim": ae.input_dim, "ops": ops}

def forward_from_export(export, z):
    x = np.asarray(z, dtype=float).reshape(1, -1)
    for op in export["ops"]:
        k = op["op"]
        if k == "dense":
            x = x.dot(np.array(op["W"])) + np.array(op["b"])
        elif k == "relu":
            x = _relu(x)
        elif k == "sigmoid":
            x = _sigmoid(x)
        elif k == "bn":
            mean = np.array(op["running_mean"]); var = np.array(op["running_var"])
            g = np.array(op["gamma"]); b = np.array(op["beta"]); eps = op["eps"]
            x = g * ((x - mean) / np.sqrt(var + eps)) + b
        else:
            raise ValueError(k)
    return x[0]

def reference_vectors(ae, n=8, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        z = rng.random(4)
        y = ae.decoder.predict(z.reshape(1, -1))[0]
        out.append({"latent": z.tolist(), "output": y.tolist()})
    return out

if __name__ == "__main__":
    dataset_csv, weights_out, refs_out = sys.argv[1], sys.argv[2], sys.argv[3]
    ds = np.loadtxt(dataset_csv, delimiter=",")
    ae = build_autoencoder()
    ae.autoencoder.fit(ds, ds, n_epochs=200, batch_size=16)
    json.dump(export_decoder(ae), open(weights_out, "w"))
    json.dump(reference_vectors(ae), open(refs_out, "w"))
    print(f"exported {weights_out} + {refs_out}")
```

> Verify the BN forward path: `forward_from_export`'s `bn` branch must match
> `BatchNormalization.forward_pass` with `training=False` (AE_init.py:170-180).
> If `running_mean` is `None` after `fit` (it should not be — `fit` runs training
> passes that populate it), the export must fail loudly rather than emit `null`.

**Step 4: Run test to verify it passes**

Run: `uv run pytest test_train_export.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add Deep_Steps_project/tools/train_export.py Deep_Steps_project/tools/test_train_export.py
git commit -m "Stage 2: train + export decoder weights & reference vectors (roundtrip-tested)"
```

---

### Task 5: Produce the committed weights artifacts

Generate the real `decoder.json` + `reference_vectors.json` the Rust crate consumes.
Without an audio corpus, generate from a deterministic synthetic dataset so the build
is reproducible now; swap in a real corpus later by re-running with `build_dataset.py`.

**Files:**
- Create: `deepsteps-plugin/weights/decoder.json`
- Create: `deepsteps-plugin/weights/reference_vectors.json`
- Create: `Deep_Steps_project/tools/make_synth_dataset.py`

**Step 1: Synthetic dataset generator** (deterministic, no aubio)

```python
# make_synth_dataset.py  — placeholder corpus: random 16-step patterns + grid substeps
import sys, numpy as np
rng = np.random.default_rng(42)
n = int(sys.argv[2]) if len(sys.argv) > 2 else 256
onehot = (rng.random((n, 16)) > 0.6).astype(float)
substeps = np.where(onehot == 1, rng.uniform(0.3, 0.7, (n, 16)), 0.0)
ds = np.concatenate((onehot, substeps), axis=1)
np.savetxt(sys.argv[1], ds, delimiter=",")
print(f"wrote {ds.shape} -> {sys.argv[1]}")
```

**Step 2: Generate dataset + train + export**

```bash
cd Deep_Steps_project/tools
uv run python make_synth_dataset.py /tmp/dataset.csv 256
mkdir -p ../../deepsteps-plugin/weights
uv run python train_export.py /tmp/dataset.csv \
    ../../deepsteps-plugin/weights/decoder.json \
    ../../deepsteps-plugin/weights/reference_vectors.json
```

Expected: prints `exported ...`; both JSON files exist and are non-empty.

**Step 3: Sanity-check the artifacts**

Run: `uv run python -c "import json; d=json.load(open('../../deepsteps-plugin/weights/decoder.json')); print(d['latent_dim'], d['input_dim'], [o['op'] for o in d['ops']])"`
Expected: `4 32 ['dense', 'relu', 'bn', 'dense', 'relu', 'bn', 'dense', 'sigmoid']`

**Step 4: Commit**

```bash
git add Deep_Steps_project/tools/make_synth_dataset.py deepsteps-plugin/weights/decoder.json deepsteps-plugin/weights/reference_vectors.json
git commit -m "Stage 2: commit reproducible synthetic-trained decoder weights + reference vectors"
```

---

## Phase B — Rust decoder

### Task 6: Cargo crate scaffold

**Files:**
- Create: `deepsteps-plugin/Cargo.toml`
- Create: `deepsteps-plugin/src/lib.rs` (stub)
- Create: `deepsteps-plugin/.gitignore`

**Step 1: Cargo.toml**

```toml
[package]
name = "deepsteps-plugin"
version = "0.1.0"
edition = "2021"
license = "see repo"

[lib]
crate-type = ["cdylib", "lib"]

[dependencies]
nih_plug = { git = "https://github.com/robbert-vdh/nih-plug.git" }
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[profile.release]
lto = "thin"
strip = "symbols"
```

> Pin `nih_plug` to a specific rev once it builds (replace the bare `git` with
> `rev = "<sha>"`) so the plugin is reproducible.

**Step 2: lib.rs stub + .gitignore**

```rust
// src/lib.rs
pub mod decoder;
```

```
# deepsteps-plugin/.gitignore
/target
```

**Step 3: Verify it compiles (empty decoder module will fail until Task 7 — make a stub)**

Create `src/decoder.rs` with `// placeholder` for now.

Run: `cd deepsteps-plugin && cargo build`
Expected: builds (may pull a large nih-plug tree on first run).

**Step 4: Commit**

```bash
git add deepsteps-plugin/Cargo.toml deepsteps-plugin/src/lib.rs deepsteps-plugin/src/decoder.rs deepsteps-plugin/.gitignore deepsteps-plugin/Cargo.lock
git commit -m "Stage 2: scaffold deepsteps-plugin Rust crate (nih-plug)"
```

---

### Task 7: Decoder weights load + forward pass

**Files:**
- Modify: `deepsteps-plugin/src/decoder.rs`
- Test: inline `#[cfg(test)]` module in `decoder.rs`
- Uses fixture: `deepsteps-plugin/weights/reference_vectors.json`, `weights/decoder.json`

**Step 1: Write the failing test**

```rust
// in src/decoder.rs, bottom
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn forward_matches_reference_vectors() {
        let dec = Decoder::from_json_str(include_str!("../weights/decoder.json")).unwrap();
        let refs: Vec<RefVec> =
            serde_json::from_str(include_str!("../weights/reference_vectors.json")).unwrap();
        for r in &refs {
            let out = dec.forward(&r.latent);
            assert_eq!(out.len(), 32);
            for (a, b) in out.iter().zip(r.output.iter()) {
                assert!((a - b).abs() < 1e-5, "mismatch {a} vs {b}");
            }
        }
    }
    #[derive(serde::Deserialize)]
    struct RefVec { latent: Vec<f64>, output: Vec<f64> }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p deepsteps-plugin forward_matches_reference_vectors`
Expected: FAIL — `Decoder`/`from_json_str`/`forward` undefined.

**Step 3: Write minimal implementation**

Mirror `forward_from_export` (Task 4) exactly. Work in `f64` to match numpy, then
the caller can downcast. Matrix is row-major; `b` is shape `(1, n)`.

```rust
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(tag = "op")]
enum Op {
    #[serde(rename = "dense")]
    Dense { W: Vec<Vec<f64>>, b: Vec<Vec<f64>> },
    #[serde(rename = "relu")]
    Relu,
    #[serde(rename = "sigmoid")]
    Sigmoid,
    #[serde(rename = "bn")]
    Bn { gamma: Vec<f64>, beta: Vec<f64>, running_mean: Vec<f64>, running_var: Vec<f64>, eps: f64 },
}

#[derive(Deserialize)]
pub struct Decoder { pub latent_dim: usize, pub input_dim: usize, ops: Vec<Op> }

impl Decoder {
    pub fn from_json_str(s: &str) -> Result<Self, serde_json::Error> { serde_json::from_str(s) }

    pub fn forward(&self, latent: &[f64]) -> Vec<f64> {
        let mut x = latent.to_vec();
        for op in &self.ops {
            x = match op {
                Op::Dense { W, b } => {
                    let n_out = W[0].len();
                    let mut y = vec![0.0; n_out];
                    for j in 0..n_out {
                        let mut acc = b[0][j];
                        for i in 0..x.len() { acc += x[i] * W[i][j]; }
                        y[j] = acc;
                    }
                    y
                }
                Op::Relu => x.iter().map(|&v| if v >= 0.0 { v } else { 0.0 }).collect(),
                Op::Sigmoid => x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect(),
                Op::Bn { gamma, beta, running_mean, running_var, eps } =>
                    (0..x.len()).map(|i|
                        gamma[i] * ((x[i] - running_mean[i]) / (running_var[i] + eps).sqrt()) + beta[i]
                    ).collect(),
            };
        }
        x
    }

    /// steps[16] thresholded >0.5, substeps[16] raw — mirrors receive_latent().
    pub fn generate(&self, latent: &[f64]) -> ([bool; 16], [f64; 16]) {
        let out = self.forward(latent);
        let mut steps = [false; 16];
        let mut substeps = [0.0; 16];
        for i in 0..16 {
            steps[i] = out[i] > 0.5;
            substeps[i] = out[16 + i];
        }
        (steps, substeps)
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p deepsteps-plugin forward_matches_reference_vectors`
Expected: PASS.

**Step 5: Commit**

```bash
git add deepsteps-plugin/src/decoder.rs
git commit -m "Stage 2: Rust decoder forward pass, matches Python reference vectors"
```

---

## Phase C — Rust sequencer

### Task 8: Extract exact sequencer constants from main.pd

Before coding the sequencer, pin the numbers from the patch and Stage 1 capture so
later tasks aren't guessing. Produce a short notes file; no code yet.

**Files:**
- Create: `deepsteps-plugin/NOTES-sequencer.md`

**Step 1: Read the patch and record, with `main.pd` line cites:**
- The PPQN base and the pulses-per-step divisor (`mod 192`, `mod 16` chain).
- How `select 248 250 252` (clock/start/stop) advances the counter — confirm one
  step = how many clock pulses, and that 16 steps = one bar.
- The substep lerp: the 16 `expr $f1*($f2-$f3)+$f3` — identify `$f2`/`$f3` (the
  min/max bounds) and where the substep param feeds in. Record the pulse range a
  `substep ∈ [0,1]` maps to.
- The scale-quantize tables (`sel 0 3 5 7 10`, `sel 0 2 4 7 9`, `sel 0..11`), the
  `%12` wrap, and how `key` offsets pitch.
- `makenote 100 200` defaults (vel, dur ms) and which slider overrides the gate.

**Step 2: Cross-check against Stage 1**
Re-run `Deep_Steps_project/tools/run_midi_test.sh`, capture the MIDI, and note the
observed step timing/pitches to validate the constants above.

**Step 3: Commit**

```bash
git add deepsteps-plugin/NOTES-sequencer.md
git commit -m "Stage 2: document exact sequencer constants extracted from main.pd"
```

---

### Task 9: Scale quantization

**Files:**
- Create: `deepsteps-plugin/src/sequencer.rs`
- Modify: `deepsteps-plugin/src/lib.rs` (add `pub mod sequencer;`)

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quantize_snaps_to_scale_and_key() {
        // pentatonic minor [0,3,5,7,10]; key=0; octave 5 (base 60)
        // raw degree 1 -> nearest scale tone; assert membership + key offset.
        let q = quantize(62, Scale::PentMinor, 0);  // D
        assert!([60,63,65,67,70].contains(&(q % 12 + 60).min(q)));  // refine per NOTES
    }
    #[test]
    fn chromatic_is_identity_mod_key() {
        assert_eq!(quantize(64, Scale::Chromatic, 0), 64);
    }
}
```

> Tighten these asserts using the exact mapping recorded in `NOTES-sequencer.md`
> (Task 8). The patch maps a raw pitch to a scale degree via `%12` + `sel`; replicate
> that precisely rather than inventing a nearest-neighbour rule if they differ.

**Step 2: Run test to verify it fails**

Run: `cargo test -p deepsteps-plugin quantize`
Expected: FAIL — `quantize`/`Scale` undefined.

**Step 3: Write minimal implementation** (adjust to match NOTES)

```rust
#[derive(Clone, Copy)]
pub enum Scale { Chromatic, PentMajor, PentMinor }

impl Scale {
    fn table(self) -> &'static [i32] {
        match self {
            Scale::Chromatic => &[0,1,2,3,4,5,6,7,8,9,10,11],
            Scale::PentMajor => &[0,2,4,7,9],
            Scale::PentMinor => &[0,3,5,7,10],
        }
    }
}

/// Map a raw MIDI note to the nearest in-scale note, then apply key offset.
pub fn quantize(note: i32, scale: Scale, key: i32) -> i32 {
    let pc = ((note % 12) + 12) % 12;
    let octave = note - pc;
    let nearest = scale.table().iter().copied()
        .min_by_key(|&d| (d - pc).abs()).unwrap();
    octave + nearest + key
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p deepsteps-plugin quantize`
Expected: PASS.

**Step 5: Commit**

```bash
git add deepsteps-plugin/src/sequencer.rs deepsteps-plugin/src/lib.rs
git commit -m "Stage 2: scale quantization (pent maj/min, chromatic) + key offset"
```

---

### Task 10: Transport-driven step advance

Pure, testable stepping: given previous and current beat position, return the step
boundaries crossed within a block. No nih-plug types here.

**Files:**
- Modify: `deepsteps-plugin/src/sequencer.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn steps_per_beat_and_boundary_crossing() {
    // 16 steps per bar (4 beats) -> 4 steps per beat; step = 0.25 beat.
    // Block from beat 0.0 to 0.30 crosses step boundaries at 0.0 and 0.25.
    let crossed = steps_in_range(0.0, 0.30, 16);
    assert_eq!(crossed, vec![0, 1]);            // step indices whose onset is in [0,0.30)
}
#[test]
fn step_index_wraps_mod_seq_length() {
    assert_eq!(step_at_beat(4.25, 16) , 1);     // 4.25 beats -> 17th step -> mod 16 = 1
}
```

> Confirm "16 steps per bar / 4 per beat" against `NOTES-sequencer.md`. If the patch
> uses a different step length, fix the constant and the test together.

**Step 2: Run test to verify it fails**

Run: `cargo test -p deepsteps-plugin steps`
Expected: FAIL — functions undefined.

**Step 3: Write minimal implementation**

```rust
pub const STEPS_PER_BEAT: f64 = 4.0;   // 16 steps per 4-beat bar; verify vs NOTES

pub fn step_at_beat(beat: f64, seq_len: usize) -> usize {
    let idx = (beat * STEPS_PER_BEAT).floor() as i64;
    ((idx.rem_euclid(seq_len as i64)) as usize)
}

/// Absolute step indices (pre-wrap) whose onset time lies in [start, end).
pub fn steps_in_range(start_beat: f64, end_beat: f64, _seq_len: usize) -> Vec<i64> {
    let step = 1.0 / STEPS_PER_BEAT;
    let first = (start_beat / step).ceil() as i64;
    let mut out = vec![];
    let mut k = if (first as f64) * step < start_beat { first + 1 } else { first };
    // include a step exactly at start_beat
    if (k as f64 - 1.0) * step >= start_beat { k -= 1; }
    let mut idx = (start_beat / step).floor() as i64;
    if (idx as f64) * step >= start_beat { /* boundary at start */ }
    out.clear();
    let mut s = (start_beat / step).floor() as i64;
    if (s as f64) * step < start_beat { s += 1; }
    while (s as f64) * step < end_beat {
        out.push(s);
        s += 1;
    }
    out
}
```

> The `steps_in_range` sketch is fiddly at boundaries — let the test drive the exact
> half-open semantics. Simplify the body once the test passes; do not ship the dead
> scratch lines above.

**Step 4: Run test to verify it passes**

Run: `cargo test -p deepsteps-plugin steps`
Expected: PASS.

**Step 5: Commit**

```bash
git add deepsteps-plugin/src/sequencer.rs
git commit -m "Stage 2: transport beat -> step boundary mapping"
```

---

### Task 11: Note scheduling (substep offset + gate)

**Files:**
- Modify: `deepsteps-plugin/src/sequencer.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn schedules_noteon_with_substep_offset_and_noteoff_after_gate() {
    // step length in beats = 0.25; substep 0.5 -> centered (zero net offset per NOTES);
    // gate 0.5 -> noteoff at on + 0.5*step.
    let ev = schedule_step(/*step_onset_beat*/ 1.0, /*substep*/ 0.5,
                           /*gate*/ 0.5, /*note*/ 60, /*vel*/ 100);
    assert_eq!(ev.note, 60);
    assert_eq!(ev.vel, 100);
    assert!((ev.on_beat - 1.0).abs() < 1e-9);          // refine offset per NOTES
    assert!((ev.off_beat - (ev.on_beat + 0.125)).abs() < 1e-9);
}
```

> Replace the substep→offset rule and gate scaling with the exact lerp from
> `NOTES-sequencer.md` (the `expr $f1*($f2-$f3)+$f3` bounds). The numbers above are
> placeholders to be pinned by Task 8.

**Step 2: Run test to verify it fails** → undefined `schedule_step`.

**Step 3: Write minimal implementation** (fill bounds from NOTES)

```rust
pub struct StepEvent { pub note: i32, pub vel: u8, pub on_beat: f64, pub off_beat: f64 }

pub const STEP_BEATS: f64 = 1.0 / STEPS_PER_BEAT;
// substep in [0,1] maps to +-HALF_STEP*range around the grid; bounds from NOTES.
pub const SUBSTEP_RANGE_BEATS: f64 = STEP_BEATS; // placeholder; set per expr bounds

pub fn schedule_step(step_onset_beat: f64, substep: f64, gate: f64, note: i32, vel: u8) -> StepEvent {
    let offset = (substep - 0.5) * SUBSTEP_RANGE_BEATS;   // 0.5 = on-grid per encoder
    let on = step_onset_beat + offset;
    StepEvent { note, vel, on_beat: on, off_beat: on + gate * STEP_BEATS }
}
```

**Step 4: Run test to verify it passes** → PASS.

**Step 5: Commit**

```bash
git add deepsteps-plugin/src/sequencer.rs
git commit -m "Stage 2: note scheduling with substep timing offset + gate"
```

---

## Phase D — nih-plug shell

### Task 12: Parameters

**Files:**
- Create: `deepsteps-plugin/src/params.rs`
- Modify: `deepsteps-plugin/src/lib.rs`

**Step 1: Define params** (no standalone test — exercised via plugin in Task 14)

```rust
use nih_plug::prelude::*;
use std::sync::Arc;

#[derive(Enum, PartialEq, Clone, Copy)]
pub enum ScaleParam { #[id="chromatic"] Chromatic, #[id="pentmaj"] PentMajor, #[id="pentmin"] PentMinor }

#[derive(Params)]
pub struct DeepStepsParams {
    #[id = "latentA"] pub latent_a: FloatParam,
    #[id = "latentB"] pub latent_b: FloatParam,
    #[id = "latentC"] pub latent_c: FloatParam,
    #[id = "latentD"] pub latent_d: FloatParam,
    #[id = "gate"]    pub gate: FloatParam,
    #[id = "substep"] pub substep_scale: FloatParam,
    #[id = "seqlen"]  pub seq_len: IntParam,
    #[id = "key"]     pub key: IntParam,
    #[id = "scale"]   pub scale: EnumParam<ScaleParam>,
    #[nested(array, group = "notes")] pub notes: [NoteParam; 16],
}

#[derive(Params)]
pub struct NoteParam { #[id = "pitch"] pub pitch: IntParam }

impl Default for DeepStepsParams {
    fn default() -> Self {
        let lat = || FloatParam::new("latent", 0.5, FloatRange::Linear { min: 0.0, max: 1.0 });
        Self {
            latent_a: lat(), latent_b: lat(), latent_c: lat(), latent_d: lat(),
            gate: FloatParam::new("gate", 0.5, FloatRange::Linear { min: 0.05, max: 1.0 }),
            substep_scale: FloatParam::new("substep", 1.0, FloatRange::Linear { min: 0.0, max: 1.0 }),
            seq_len: IntParam::new("seqlen", 16, IntRange::Linear { min: 1, max: 16 }),
            key: IntParam::new("key", 0, IntRange::Linear { min: 0, max: 11 }),
            scale: EnumParam::new("scale", ScaleParam::PentMinor),
            notes: std::array::from_fn(|_| NoteParam {
                pitch: IntParam::new("pitch", 60, IntRange::Linear { min: 24, max: 96 }),
            }),
        }
    }
}
```

> Verify `#[nested(array, ...)]` syntax against the nih-plug version pulled in Task 6
> (the derive API for arrays of nested params has changed across revisions). If
> unsupported, fall back to 16 explicitly-`#[id]`'d pitch params.

**Step 2: Wire into lib.rs** (`pub mod params;`) and `cargo build`.
Expected: compiles.

**Step 3: Commit**

```bash
git add deepsteps-plugin/src/params.rs deepsteps-plugin/src/lib.rs
git commit -m "Stage 2: nih-plug parameters (latent, gate, substep, seq, key, scale, 16 notes)"
```

---

### Task 13: Plugin impl — process loop, transport, MIDI out

**Files:**
- Modify: `deepsteps-plugin/src/lib.rs`

**Step 1: Implement the plugin**

Key behaviors: load weights via `include_str!` with a fallback; regenerate the
pattern when any latent param changes (compare a cached `[f64;4]`); on each block,
map transport beats → step boundaries → `NoteOn`/`NoteOff` with sample-offset timing.

```rust
use nih_plug::prelude::*;
use std::sync::Arc;
mod decoder; mod sequencer; mod params;
use decoder::Decoder;
use params::{DeepStepsParams, ScaleParam};
use sequencer::*;

struct DeepSteps {
    params: Arc<DeepStepsParams>,
    decoder: Decoder,
    last_latent: [f64; 4],
    steps: [bool; 16],
    substeps: [f64; 16],
    sample_rate: f32,
}

impl Default for DeepSteps {
    fn default() -> Self {
        let decoder = Decoder::from_json_str(include_str!("../weights/decoder.json"))
            .unwrap_or_else(|_| Decoder::empty()); // add Decoder::empty() fallback
        Self {
            params: Arc::new(DeepStepsParams::default()),
            decoder, last_latent: [-1.0; 4], steps: [false; 16], substeps: [0.0; 16],
            sample_rate: 44100.0,
        }
    }
}

impl DeepSteps {
    fn maybe_regen(&mut self) {
        let p = &self.params;
        let z = [p.latent_a.value() as f64, p.latent_b.value() as f64,
                 p.latent_c.value() as f64, p.latent_d.value() as f64];
        if z != self.last_latent {
            let (s, ss) = self.decoder.generate(&z);
            self.steps = s; self.substeps = ss; self.last_latent = z;
        }
    }
}

impl Plugin for DeepSteps {
    const NAME: &'static str = "DeepSteps";
    const VENDOR: &'static str = "DeepSteps";
    const URL: &'static str = "https://github.com/gustavokch/DeepSteps";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: None, main_output_channels: None, ..AudioIOLayout::const_default()
    }];
    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const MIDI_OUTPUT: MidiConfig = MidiConfig::Basic;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;
    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> { self.params.clone() }

    fn initialize(&mut self, _: &AudioIOLayout, cfg: &BufferConfig, _: &mut impl InitContext<Self>) -> bool {
        self.sample_rate = cfg.sample_rate; true
    }

    fn process(&mut self, buffer: &mut Buffer, _aux: &mut AuxiliaryBuffers,
               context: &mut impl ProcessContext<Self>) -> ProcessStatus {
        self.maybe_regen();
        let t = context.transport();
        if !t.playing { return ProcessStatus::Normal; }
        let (Some(tempo), Some(pos_beats)) = (t.tempo, t.pos_beats()) else { return ProcessStatus::Normal; };

        let nframes = buffer.samples() as f64;
        let beats_per_sample = tempo / 60.0 / self.sample_rate as f64;
        let block_end = pos_beats + nframes * beats_per_sample;
        let seq_len = self.params.seq_len.value() as usize;
        let gate = self.params.gate.value() as f64;
        let scale = match self.params.scale.value() {
            ScaleParam::Chromatic => Scale::Chromatic,
            ScaleParam::PentMajor => Scale::PentMajor,
            ScaleParam::PentMinor => Scale::PentMinor,
        };
        let key = self.params.key.value();

        for abs_step in steps_in_range(pos_beats, block_end, seq_len) {
            let idx = (abs_step.rem_euclid(seq_len as i64)) as usize;
            if !self.steps[idx] { continue; }
            let onset_beat = abs_step as f64 * (1.0 / STEPS_PER_BEAT);
            let raw = self.params.notes[idx].pitch.value();
            let note = quantize(raw, scale, key);
            let ev = schedule_step(onset_beat, self.substeps[idx], gate, note,
                                   100 /* vel; or a param */);
            let on_smp = ((ev.on_beat - pos_beats) / beats_per_sample).round() as u32;
            let off_smp = ((ev.off_beat - pos_beats) / beats_per_sample).round() as u32;
            context.send_event(NoteEvent::NoteOn { timing: on_smp.min(buffer.samples() as u32 - 1),
                voice_id: None, channel: 0, note: ev.note.clamp(0,127) as u8, velocity: ev.vel as f32 / 127.0 });
            if off_smp < buffer.samples() as u32 {
                context.send_event(NoteEvent::NoteOff { timing: off_smp,
                    voice_id: None, channel: 0, note: ev.note.clamp(0,127) as u8, velocity: 0.0 });
            }
            // NoteOff landing in a later block must be tracked; see step 2.
        }
        ProcessStatus::Normal
    }
}
```

> This first cut drops NoteOffs that fall outside the current block. Step 2 fixes that.

**Step 2: Track pending NoteOffs across blocks**

Add `pending_off: Vec<(f64 /*off_beat*/, u8 /*note*/)>` to the struct; each block,
emit any whose `off_beat` falls within `[pos_beats, block_end)` and retain the rest.
This is required for correct note durations and to avoid stuck notes on transport stop
(flush all pending offs when `!t.playing`).

**Step 3: `cargo build`** — Expected: compiles.

**Step 4: Commit**

```bash
git add deepsteps-plugin/src/lib.rs
git commit -m "Stage 2: plugin process loop — transport clock, decoder regen, MIDI out"
```

---

### Task 14: CLAP + VST3 export + bundler

**Files:**
- Modify: `deepsteps-plugin/src/lib.rs`
- Create: `deepsteps-plugin/Cargo.toml` xtask wiring (or a `bundler.toml`)

**Step 1: Add export impls + macros**

```rust
impl ClapPlugin for DeepSteps {
    const CLAP_ID: &'static str = "dev.gruber.deepsteps";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("Autoencoder MIDI step sequencer");
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[ClapFeature::NoteEffect, ClapFeature::Utility];
}
impl Vst3Plugin for DeepSteps {
    const VST3_CLASS_ID: [u8; 16] = *b"DeepStepsGenMidi";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Instrument, Vst3SubCategory::Tools];
}
nih_export_clap!(DeepSteps);
nih_export_vst3!(DeepSteps);
```

**Step 2: Set up the nih-plug bundler** per nih-plug docs (xtask). Typically add an
`xtask` member crate or use `cargo-nih-plug`. Follow the current nih-plug README
exactly for the version pulled in Task 6.

**Step 3: Bundle**

Run: `cd deepsteps-plugin && cargo xtask bundle deepsteps-plugin --release`
Expected: produces `target/bundled/deepsteps-plugin.clap` and `.vst3`.

**Step 4: Commit**

```bash
git add deepsteps-plugin/src/lib.rs deepsteps-plugin/Cargo.toml deepsteps-plugin/xtask
git commit -m "Stage 2: export CLAP + VST3, wire nih-plug bundler"
```

---

## Phase E — Validation

### Task 15: clap-validator + host smoke test

**Step 1: Validate**

```bash
clap-validator validate deepsteps-plugin/target/bundled/deepsteps-plugin.clap
```
Expected: all checks pass (no errors).

**Step 2: Host smoke test in Carla**
- Load the `.clap` (or `.vst3`) in Carla as a MIDI plugin.
- Route its MIDI out to `aseqdump -p <carla port>`.
- Press play in Carla's transport.
- Expected: note-on/off stream in `aseqdump`; changing latent params changes the
  pattern; changing key/scale changes pitches.

**Step 3: Record findings** in `deepsteps-plugin/NOTES-sequencer.md` (observed vs expected).

**Step 4: Commit any fixes**, then:

```bash
git commit -am "Stage 2: validation fixes from clap-validator + Carla smoke test"
```

---

### Task 16: A/B against Stage 1 reference

**Step 1: Capture Stage 1 reference** for a fixed latent + params (reuse
`Deep_Steps_project/tools/run_midi_test.sh`, but pin the latent values rather than the
GUI defaults). Save the note sequence (pitch, step, timing).

**Step 2: Capture Stage 2** plugin output in Carla for the *same* latent + params
(same note pitches, key, scale, gate).

**Step 3: Compare**
- Active steps must match (same decoder + same threshold).
- Pitches after quantize must match.
- Relative step timing + substep offsets must match within a small tolerance.

**Step 4: Reconcile** any divergence by correcting the Rust sequencer constants
(Task 8 NOTES) — the Pd patch is the source of truth. Re-run until they agree.

**Step 5: Commit + document**

```bash
git add deepsteps-plugin/NOTES-sequencer.md
git commit -m "Stage 2: A/B verified against Stage 1 reference capture"
```

---

## Done criteria

- `cargo test -p deepsteps-plugin` green (decoder matches Python; sequencer unit tests).
- `clap-validator` passes on the `.clap`.
- Plugin loads in Carla, emits MIDI off host transport, params reactive.
- Stage 2 MIDI A/B-matches the Stage 1 reference for identical latent + params.
- `decoder.json` + `reference_vectors.json` committed; offline pipeline reproducible
  via `make_synth_dataset.py` → `train_export.py` (and `build_dataset.py` for a real
  audio corpus).

## Notes for the implementer

- The biggest fidelity risk is the **sequencer constants** (Task 8). Do that task
  carefully and let Tasks 9–11 tests encode what you find; the placeholder numbers in
  those tasks are explicitly marked to be replaced.
- nih-plug's API moves; verify `#[nested(array)]`, `transport()` accessors, and the
  bundler against the exact rev you pin in Task 6.
- Keep everything in `f64` through the decoder to match numpy; only convert to plugin
  types at the boundary.
