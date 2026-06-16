# DeepSteps

<img src="DS-UI.png" width="400" height="400">

A MIDI step sequencer whose patterns are produced by an integrated, user-trainable
generative neural network (a small autoencoder).

Originally created by **Alex Wasatnidge** as part of their Master's thesis for the
Music, Communication and Technology programme at the University of Oslo, as a macOS
[openFrameworks](https://openframeworks.cc) standalone app. You can read the original
blog post [here](https://mct-master.github.io/masters-thesis/2024/05/14/alexanjw-DeepSteps.html).

> **This fork is a Linux x86_64 port.** It adds (Stage 1) a Linux build of the
> original standalone and (Stage 2) a from-scratch Rust rewrite as a **CLAP + VST3
> plugin**. See **Project status** below.

## Project status

| Stage | What | State |
|-------|------|-------|
| **Stage 1** | Original openFrameworks standalone, building/running on Linux x86_64 (runtime only — offline training UI disabled) | Builds & runs on CachyOS/Arch with gcc 16 + openFrameworks 0.12.1. See [`docs/BUILDING-linux.md`](docs/BUILDING-linux.md). |
| **Stage 2** | Rust rewrite as a CLAP + VST3 **MIDI-generator** plugin (nih-plug). Reuses no C++/Pd/Python at runtime. | Working. 34 cargo + 8 pytest tests, clippy clean, `clap-validator` 18/0/3, `pluginval` (VST3) SUCCESS, and headless CLAP + VST3 host tests passing all 14 scales. CI green. |
| **Stage 3** | **In-plugin training**: build a dataset and train the autoencoder live in the DAW, then play *your* model. The Python training pipeline reimplemented from scratch in Rust. | Working. Trains off the audio thread, hot-swaps the result, persists with the session. See [`docs/plans/2026-06-16-runtime-training-design.md`](docs/plans/2026-06-16-runtime-training-design.md). |

The two stages share no runtime code. Stage 1 is the behavioural reference; Stage 2
is the plugin you actually install in a DAW.

## The plugin (Stage 2)

A **MIDI generator**: it emits notes; your host/synth makes the sound. It has **no
internal clock** — it follows the **host transport** (tempo + playhead). Press play
in your DAW and it sequences.

**How it works.** An autoencoder **decoder** turns 4 latent parameters into a 16-step
pattern (which steps fire + a per-step "groove" sub-step offset). The sequencer plays
that pattern at 4 steps/beat (16 per bar), quantising each step's pitch to a selected
scale + key. The plugin ships with a frozen, offline-trained decoder as the default, and
you can **train your own model in-session** (see *Training*, below).

**Custom GUI** (egui editor): a 16-step grid with a live playhead and click-to-toggle
cells (a click forces a step on/off, overriding the decoder until the next latent-driven
regeneration), plus sliders for the latent vector, per-step pitches, timing, and tuning,
and a **Training** panel.

## Training (in-plugin)

The plugin reimplements the original's autoencoder *training* in pure Rust — no Python,
no offline step. In the **Training** panel:

1. **Build a dataset.** *Capture pattern* snapshots the current 16-step grid (steps +
   sub-step offsets) as a training sample, and/or *Add audio…* loads `.wav`/`.flac` files,
   detects onsets (spectral-flux), and encodes each file as a sample. The dataset
   accumulates across both sources.
2. **Train.** Set epochs/batch and press *Train*. Training runs on a background thread
   (never the audio thread), showing a live epoch/loss progress bar; *Cancel* stops it.
3. **Play your model.** On finish the new decoder is **hot-swapped** into the audio path
   (lock-free) and drives the latent sliders immediately. *Encode pattern → latent* runs
   the encoder on the current grid to set the 4 latents to that pattern's latent code.

The trained model is **saved with the DAW session** (and travels with presets); reloading
restores it. The baked default decoder remains the fallback when no model has been trained.

> Faithfulness note: the Rust training fixes two bugs in the original Python (broken Adam
> bias correction; no batch shuffle), so it converges better but does not reproduce the
> Python numerically. The offline Python pipeline under `Deep_Steps_project/tools/` still
> exists for reference.

**Parameters** (also host-automatable): Latent A–D, Gate length (ms),
Sub-step scale, Sequence length (1–16), Key (0–11), Scale (14 options: Chromatic,
Pentatonic Major/Minor, Major, Natural/Harmonic/Melodic Minor, Dorian, Phrygian,
Lydian, Mixolydian, Locrian, Blues, Whole Tone), and 16 per-step note pitches.
Output: NoteOn/NoteOff, velocity 100, MIDI channel 1.

## Installation (Linux x86_64)

### Plugin — from a release

Download `deepsteps-plugin-v0.1.1-linux-x86_64.zip` from
[Releases](https://github.com/gustavokch/DeepSteps/releases), then:

```bash
mkdir -p ~/.clap ~/.vst3
unzip deepsteps-plugin-v0.1.1-linux-x86_64.zip
cp    deepsteps-plugin.clap  ~/.clap/
cp -r deepsteps-plugin.vst3  ~/.vst3/
```

Rescan plugins in your host (Carla, Bitwig, Reaper, …). It appears as **DeepSteps**.

### Plugin — from source

Needs a [Rust](https://rustup.rs) toolchain (stable) and these system packages
(Debian/Ubuntu names; the CI installs the same set):

```bash
sudo apt-get install -y libasound2-dev libgl-dev libjack-jackd2-dev \
  libx11-xcb-dev libxcb1-dev libxcb-icccm4-dev \
  libxcursor-dev libxkbcommon-dev libxcb-shape0-dev libxcb-xfixes0-dev
```

Then build the bundle and install it:

```bash
cd deepsteps-plugin
cargo xtask bundle deepsteps-plugin --release
cp    target/bundled/deepsteps-plugin.clap ~/.clap/
cp -r target/bundled/deepsteps-plugin.vst3 ~/.vst3/
```

### Standalone (Stage 1)

The original openFrameworks app on Linux — see
[`docs/BUILDING-linux.md`](docs/BUILDING-linux.md) for the full toolchain (OF 0.12.1,
addons, embedded Python via [uv](https://docs.astral.sh/uv/)). It has no internal
clock and sequences off **incoming MIDI clock**.

## Known issues / pending

- **Step toggles are not preset-persisted.** Grid clicks override the decoder at
  runtime but are not saved in presets (they are runtime state, not params) and a
  latent move regenerates over them. Promoting them to params is a possible later pass.
- **Default weights are from a synthetic dataset.** The committed default decoder
  `deepsteps-plugin/weights/decoder.json` was trained offline on a deterministic
  *synthetic* corpus (`Deep_Steps_project/tools/make_synth_dataset.py`), so out-of-the-box
  patterns are reproducible but not musically trained. Train your own **in the plugin**
  (see *Training*) — or, offline, with the Python tools
  `Deep_Steps_project/tools/build_dataset.py` + `train_export.py` ([librosa](https://librosa.org)
  onsets). The in-plugin onset detector is spectral-flux based and intentionally not a
  librosa clone, so audio-derived datasets are approximate.
- **Two sequencer timing approximations** (flagged for A/B in
  [`deepsteps-plugin/NOTES-sequencer.md`](deepsteps-plugin/NOTES-sequencer.md) and
  [`VALIDATION.md`](deepsteps-plugin/VALIDATION.md)): the sub-step offset uses a
  continuous beat offset vs the Pd patch's integer-pulse (48 PPQN) truncation; and a
  step landing exactly on a process-block boundary could in principle double/drop.
  Neither has been observed; both want a host A/B check.

## Validation

See [`deepsteps-plugin/VALIDATION.md`](deepsteps-plugin/VALIDATION.md). Automated:
`cargo test` (34), `clap-validator` (18/0/3) and `pluginval` (VST3, strictness 8,
SUCCESS), plus headless host scale tests that load the **shipped** binaries and assert
all 14 scales quantise correctly through both plugin formats — `clap-host-test/` (CLAP)
and `vst3-host-test/` (VST3). All run in CI on every push/PR.

## Original macOS build

The repo still contains the Xcode project (`Deep Steps.xcodeproj`) and the
openFrameworks sources under `Deep_Steps_project/`. To build the original on macOS
you need the matching openFrameworks plus the add-ons/libraries below; Python is
embedded via the ["Very High Level Embedding"](https://docs.python.org/3/extending/embedding.html#)
API, and aubio is used for the offline corpus.

- [openFrameworks](https://openframeworks.cc) — C++ creative-coding toolkit
- [ofxMidi](https://github.com/danomatika/ofxMidi) — MIDI in/out add-on
- [Pure Data](https://puredata.info) + [ofxPd](https://github.com/danomatika/ofxPd) — embedded Pd patch (the sequencer brain)
- [aubio](https://aubio.org) — audio onset analysis (offline corpus only)
- [Python](https://www.python.org) + [python-osc](https://pypi.org/project/python-osc/) — embedded interpreter + OSC

## Credits

Original concept, design, and implementation: **Alex Wasatnidge** (University of Oslo).
Linux/CLAP/VST3 port: this fork.
