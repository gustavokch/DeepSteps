# Building DeepSteps on Linux x86_64

Stage 1 of the Linux port: the original openFrameworks standalone app, building
and running on Linux x86_64 as a **runtime-only** MIDI generator (the offline
aubio corpus/dataset build and in-app training are disabled). Verified on
CachyOS (Arch) with **gcc 16.1.1** and openFrameworks **0.12.1**.

> Design + staging rationale: `docs/plans/2026-06-13-linux-port.md`.

## 1. openFrameworks

```bash
cd ~
curl -fL -o of.tar.gz \
  https://github.com/openframeworks/openFrameworks/releases/download/0.12.1/of_v0.12.1_linux64_gcc6_release.tar.gz
tar xzf of.tar.gz                      # -> ~/of_v0.12.1_linux64_gcc6_release  (= OF_ROOT)
```

System dependencies (Arch/CachyOS):

```bash
sudo pacman -S --needed glew assimp glfw-x11 uriparser rtaudio poco \
                        make pkgconf gcc openal freeglut curl pugixml brotli
# freeimage is AUR-only on Arch; OF links system -lfreeimage:
paru -S --needed freeimage
```

Build the OF core library:

```bash
cd ~/of_v0.12.1_linux64_gcc6_release/libs/openFrameworksCompiled/project
make Release -j$(nproc)
```

## 2. Addons

Clone into `OF_ROOT/addons/` (ofxGui + ofxOsc ship with OF):

```bash
cd ~/of_v0.12.1_linux64_gcc6_release/addons
git clone --depth 1 https://github.com/danomatika/ofxMidi.git
git clone --depth 1 https://github.com/danomatika/ofxPd.git          # vendors libpd + pure-data
git clone --depth 1 https://github.com/moebiussurfing/ofxSimpleSlider.git
```

### Required addon patches (gcc16 / Linux case-sensitivity)

These edits live in `OF_ROOT` (outside this repo), so they must be reapplied on
a fresh OF install:

1. **ofxPd — drop the pd "extra" externals.** They are ancient K&R C that gcc16
   rejects (`fiddle~.c`: `conflicting types for 'sqrt'`), and DeepSteps uses
   none of them. In `addons/ofxPd/addon_config.mk`, `common:` section:
   - remove `-DLIBPD_EXTRA` from `ADDON_CFLAGS`
   - set `ADDON_SOURCES_EXCLUDE = libs/libpd/pure-data/extra/%`
   (The `*_setup()` calls in `z_libpd.c` are guarded by `#ifdef LIBPD_EXTRA`, so
   dropping the define removes the references too.)

2. **ofxSimpleSlider — fix a case-sensitive include.** Linux is case-sensitive;
   `src/layoutCanvas.cpp` line 1 includes `"LayoutCanvas.h"` but the file is
   `layoutCanvas.h`. Change the include to lowercase `"layoutCanvas.h"`.

## 3. Python runtime (embedded, uv-managed)

The app embeds CPython (system `libpython3.14`) and at runtime imports `numpy`
and `python-osc`. Managed with **uv** (project `Deep_Steps_project/pyproject.toml`):

```bash
cd Deep_Steps_project
uv sync                 # creates .venv (python 3.14) with numpy + python-osc
```

`config.make` adds the embed flags via `python3-config --includes` /
`python3-config --embed --ldflags`, so the interpreter version is taken from the
system `python3`.

## 4. Build the app

```bash
cd Deep_Steps_project
make Release -j$(nproc)          # -> bin/Deep_Steps_project
```

## 5. Run

Force the GLFW **X11** backend (the bundled GLFW's Wayland path fails), point the
embedded interpreter at the uv venv, and run from `bin/` (the app resolves `data/`
relative to the cwd):

```bash
cd Deep_Steps_project/bin
env -u WAYLAND_DISPLAY DISPLAY=:1 \
    PYTHONPATH="$PWD/../.venv/lib/python3.14/site-packages" \
    ./Deep_Steps_project
```

The app has **no internal clock** — it sequences off incoming MIDI clock. Send
MIDI START + clock to its `ofxMidiIn` port and click **Generate**; notes appear
on its `ofxMidiOut` port.

## 6. Verify (automated)

`tools/run_midi_test.sh` launches the app, clicks Generate via `xdotool`, streams
MIDI clock, and captures note output with `tools/midi_verify.py` (rtmidi). Needs
`xdotool` (`sudo pacman -S xdotool`) and the venv extras (`uv add mido python-rtmidi`).
Expected: tens of `note-on`/`note-off` on the app's MIDI OUT.

## Disabled in this build

- **aubio** corpus/dataset build (`#if 0` in `ofApp.cpp`) — runtime-only port.
- **Audio**: the app still inits RtAudio (harmless underruns); it is a MIDI
  generator, the host/synth makes sound.
