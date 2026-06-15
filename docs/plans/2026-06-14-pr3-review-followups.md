# PR #3 review follow-ups (egui editor)

Plan to address findings from the PR #3 review
(<https://github.com/gustavokch/DeepSteps/pull/3#issuecomment-4702798431>).
PR is merged; these are follow-up fixes on `main`. Ordered by priority.

**Status:** items 1-6 done and pushed to `main` (commits `bab0bce`, `110578b`,
`23b1b38`). Item 7: VST3 render in Carla confirmed (screenshot in `docs/img/`);
interactive click/playhead + headless snapshot test still open, plus a new
oversized-host-window observation logged below. `cargo test` 16 pass,
`cargo build --release` warning-clean.

## 1. Clean stale scaffolding in `src/params.rs` — Medium ✅ (bab0bce)
The params struct is now wired into the plugin, so the "not yet wired" scaffolding
is obsolete and the blanket `allow(dead_code)` masks real dead code.

- [x] Remove `#![allow(dead_code)]` — build stays warning-clean, no dead items.
- [x] Rewrite the module doc — drop "not wired into a plugin yet (Task 13/14)".
- [x] Delete the `_ParamsArc` marker type + comment and the unused-`Arc` note.
- [x] `cargo build --release` clean.

## 2. Stop unconditional repaint in `src/editor.rs` — Low/Medium ✅ (110578b)
`ctx.request_repaint()` spins the GUI at full framerate even when stopped.

- [x] Gate the repaint on `shared.current() != NO_STEP` (`NO_STEP` imported).
- [ ] Manually confirm the playhead still animates during playback and the
      editor goes idle when stopped. *(covered by item 7 GUI render)*

## 3. Fix misleading latent-range comment in `src/params.rs` — Low ✅ (bab0bce)
Reference latents are all in `0.028..0.981`, so `0.0..1.0`/default 0.5 is correct;
no rescale is pending.

- [x] Reworded the `latent` closure comment: range matches the decoder's training
      domain (verified against `reference_vectors.json`); "-10..10 … rescale" gone.

## 4. Guard public bit-index API in `src/shared.rs` — Low ✅ (23b1b38)
`get`/`toggle` are `pub`; `1 << i` is unsound for `i >= 16`.

- [x] Added `debug_assert!(i < 16)` to `get` and `toggle` documenting the
      `0..16` invariant.

## 5. Note playhead last-write-wins in `src/lib.rs` — Low ✅ (23b1b38)
`set_current(idx)` in the step loop only leaves the last step of a block visible.

- [x] Added a comment at the `set_current` call documenting the last-write-wins
      limitation (deferred the wall-clock rewrite — comment sufficient for now).

## 6. Doc drift — Cosmetic ✅ (bab0bce + 110578b)
- [x] `params.rs`: reworded the `EDITOR_HEIGHT`/canvas comment — 600×520 (c42a1ff).
- [x] `editor.rs`: "4 columns" → "4 step columns" (grid is `num_columns(8)`).
- [x] No `600×640` figure remained in tracked docs.

## 7. Real verification gap — manual GUI render — Medium (process, not code) ✅ render confirmed
The PR shipped without a manual Carla render ("not yet done, pending reviewer check").

- [x] Loaded the VST3 in Carla (`carla-single native vst3`) on a live display;
      grid, all panels, and param values render correctly. Screenshot:
      [`docs/img/stage3-egui-carla-render.png`](../img/stage3-egui-carla-render.png).
      (Carla has no CLAP support on this box; CLAP render unverified but shares the
      same egui editor code path.)
- [ ] Still unverified interactively: click-to-toggle a cell, and playhead tracking
      during host playback (standalone load has no running transport).
- [ ] If feasible, add a headless egui render/snapshot smoke test so this isn't
      manual-only going forward.

### New observation from the render
- **Host window oversized vs editor.** Carla opens the plugin window far larger than
  the 600×520 `EguiState`; the egui content sits top-left and the rest is black (see
  screenshot). The egui layout itself is fine — the VST3 wrapper isn't reporting/
  constraining the host window to the `EguiState` size. Worth a follow-up: confirm
  whether `EguiState::from_size` is honored by the nih-plug VST3 wrapper at this rev,
  or whether the window needs an explicit resize request.

### Known issue: pluginval Editor test segfaults headless (fixed in CI)
- **Symptom.** CI pluginval step crashed with `Segmentation fault (core dumped)`
  right at `Starting tests in: pluginval / Editor...` (strictness 8). The earlier
  `Open plugin (cold/warm)` tests passed, so the crash was in editor *paint*, not
  creation.
- **Root cause.** Not a plugin bug. pluginval's Editor test opens the egui editor,
  which creates a real X11/GL window via baseview. CI ran pluginval with no
  `DISPLAY`, so window creation derefs null → SIGSEGV. Confirmed by reproduction:
  `xvfb-run pluginval … --strictness-level 8` → SUCCESS; `env -u DISPLAY
  -u WAYLAND_DISPLAY pluginval …` → exit 139 (SIGSEGV) at the Editor test. The
  resizable-window / text-scale commits are *not* implicated — they run fine with
  a display.
- **Resolution: `--skip-gui-tests`.** `.github/workflows/ci.yml` runs pluginval
  with `--skip-gui-tests`; no display/GL needed. Verified locally: headless
  `pluginval --strictness-level 8 --skip-gui-tests` → SUCCESS (exit 0). All
  non-GUI conformance (params, buses, state, fuzz, threading) still runs. The
  editor itself is verified manually in Carla (screenshot above).
- **Why not headless GL (xvfb) — investigation trail.** Two attempts to give CI a
  real GL stack both failed; recorded so nobody re-treads them:
  1. **`xvfb` alone.** With a display the Editor test still aborted (exit 134):
     `Could not fetch framebuffer config: CreationFailed(InvalidFBConfig)`
     (`baseview/src/x11/visual_info.rs:30`), then a non-unwinding panic across the
     C FFI boundary → `Aborted (core dumped)`.
  2. **`xvfb` + `libgl1-mesa-dri` + `LIBGL_ALWAYS_SOFTWARE=1` + 24-bit GLX screen.**
     Same `InvalidFBConfig`. CI logs confirmed the Mesa software stack (llvmpipe /
     `mesa-libgallium`) *did* install — so "missing DRI driver" was not the cause.
  - **Actual root cause.** baseview asks GLX (`glXChooseFBConfig`) for an RGBA
    config. Client-side direct software rendering (llvmpipe/drisw) is what supplies
    FBConfigs; it works on a dev box with a `/dev/dri` render node but the GHA
    runner has none, so GLX falls back to the X server's *indirect* path. Modern
    Mesa has dropped indirect/server-side software GLX, so the server advertises
    GLX with **zero** FBConfigs → `InvalidFBConfig`. Reproduced locally by forcing
    `LIBGL_ALWAYS_INDIRECT=1` (0 FBConfigs); confirmed `+iglx` does **not** help
    (still 0). Not fixable without a GPU/DRM node on the runner — hence skip.

## Suggested batching
- One commit: items 1 + 3 + 6 (params.rs cleanup, all doc/comment).
- One commit: item 2 (editor repaint gating) + item 6's editor.rs comment.
- One commit: items 4 + 5 (shared.rs guard + lib.rs playhead comment).
- Item 7 is a separate verification task, not a code commit.
