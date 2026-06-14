# PR #3 review follow-ups (egui editor)

Plan to address findings from the PR #3 review
(<https://github.com/gustavokch/DeepSteps/pull/3#issuecomment-4702798431>).
PR is merged; these are follow-up fixes on `main`. Ordered by priority.

## 1. Clean stale scaffolding in `src/params.rs` — Medium
The params struct is now wired into the plugin, so the "not yet wired" scaffolding
is obsolete and the blanket `allow(dead_code)` masks real dead code.

- [ ] Remove `#![allow(dead_code)]` (line 6). Rebuild; fix or `#[allow]` each
      genuinely-unused item the compiler then flags (narrow, not blanket).
- [ ] Rewrite the module doc (lines 4-6) — drop "not wired into a plugin yet
      (Task 13/14)". Describe what the module is now.
- [ ] Delete the `_ParamsArc` marker type + comment (lines 165-168) and the
      "silence the unused-`Arc` import" note — `Arc<EguiState>` is a live field.
- [ ] `cargo build --release` + `cargo clippy` clean.

## 2. Stop unconditional repaint in `src/editor.rs` — Low/Medium
`ctx.request_repaint()` spins the GUI at full framerate even when stopped.

- [ ] Only request a repaint while the transport is playing:
      `if state.shared.current() != NO_STEP { ctx.request_repaint(); }`
      (import `NO_STEP` from `crate::shared`).
- [ ] Manually confirm the playhead still animates during playback and the
      editor goes idle when stopped.

## 3. Fix misleading latent-range comment in `src/params.rs` — Low
Reference latents are all in `0.028..0.981`, so `0.0..1.0`/default 0.5 is correct;
no rescale is pending.

- [ ] Reword the `latent` closure comment (lines ~129-135): the range matches the
      decoder's training domain (verified against `reference_vectors.json`); remove
      the "-10..10 … Task 13 will rescale" claim.

## 4. Guard public bit-index API in `src/shared.rs` — Low
`get`/`toggle` are `pub`; `1 << i` is unsound for `i >= 16`.

- [ ] Add `debug_assert!(i < 16)` to `get` and `toggle`, or document the
      `0..16` invariant on both.

## 5. Note playhead last-write-wins in `src/lib.rs` — Low
`set_current(idx)` in the step loop only leaves the last step of a block visible.

- [ ] Add a comment at `lib.rs:210` documenting the limitation, **or** publish the
      playhead from wall-clock transport position instead of the step loop so it's
      block-size-independent. Comment is sufficient for now.

## 6. Doc drift — Cosmetic
- [ ] `params.rs:12`: reword the `EDITOR_HEIGHT`/canvas comment — current size is
      600×520 (commit c42a1ff), not the Stage-1 canvas.
- [ ] `editor.rs:73`: "4 columns" → "4 step columns" (grid is `num_columns(8)` =
      4 label+slider pairs).
- [ ] If kept anywhere, correct the `600×640` figure in docs to `600×520`.

## 7. Real verification gap — manual GUI render — Medium (process, not code)
The PR shipped without a manual Carla render ("not yet done, pending reviewer check").

- [ ] Load the bundled CLAP/VST3 in Carla; confirm the grid draws, cells toggle on
      click, and the playhead tracks during host playback. Capture a screenshot.
- [ ] If feasible, add a headless egui render/snapshot smoke test so this isn't
      manual-only going forward.

## Suggested batching
- One commit: items 1 + 3 + 6 (params.rs cleanup, all doc/comment).
- One commit: item 2 (editor repaint gating) + item 6's editor.rs comment.
- One commit: items 4 + 5 (shared.rs guard + lib.rs playhead comment).
- Item 7 is a separate verification task, not a code commit.
