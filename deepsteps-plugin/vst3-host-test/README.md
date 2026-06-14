# vst3-host-test — headless scale verification

A minimal standalone VST3 host that `dlopen`s the **shipped** `deepsteps-plugin.vst3`
and verifies pitch quantization for all 14 scales end-to-end, in a real host
`process()` loop — not via the plugin's own unit tests. It mirrors the sibling
`clap-host-test` harness, feeding a VST3 `ProcessContext` instead of a CLAP transport.

For each scale it instantiates a fresh component, sets the `Scale` / `Key` params plus a
spread of 16 note pitches (covering every pitch class) via VST3 input parameter changes,
drives a *playing* host transport across one 4-beat bar, collects the emitted NoteOn
events, and asserts every emitted pitch is in-scale (its pitch class is a member of the
scale's table) and equals the reference snap-down — the same contract the CLAP harness
checks.

This exercises the real artifact: the bundled `.vst3` shared object, the VST3 COM ABI,
the nih-plug wrapper, param plumbing, and the sequencer/quantizer.

## Run

```bash
# Build the plugin bundle first (from deepsteps-plugin/):
cargo xtask bundle deepsteps-plugin --release

# Then run the host test (from this dir):
cd vst3-host-test
cargo run --release
# optional: pass an explicit path to a .vst3 .so
cargo run --release -- /path/to/deepsteps-plugin.vst3/Contents/x86_64-linux/deepsteps-plugin.so
```

Exit code 0 and `ALL 14 SCALES PASS` on success; non-zero with the offending
out-of-scale pitches on failure.

## Notes

- Pinned to the **same** `vst3-sys` git rev the plugin links, so the COM/C-ABI struct
  layouts are guaranteed to match. (cargo forbids `branch` + `rev` together, so we pin
  by `rev`; it resolves to the same commit as nih-plug's `fix/drop-box-from-raw` branch.)
- **Linux module lifecycle**: the host must call the exported `ModuleEntry(handle)` before
  `GetPluginFactory()`, and `ModuleExit()` at the end. nih-plug ignores the handle arg.
- **Host COM callbacks**: the input parameter changes / value queues and the output event
  list are real VST3 COM objects, built with the same `#[VST3(implements(..))]` co_class
  macro the SDK examples use. `Box::into_raw(Obj::allocate())` yields a `*mut Obj` whose
  first member is the vtable-ptr-ptr, which transmutes cleanly into the `StaticVstPtr<dyn
  I>` fields of `ProcessData`. The objects are kept alive across the whole run and freed
  afterwards.
- **Transport flags** are not in the bindings; we use SDK values: `kPlaying` (1<<1) |
  `kProjectTimeMusicValid` (1<<9) | `kTempoValid` (1<<10) = 1538, with `project_time_music`
  advanced in beats each block.
- Param values are normalized 0..1 via `controller.plain_param_to_normalized(id, plain)` —
  no hand-rolled normalization. Scale plain = scale index 0..13, Key = 0, Pitch = 60+i.
- Detached from the plugin workspace (own empty `[workspace]` table) so it is never pulled
  into the plugin's `cargo test`.
