# clap-host-test — headless scale verification

A minimal standalone CLAP host that `dlopen`s the **shipped** `deepsteps-plugin.clap`
and verifies pitch quantization for all 14 scales end-to-end, in a real host
`process()` loop — not via the plugin's own unit tests.

For each scale it instantiates a fresh plugin, sets the `Scale` / `Key` params plus a
spread of 16 note pitches (covering every pitch class) via CLAP param-value events,
drives a *playing* host transport across one 4-beat bar, collects the emitted NoteOn
events, and asserts every emitted pitch is in-scale (its pitch class is a member of
the scale's table) and equals the reference snap-down.

This exercises the real artifact: the bundled `.clap` shared object, the CLAP ABI,
the nih-plug wrapper, param plumbing, and the sequencer/quantizer.

## Run

```bash
# Build the plugin bundle first (from deepsteps-plugin/):
cargo xtask bundle deepsteps-plugin --release

# Then run the host test (from this dir):
cd clap-host-test
cargo run --release
# optional: pass an explicit path to a .clap
cargo run --release -- /path/to/deepsteps-plugin.clap
```

Exit code 0 and `ALL 14 SCALES PASS` on success; non-zero with the offending
out-of-scale pitches on failure.

## Notes

- Pinned to the **same** `clap-sys` git rev the plugin links, so the C-ABI struct
  layouts are guaranteed to match.
- nih-plug zero-bases `IntParam` CLAP values: the note Pitch param's CLAP range is
  `0..72` (= `midi_note - 24`), so the harness offsets pitch values by `PITCH_MIN`.
- Detached from the plugin workspace (own empty `[workspace]` table) so it is never
  pulled into the plugin's `cargo test`.
