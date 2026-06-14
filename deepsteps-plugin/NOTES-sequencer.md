# NOTES: DeepSteps Sequencer — exact constants extracted from `main.pd`

Source of truth for the Rust sequencer port (Stage 2, Tasks 9–11). Every claim below
cites line numbers in `Deep_Steps_project/bin/data/pd/main.pd` (top-level object indices
in brackets, assigned in file order starting at 0 — `#X obj/msg/floatatom/...` increment
the index; `#N canvas ... #X restore` subpatches count as one object at their `#X restore`).

Cross-referenced against:
- `Deep_Steps_project/bin/data/AE_init.py` `receive_latent` (lines 383–402) — what the decoder sends.
- `Deep_Steps_project/src/ofApp.cpp` — what C++ pushes into Pd and how Pd output reaches MIDI.

## 0. Object index map (top-level, the sequencer chain)

| idx | line | object | role |
|----|------|--------|------|
| 0  | 2   | `f`              | step-counter accumulator (current pulse) |
| 1  | 3   | `+ 1`            | increments counter each clock bang |
| 9  | 11  | `r tempo`        | BPM in (from C++ `clock.getBpm()`) |
| 10 | 12  | `expr 60/$f1*1000` | BPM → ms/quarter (top-level copy; the live one is inside `pd midi in`) |
| 11 | 13  | `mod 192`        | wrap counter to pattern length (divisor overridden by `len*12`) |
| 12 | 14  | `loadbang`       | inits step-index constants + first sequence bit |
| 13 | 15  | `s clock`        | broadcasts wrapped pulse 0..(len*12-1) to all `pd step` |
| 17 | 19  | `sel 5`          | GUI display step counter (`s toOF`), display-only |
| 18 | 20  | `mod 16`         | GUI display step wrap (divisor overridden by `len`) |
| 22 | 24  | `s toOF`         | GUI step highlight → OF, NOT note logic |
| 23 | 25  | `unpack f...f, f 55` | `/sequence` 16 on/off bits → each `pd step` inlet 1 |
| 25 | 27  | `mod 12`         | feeds GUI display counter (`sel 5`) |
| 26,29..43 | 57,89.. | 16× `pd step`  | per-step timing + gate; outlet bangs the pitch `int` |
| 27 | 28  | `unpack f...f, f 50` | `/steps` 16 substep floats → each `pd step` inlet 2 |
| 28 | 59  | `noteout`        | final MIDI note out (to OF `receiveNoteOn`) |
| 44..59 | 512.. | 16× `r noteN`  | per-step base pitch from OF sliders (24..127) |
| 60..75 | 528.. | 16× `int`      | holds noteN; banged by matching `pd step` outlet |
| 76 | 544 | `msg 1`          | MIDI channel 1 → `noteout` inlet 2 |
| 77 | 545 | `makenote 100 200` | velocity 100, duration 200 ms (duration overridden by `gate`) |
| 78..93 | 546.. | 16× `f 0`..`f 15` | constant step index N → `pd step` inlet 0 |
| 96 | 578 | `route sequence` | splits `/sequence` OSC → `unpack`(23) |
| 97 | 579 | `route steps`    | splits `/steps` OSC → `unpack`(27) |
| 98 | 580 | `r gate`         | gate-length ms (from OF `gateLength` slider) |
| 99 | 581 | `change`         | dedupe gate value → `makenote` inlet 2 (duration) |
| 100| 582 | `vradio` 0..11   | quantised key selector → `s key` |
| 101| 583 | `s key`          | key offset 0..11 |
| 110| 738 | `pd quant`       | pitch quantizer; outlet → `makenote`(77) inlet 0 |
| 111| 739 | `hsl 0..6`       | sub-steps scaling slider → `s scale` (note: separate from OF `scale`) |
| 112| 740 | `s scale`        | substep swing amount → each `pd step`'s `r scale` |
| 115| 743 | `pd midi out`    | local play/stop transport (UI toggle), not used by OF clock path |
| 119| 780 | `pd midi in`     | parses incoming MIDI clock; emits pulses to counter |
| 124| 810 | `r len`          | sequence length 1..16 (from OF `seqLength` slider) |
| 125| 811 | `* 12`           | len → `mod 192` right inlet (sets pattern length in pulses) |

Connects that establish the chain (lines 812–979): `119 0 0 0` (clock→counter),
`0 0 1 0`/`1 0 0 1` (accumulate), `0 0 11 0`→`11 0 13 0` (counter→mod192→`s clock`),
`23 N 26..43 inlet1` (sequence bits→steps), `27 N 26..43 inlet1/2` (substeps→steps),
`78..93 0 26..43 0` (step index→steps), `44..59 0 60..75 1` (noteN→int right inlet),
`26..43 0 60..75 0` (step bang→int trigger), `60..75 0 110 0` (pitch→quant),
`110 0 77 0` (quant→makenote), `77 0 28 0`/`77 1 28 1`/`76 0 28 2` (makenote→noteout),
`98 0 99 0 0 77 2` (gate→change→makenote duration), `124 0 18 1`+`124 0 125 0 0 11 1` (len).

---

## 1. Clock → step counter

### How pulses arrive (C++ side)
`ofApp::newMidiMessage` (ofApp.cpp:1200) forwards **one** Pd MIDI byte `0xF8` (248) per
incoming MIDI realtime clock tick: `pd.sendMidiByte(0, 248);` (ofApp.cpp:1251), only when
`clock.update(message.bytes)` reports a new tick. Standard MIDI clock = **24 PPQN**, so OF
delivers 24 `0xF8` per quarter note into Pd. Live tempo is pushed every message via
`pd.sendFloat("tempo", clock.getBpm());` (ofApp.cpp:1245). The GUI `tempoSlider` is **dead**:
`//pd.sendFloat("tempo", tempoSlider.getValue());` is commented out (ofApp.cpp:263).

### `pd midi in` doubles the clock (the subtle bit) — lines 780–805
Locals: 1=`select 248 250 252`, 5=`midiin`, 7=`t b b`, 8=`expr 60/$f1*1000`,
9=`/ 48`, 10=`delay`, 11=`int` (tempo), 2=outlet(clock), 3=outlet(start).

- `midiin`(5) → `select 248 250 252`(1) (line 782): 248=clock, 250=start, 252=stop.
- On a 248 match: `1 0 7 0` → `t b b`(7). `t b b` outlet **1** (fires first) → `7 1 2 0` →
  **clock outlet immediately** (one pulse). `t b b` outlet **0** → `7 0 10 0` → `delay`(10);
  `delay`(10) → `10 0 2 0` → **clock outlet again after the delay** (a second pulse).
- Delay time = `r tempo`→`int`(11)→`expr 60/$f1*1000`(8)→`/ 48`(9)→`delay` right inlet
  (`8 0 9 0`, `9 0 10 1`). `60/bpm*1000` = ms per quarter; `/48` = ms per 1/48-quarter.

Net effect: each incoming 24-PPQN tick produces **two** `s clock` pulses (one now, one a
half-tick later). Effective internal resolution = **48 PPQN**.

> VERIFY in Task 16: confirm by A/B that the doubling actually halves step duration as
> reasoned (i.e. step = 16th note, not 8th note). The reading is consistent but the
> single-shot `delay` re-trigger behavior is the one place worth a live check.

### Counter and wrap — lines 2,3,11,13
`pd midi in` clock outlet → counter `f`(0) (`119 0 0 0`). Each pulse bangs `f`(0) →
`+ 1`(1) → back into `f` right inlet, so the counter increments by 1 per pulse.
Counter → `mod 192`(11) (`0 0 11 0`) → `s clock`(13) (`11 0 13 0`).

- 192 pulses per pattern ÷ 48 PPQN = **4 quarter notes = 1 bar (4/4)**.
- 192 ÷ 16 steps = **12 pulses per step**. 12 pulses ÷ 48 PPQN = a **16th note** per step.
- So: **one full 16-step pattern = exactly 1 bar; 12 pulses (= 1/16 note) per step.**

The `mod 192` divisor is **not hardcoded for variable length** — see §6.

---

## 2. Step → which note fires (the `pd step` subpatch)

Each of the 16 `pd step` instances is identical (subpatch at lines 28–57, replicated
89–511). Local indices: 0=inlet(step idx N), 1=`+`, 2=`sel`, 3=`* 12`, 4=`r clock`,
5=outlet, 6=`spigot`, 7=inlet(gate bit), 8=`tgl`, 9=inlet(substep), 10=`* -1`,
11=`expr $f1*($f2-$f3)+$f3`, 12=`int`, 13=`r scale`.

Data flow (connects lines 43–56):
- inlet0 (step index N, a constant from `f N` boxes 78–93) → `* 12`(3) → `+`(1) right inlet.
  So the base trigger pulse for step N = **N × 12** (the on-grid pulse for that 16th).
- `+`(1) → `sel`(2) right inlet (sets the value to match).
- `r clock`(4) → `sel`(2) left inlet. When the live pulse from `s clock` equals
  `N*12 + substep_offset`, `sel` fires a bang → `spigot`(6) → outlet(5).
- inlet1 (the `/sequence` on/off bit for this step) → `tgl`(8) → `spigot`(6) right inlet
  (`8 0 6 1`). **The spigot is the per-step on/off gate**: if the sequence bit is 0 the
  bang is blocked and the step is silent. (16 spigots total, one per `pd step`.)
- The outlet bang (`pd step` outlet → top-level) triggers the pitch `int`(60–75) which
  emits the held `r noteN` value into `pd quant` → `makenote`.

So: **step index → on-grid pulse N*12; spigot gates it by the `/sequence` bit; when the
clock reaches the (offset) pulse the step's `int` releases its base pitch noteN.**

---

## 3. Substep timing (the lerp)

Inside `pd step` (lines 40, 9–11, 39): `expr $f1 * ($f2 - $f3) + $f3` with
- `$f1` = inlet2 = the **substep value 0..1** for this step (from `/steps`).
- `$f2` = `r scale`(13) value = the **swing/scaling amount** S (from `s scale`, the
  Sub-Steps Scaling `hsl 0..6`, idx 111).
- `$f3` = `r scale` × −1 via `* -1`(10) = **−S** (`13 0 10 0`, `10 0 11 2`).

So offset = `substep*(S − (−S)) + (−S)` = **`S * (2*substep − 1)`**.

| substep | offset (pulses) |
|---------|-----------------|
| 0.0     | −S |
| **0.5** | **0  (on-grid)** ✓ |
| 1.0     | +S |

Then `expr` → `int`(11→12, truncates to whole pulses) → `+`(1) left inlet, added to
`N*12`. The result `N*12 + round(S*(2*substep−1))` is the pulse the `sel` waits for.

- Corpus encoder convention (substep 0.5 = on grid) is satisfied: at 0.5 the offset is 0.
- Range: substep ∈ [0,1] maps to offset ∈ **[−S, +S] pulses**, S = the scaling slider
  (0..6). With 12 pulses/step and default S, e.g. S=6 gives ±6 pulses = ±half a step.
- The offset is applied **by shifting which clock pulse the `sel` matches** (i.e. it
  advances/retards the trigger by whole pulses), NOT via a separate `delay` per step.
  (The only `delay` is the clock-doubling delay in `pd midi in`, §1.)

> The substep float is whatever the decoder outputs in `substeps` (AE_init.py:391, 402) —
> a sigmoid output, nominally [0,1]. `/steps` is sent raw (not thresholded), unlike
> `/sequence` which is thresholded at 0.5 (AE_init.py:395–401).

> VERIFY in Task 16: exact rounding of `int` (truncation toward zero vs floor) for
> negative offsets, and the effective S range actually in use.

---

## 4. Pitch quantization (`pd quant`, lines 592–738)

The raw note (held `r noteN`, 24..127) enters `pd quant` inlet (local 10). Three parallel
branches, each gated by a spigot driven by `r scalenotes`(local 53) compared with
`== 0` / `== 1` / `== 2`. `scalenotes` comes from OF `pd.sendFloat("scalenotes", scale)`
(ofApp.cpp:282) — an integer scale-type selector.

Scale tables present:
- `sel 0 3 5 7 10` (lines 598, 622, 629) — **pentatonic minor**.
- `sel 0 2 4 7 9` (lines 638, 639) — **pentatonic major**.
- `sel 0 1 2 3 4 5 6 7 8 9 10 11` (line 619) — **chromatic** (always matches → pass-through).

### Quantization rule (snap-down to nearest in-scale pitch class)
Per branch (minor branch shown, locals 1,29,2,3,6,5,7,8,4):
1. note → `% 12`(1) → pitch class pc.
2. `sel 0 3 5 7 10`(29): if pc ∈ scale, match outlet → `msg 0`(2) → add **0** offset
   (already in scale).
3. if pc ∉ scale, `sel` right (no-match) outlet → floatatom(3) → `- 1`(6) → a second
   `sel 0 3 5 7 10`(5) re-tests pc−1; on match emits the accumulated negative offset
   (via `msg -1`/`msg -2`, lines 623/601), looping down until an in-scale pc is found.
4. The accumulated offset (0, −1, or −2…) is added to the **original** note at `+`(4/12),
   so the note is lowered to the nearest in-scale pitch **at or below** it.

The chromatic branch (`sel 0..11`) always matches → offset 0 → note passes unchanged.

### Stage-2 addition: extra scales (not in the Pd patch)
The Pd patch ships only the three scales above. The Rust port (commit `59afbc5`)
keeps those three and adds 11 more, using the **same snap-down rule** (snap pitch
class to the nearest scale member at or below it, keep octave, add key). Each table
is sorted ascending from 0, so `sequencer::quantize` is unchanged — only the table set
grows. Full set (14), with semitone tables:

| scale | pitch classes |
|-------|---------------|
| Chromatic | 0 1 2 3 4 5 6 7 8 9 10 11 |
| Pentatonic Major | 0 2 4 7 9 |
| Pentatonic Minor | 0 3 5 7 10 |
| Major (Ionian) | 0 2 4 5 7 9 11 |
| Natural Minor (Aeolian) | 0 2 3 5 7 8 10 |
| Harmonic Minor | 0 2 3 5 7 8 11 |
| Melodic Minor | 0 2 3 5 7 9 11 |
| Dorian | 0 2 3 5 7 9 10 |
| Phrygian | 0 1 3 5 7 8 10 |
| Lydian | 0 2 4 6 7 9 11 |
| Mixolydian | 0 2 4 5 7 9 10 |
| Locrian | 0 1 3 5 6 8 10 |
| Blues | 0 3 5 6 7 10 |
| Whole Tone | 0 2 4 6 8 10 |

To add another: edit `Scale` enum + `table()` (`src/sequencer.rs`), `ScaleParam` enum
(`src/params.rs`), and `map_scale` (`src/lib.rs`). Keep the table ascending from 0.

### Key offset
`r key`(local 28) ← `s key`(idx 101) ← `vradio 0..11`(idx 100, "Quantised Key", C..B,
lines 582–591). `key` → `sel 0..11`(26) → one of `msg 0..11`(13–25,27) → `+`(12) right
inlet. So **key adds 0..11 semitones** as a transpose on top of the snapped note. OF also
pushes it directly: `pd.sendFloat("key", key)` (ofApp.cpp:281).

**Rule summary:** quantize = snap the note's pitch class DOWN to the nearest scale member
(pent-minor / pent-major / chromatic per `scalenotes`), keeping the octave, then add `key`
semitones. Chromatic = identity (key only).

> VERIFY in Task 16 (medium priority): the exact mapping of `scalenotes` integer →
> which branch (`==0` chromatic vs `==1` minor vs `==2` major). The three `==N`/spigot
> wires (connects 729–737) are dense; the *rule* (snap-down + key) is solid, but confirm
> the index→scale assignment and that snapping is strictly downward (no nearest-either-way)
> via an A/B with a known out-of-scale input note.

---

## 5. Note output

Chain: `pd quant`(110) → `makenote 100 200`(77) inlet0 (pitch) → `noteout`(28).
- `makenote 100 200` (line 545): **velocity = 100**, **duration = 200 ms** (defaults/args).
- Velocity 100 is **fixed** — `makenote` inlet1 (velocity) has no incoming connection.
- **Duration is overridden by the Gate Length slider**: `r gate`(98) → `change`(99) →
  `makenote` inlet2 (`98 0 99 0`, `99 0 77 2`). OF: `pd.sendFloat("gate", gateLength.getValue())`
  (ofApp.cpp:286); `gateLength` range 1..1000 ms, default 100 (ofApp.cpp:214). `change`
  only forwards when the value differs (dedupe).
- `makenote` outlet0 (note-on pitch) → `noteout`(28) inlet0; outlet1 (velocity) → inlet1
  (`77 0 28 0`, `77 1 28 1`); `msg 1`(76) → inlet2 = **MIDI channel 1** (`76 0 28 2`).
- `makenote` schedules the matching note-off after the duration automatically.

### Reaching real MIDI out (ofApp.cpp)
Pd `noteout` is delivered to OF via `pd.receiveMidi()` (ofApp.cpp:259), surfacing as
`ofApp::receiveNoteOn(channel, pitch, velocity)` (ofApp.cpp:1171), which calls
`midiOut.sendNoteOn(channel, pitch, velocity)` (ofApp.cpp:1173). Channel is whatever Pd
sent (1), pitch = quantized+key note, velocity = 100.

**For the Rust port:** emit NoteOn(ch=1, pitch=quantized, vel=100) at the step's
offset pulse, and a NoteOff after `gate` ms (1..1000, default 100), NOT after the makenote
default 200 (the slider overrides it whenever the user has touched gate; default slider =
100 ms, so effective default gate = 100 ms, not 200 ms).

---

## 6. Sequence length & tempo

### Sequence length (variable, overrides `mod 192` / `mod 16`)
`r len`(124) ← OF `pd.sendFloat("len", seqLength.getValue())` (ofApp.cpp:288);
`seqLength` range **1..16, default 16** (ofApp.cpp:218).
- `124 0 125 0` → `* 12`(125) → `125 0 11 1` → **`mod 192` right inlet** = sets the
  modulo to `len * 12` pulses. Default len=16 → mod 192 (= 1 bar). len=8 → mod 96, etc.
- `124 0 18 1` → **`mod 16` right inlet** (the GUI display step counter divisor).

So the "192" and "16" written in the patch are just the **default args**; the live divisor
is `len*12` and `len`. The pattern always plays `len` steps of 12 pulses each.

### Tempo
- Live tempo enters Pd only from MIDI clock: `pd.sendFloat("tempo", clock.getBpm())`
  (ofApp.cpp:1245). `r tempo` inside `pd midi in`(idx 119) feeds the clock-doubling delay
  (§1) via `expr 60/$f1*1000` → `/48`.
- There is also a top-level `r tempo`(9) → `expr 60/$f1*1000`(10) (line 12), but ms/quarter
  there is not on the note-timing critical path (timing is driven entirely by pulse counting
  off `s clock`, §1–2). The Rust port derives step duration from clock pulses, not from BPM
  math; BPM only matters for the `delay` half-tick (which the port can replace by counting at
  48 PPQN directly).
- The GUI `tempoSlider` (ofApp.cpp:188, range 80..200, default 120) is **not used** (its
  `sendFloat` is commented out, ofApp.cpp:263). Tempo = host/MIDI clock.

> VERIFY in Task 16: that the port reproducing 48 PPQN counting (rather than literally
> emulating the doubling `delay`) yields identical step times across tempo changes.

---

## Constants cheat-sheet for the Rust port

| quantity | value | source |
|----------|-------|--------|
| Internal resolution | **48 PPQN** (24 PPQN in, doubled by `pd midi in`) | main.pd:780–805; ofApp.cpp:1251 |
| Pulses per step | **12** (= 1/16 note) | `mod 192` ÷ 16 steps; `* 12` in `pd step` |
| Pattern length | **len × 12 pulses** (default 16 → 192 = 1 bar) | main.pd:810–811,977–979 |
| Step on-grid pulse | **N × 12** | `pd step` `* 12` |
| Substep offset | **S × (2·substep − 1)** pulses, S = scaling slider 0..6, 0.5=on-grid | `expr $f1*($f2-$f3)+$f3`, main.pd:40 |
| Per-step gate | spigot gated by `/sequence` bit (1=on) | `pd step` spigot |
| Base pitch | `noteN` slider, 24..127, default 50 | ofApp.cpp:190–205 |
| Quantize | snap pitch-class DOWN to scale (pentMinor/pentMajor/chromatic), + key 0..11 | `pd quant`, main.pd:592–738 |
| Scale select | `scalenotes` int 0/1/2 (chromatic/minor/major — **verify mapping**) | ofApp.cpp:282 |
| Velocity | **100** (fixed) | `makenote 100 200`, main.pd:545 |
| Note duration | **gate** ms (slider 1..1000, default 100; overrides makenote 200) | main.pd:545,580–581; ofApp.cpp:214,286 |
| MIDI channel | **1** | `msg 1`→`noteout` inlet2, main.pd:544 |
| Tempo source | MIDI clock BPM (tempoSlider dead) | ofApp.cpp:263,1245 |

### Solid vs needs-A/B
- **Solid:** 12 pulses/step, 1 bar/pattern, substep lerp `S*(2x−1)` with 0.5=on-grid,
  spigot gating by `/sequence`, velocity 100 fixed, gate-slider overrides duration,
  channel 1, variable length = `len*12`, tempo from MIDI clock only.
- **Needs A/B (Task 16):** (a) the clock-doubling `delay` actually yielding 48 PPQN /
  16th-note steps; (b) `scalenotes` integer → which scale-branch mapping; (c) snap direction
  is strictly downward; (d) `int` truncation of negative substep offsets.
