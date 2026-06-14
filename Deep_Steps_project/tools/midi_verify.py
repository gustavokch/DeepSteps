#!/usr/bin/env python3
"""Stage-1 MIDI capture harness for the DeepSteps Linux build.

Assumes a Deep_Steps_project instance is already running AND that 'Generate'
has been clicked (so pd holds a pattern + note params). This harness only:
  1. connects to the app's ALSA MIDI ports by name (rtmidi)
  2. streams MIDI START + clock into the app's MIDI IN
  3. captures and classifies what the app emits on its MIDI OUT
"""
import sys, time, threading, collections
import rtmidi

CLOCKS_PER_QUARTER = 24
BPM = 120.0
CLOCK_INTERVAL = 60.0 / (BPM * CLOCKS_PER_QUARTER)
RUN_SECONDS = 6.0

def find_port(ports, needle):
    for i, n in enumerate(ports):
        if needle in n:
            return i, n
    return None, None

mo = rtmidi.MidiOut()
oi, on = find_port(mo.get_ports(), "ofxMidi Input")
if oi is None:
    print("FAIL: app 'ofxMidi Input' port not found. Is the app running?")
    print("targets:", mo.get_ports()); sys.exit(2)
mo.open_port(oi); print(f"clock   -> [{oi}] {on}")

mi = rtmidi.MidiIn()
mi.ignore_types(timing=False, sysex=True, active_sense=True)
ji, jn = find_port(mi.get_ports(), "ofxMidi Output")
if ji is None:
    print("FAIL: app 'ofxMidi Output' port not found.")
    print("sources:", mi.get_ports()); sys.exit(2)
mi.open_port(ji); print(f"monitor <- [{ji}] {jn}")

captured = []
mi.set_callback(lambda ev, d=None: captured.append(ev[0]))

MIDI_START, MIDI_CLOCK, MIDI_STOP = 0xFA, 0xF8, 0xFC
print(f"\nMIDI START + clock @ {BPM:.0f} bpm for {RUN_SECONDS:.0f}s ...")
mo.send_message([MIDI_START])
stop = threading.Event()
def clockgen():
    nxt = time.perf_counter()
    while not stop.is_set():
        mo.send_message([MIDI_CLOCK])
        nxt += CLOCK_INTERVAL
        d = nxt - time.perf_counter()
        if d > 0: time.sleep(d)
t = threading.Thread(target=clockgen, daemon=True); t.start()
time.sleep(RUN_SECONDS)
stop.set(); t.join(timeout=1)
mo.send_message([MIDI_STOP]); time.sleep(0.2)
mi.cancel_callback()

# classify
hist = collections.Counter()
for m in captured:
    hist[m[0] & 0xF0 if m and m[0] < 0xF0 else (m[0] if m else 0)] += 1
note_on  = [m for m in captured if 0x90 <= m[0] <= 0x9F and len(m) == 3 and m[2] > 0]
note_off = [m for m in captured if (0x80 <= m[0] <= 0x8F) or (0x90 <= m[0] <= 0x9F and len(m) == 3 and m[2] == 0)]
print("\n================ RESULT ================")
print(f"total messages on app OUT: {len(captured)}")
print("status histogram:", {hex(k): v for k, v in hist.most_common()})
print(f"note-ON:  {len(note_on)}")
print(f"note-OFF: {len(note_off)}")
if note_on:
    print("first note-ONs (status,pitch,vel):", note_on[:8])
    print("distinct pitches:", sorted({m[1] for m in note_on}))
    print("distinct velocities:", sorted({m[2] for m in note_on}))
    print("\nPASS: clocked DeepSteps emits MIDI notes on Linux x86_64.")
else:
    print("\nNO NOTES. (Generate not clicked, or pd note params unset)")
del mi, mo
