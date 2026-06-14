#!/usr/bin/env bash
# Stage-1 end-to-end MIDI test: launch app, click Generate, clock it, capture notes.
set -u
APP=/home/gustavo/Git/DeepSteps/.worktrees/linux-port/Deep_Steps_project
VENV_SP="$APP/.venv/lib/python3.14/site-packages"
PY="$APP/.venv/bin/python"
export DISPLAY=:1
unset WAYLAND_DISPLAY

cd "$APP/bin"
PYTHONPATH="$VENV_SP" ./Deep_Steps_project >/tmp/ds_run.log 2>&1 &
PID=$!
echo "app pid $PID"
# wait for pd patch ready
for i in $(seq 1 30); do grep -q 'Patch Open' /tmp/ds_run.log 2>/dev/null && break; sleep 1; done
sleep 2
kill -0 $PID 2>/dev/null || { echo "app died early"; tail -20 /tmp/ds_run.log; exit 1; }

# locate the GLFW window for this pid
WID=$(xdotool search --pid $PID 2>/dev/null | tail -1)
echo "window id: ${WID:-NONE}"
if [ -n "$WID" ]; then
  xdotool windowactivate --sync "$WID" 2>/dev/null
  sleep 0.5
  # generateButton.set(20,300,100,50) -> center (70,325) in window coords
  xdotool mousemove --window "$WID" 70 325 click 1
  echo "clicked Generate at window(70,325)"
  sleep 1.0
fi

# break MIDI feedback loop: ofxMidiOut -> Midi Through (stops app hearing its own out)
OUTC=$(aconnect -l | awk '/ofxMidiOut Client/{print $2}')
THRU=$(aconnect -l | awk '/Midi Through/{print $2}')
if [ -n "${OUTC:-}" ] && [ -n "${THRU:-}" ]; then
  aconnect -d "${OUTC}:0" "${THRU}:0" 2>/dev/null && echo "broke feedback ${OUTC}:0 -> ${THRU}:0"
fi

echo "=== harness ==="
PYTHONPATH="$VENV_SP" "$PY" "$APP/tools/midi_verify.py"
RC=$?

echo "=== app pd-side log (last lines) ==="; tail -4 /tmp/ds_run.log
kill $PID 2>/dev/null; sleep 1; kill -9 $PID 2>/dev/null
exit $RC
