#!/usr/bin/env bash
cd "$(dirname "$0")"

set -uo pipefail   # note: no -e; we handle exits manually below

# ---------------------------------------------------------------------------
# Launch both processes with unbuffered output (-u) so logs update live.
# Each writes stdout+stderr to its own file.
# ---------------------------------------------------------------------------
python3 -u preprocess.py > preprocess.log 2>&1 &
PREPROCESS_PID=$!

python3 -u detect.py --input_dir /media/sd/processed \
    --module_dir /media/sd/modules --window 120 your_module \
    > detect.log 2>&1 &
DETECT_PID=$!

echo "preprocess -> preprocess.log (PID $PREPROCESS_PID)"
echo "detect     -> detect.log (PID $DETECT_PID)"
echo "Follow logs with: tail -f preprocess.log detect.log"

# ---------------------------------------------------------------------------
# Shut down cleanly: send SIGINT to both (triggers their KeyboardInterrupt
# handlers), then wait for them to actually exit. Idempotent so it's safe to
# call from both the trap and the monitor logic below.
# ---------------------------------------------------------------------------
shutting_down=0
shutdown() {
    [ "$shutting_down" -eq 1 ] && return    # don't run twice
    shutting_down=1
    echo "Shutting down both processes..."
    kill -INT "$PREPROCESS_PID" "$DETECT_PID" 2>/dev/null || true
    wait "$PREPROCESS_PID" "$DETECT_PID" 2>/dev/null || true
    echo "Both stopped."
}

# Ctrl-C / termination -> clean shutdown of both children.
trap 'shutdown; exit 130' INT TERM

# ---------------------------------------------------------------------------
# Monitor: poll both PIDs. The instant either one is gone, report which,
# kill the survivor, and exit. kill -0 tests existence without signaling.
# ---------------------------------------------------------------------------
while true; do
    if ! kill -0 "$PREPROCESS_PID" 2>/dev/null; then
        echo "preprocess (PID $PREPROCESS_PID) exited — stopping detect."
        shutdown
        exit 1
    fi
    if ! kill -0 "$DETECT_PID" 2>/dev/null; then
        echo "detect (PID $DETECT_PID) exited — stopping preprocess."
        shutdown
        exit 1
    fi
    sleep 1
done