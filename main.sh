#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-/path/to/data_dir}"
OUTPUT_DIR="${2:-/path/to/output_dir}"
MAIN_SCRIPT='run_naq.py'

# Ensure the directories exist (neither run_naq.py nor np.savez/savefig create them).
mkdir -p "$DATA_DIR" "$OUTPUT_DIR"

# --- start DAQ ---
python run_daq.py \
    --arg1 "" \
    --arg2 "" \
    --arg3 "" &
DAQ_PID=$!
echo "DAQ started (PID $DAQ_PID)"

# --- start run_naq.py in background, tell it the DAQ PID ---
python "$MAIN_SCRIPT" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --daq_pid "$DAQ_PID" &
MAIN_PID=$!
echo "Main started (PID $MAIN_PID)"

# --- clean shutdown handling ---
cleanup() {
    echo "Shutting down..."
    kill "$DAQ_PID" 2>/dev/null || true
    # give main time to drain remaining chunks, then stop.
    # run_naq.py notices the dead DAQ via daq_alive() and exits after one final pass.
    wait "$MAIN_PID" 2>/dev/null || true
}
trap cleanup INT TERM

# wait for DAQ to finish naturally.
# '|| true' so a nonzero DAQ exit under `set -e` doesn't skip the drain below.
wait "$DAQ_PID" || true
echo "DAQ finished. Waiting for main to drain..."
wait "$MAIN_PID" || true
echo "Done."