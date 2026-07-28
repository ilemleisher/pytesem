#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="/data/run45/raw"
OUTPUT_DIR="${1:-/path/to/output_dir}"
MAIN_SCRIPT='run_naq.py'

# --- find the most recent timestamped directory in RAW_DIR ---
DATA_DIR=$(find "$RAW_DIR" -mindepth 1 -maxdepth 1 -type d \
    -name 'continuous_I*_D*_T*' | sort | tail -n 1)

if [[ -z "$DATA_DIR" ]]; then
    echo "Error: no matching directories found in $RAW_DIR" >&2
    exit 1
fi

echo "Using most recent data directory: $DATA_DIR"

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
    wait "$MAIN_PID" 2>/dev/null || true
}
trap cleanup INT TERM

wait "$DAQ_PID" || true
echo "DAQ finished. Waiting for main to drain..."
wait "$MAIN_PID" || true
echo "Done."