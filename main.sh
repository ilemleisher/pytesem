#!/usr/bin/env bash
set -euo pipefail

cd /home/mwilliams/081425pytesdaq/pytesdaq/bin/pytesem

# --- start DAQ ---
python ../run_daq.py -c top_qet1, top_qet2, bot_qet1, bot_qet2 --acquire-cont --duration 20m --comment "Old pytesdaq. Chs 0,1,6,7 continuous transition with thin film, MXC <7 mK, still heater 8 mW" &
DAQ_PID=$!
echo "DAQ started (PID $DAQ_PID)"

RUN="run45"
RAW_DIR="/home/mwilliams/data/$RUN/raw"
MAIN_SCRIPT='run_naq.py'

# --- wait for directory to be generated ---
sleep 10

# --- find the most recent timestamped directory in RAW_DIR ---
DATA_DIR=$(find "$RAW_DIR" -mindepth 1 -maxdepth 1 -type d \
    -name 'continuous_I*_D*_T*' | sort | tail -n 1)

if [[ -z "$DATA_DIR" ]]; then
    echo "Error: no matching directories found in $RAW_DIR" >&2
    exit 1
fi

NAME=$(basename "$DATA_DIR")
OUTPUT_DIR="${1:-/home/mwilliams/081425pytesdaq/pytesdaq/bin/pytesem/em_output/$RUN/$NAME}"
mkdir -p "$OUTPUT_DIR"

echo "Using most recent data directory: $DATA_DIR"
echo "Saving output to $OUTPUT_DIR"

# --- start run_naq.py in background, tell it the DAQ PID ---
python "$MAIN_SCRIPT" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --daq_pid "$DAQ_PID" &
NAQ_PID=$!
echo "NAQ started (PID $NAQ_PID)"

# --- clean shutdown handling ---
cleanup() {
    echo "Shutting down..."
    kill "$DAQ_PID" 2>/dev/null || true
    wait "$NAQ_PID" 2>/dev/null || true
}
trap cleanup INT TERM

wait "$DAQ_PID" || true
echo "DAQ finished. Waiting for main to drain..."
wait "$NAQ_PID" || true
echo "Done."

# --- post-run analysis ---
python post_run_analysis.py --output_dir "$OUTPUT_DIR"