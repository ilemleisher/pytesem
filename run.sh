#!/usr/bin/env bash
set -euo pipefail

cd /home/mwilliams/081425pytesdaq/pytesdaq/bin/pytesem

# --- defaults ---
RUN="run45"
OUTPUT_DIR_OVERRIDE=""

DOWNSAMPLE_FACTOR=10
SAMPLING_RATE=1.25e6
FREQ_CUTOFF=1000.0
CHANNEL_NUMBER=0
MAX_BINS=5
THRESHOLD_LOW=2.0
THRESHOLD_HIGH=3.0
BAND_WIDTH=2.0

MAIN_SCRIPT='main.py'

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --run NAME                  Run name (default: $RUN)
  --output-dir DIR            Override output directory
  --downsample-factor N       Downsample reduction factor (default: $DOWNSAMPLE_FACTOR)
  --sampling-rate HZ          Sampling rate of raw data in Hz (default: $SAMPLING_RATE)
  --freq-cutoff HZ            Frequency separating low/high bands (default: $FREQ_CUTOFF)
  --channel-number N          Channel number to read (0-indexed) (default: $CHANNEL_NUMBER)
  --max-bins N                Max bins to display in live figure (default: $MAX_BINS)
  --threshold-low VAL         Lower threshold for bin selection (default: $THRESHOLD_LOW)
  --threshold-high VAL        Upper threshold for bin selection (default: $THRESHOLD_HIGH)
  --band-width VAL            Dex-width of caution band above threshold (default: $BAND_WIDTH)
  -h, --help                  Show this help message

Example:
  $0 --run run46 --threshold-low 1.5 --threshold-high 2.5 --freq-cutoff 500
EOF
}

# --- parse command-line arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)
            RUN="$2"; shift 2 ;;
        --output-dir)
            OUTPUT_DIR_OVERRIDE="$2"; shift 2 ;;
        --downsample-factor)
            DOWNSAMPLE_FACTOR="$2"; shift 2 ;;
        --sampling-rate)
            SAMPLING_RATE="$2"; shift 2 ;;
        --freq-cutoff)
            FREQ_CUTOFF="$2"; shift 2 ;;
        --channel-number)
            CHANNEL_NUMBER="$2"; shift 2 ;;
        --max-bins)
            MAX_BINS="$2"; shift 2 ;;
        --threshold-low)
            THRESHOLD_LOW="$2"; shift 2 ;;
        --threshold-high)
            THRESHOLD_HIGH="$2"; shift 2 ;;
        --band-width)
            BAND_WIDTH="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1 ;;
    esac
done

RAW_DIR="/home/mwilliams/data/$RUN/raw"

# --- start DAQ ---
python ../run_daq.py -c top_qet1, top_qet2, bot_qet1, bot_qet2 --acquire-cont --duration 20m --comment "Old pytesdaq. Chs 0,1,6,7 continuous transition with thin film, MXC <7 mK, still heater 8 mW" &
DAQ_PID=$!
echo "pytesem: DAQ started (PID $DAQ_PID)"

# --- wait for directory to be generated ---
sleep 10

# --- find the most recent timestamped directory in RAW_DIR ---
DATA_DIR=$(find "$RAW_DIR" -mindepth 1 -maxdepth 1 -type d \
    -name 'continuous_I*_D*_T*' | sort | tail -n 1)

if [[ -z "$DATA_DIR" ]]; then
    echo "pytesem Error: no matching directories found in $RAW_DIR" >&2
    exit 1
fi

NAME=$(basename "$DATA_DIR")
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-/home/mwilliams/081425pytesdaq/pytesdaq/bin/pytesem/em_output/$RUN/$NAME}"
mkdir -p "$OUTPUT_DIR"

echo "pytesem: Using most recent data directory: $DATA_DIR"
echo "pytesem: Saving output to $OUTPUT_DIR"

# --- start main.py in background, tell it the DAQ PID ---
python "$MAIN_SCRIPT" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --daq_pid "$DAQ_PID" \
    --downsample_factor "$DOWNSAMPLE_FACTOR" \
    --sampling_rate "$SAMPLING_RATE" \
    --freq_cutoff "$FREQ_CUTOFF" \
    --channel_number "$CHANNEL_NUMBER" \
    --max_bins "$MAX_BINS" \
    --threshold_low "$THRESHOLD_LOW" \
    --threshold_high "$THRESHOLD_HIGH" &
NAQ_PID=$!
echo "pytesem: main started (PID $NAQ_PID)"

# --- clean shutdown handling ---
cleanup() {
    echo "pytesem: Shutting down..."
    kill "$DAQ_PID" 2>/dev/null || true
    wait "$NAQ_PID" 2>/dev/null || true
}
trap cleanup INT TERM

wait "$DAQ_PID" || true
echo "pytesem: DAQ finished. Waiting for main to drain..."
wait "$NAQ_PID" || true
echo "pytesem: Done."

# --- post-run analysis ---
echo "pytesem: Running post run analysis..."
python post_run_analysis.py \
    --output_dir "$OUTPUT_DIR" \
    --threshold_low "$THRESHOLD_LOW" \
    --threshold_high "$THRESHOLD_HIGH" \
    --band_width "$BAND_WIDTH" \
    --freq_cutoff "$FREQ_CUTOFF" &
