"""
Live noise-monitoring driver for a DAQ run.

Polls a directory of HDF5 files being written by run_daq.py, incrementally
preprocesses new data chunks as they become available, reduces a sliding
window of chunks via PCA, and updates live diagnostic figures. Exits once
the DAQ process has ended and one final pass over any remaining chunks has
completed, then saves the last figure to disk.
"""
import os, time, glob, argparse
import numpy as np
from collections import deque
from preprocessing.preprocess import list_chunks, process_chunk, daq_alive, _is_latest_open_chunk
from feature_extraction.feature_extraction import reduce_window
from analysis.analysis import analyze
import matplotlib.pyplot as plt

# How long to wait between filesystem scans when the DAQ is still running.
CHUNK_POLL_SEC = 2
# Number of consecutive chunks held in the sliding window. Analysis only
# runs once this many chunks have accumulated, and thereafter on a rolling basis.
WINDOW_SIZE = 60


def parse_args():
    """
    Define and parse command-line arguments controlling I/O paths, the DAQ
    PID handshake, downsampling/sampling-rate settings, and the analysis
    thresholds/bin limits used by the live figures.
    """
    parser = argparse.ArgumentParser(description="Preprocess raw data files.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the folder containing the .hdf5 files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the folder where output is saved.')
    parser.add_argument('--daq_pid', type=int, required=True, help='run_daq.py PID.')
    parser.add_argument('--downsample_factor', type=int, default=10, help='Downsample reduction factor.')
    parser.add_argument('--sampling_rate', type=float, default=1.25e6, help='Sampling rate of the raw data in Hz.')
    parser.add_argument('--freq_cutoff', type=float, default=1000.0, help='Frequency (Hz) separating low/high bands.')
    parser.add_argument('--channel_number', type=int, default=0, help='Channel number to read from the raw data file (0-indexed).')
    parser.add_argument('--max_bins', type=int, default=5, help='Maximum number of bins to display in the live figure.')
    parser.add_argument('--threshold_low', type=float, default=2.0, help='Lower threshold for bin selection.')
    parser.add_argument('--threshold_high', type=float, default=3.0, help='Upper threshold for bin selection.')
    parser.add_argument('--band_width', type=float, default=2.0, help='Width of the frequency band for bin selection.')
    parser.add_argument('--figures', type=str, nargs='+', choices=['worst_bins', 'psd', 'none'], default=['worst_bins', 'psd'], 
                        help='Which live figures generate: worst_bins, psd, none, both (default: both).')
    return parser.parse_args()


def main():

    args = parse_args()

    # Unpack CLI arguments into locals for readability.
    data_dir = args.data_dir
    output_dir = args.output_dir
    daq_pid = args.daq_pid
    sampling_rate = args.sampling_rate
    freq_cutoff = args.freq_cutoff
    channel_number = args.channel_number
    downsample_factor = args.downsample_factor
    max_bins = args.max_bins
    threshold_low = args.threshold_low
    threshold_high = args.threshold_high
    band_width = args.band_width

    # Convert the CLI list into a dict of booleans matching analyze()'s
    # expected "figures" argument.
    if 'none' in args.figures:
        args.figures = []
    figures = {
        "worst_bins": "worst_bins" in args.figures,
        "psd": "psd" in args.figures,
    }

    # Sliding window of the most recent WINDOW_SIZE processed chunks.
    # deque(maxlen=...) auto-evicts the oldest entry once full, so the
    # window always represents the latest WINDOW_SIZE chunks in time order.
    window = deque(maxlen=WINDOW_SIZE)

    # Set of (file, chunk) keys we've already processed, so we never
    # reprocess a chunk across poll iterations.
    done = set()

    # Handle to the live matplotlib figure; stays None until the first analyze() call.
    # Used at the end to decide whether there's anything to save.
    fig = fig2 = None

    # Initialize first fit PCA and scaler for use with long term analysis
    first_pca = None
    first_scaler = None

    # ------------------------------------------------------------------
    # MAIN POLLING LOOP
    # Repeatedly scan the data directory for new chunks, process them,
    # and run reduction + analysis once the window is full. Exits after
    # the DAQ process has died and one final full scan has completed.
    # ------------------------------------------------------------------
    while True:

        # -------------------
        # PREPROCESSING
        # -------------------
        # Snapshot DAQ liveness once per iteration. Captured before the scan
        # so that the "final pass" logic at the bottom is based on the same
        # state we scanned under.
        alive = daq_alive(daq_pid)

        # Enumerate candidate files. sorted() gives deterministic, roughly
        # chronological ordering assuming timestamped filenames.
        files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
        for h5path in files:
            # Each HDF5 file contains multiple named chunks (datasets/groups).
            chunks = list_chunks(h5path)
            # Skip the most recent chunk in the *newest* file while DAQ is live,
            # since it may still be being written.
            for chunk_name in chunks:
                key = (h5path, chunk_name)
                # Already handled on a previous iteration → skip.
                if key in done:
                    continue
                # While the DAQ is running, don't touch the currently-open
                # (still-being-written) chunk to avoid reading a partial write.
                if alive and _is_latest_open_chunk(h5path, chunk_name, files, chunks):
                    continue
                # Read + preprocess this chunk (downsample, select channel, etc.).
                # Returns None if the chunk isn't usable/complete.
                processed = process_chunk(h5path, chunk_name, sampling_rate, channel_number, downsample_factor)
                if processed is not None:
                    # Push into the sliding window and mark as handled.
                    window.append(processed)
                    done.add(key)

                    # -------------------
                    # DATA REDUCTION
                    # -------------------
                    # Only reduce/analyze once the window is completely full,
                    # so every reduction operates over a fixed WINDOW_SIZE span.
                    if len(window) == window.maxlen:      # full → safe to analyze
                        # Collapse the window of chunks into a reduced summary
                        # (expected to be a dict of named arrays; see note below).
                        reduced_data, pca, scaler = reduce_window(list(window), first_pca, first_scaler)
                        # Timestamp the output using the newest chunk in the window.
                        timestamp = processed['timestamp']
                        ts_str = timestamp.strftime("%Y%m%dT%H%M%S")
                        # Persist the reduced data. **reduced_data requires a
                        # dict with string keys.
                        np.savez(f"{output_dir}/rd_{ts_str}.npz", **reduced_data)
                        # Store first PCA components in memory for long term analysis
                        if first_pca is None:
                            first_pca = pca
                            first_scaler = scaler

                        # -------------------
                        # DATA ANALYSIS
                        # -------------------                     
                        # Produce/update the live figure from the reduced data.
                        # Returns a matplotlib Figure we hold onto for final save.
                        fig, fig2 = analyze(reduced_data, timestamp, threshold_low=threshold_low, threshold_high=threshold_high, figures=figures,
                                             freq_cutoff=freq_cutoff, max_bins=max_bins, band_width=band_width)

        # If the DAQ has exited, the scan we just completed already covered
        # every finalized chunk (nothing is "open" anymore), so we're done.
        if not alive:
            # DAQ gone: one final full pass already ran above; exit.
            break
        # DAQ still alive → wait before the next scan to avoid busy-spinning.
        time.sleep(CHUNK_POLL_SEC)

    # ------------------------------------------------------------------
    # FINALIZATION
    # ------------------------------------------------------------------
    # Collect whichever final figures were actually generated, since the
    # user may have disabled one or both via --figures.
    figs_to_save = []
    if fig is not None:
        figs_to_save.append((fig, "live_bins_final.png"))
    if fig2 is not None:
        figs_to_save.append((fig2, "live_psd_final.png"))

    if figs_to_save:
        plt.ioff()
        for f, filename in figs_to_save:
            path = os.path.join(output_dir, filename)
            f.savefig(path, dpi=150)
            print(f"Saved final plot to {path}", flush=True)
    else:
        print("No full window was ever reached; no plot to save.", flush=True)


if __name__ == "__main__":
    main()