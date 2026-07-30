import os, time, glob, argparse
import numpy as np
from collections import deque
from preprocessing.preprocess import list_chunks, process_chunk, daq_alive, _is_latest_open_chunk
from feature_extraction.feature_extraction import reduce_window
from analysis.analysis import live_figure_worst_bins, live_figure_sum_stats, live_figure_psd
import matplotlib.pyplot as plt

# How long to wait between filesystem scans when the DAQ is still running.
CHUNK_POLL_SEC = 2
# Number of consecutive chunks held in the sliding window. Analysis only
# runs once this many chunks have accumulated, and thereafter on a rolling basis.
WINDOW_SIZE = 30


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess raw data files.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the folder containing the .hdf5 files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the folder where output is saved.')
    parser.add_argument('--daq_pid', type=int, required=True, help='run_daq.py PID.')
    parser.add_argument('--downsample_factor', type=int, default=10, help='Downsample reduction factor.')
    parser.add_argument('--sampling_rate', type=float, default=1.25e6, help='Sampling rate of the raw data in Hz.')
    parser.add_argument('--channel_number', type=int, default=0, help='Channel number to read from the raw data file (0-indexed).')
    return parser.parse_args()


def main():

    args = parse_args()

    # Unpack CLI arguments into locals for readability.
    data_dir = args.data_dir
    output_dir = args.output_dir
    daq_pid = args.daq_pid
    sampling_rate = args.sampling_rate
    channel_number = args.channel_number
    downsample_factor = args.downsample_factor

    # Sliding window of the most recent WINDOW_SIZE processed chunks.
    # deque(maxlen=...) auto-evicts the oldest entry once full, so the
    # window always represents the latest WINDOW_SIZE chunks in time order.
    window = deque(maxlen=WINDOW_SIZE)

    # Set of (file, chunk) keys we've already processed, so we never
    # reprocess a chunk across poll iterations.
    done = set()

    # Handle to the live matplotlib figure; stays None until the first analyze() call.
    # Used at the end to decide whether there's anything to save.
    fig1 = fig2 = fig3 = None

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
                        fig1 = live_figure_sum_stats(reduced_data, timestamp)
                        fig2 = live_figure_worst_bins(reduced_data, timestamp)
                        fig3 = live_figure_psd(reduced_data, timestamp)

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
    if fig1 is not None:
        plt.ioff()

        figs_to_save = [
            (fig1, "noise_levels_final.png"),
            (fig2, "worst_bins_final.png"),
            (fig3, "psd_final.png"),
        ]
        for fig, filename in figs_to_save:
            path = os.path.join(output_dir, filename)
            fig.savefig(path, dpi=150)
            print(f"Saved final plot to {path}", flush=True)
    else:
        print("No full window was ever reached; no plot to save.", flush=True)


if __name__ == "__main__":
    main()