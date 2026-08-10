"""
Post-run analysis over the reduced-data files saved during a run.

Loads every rd_*.npz file written by main.py, reconstructs a chronological
time series of short-term and long-term PCA residuals, and reports every
instance where a bin's residual crossed into the "red zone" (i.e. exceeded
threshold + 2*band_width) for its frequency band.
"""
import os
import glob
import re
import numpy as np
import argparse
from datetime import datetime


def parse_args():
    """
    Define and parse command-line arguments controlling the output
    directory to scan and the thresholds/band settings used to define
    the red-zone crossing criteria.
    """
    parser = argparse.ArgumentParser(description="Post-run analysis of reduced data files.")
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the folder where output is saved.')
    parser.add_argument('--threshold_low', type=float, default=2.0, help='Lower threshold for bin selection.')
    parser.add_argument('--threshold_high', type=float, default=3.0, help='Upper threshold for bin selection.')
    parser.add_argument('--band_width', type=float, default=2.0,
                         help='Dex-width of the yellow caution band above threshold. '
                              'The red zone begins at threshold + 2 * band_width.')
    parser.add_argument('--freq_cutoff', type=float, default=1000.0,
                         help='Frequency (Hz) separating low/high bands.')
    return parser.parse_args()


def _timestamp_from_filename(path):
    """
    Extract the run timestamp embedded in filenames of the form
    rd_YYYYmmddTHHMMSS.npz (see run script: ts_str = timestamp.strftime(...)).
    Returns None if the filename doesn't match the expected pattern.
    """
    match = re.search(r"rd_(\d{8}T\d{6})\.npz$", os.path.basename(path))
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")


def load_all_reduced_data(output_dir):
    """
    Load every rd_*.npz file in output_dir, sorted chronologically by the
    timestamp embedded in each filename.

    Returns:
    - timestamps: list[datetime], one per file, in chronological order.
    - files: list[str], the corresponding file paths (same order).
    - data_list: list[dict], each file's arrays as a plain dict (NpzFile
      objects are lazy/file-backed, so we materialize + close them here
      rather than holding many open file handles at once).
    """
    pattern = os.path.join(output_dir, "rd_*.npz")
    paths = sorted(glob.glob(pattern), key=lambda p: _timestamp_from_filename(p) or p)

    if not paths:
        raise FileNotFoundError(f"No files matching {pattern} were found.")

    timestamps = []
    data_list = []
    for path in paths:
        ts = _timestamp_from_filename(path)
        # Materialize arrays into a plain dict and close the file
        # immediately, rather than keeping many NpzFile handles open.
        with np.load(path) as npz:
            data_list.append({key: npz[key] for key in npz.files})
        timestamps.append(ts)

    return timestamps, paths, data_list


def _stack_with_timestamps(data_list, timestamps, key):
    """
    Stack a per-window array across only the files where `key` is present,
    keeping the aligned subset of timestamps. Needed because long_term_*
    keys only start appearing once first_pca has been frozen, so they
    can't be naively stacked against the full timestamp list.

    Returns:
    - (aligned_timestamps, stacked_array) or (None, None) if `key` is
      absent from every file.
    """
    aligned_ts = [ts for ts, d in zip(timestamps, data_list) if key in d]
    present = [d[key] for d in data_list if key in d]
    if not present:
        return None, None
    return aligned_ts, np.stack(present)


def find_red_zone_crossings(data_list, timestamps, bins, threshold_low,
                             threshold_high, band_width, freq_cutoff):
    """
    Find every (timestamp, bin) at which a residual's abs value entered
    the red zone (i.e. exceeded threshold + band_width for its band),
    across both short-term and long-term residuals and both frequency
    bands.

    Returns:
    - list of dicts, each with keys: timestamp, term ("short"/"long"),
      band ("low"/"high"), bin_index, freq_hz, residual. Sorted
      chronologically.
    """
    # Split bins into low/high frequency bands, same convention as analysis.py.
    low_mask = bins < freq_cutoff
    high_mask = ~low_mask

    # Red-zone thresholds sit two caution-band-widths above the base thresholds.
    red_low = threshold_low + 2 * band_width
    red_high = threshold_high + 2 * band_width

    band_specs = [
        ("low", low_mask, red_low),
        ("high", high_mask, red_high),
    ]

    term_keys = [
        ("short", "residual"),
        ("long", "long_term_residual"),
    ]

    events = []
    for term_label, key in term_keys:
        # Long-term residuals may be missing from early files; align
        # timestamps to only the files that actually contain this key.
        aligned_ts, arr = _stack_with_timestamps(data_list, timestamps, key)
        if arr is None:
            print(f"No '{key}' data found (term={term_label}); skipping.")
            continue

        abs_arr = np.abs(arr)  # (n_windows, n_bins)

        for band_label, mask, red_thresh in band_specs:
            # (n_windows, n_bins_in_band) boolean of crossings
            crossings = abs_arr[:, mask] > red_thresh
            t_idx, local_bin_idx = np.where(crossings)
            # Map band-local bin indices back to global bin indices.
            global_bin_idx = np.where(mask)[0][local_bin_idx]

            for t, b in zip(t_idx, global_bin_idx):
                events.append({
                    "timestamp": aligned_ts[t],
                    "term": term_label,
                    "band": band_label,
                    "bin_index": int(b),
                    "freq_hz": float(bins[b]),
                    "residual": float(arr[t, b]),
                })

    # Present crossings in chronological order across all terms/bands.
    events.sort(key=lambda e: e["timestamp"])
    return events


def main():
    """
    Load all reduced-data files for a run, scan them for red-zone
    residual crossings, and print a chronological report.
    """

    args = parse_args()

    # Unpack CLI arguments into locals for readability.
    output_dir = args.output_dir
    threshold_low = args.threshold_low
    threshold_high = args.threshold_high
    band_width = args.band_width
    freq_cutoff = args.freq_cutoff

    timestamps, paths, data_list = load_all_reduced_data(output_dir)
    print(f"Loaded {len(paths)} reduced-data files from {output_dir}")

    # bins is expected to be identical across files (same frequency axis
    # every window); take it from the first file rather than re-stacking.
    bins = data_list[0]["bins"]

    events = find_red_zone_crossings(
        data_list, timestamps, bins,
        threshold_low=threshold_low, threshold_high=threshold_high,
        band_width=band_width, freq_cutoff=freq_cutoff,
    )

    print(f"\nRed zone crossings "
          f"(threshold_low+2*band_width={threshold_low + 2 * band_width:g} dex, "
          f"threshold_high+2*band_width={threshold_high + 2 * band_width:g} dex):")
    if not events:
        print("  None.")
    else:
        for e in events:
            print(f"  {e['timestamp']:%Y-%m-%d %H:%M:%S}  "
                  f"[{e['term']}-term, {e['band']}-freq]  "
                  f"bin {e['bin_index']} ({e['freq_hz']:.3g} Hz)  "
                  f"|residual|={abs(e['residual']):.3g} dex")


if __name__ == "__main__":
    main()