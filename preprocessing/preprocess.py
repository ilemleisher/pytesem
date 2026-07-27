import h5py, os, psutil, re
import numpy as np
from scipy.ndimage import binary_dilation
from datetime import datetime, timedelta


def event_index(name):
    """
    Convert an event/chunk name into its integer index.

    e.g. "event10" -> 10. Used as a sort key so chunks come back in
    acquisition order rather than lexicographic order (which would put
    "event10" before "event2").

    Parameters:
    - name: chunk name string of the form "event<N>".

    Returns:
    - int: the numeric index N.
    """
    # "event10" -> 10
    return int(name.replace("event", ""))


def list_chunks(h5path):
    """
    Return the event/chunk names in an HDF5 file, sorted in acquisition order.

    Reads the keys under the 'adc1' group and sorts them numerically via
    event_index (event1, event2, ..., event10, ...). Fails soft: if the file
    is still being created / mid-write, or the 'adc1' group doesn't exist yet,
    returns an empty list instead of raising, so the caller can just retry
    on the next poll.

    Parameters:
    - h5path: path to the HDF5 file.

    Returns:
    - list[str]: sorted chunk names, or [] if the file/group isn't readable yet.
    """
    try:
        with h5py.File(h5path, "r") as f:
            return sorted(f['adc1'].keys(), key=event_index)
    except (OSError, KeyError):
        return []   # file mid-write / adc1 not created yet


def chunk_timestamp(h5path, chunk_name, duration):
    """
    Compute the absolute start time of a given chunk.

    Combines the DAQ start time parsed from the filename with the chunk's
    position in the sequence. Assumes chunks are contiguous and each spans
    `duration` seconds, so chunk N starts at start_time + (N-1)*duration.

    Parameters:
    - h5path: path to the HDF5 file (used to parse the DAQ start time).
    - chunk_name: chunk name of the form "event<N>".
    - duration: length of a single chunk in seconds.

    Returns:
    - datetime: absolute UTC-naive start time of the chunk.
    """
    start = parse_file_start_time(h5path)
    idx = event_index(chunk_name)               # "event1" -> 1
    return start + timedelta(seconds=(idx - 1) * duration)


def parse_file_start_time(h5path):
    """
    Extract the DAQ acquisition start time encoded in the filename.

    Expects a filename containing a "_D<YYYYMMDD>_T<HHMMSS>_" segment,
    e.g. "..._D20260707_T175657_...". Raises ValueError if the pattern
    isn't found.

    Parameters:
    - h5path: path (or filename) to parse.

    Returns:
    - datetime: parsed start time (naive, no timezone).

    Raises:
    - ValueError: if no timestamp pattern is present in the filename.
    """
    name = os.path.basename(h5path)
    m = re.search(r"_D(\d{8})_T(\d{6})_", name)
    if not m:
        raise ValueError(f"Could not parse timestamp from filename: {name}")
    date_str, time_str = m.group(1), m.group(2)          # "20260707", "175657"
    return datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")


def preprocess(tdata, data, target_len, sigma_thresh=5, radius=100):
    """
    Clean and downsample a raw waveform by removing large transient excursions.

    Pipeline:
    - Estimate a robust noise level from the median and MAD.
    - Mark all points above median + sigma_thresh * robust_sigma.
    - Expand each marked point by +/- `radius` samples (binary dilation),
      so the shoulders around a spike are also removed.
    - Drop all marked points.
    - Downsample the survivors to exactly `target_len` points by picking
      evenly-spaced indices.

    NOTE (correctness caveat): because points are *removed* rather than
    interpolated, the surviving samples are NOT uniformly spaced in time.
    The returned filtered_tdata reflects this. Downstream FFT-based analysis
    that assumes a constant sample rate will therefore see a distorted
    frequency axis. Either use the returned time array with a non-uniform
    method (e.g. Lomb-Scargle) or replace spikes in-place to keep uniform
    spacing.

    Parameters:
    - tdata: time-data array (same length as `data`).
    - data: raw waveform array.
    - target_len: desired number of output points.
    - sigma_thresh: number of robust sigmas above the median for the cut.
    - radius: number of samples on each side of a marked point to also remove.

    Returns:
    - (filtered_tdata, filtered_data): tuple of arrays, each length target_len.

    Raises:
    - ValueError: if nothing survives filtering, or fewer than target_len
      points survive.
    """
    raw_data = np.asarray(data).copy()
    tdata = np.asarray(tdata)

    # Robust noise estimate: median absolute deviation scaled to a Gaussian sigma.
    med = np.median(raw_data)
    mad = np.median(np.abs(raw_data - med))
    sigma_robust = 1.4826 * mad          # scales MAD to be a std estimate for Gaussian noise
    height = med + sigma_thresh * sigma_robust

    print("Height mask")
    # Initial keep mask: True where the sample is below the spike threshold.
    keep = raw_data < height
    print("Expand by radius")
    # Expand removals by radius: grow the removed region outward so the
    # rising/falling shoulders of each spike are excluded too.
    if radius > 0:
        remove = ~keep
        kernel = np.ones(2 * radius + 1, dtype=bool)
        out = binary_dilation(remove, structure=kernel)
        keep = ~out

    filtered_data = raw_data[keep]
    filtered_tdata = tdata[keep]

    n = len(filtered_data)
    if n == 0:
        raise ValueError("No data left after filtering.")
    if n < target_len:
        raise ValueError(f"Filtered length ({n}) is smaller than target_len ({target_len}).")
    print("Downsample")
    # Downsample to exactly target_len points by evenly-spaced index selection.
    idx = np.linspace(0, n - 1, target_len, dtype=int)
    return filtered_tdata[idx], filtered_data[idx]


def fft(data, fs):
    """
    Compute the one-sided Amplitude Spectral Density (ASD) of a signal.

    Removes the DC offset, takes a real FFT, forms the one-sided power
    spectral density with the appropriate single-sided doubling of the
    interior bins, and returns its square root (the ASD).

    Assumes the input is uniformly sampled at `fs` (see the caveat in
    preprocess about non-uniform spacing after spike removal).

    Parameters:
    - data: time-domain signal array.
    - fs: sampling frequency in Hz.

    Returns:
    - (freqs, asd): frequency bins (Hz) and ASD values, both length N//2 + 1.
    """
    print("Computing ASD")
    # Remove DC offset so bin 0 doesn't dominate.
    x = data - np.mean(data)
    N = len(x)

    # Real FFT and its corresponding frequency bins.
    Y = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1/fs)

    # One-sided PSD. The factor 1/(fs*N) normalizes to a density; interior
    # bins are doubled to account for the folded negative frequencies.
    # For even N the Nyquist bin (last) is not doubled; for odd N there is
    # no exact Nyquist bin so all bins from 1 on are doubled.
    psd = (1 / (fs * N)) * np.abs(Y)**2
    if N % 2 == 0:
        psd[1:-1] *= 2
    else:
        psd[1:] *= 2
    asd = np.sqrt(psd)

    return freqs, asd


def process_chunk(h5path, chunk_name, sampling_rate, channel_number, downsample_factor):
    """
    Read, clean, and spectrally reduce a single chunk from an HDF5 file.

    Loads one channel of one chunk, builds a time axis, runs spike removal +
    downsampling via preprocess, then computes the ASD via fft. Also derives
    the chunk's absolute timestamp from the filename and sequence index.

    Parameters:
    - h5path: path to the HDF5 file.
    - chunk_name: chunk name of the form "event<N>".
    - sampling_rate: raw acquisition sample rate in Hz.
    - channel_number: which channel (row) of the 2D dataset to read.
    - downsample_factor: raw length is divided by this to get target_len.

    Returns:
    - dict with keys:
        "timestamp": datetime start time of the chunk,
        "bins": frequency bins (Hz),
        "asd": amplitude spectral density values.

    Notes:
    - `duration` is n_samples / sampling_rate seconds.
    - `fft_fs` is the effective post-downsample rate (target_len / duration);
      see the uniform-sampling caveat in preprocess.
    - Assumes the dataset is 2D (channels x samples).
    """
    with h5py.File(h5path, "r") as f:
        data = f['adc1'][chunk_name]
        n_samples = data.shape[1]                     # samples per channel
        time_data = np.arange(n_samples) / sampling_rate
        waveform_data = data[channel_number]          # select the channel
        target_len = n_samples // downsample_factor   # integer count of output points
        duration = n_samples / sampling_rate          # chunk length in seconds
        fft_fs = target_len / duration                # effective rate after downsampling
        preprocessed_time, preprocessed_waveform = preprocess(time_data, waveform_data, target_len)
        bins, asd = fft(preprocessed_waveform, fft_fs)

    ts = chunk_timestamp(h5path, chunk_name, duration)

    print(f"Processed {os.path.basename(h5path)}::{chunk_name} @ {ts.isoformat()}", flush=True)

    return {
        "timestamp": ts,
        "bins": bins,
        "asd": asd,
    }


def daq_alive(pid):
    """
    Check whether the DAQ process is still running.

    Returns True only if a process with the given PID exists and is not a
    zombie. Any race where the process disappears mid-check is treated as
    "not alive".

    Parameters:
    - pid: process ID of the DAQ writer.

    Returns:
    - bool: True if the process is live, False otherwise.
    """
    try:
        return psutil.pid_exists(pid) and \
               psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _is_latest_open_chunk(h5path, chunk_name, files, chunks):
    """
    Heuristic: is this the chunk currently being written?

    The last chunk of the last (newest) file is assumed to still be open for
    writing, so callers skip it while the DAQ is live to avoid reading a
    partial write.

    Parameters:
    - h5path: the file being considered.
    - chunk_name: the chunk being considered.
    - files: full sorted list of candidate files.
    - chunks: full sorted list of chunks within h5path.

    Returns:
    - bool: True if (h5path, chunk_name) is the newest file's newest chunk.
    """
    return h5path == files[-1] and chunk_name == chunks[-1]