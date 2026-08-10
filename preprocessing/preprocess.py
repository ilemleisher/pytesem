"""
Chunk-level preprocessing for raw DAQ HDF5 files.

Provides helpers to discover which chunks exist in a run (possibly split
across multiple sequential files), reconstruct each chunk's absolute
timestamp, clean/downsample a single channel's waveform via spike removal,
convert it to an amplitude spectral density, and check whether the DAQ
writer process is still alive.
"""
import h5py, os, psutil, re
import numpy as np
from scipy.ndimage import binary_dilation
from datetime import datetime, timedelta


# Cache of per-file chunk counts, keyed by file path. Only ever populated
# for finalized (closed) files, since counting chunks in an actively-written
# file would be unstable.
_chunk_count_cache = {}


def _cached_chunk_count(h5path):
    """
    Get the number of chunks in a finalized (closed) HDF5 file, caching
    the result so repeated lookups don't re-open the file.

    Only call this on files guaranteed to no longer be written to;
    calling it on the currently-open file would cache a stale count.

    Parameters:
    - h5path: path to a finalized HDF5 file.
Returns:
    - int: number of chunks in the file.
    """
    if h5path not in _chunk_count_cache:
        _chunk_count_cache[h5path] = len(list_chunks(h5path))
    return _chunk_count_cache[h5path]


def event_index(name):
    """
    Convert an event/chunk name into its integer index.

    e.g. "event_10" -> 10. Used as a sort key so chunks come back in
    acquisition order rather than lexicographic order (which would put
    "event10" before "event2").

    Parameters:
    - name: chunk name string of the form "event<N>".

    Returns:
    - int: the numeric index N.
    """
    # "event_10" -> 10
    return int(name.replace("event_", ""))


def list_chunks(h5path):
    """
    Return the event/chunk names in an HDF5 file, sorted in acquisition order.

    Reads the keys under the 'adc1' group and sorts them numerically via
    event_index (event_1, event_2, ..., event_10, ...). Fails soft: if the file
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


def base_run_id(h5path):
    """
    Extract the run identifier shared by all files in a multi-file
    acquisition run.

    Files belong to the same run if they share the same
    sequence number (e.g. "..._F0001.hdf5" vs "..._F0002.hdf5").

    Parameters:
    - h5path: path (or filename) to parse.

    Returns:
    - str: the shared "D<YYYYMMDD>_T<HHMMSS>" run identifier.

    Raises:
    - ValueError: if the filename doesn't match the expected pattern.
    """
    name = os.path.basename(h5path)
    m = re.search(r"(D\d{8}_T\d{6})_F\d+", name)
    if not m:
        raise ValueError(f"Could not parse run id from filename: {name}")
    return m.group(1)


def file_sequence_number(h5path):
    """
    Extract the per-file sequence number from the filename.

    Expects "_D<YYYYMMDD>_T<HHMMSS>_F<NNNN>", where NNNN increases
    consecutively across files belonging to the same acquisition run.

    Parameters:
    - h5path: path (or filename) to parse.

    Returns:
    - int: the sequence number (e.g. "..._F0002" -> 2).

    Raises:
    - ValueError: if no sequence number pattern is present.
    """
    name = os.path.basename(h5path)
    m = re.search(r"_D\d{8}_T\d{6}_F(\d+)", name)
    if not m:
        raise ValueError(f"Could not parse sequence number from filename: {name}")
    return int(m.group(1))


def _sibling_files(h5path):
    """
    Find all files in the same directory belonging to the same
    acquisition run as h5path, sorted by sequence number.

    Parameters:
    - h5path: path to a file in the run.

    Returns:
    - list[str]: full paths of sibling files, in run order.
    """
    d = os.path.dirname(h5path)
    run_id = base_run_id(h5path)
    siblings = []
    for f in os.listdir(d):
        full = os.path.join(d, f)
        try:
            if base_run_id(full) == run_id:
                siblings.append(full)
        except ValueError:
            continue  # not part of any run (or unrelated file in this dir)
    return sorted(siblings, key=file_sequence_number)


def chunk_timestamp(h5path, chunk_name, duration):
    """
    Compute the absolute start time of a given chunk.

    Combines the DAQ run start time (shared across all files in a
    multi-file run) with the chunk's *global* position in the sequence.
    Chunks are contiguous within a file, and files sharing the same base
    timestamp are contiguous continuations of the same run rather than
    separate start times, so earlier files' chunk counts must be added
    to get the true offset. Earlier files' chunk counts are cached since
    the DAQ writes files sequentially, meaning any file earlier in the run
    than h5path is guaranteed already closed and won't grow further.

    Parameters:
    - h5path: path to the HDF5 file (used to parse the run start time and
      sequence number).
    - chunk_name: chunk name of the form "event<N>", local to h5path.
    - duration: length of a single chunk in seconds.

    Returns:
    - datetime: absolute start time of the chunk within the full run.
    """
    # Run-level start time, parsed once from this file's name.
    start = parse_file_start_time(h5path)
    # This chunk's position within its own file.
    local_idx = event_index(chunk_name)

    # Add up chunk counts from every file that precedes h5path in the run,
    # so local_idx can be converted into a run-wide chunk index.
    prior_chunks = 0
    for f in _sibling_files(h5path):
        if f == h5path:
            break
        prior_chunks += _cached_chunk_count(f)

    global_idx = prior_chunks + local_idx
    return start + timedelta(seconds=(global_idx - 1) * duration)


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


def preprocess(tdata, data, target_len, sigma_thresh=5, radius=100, default_value=None):
    """
    This function reads in raw data file and first:
    - Marks all points above a defined height threshold
    - Marks all points within a defined radius around each marked point from the previous step
    - Replaces all marked points with a default value (preserving uniform sampling)
    - Downsamples the resulting data array to a defined target length

    Parameters:
    - tdata: time data array
    - data: raw data array
    - target_len: desired length of the output data array
    - sigma_thresh: number of standard deviations above the median to define the height threshold
    - radius: number of points around each marked point to also mark for replacement
    - default_value: value to substitute for marked points (defaults to the median of the data)

    Returns:
    - filtered_tdata: time data array after downsampling (uniform sampling preserved throughout)
    - filtered_data: data array after peak replacement and downsampling
    """
    raw_data = np.asarray(data).copy()
    tdata = np.asarray(tdata)

    # Robust (outlier-resistant) estimate of the noise floor and its spread,
    # used instead of mean/std so a few large spikes don't skew the threshold.
    med = np.median(raw_data)
    mad = np.median(np.abs(raw_data - med))
    sigma_robust = 1.4826 * mad          # scales MAD to be a std estimate for Gaussian noise
    height = med + sigma_thresh * sigma_robust

    if default_value is None:
        default_value = med

    # Points that exceed the threshold
    remove = raw_data >= height

    # Expand marked region by radius
    if radius > 0:
        kernel = np.ones(2 * radius + 1, dtype=bool)
        remove = binary_dilation(remove, structure=kernel)

    # Replace marked points with default_value instead of deleting them
    raw_data[remove] = default_value

    n = len(raw_data)
    if n < target_len:
        raise ValueError(f"Data length ({n}) is smaller than target_len ({target_len}).")

    # Downsample to exactly target_len points (uniform spacing preserved)
    idx = np.linspace(0, n - 1, target_len, dtype=int)
    return tdata[idx], raw_data[idx]


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

    # Absolute wall-clock start time of this chunk, for plotting/saving.
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