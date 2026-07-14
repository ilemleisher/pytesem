import numpy as np
from dev.utils import track_runtime
from scipy.ndimage import binary_dilation

def downsample(tdata, data, target_len):
    """
    This function reads in the raw data and uniformly downsamples it to a specified target length.

    Parameters:
    - tdata: time data array
    - data: raw data array
    - target_len: desired length of the output data array

    Returns:
    - downsampled_tdata: time data array after downsampling
    - downsampled_data: raw data array after downsampling
    """
    raw_data = np.asarray(data).copy()
    tdata = np.asarray(tdata)

    # Downsample to exactly target_len points
    idx = np.linspace(0, len(tdata) - 1, target_len, dtype=int)
    return tdata[idx], raw_data[idx]

def preprocess(tdata, data, target_len, sigma_thresh, radius):
    """
    This function reads in raw data file and:
    - Marks all points above a defined height threshold
    - Marks all points within a defined radius around each marked point from the previous step
    - Removes all marked points
    - Downsamples the remaining points to a defined target length

    Parameters:
    - tdata: time data array
    - data: raw data array
    - target_len: desired length of the output data array
    - sigma_thresh: number of standard deviations above the median to define the height threshold
    - radius: number of points around each marked point to also mark for removal

    Returns:
    - filtered_tdata: time data array after filtering and downsampling
    - filtered_data: raw data array after filtering and downsampling
    """
    raw_data = np.asarray(data).copy()
    tdata = np.asarray(tdata)

    med = np.median(raw_data)
    height = med + np.std(raw_data) * sigma_thresh

    # Initial keep mask
    keep = raw_data < height

    # Expand removals by radius
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

    # Downsample to target_len points
    filtered_tdata, filtered_data = downsample(filtered_tdata, filtered_data, target_len)
    
    return filtered_tdata, filtered_data

def chunk(tdata,data,n_chunks):
    """  
    This function reads in a raw data file and divides it uniformly into a specified number of chunks.

    Parameters:
    - tdata: time data array
    - data: raw data array
    - n_chunks: number of chunks to divide the data into

    Returns:
    - chunks: list of tuples, where each tuple contains a chunk of time data and the corresponding chunk of raw data
    """
    chunk_size = len(data) // n_chunks
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < n_chunks - 1 else len(data)
        chunks.append((tdata[start:end], data[start:end]))
    return chunks

def fft(data, fs):
    """
    This function computes the Amplitude Spectral Density (ASD) of a given time-domain signal using the 
    Fast Fourier Transform (FFT).

    Parameters:
    - data: time-domain signal array
    - fs: sampling frequency of the signal in Hz

    Returns:
    - freqs: frequency bins
    - asd: ASD values corresponding to the frequency bins
    """
    n = len(data)

    # Remove DC offset
    x = data - np.mean(data)
    N = len(x)

    # Use Hann window
    w = np.hanning(N)
    xw = x * w

    # Perform FFT
    Y = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(N, d=1/fs)

    # Window power normalization
    U = (1/N) * np.sum(w**2)

    # Compute one-sided PSD and convert to ASD
    psd = (1 / (fs * N * U)) * np.abs(Y)**2
    if N % 2 == 0:
        psd[1:-1] *= 2
    else:
        psd[1:] *= 2
    asd = np.sqrt(psd)
    
    return freqs,asd

def filter_chunks(chunks, sigma_thresh):
    """
    This function reads in a list of data chunks and removes chunks that have points surpassing a defined height threshold.

    Parameters:
    - chunks: list of data chunks
    - sigma_thresh: number of standard deviations above the median to define the height threshold

    Returns:
    - filtered_chunks: list containing chunks that pass the threshold
    """
    filtered_chunks = []
    for chunk in chunks:
        data = chunk[1]             # Only look at the waveform data
        med = np.median(data)
        height = med + np.std(data) * sigma_thresh
        if np.any(data > height):
            continue
        filtered_chunks.append(chunk)
    
    num_filtered_chunks = len(chunks)-len(filtered_chunks)

    return filtered_chunks, num_filtered_chunks