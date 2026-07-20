import h5py, os
import numpy as np
from utils import get_files
from scipy.ndimage import binary_dilation

# Variables
n_chunks = 1                        # Number of chunks to divide each data file into
downsample_factor = 10              # Downsample reduction factor
sampling_rate = 1.25e6              # Sampling rate of the raw data in Hz
channel_number = 1                  # Channel number to read from the raw data file (0-indexed)

def preprocess(tdata, data, target_len, sigma_thresh=4, radius=100):
    """
    This function reads in raw data file and first:
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
    print("Height mask")
    # Initial keep mask
    keep = raw_data < height
    print("Expand by radius")
    # Expand removals by radius
    if radius > 0:
        remove = ~keep
        kernel = np.ones(2 * radius + 1, dtype=bool)
        out = binary_dilation(remove, structure=kernel)
        keep = ~out

        # remove = ~keep
        # kernel = np.ones(2 * radius + 1, dtype=int)
        # expanded_remove = np.convolve(remove.astype(int), kernel, mode="same") > 0
        # keep = ~expanded_remove

    filtered_data = raw_data[keep]
    filtered_tdata = tdata[keep]

    n = len(filtered_data)
    if n == 0:
        raise ValueError("No data left after filtering.")
    if n < target_len:
        raise ValueError(f"Filtered length ({n}) is smaller than target_len ({target_len}).")
    print("Downsample")
    # Downsample to exactly target_len points
    idx = np.linspace(0, n - 1, target_len, dtype=int)
    return filtered_tdata[idx], filtered_data[idx]

def downsample(tdata, data, target_len=12000):
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

def chunk(tdata,data,n_chunks=n_chunks):
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
    print("Dividing into chunks")
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < n_chunks - 1 else len(data)
        chunks.append((tdata[start:end], data[start:end]))
    return chunks

def fft(data, fs=sampling_rate):
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
    print("Computing ASD")
    # Remove DC offset
    x = data - np.mean(data)
    N = len(x)

    # Use Hann window
    #w = np.hanning(N)
    #xw = x * w
    xw = x

    # Perform FFT
    Y = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(N, d=1/fs)

    # Window power normalization
    # U = (1/N) * np.sum(w**2)
    U = 1
    # Compute one-sided PSD and convert to ASD
    psd = (1 / (fs * N * U)) * np.abs(Y)**2
    if N % 2 == 0:
        psd[1:-1] *= 2
    else:
        psd[1:] *= 2
    asd = np.sqrt(psd)
    
    return freqs,asd

def filter_chunks(chunks, sigma_thresh=4):
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


if __name__ == '__main__':

    # Path to the folder containing the .hdf5 files
    base_path = "/data/lbl/run45/raw/"
    folder = "continuous_I4_D20260707_T160503/"
    output_path = '/home/ilemleisher/data/test_noise/'
    output_directory = os.path.join(output_path, folder)
    os.makedirs(output_directory, exist_ok=True)

    path = os.path.join(base_path, folder)
    filenames = get_files(path)
    
    # Loop over each file in the folder
    for filename in filenames:
        filepath = path+filename
        print(f"Reading: {filename}")
        num_filtered = 0
        num_chunks = 0

        with h5py.File(filepath, "r") as data:

            events = data['adc1'].keys()
            print("Found", len(events), "events")

            # Data containers
            freqs_list, asd_list, waveform_data_list = [], [], []
            new_tdata_list, new_data_list, time_data_list = [], [], []
            labels, og_freqs_list, og_asd_list = [], [], []

            # Loop over each event in the file
            for event in events:

                # Read ADC1 output from the specified channel
                waveforms = np.array(data["adc1"][str(event)])
                n_samples = waveforms.shape[1]
                time_data = np.arange(n_samples) / sampling_rate
                waveform_data = waveforms[channel_number]

                # Calculations
                target_len = n_samples // downsample_factor
                duration = n_samples // sampling_rate
                fft_fs = target_len // duration

                og_freqs, og_asd = fft(waveform_data, sampling_rate)

                # Downsample the raw waveform data
                new_tdata, new_data = preprocess(time_data,waveform_data,target_len=target_len)
                #new_tdata, new_data = downsample(time_data, waveform_data, target_len=target_len)

                # Divide the raw data into chunks
                chunks = chunk(new_tdata, new_data)

                # Count number of chunks
                num_chunks += len(chunks)

                # Discard any chunks that contains peaks above 4 sigma
                #filtered_chunks, num_filtered_chunks = filter_chunks(chunks, 4)
                filtered_chunks = chunks
                num_filtered_chunks = 0
                # Count number of filtered chunks
                #num_filtered += num_filtered_chunks

                # Loop over each remaining chunk
                for data_chunk in filtered_chunks:

                    # Compute the ASD for each chunk
                    freqs, asd = fft(data_chunk[1], fft_fs)

                    freqs_list.append(freqs.astype(np.float32))
                    asd_list.append(asd.astype(np.float32))
                    waveform_data_list.append(waveform_data.astype(np.float32))
                    new_tdata_list.append(new_tdata.astype(np.float32))
                    new_data_list.append(new_data.astype(np.float32))
                    time_data_list.append(time_data.astype(np.float32))
                    og_freqs_list.append(og_freqs.astype(np.float32))
                    og_asd_list.append(og_asd.astype(np.float32))

        print(f"Filtered out {num_filtered}/{num_chunks} chunks in file {filename} ({num_filtered/num_chunks*100:.1f}% removal rate)")

        print(f'Saving...')

        # Stack data and save to .npz file
        np.savez(
            f"{output_directory}/{filename[:-5]}.npz",
            freqs_list=np.stack(freqs_list),         # (N, F) or (F,)
            asd_list=np.stack(asd_list),             # (N, F)
            # optional metadata
            new_tdata_list=np.stack(new_tdata_list),
            new_data_list=np.stack(new_data_list),
            time_data_list=np.stack(time_data_list),
            waveform_data_list=np.stack(waveform_data_list),
            og_freqs_list=np.stack(og_freqs_list),
            og_asd_list=np.stack(og_asd_list)
)
        print(f'Saved.')