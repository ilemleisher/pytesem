import h5py, sys
import numpy as np
from utils import get_files
sys.path.append("/home/ilemleisher/em_project/modules/") 
from ema import get_drift_score
import matplotlib.pyplot as plt
import colorednoise as cn

# Variables
n_chunks = 2                        # Number of chunks to divide each data file into
post_downsample_length = 12000      # Target length of the raw data file for downsampling
sampling_rate = 1.25e6              # Sampling rate of the raw data in Hz
channel_number = 0                  # Channel number to read from the raw data file (0-indexed)

def spectral_peak_noise(t, a, f_hz):
    a_rand = a * (1 + np.random.normal(0, 0.01))      # ~1% amplitude variation
    f_rand = f_hz * (1 + np.random.normal(0, 0.001))  # ~0.1% frequency variation
    phi = np.random.uniform(-0.05, 0.05)              # very small phase variation (rad)
    return a_rand * np.sin(f_rand*t)

def broadband_noise(x, s):
    return np.random.normal(np.mean(x), s, size=len(x))

def colored_noise(x, b, a):
    drift = cn.powerlaw_psd_gaussian(b, len(x))
    drift = drift / np.std(drift) * a
    return drift

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

# Path to the folder containing the .hdf5 files
path = "/data/lbl/run21/raw/continuous_I4_D20250102_T224744/"

filenames = get_files(path)

# Loop over each file in the folder
for filename in filenames[:8]:
    filepath = path+filename
    print("Reading:", filename)
    num_filtered = 0
    num_chunks = 0

    with h5py.File(filepath, "r") as data:
        events = data['adc1'].keys()
        print("Found", len(events), "events")
        labels = []
        # Data containers
        freqs_list, asd_list, waveform_data_list = [], [], []
        new_tdata_list, new_data_list, time_data_list = [], [], []
        baseline_list, residual_list, drift_score_list = [], [], []

        # Loop over each event in the file
        for event in events:

            # Read ADC1 output from the specified channel
            waveforms = np.array(data["adc1"][str(event)])
            time_data = np.arange(waveforms.shape[1]) / sampling_rate
            waveform_data = waveforms[channel_number]

            if np.random.random() < 0.8:

                waveform_data = spectral_peak_noise(time_data, 5, 100) + waveform_data
                print("Added noise to", event)

                noise_tag = True
            else:
                noise_tag = False

            new_tdata, new_data = downsample(time_data, waveform_data)

            # Divide the raw data into chunks
            chunks = chunk(new_tdata, new_data)

            # Count number of chunks
            num_chunks += len(chunks)

            # Discard any chunks that contains peaks above 4 sigma
            filtered_chunks, num_filtered_chunks = filter_chunks(chunks, 4)

            # Count number of filtered chunks
            num_filtered += num_filtered_chunks

            # Loop over each remaining chunk
            for data_chunk in filtered_chunks:
                
                if noise_tag == True:
                    labels.append(1)
                elif noise_tag == False:
                    labels.append(0)

                # Compute the ASD for each chunk
                freqs, asd= fft(data_chunk[1])
                baseline, residual, drift_score = get_drift_score(freqs, np.log10(asd + 1e-12).astype(np.float32))

                freqs_list.append(freqs.astype(np.float32))
                asd_list.append(asd.astype(np.float32))
                waveform_data_list.append(waveform_data.astype(np.float32))
                new_tdata_list.append(new_tdata.astype(np.float32))
                new_data_list.append(new_data.astype(np.float32))
                time_data_list.append(time_data.astype(np.float32))

                baseline_list.append(baseline)
                residual_list.append(residual)
                drift_score_list.append(drift_score)

    print(f"Filtered out {num_filtered}/{num_chunks} chunks in file {filename} ({num_filtered/num_chunks*100:.1f}% removal rate)")

    print(f'Saving...')
    # Stack data and save to .npz file
    np.savez(
        f"/home/ilemleisher/data/artificial_noise/data_{filename[:-5]}.npz",
        freqs_list=np.stack(freqs_list),         # (N, F) or (F,)
        asd_list=np.stack(asd_list),             # (N, F)
        # optional metadata
        new_tdata_list=np.stack(new_tdata_list),
        new_data_list=np.stack(new_data_list),
        time_data_list=np.stack(time_data_list),
        waveform_data_list=np.stack(waveform_data_list),
        labels=np.array(labels, dtype=np.int32),
)
    print(f'Saved.')

# for i in range(len(freqs_list[:20])):

#     print(f"Plotting chunk {str(i)}")
#     fig, axs = plt.subplots(2, 2, figsize=(12, 8))

#     axs[0, 0].plot(time_data_list[i], waveform_data_list[i], label="original data")
#     axs[0, 0].set_title("Original waveform")
#     axs[0, 0].set_xlabel("Time [s]")
#     axs[0, 0].set_ylabel("Amplitude")
#     axs[0, 0].legend()

#     axs[1, 0].plot(new_tdata_list[i], new_data_list[i], label="peakless data")
#     axs[1, 0].set_title("Waveform after peak removal/downsampling")
#     axs[1, 0].set_xlabel("Time [s]")
#     axs[1, 0].set_ylabel("Amplitude")
#     axs[1, 0].legend()

#     axs[0, 1].loglog(freqs_list[i], asd_list[i], label="fft data")
#     axs[0, 1].plot(freqs_list[i],10**baseline_list[i], 'r--', label=f"drift score={drift_score_list[i]:.3f}")
#     axs[0, 1].set_title("FFT magnitude")
#     axs[0, 1].set_xlabel("Frequency [Hz]")
#     axs[0, 1].set_ylabel(r"[A/$\sqrt{\mathrm{Hz}}$]")
#     axs[0, 1].legend()

#     axs[1, 1].hist(new_data_list[i], bins=100, label="histogram of data")
#     axs[1, 1].set_title("Noise histogram")
#     axs[1, 1].set_xlabel("Amplitude")
#     axs[1, 1].set_ylabel("Count")
#     axs[1, 1].legend()

#     fig.savefig("/home/ilemleisher/plots/artificial_noise/plot_noise_"+str(filename[:-4])+"_chunk_"+str(i)+".png")
#     plt.close(fig)

    
    


