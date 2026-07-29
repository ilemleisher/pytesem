import h5py, sys
import numpy as np
from utils import get_files
from preprocess import fft, preprocess
import matplotlib.pyplot as plt
import colorednoise as cn

def spectral_peak_noise(t, a, f_hz):
    return a * np.sin(2*np.pi*f_hz*t)

# Parameters
duration = 10          # seconds
sample_rate = 1250000    # Hz
n_samples = duration * sample_rate
downsample_factor = 10
target_len = n_samples // downsample_factor
output_directory = "/home/ilemleisher/plots/artificial_noise"

# Generate white noise (mean 0, std 1)
white_noise = np.random.normal(0, 1, n_samples)
time_data = np.arange(len(white_noise)) / sample_rate

white_noise += spectral_peak_noise(time_data, 1, 100)

#white_noise[1000000:1020000] += 5

freqs,asd = fft(white_noise,sample_rate)

# Pre process
time_data_pp,white_noise_pp = preprocess(time_data,white_noise,target_len)

freqs_pp,asd_pp = fft(white_noise_pp,target_len/duration)

fig, axs = plt.subplots(2, 2, figsize=(15, 15))

axs[0, 0].plot(time_data,white_noise)
axs[0, 0].set_title(f"White noise")
axs[0, 0].set_xlabel("Time [s]")
axs[0, 0].set_ylabel("ADC1 bins")

axs[0, 1].loglog(freqs, asd)
axs[0, 1].set_title(f"White noise")
axs[0, 1].set_xlabel("Frequency [Hz]")
axs[0, 1].set_ylabel(r"[ADC/$\sqrt{\mathrm{Hz}}$]")

axs[1, 0].plot(time_data_pp,white_noise_pp, color='orange')
axs[1, 0].set_title(f"Pre-Processed White Noise")
axs[1, 0].set_xlabel("Time [s]")
axs[1, 0].set_ylabel("ADC1 bins")

axs[1, 1].loglog(freqs_pp, asd_pp, color='orange')
axs[1, 1].set_title(f"Pre-Processed White Noise")
axs[1, 1].set_xlabel("Frequency [Hz]")
axs[1, 1].set_ylabel(r"[ADC/$\sqrt{\mathrm{Hz}}$]")

fig.savefig(f"{output_directory}/fft_test.png")
plt.close(fig)