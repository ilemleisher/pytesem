import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from utils import get_files, filter_files,  stitch_files
import os
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler

def pca(X_train, X_test):
    # Data already in log space; standardize using training stats only
    scaler = StandardScaler().fit(X_train)
    X_test = X_test.reshape(1, -1)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca = PCA(n_components=0.99, svd_solver="full")
    pca.fit(X_train_s)

    Z = pca.transform(X_test_s)
    Xhat_s = pca.inverse_transform(Z)

    residual = X_test_s - Xhat_s[0]

    eigenvalues = pca.explained_variance_
    t2 = np.sum(Z[0]**2 / eigenvalues)

    metadata = {
        "pca_components": pca.components_,
        "n_components": pca.n_components_,
        "residuals": residual,
        "t2": t2,
        "mean": scaler.mean_,
        "scale": scaler.scale_,
    }
    return metadata

# freqs_total,asd_total=[],[]
# # TRAINING SET
# # Path to regular data
# path = "/home/ilemleisher/data/overnight/"
# for name in os.listdir(path):

#     print("LOADING DATA")
#     filenames = get_files(path+name)

#     containers = stitch_files(path+name+"/", filenames, 'freqs_list','asd_list')
#     freqs_total.append(containers['freqs_list'])
#     asd_total.append(containers['asd_list'])

# asd_total=np.concatenate(asd_total)
# freqs_total=np.concatenate(freqs_total)

# n_chunks = len(freqs_total)
# print(f"FOUND {n_chunks} CHUNKS")

# X_clean = np.log10(asd_total + 1e-12).astype(np.float32)

path = "/home/ilemleisher/data/test_noise/continuous_I4_D20260706_T150239/"

# Dataset
target = 'I4_D20260706_T150246'

print("LOADING VALIDATION SET")
# Load continuous data from preprocessed files following naming format from preprocess.py
filenames = filter_files(get_files(path),target)

# Stitch together continuous dataset
containers = stitch_files(path, filenames, 'freqs_list','asd_list')
freqs_total = containers['freqs_list']
asd_total = containers['asd_list']
X_clean = np.log10(asd_total + 1e-12).astype(np.float32)
n_chunks = len(freqs_total)

print(f"Found {n_chunks} chunks")

metadata_list=[]

window_len = 20
for i in range(n_chunks-window_len):

    print(f"WINDOW: {i} TO {i+window_len}")

    X_train = X_clean[i:window_len+i]
    X_test = X_clean[window_len+i] 

    metadata = pca(X_train, X_test)
    metadata_list.append(metadata)
output_directory = "/home/ilemleisher/data/pca_dev"
print('Saving...')
np.savez(f"{output_directory}/overnight_residuals_control.npz", metadata_list=metadata_list)
print('Saved.')
# fig, ax = plt.subplots(1,1)
# ax.plot(range(window_len,len(residuals)+window_len),residuals)
# output_directory = "/home/ilemleisher/plots/pca_dev/"
# ax.set_xlabel("Data Chunk")
# ax.set_ylabel("Abs(sum(residuals))")
# ax.set_title("PCA reconstruction residual per chunk")
# fig.savefig(f"{output_directory}/overnight_residuals.png")
# plt.close(fig)

# end = 60

# # Stitch together continuous dataset
# containers = stitch_files(path, filenames, 'freqs_list','asd_list')
# freqs_total_train = containers['freqs_list'][:end]
# asd_total_train = containers['asd_list'][:end]

# X_train_clean = np.log10(asd_total_train + 1e-12).astype(np.float32)
# print(f"{len(X_train_clean)} training chunks")

# chunk = end + 1

# freqs_total_val = containers['freqs_list'][chunk]
# asd_total_val = containers['asd_list'][chunk]

# X_val = np.log10(asd_total_val + 1e-12).astype(np.float32)

# flag, metadata = pca(X_train_clean, X_val)
# residuals = metadata['residual_data']

# print(f"Chunk {chunk}: {['NORMAL', 'ANOMALY'][flag[0]]}")

# fig, axs = plt.subplots(1, 1, figsize=(10, 10))

# axs.loglog(freqs_total_val, asd_total_val, color='magenta', label='Original PSD', alpha= 0.8)
# axs.set_title(f"Data Chunk {str(chunk)} (Anomalous)")
# axs.set_xlabel("Frequency [Hz]")
# axs.set_ylabel(r"[ADC/$\sqrt{\mathrm{Hz}}$]")
# axs.set_xlim(0.01, 10**6)
# axs.set_ylim(10**(-5), 100)
# axs.grid(True, ls='--')

# # Reconstruction (invert log10 for the linear ASD axis)
# Xhat = 10 ** metadata['reconstruction']
# axs.loglog(freqs_total_val, Xhat, color='cyan', lw=1.0, alpha=0.8,
#            label=f'PCA reconstruction (fit on chunks 0 to {end})')

# # Anomalous bins, colored by signed log-space residual
# res = metadata['residual_data']
# anom_bins = np.where(res > np.percentile(res, 99.99))[0]

# trans = axs.get_xaxis_transform()
# vmax = np.max(np.abs(res[anom_bins]))
# norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
# sc = axs.scatter(freqs_total_val[anom_bins], np.zeros(len(anom_bins)),
#                     c=res[anom_bins], cmap='Spectral', norm=norm,
#                     transform=trans, marker='|', s=350, linewidths=2,
#                     clip_on=False, zorder=5)
# cbar = axs.figure.colorbar(sc, ax=axs, pad=0.1)
# cbar.set_label(r'Residual [$\log_{10}$]')

# axs.legend(loc='upper right')
# output_directory = "/home/ilemleisher/plots/pca_dev/"

# fig.savefig(f"{output_directory}/{str(target)}_chunk_{str(chunk)}.png")
# plt.close(fig)