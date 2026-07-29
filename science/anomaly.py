import numpy as np
import sys
from sklearn.decomposition import PCA
sys.path.append("/home/ilemleisher/em_project/modules/") 
from ema import get_drift_score
from utils import get_files, filter_files,  stitch_files
import matplotlib.pyplot as plt

# Reconstruction-error anomaly score
def pca_recon_flags(X, pca, thr=0.0015):
    """
    This function computes the PCA reconstruction-error per sample and assigns flags based on a defined threshold.

    Inputs:
    - X: noramlized input spectra with shape (N, F) (N: number of samples, F: number of frequency bins)
    - pca: PCA model
    - threshold_percentile: the threshold above which to label an anomaly
    Returns:
    - peak_flags: list of anomaly labels (0 = no anomaly, 1 = anomaly)
    - idx: list of corresponding indices for the anomaly labels
    - residual_data: array of PCA error residuals for each anomalous chunk
    """
    # Project each sample into PCA space
    Z = pca.transform(X)

    # Reconstruct back into original space
    Xhat = pca.inverse_transform(Z)

    # Compute MSE
    err = np.mean((X - Xhat)**2, axis=1)

    peak_flags = (err > thr).astype(np.int32)
    idx = np.where(peak_flags == 1)[0]
    print(f"Labeled chunks with indices {idx.tolist()} as anomalous by PCA reconstruction")

    residual_data = X - Xhat

    return peak_flags, idx, residual_data

def pca_recon_flags_binlocal(
    X,
    pca,
    thr_global=0.0015,
    thr_local=4.0,
    mode="max_z",          # "max_z" | "topk_z_mean" | "band_max_z"
    topk=5,
    band_idx=None,         # tuple(start, stop) if mode="band_max_z"
    alpha=0.5,             # for combined score
    thr_combined=None      # optional threshold on combined score
):
    """
    PCA anomaly flags with both global and bin-local residual scoring.

    Inputs
    ------
    X : array (N, F)
        Normalized spectra.
    pca : fitted PCA model
    thr_global : float
        Threshold for global reconstruction MSE.
    thr_local : float
        Threshold for local z-residual score.
    mode : str
        Local score type:
          - "max_z": max abs z-residual across bins
          - "topk_z_mean": mean of top-k abs z-residual bins
          - "band_max_z": max abs z-residual in a selected band
    topk : int
        Used when mode="topk_z_mean".
    band_idx : (start, stop)
        Frequency-bin slice for mode="band_max_z".
    alpha : float in [0,1]
        Weight for combined score = alpha*global_z + (1-alpha)*local_score.
    thr_combined : float or None
        If set, use combined score threshold too.

    Returns
    -------
    flags : (N,) int32
        1 if anomalous else 0.
    idx : (M,) int
        Indices of flagged samples.
    residual : (N, F)
        X - Xhat.
    scores : dict
        global_err, global_z, local_score, combined_score
    """
    # PCA reconstruction
    Z = pca.transform(X)
    Xhat = pca.inverse_transform(Z)
    residual = X - Xhat

    # ----- Global score (MSE) -----
    global_err = np.mean(residual**2, axis=1)
    g_mu, g_sd = np.mean(global_err), np.std(global_err) + 1e-12
    global_z = (global_err - g_mu) / g_sd

    # ----- Bin-local score -----
    # Per-bin robust-ish standardization of residuals
    r_mu = np.mean(residual, axis=0, keepdims=True)
    r_sd = np.std(residual, axis=0, keepdims=True) + 1e-12
    zres = (residual - r_mu) / r_sd
    absz = np.abs(zres)

    if mode == "max_z":
        local_score = np.max(absz, axis=1)

    elif mode == "topk_z_mean":
        k = min(topk, absz.shape[1])
        topk_vals = np.partition(absz, -k, axis=1)[:, -k:]
        local_score = np.mean(topk_vals, axis=1)

    elif mode == "band_max_z":
        if band_idx is None:
            raise ValueError("band_idx=(start, stop) required for mode='band_max_z'")
        s, e = band_idx
        local_score = np.max(absz[:, s:e], axis=1)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ----- Flags -----
    flags_global = global_err > thr_global
    flags_local = local_score > thr_local

    combined_score = alpha * global_z + (1 - alpha) * local_score
    if thr_combined is None:
        flags = (flags_global | flags_local).astype(np.int32)
    else:
        flags_combined = combined_score > thr_combined
        flags = (flags_global | flags_local | flags_combined).astype(np.int32)

    idx = np.where(flags == 1)[0]

    scores = {
        "global_err": global_err,
        "global_z": global_z,
        "local_score": local_score,
        "combined_score": combined_score,
    }

    print(f"Labeled chunks with indices {idx.tolist()} as anomalous (global + bin-local).")
    return flags, idx, residual, scores

def ema_baseline_flag(freqs, X, s):
    '''
    This function calculates a drift score for each data chunk and assigns anomaly flags based on a
    defined threshold of s sigma above the mean.

    Parameters:
    - freqs: an array of frequency lists for data chunks
    - X: an array of ASD amplitude (in logspace) lists for the same data chunks
    - s: sigma threshold
    Returns:
    - drift_flags: list of anomaly labels (0 = no anomaly, 1 = anomaly) 
    - idx: list of corresponding indices for the anomaly labels
    - baselines: list of baseline value lists for data chunks
    - residuals: list of residual value lists for data chunks
    - drift_scores: list of drift scores for data chunks
    '''

    drift_scores=[]
    baselines = []
    residuals = []

    for i in range(len(X)):
        baseline, residual, drift_score = get_drift_score(freqs[i], X[i])
        drift_scores.append(drift_score)
        baselines.append(baseline)
        residuals.append(residual)

    thr = np.mean(drift_scores) + s * np.std(drift_scores)
    drift_flags = (drift_scores > thr).astype(np.int32)
    idx = np.where(drift_flags == 1)[0]
    print(f"Labeled chunks with indices {idx.tolist()} as anomalous by EMA baseline drift")

    return drift_flags, idx, baselines, residuals, drift_scores


# Path to preprocessed data (assumes format from preprocess.py)
path = "/home/ilemleisher/data/artificial_noise/"

# Dataset
target = 'I4_D20250102_T224816'

# Load continuous data from preprocessed files following naming format from preprocess.py
filenames = filter_files(get_files(path),target)
print(f"Found {len(filenames)} files")

# Stitch together continuous dataset
containers = stitch_files(path, filenames, 'freqs_list','asd_list','labels')
freqs_total = containers['freqs_list']
asd_total = containers['asd_list']
labels = containers['labels']

X = np.log10(asd_total + 1e-12).astype(np.float32)
print(f"Found {len(X)} chunks")

# PCA
pca = PCA(n_components=0.99, svd_solver="full")  # keep 99% variance
pca.fit(X)

peak_flags, idx, residuals, scores = pca_recon_flags_binlocal(X, pca, thr_global=0.0015, thr_local=4.3, mode="max_z", alpha=0.5, thr_combined=None)
print(labels[idx])
# Plot each anomalous original ASD with the residuals overlayed
for index in idx:
    
    f = np.asarray(freqs_total[index]).ravel()       # frequency
    asd = np.asarray(asd_total[index]).ravel()         # ASD
    res = np.abs(np.asarray(residuals[index]).ravel())  # residual magnitude per freq bin

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # ASD curve (log-log)
    ax1.loglog(f, asd, color="tab:blue", lw=1.4, label="ASD")
    ax1.set_xlabel("Frequency")
    ax1.set_ylabel("ASD", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, which="both", ls="--", alpha=0.3)

    # Histogram overlay sharing same x-axis (frequency), weighted by residual magnitude
    ax2 = ax1.twinx()
    ax2.plot(f, res,color='tab:red',alpha = 0.5,label='residuals')
    ax2.set_xscale("log")     # ensure same log x scaling
    ax2.set_ylabel("PCA reconstruction residual", color="tab:red")
    ax2.set_ylim(0,2)
    ax2.tick_params(axis="y", labelcolor="tab:red")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    plt.title(f"ASD (log-log) with aligned residuals")
    plt.savefig(f"/home/ilemleisher/plots/artificial_noise/residuals/{target}_residuals_chunk_{index}")
    plt.close(fig)


# Get EMA baseline drift flags
drift_flags, idx, baselines, residuals, drift_scores = ema_baseline_flag(freqs_total, X, 3)

# For each baseline anomaly, plot the nearest 5 data chunks with baselines overlayed
for center_idx in idx:
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)

    # plot chunks [center_idx-2, ..., center_idx+2]
    for offset, ax in zip(range(-2, 3), axes):
        i = center_idx + offset

        ax.loglog(freqs_total[i], asd_total[i], label="fft data")
        ax.loglog(
            freqs_total[i],
            10**baselines[i],
            "r--",
            label=f"drift score={drift_scores[i]:.3f}",
        )
        ax.set_title(f"Chunk {i}")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(r"[A/$\sqrt{\mathrm{Hz}}$]")
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"/home/ilemleisher/plots/artificial_noise/residuals/{target}_drift_chunk_{center_idx}")
    plt.close(fig)
    

np.savez_compressed(f"{path}labels/pca_labels_{target}.npz",pca_labels=peak_flags.astype(np.int8))