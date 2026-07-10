import numpy as np
from sklearn.decomposition import PCA
from utils import get_files, filter_files,  stitch_files
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from statistics import mode

def flag(freqs, X, thr=0.0015):
    """
    This function computes the PCA reconstruction-error per sample and assigns flags based on a defined threshold.

    Inputs:
    - freqs: an array of frequency lists for data chunks
    - X: an array of ASD amplitude (in logspace) lists for the same data chunks
    - thr: the threshold above which to label an anomaly
    Returns:
    - flags: array of anomaly labels (0 = no anomaly, 1 = anomaly)
    - idx: array of corresponding indices for the anomaly labels
    - metadata:
        - pca_components: PCA components used for reconstruction
        - residual_data: array of PCA error residuals for each anomalous chunk
    """
    # PCA
    pca = PCA(n_components=0.99, svd_solver="full")  # keep 99% variance
    pca.fit(X)
    
    # Project each sample into PCA space
    Z = pca.transform(X)

    # Reconstruct back into original space
    Xhat = pca.inverse_transform(Z)

    # Compute MSE
    err = np.mean((X - Xhat)**2, axis=1)

    flags = (err > thr).astype(np.int32)
    idx = np.where(flags == 1)[0]

    residual_data = X - Xhat
    metadata = {}
    metadata['pca_components'] = pca.components_
    metadata['residual_data'] = residual_data

    return flags, idx, metadata

# Path to control data
path = "/home/ilemleisher/data/test_noise/continuous_I4_D20260706_T150239/"

# Dataset
target = 'I4_D20260706_T150246'

# Load continuous data from preprocessed files following naming format from preprocess.py
filenames = filter_files(get_files(path),target)
print(f"Found {len(filenames)} files")

# Stitch together continuous dataset
containers = stitch_files(path, filenames, 'freqs_list','asd_list')
freqs_total = containers['freqs_list']
asd_total = containers['asd_list']
# labels = containers['labels']

X_val = np.log10(asd_total + 1e-12).astype(np.float32)
#y_val = labels.astype(np.int32)
print(f"Found {len(X_val)} chunks")

flags, idx, metadata = flag(freqs_total, X_val)

print(f"Anomalous chunk indices: {idx.tolist()}")
components_list=[]
components = metadata['pca_components']
for i in range(len(components)):
    components_list.append(np.where(np.abs(components[i]) > np.percentile(np.abs(components[i]), 99)))

print(f"Mode of 99th percentile anomalous components: {mode(np.concatenate(components_list)[0])}")

fig, ax = plt.subplots(1,1,figsize=(12,8))
plt.title(f"5 highest weighted PCA Components vs frequency bin for {target}")
for i in range(len(components))[:5]:
    ax.loglog(freqs_total[i], np.abs(components[i]), alpha=0.5)

ax.set_xlabel("Frequency bin (Hz)")
ax.set_ylabel("PCA component amplitude")

fig.savefig(f"/home/ilemleisher/plots/test_noise/continuous_I4_D20260706_T150239/pca_components_{target}.png", dpi=300)