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
path = "/home/ilemleisher/data/test_noise/"
dataset="continuous_I4_D20260706_T160848/"
path += dataset
# Dataset
target = 'I4_D20260706_T160855'

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

components = metadata['pca_components']

fig, axes = plt.subplots(2, 5, figsize=(24, 10), sharey=True)
fig.suptitle(f"10 highest weighted PCA Components vs frequency bin for {target}")

for i in range(5):
    axes[0,i].loglog(freqs_total[i], np.abs(components[i]))
    axes[0,i].set_xlabel("Frequency bin (Hz)")
    axes[0,i].set_title(f"Component {i}")

for i in range(5):
    axes[1,i].loglog(freqs_total[i+5], np.abs(components[i+5]))
    axes[1,i].set_xlabel("Frequency bin (Hz)")
    axes[1,i].set_title(f"Component {i+5}")

axes[0,0].set_ylabel("PCA component amplitude")
axes[1,0].set_ylabel("PCA component amplitude")

plt.tight_layout()

fig.savefig(f"/home/ilemleisher/plots/test_noise/{dataset}pca_components_{target}.png", dpi=300)