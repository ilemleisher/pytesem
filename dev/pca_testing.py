import numpy as np
from sklearn.decomposition import PCA
from utils import get_files, filter_files,  stitch_files
import numpy as np
from sklearn.decomposition import PCA

def flag(X_train, X_val, thr=0.0015):

    # PCA
    pca = PCA(n_components=0.99, svd_solver="full")  # keep 99% variance
    pca.fit(X_train)
    
    # Project each sample into PCA space
    Z = pca.transform(X_val)

    # Reconstruct back into original space
    Xhat = pca.inverse_transform(Z)

    # Compute MSE
    err = np.mean((X_val - Xhat)**2, axis=1)

    flags = (err > thr).astype(np.int32)
    idx = np.where(flags == 1)[0]

    residual_data = X_val - Xhat
    metadata = {}
    metadata['pca_components'] = pca.components_
    metadata['residual_data'] = residual_data

    return flags, idx, metadata


# VALIDATION SET
# Path to noisy data
path = "/home/ilemleisher/data/test_noise/continuous_I4_D20260707_T160503/"

# Dataset
target = 'I4_D20260707_T160511'

print("LOADING VALIDATION SET")
# Load continuous data from preprocessed files following naming format from preprocess.py
filenames = filter_files(get_files(path),target)

# Stitch together continuous dataset
containers = stitch_files(path, filenames, 'freqs_list','asd_list')
freqs_total = containers['freqs_list']
asd_total = containers['asd_list']
labels = np.ones(len(freqs_total))

X_val = np.log10(asd_total + 1e-12).astype(np.float32)
#y_val = labels.astype(np.int32)
print(f"Found {len(X_val)} chunks")

# TRAINING SET
# Path to regular data
path = "/home/ilemleisher/data/test_noise/continuous_I4_D20260706_T150239/"

# Dataset
target = 'I4_D20260706_T150246'

# Load continuous data from preprocessed files following naming format from preprocess.py
print("LOADING TRAINING SET")
filenames = filter_files(get_files(path),target)

# Stitch together continuous dataset
containers = stitch_files(path, filenames, 'freqs_list','asd_list')
freqs_total = containers['freqs_list']
asd_total = containers['asd_list']

X_train_clean = np.log10(asd_total + 1e-12).astype(np.float32)
print(f"Found {len(X_train_clean)} chunks")

print("FLAGGING ANOMALIES")
flags, idx, metadata = flag(X_train_clean, X_val)

print(f"Anomalous chunks: {np.where(flags == 1)[0]}")
#print(f"Accuracy: {np.sum(flags)/len(flags)*100}%")

matches = sum(1 for a, b in zip(flags, labels) if a == b)
smc = matches / len(flags)
print(f"Accuracy: {smc * 100}%")