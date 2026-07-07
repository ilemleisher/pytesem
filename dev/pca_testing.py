import numpy as np
from sklearn.decomposition import PCA
from utils import get_files, filter_files,  stitch_files
import matplotlib.pyplot as plt
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
path = "/home/ilemleisher/data/artificial_noise/val/"

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

X_val = np.log10(asd_total + 1e-12).astype(np.float32)
y_val = labels.astype(np.int32)
print(f"Found {len(X_val)} chunks")

# TRAINING SET
# Path to regular data
#path = "/home/ilemleisher/data/continuous_I4_D20250102_T224744/"
path = "/home/ilemleisher/data/artificial_noise/val/"

# Dataset
#target = 'I4_D20250102_T225835'

# Load continuous data from preprocessed files following naming format from preprocess.py
filenames = filter_files(get_files(path),target)
print(f"Found {len(filenames)} files")

# Stitch together continuous dataset
containers = stitch_files(path, filenames, 'freqs_list','asd_list')
freqs_total = containers['freqs_list']
asd_total = containers['asd_list']

X_train_clean = np.log10(asd_total + 1e-12).astype(np.float32)
print(f"Found {len(X_train_clean)} chunks")

flags, idx, metadata = flag(X_train_clean, X_val)

print(np.where(flags & labels == 1))

matches = sum(1 for a, b in zip(flags, labels) if a == b)
smc = matches / len(flags)
print(smc)