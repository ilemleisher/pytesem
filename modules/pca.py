import numpy as np
from sklearn.decomposition import PCA
from dev.utils import track_runtime

# This module uses PCA to detect anomalies in the ASD data. It computes the PCA reconstruction-error per sample and 
# flags chunks with errors above a defined threshold as anomalous.

@track_runtime
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
    # Divide dataset into train and val
    size = len(X)
    train = X[0:size//2]
    val = X[size//2:]

    # PCA
    pca = PCA(n_components=0.99, svd_solver="full")  # keep 99% variance
    pca.fit(train)
    
    # Project each sample into PCA space
    Z = pca.transform(val)

    # Reconstruct back into original space
    Xhat = pca.inverse_transform(Z)

    # Compute MSE
    err = np.mean((val - Xhat)**2, axis=1)

    flags = (err > thr).astype(np.int32)
    idx = np.where(flags == 1)[0]

    residual_data = val - Xhat
    metadata = {}
    metadata['pca_components'] = pca.components_
    metadata['residual_data'] = residual_data

    return flags, idx, metadata