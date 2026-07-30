import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Minimum number of rows required in the window before PCA is attempted.
# NOTE: PCA actually trains on len(window) - 1 rows (the last row is held
# out as the test sample), so the effective training count is one less than
# the window length. See reduce_window.
MIN_TRAIN_ROWS = 10


def pca_fit(X_train, X_test):

    # Standardize using training stats only; the scaler is never fit on X_test.
    scaler = StandardScaler().fit(X_train)
    X_test = X_test.reshape(1, -1)               # PCA/scaler expect 2D input
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Retain enough components to explain 99% of training variance.
    pca = PCA(n_components=0.99, svd_solver="full")
    pca.fit(X_train_s)

    return pca, X_test_s, scaler

def pca_reconstruct(pca, X_test_s):

    # Project the test spectrum into the PCA subspace and reconstruct it.
    Z = pca.transform(X_test_s)
    Xhat_s = pca.inverse_transform(Z)

    # Residual in standardized space: what the PCA model failed to capture.
    residual = X_test_s - Xhat_s[0]

    reduced_data = {
        "residual": residual
    }

    return reduced_data


def reduce_window(window, first_pca=None, first_scaler=None):
    """
    Reduce a window of processed chunks into a PCA-based anomaly summary.

    Treats the oldest chunks in the window as the training set and the most
    recent chunk as the test sample. Stacks the per-chunk ASDs, converts to
    log space, and hands off to pca(). The frequency axis (bins) is carried
    through into the result so downstream saving/plotting can align the
    residual and components to real frequencies.

    Parameters:
    - window: sequence of chunk dicts, each with keys "bins" and "asd"
              (as produced by process_chunk). Length must be >= MIN_TRAIN_ROWS.

    Returns:
    - dict: the pca() result, additionally containing "bins" (1D frequency
      axis). All values are numpy arrays, suitable for np.savez(**result).

    Raises:
    - ValueError: if the window has fewer than MIN_TRAIN_ROWS entries.
    """
    if len(window) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Window too small for PCA: got {len(window)}, "
            f"need >= {MIN_TRAIN_ROWS}"
        )

    # Frequency axis is identical across chunks, so take it from the first.
    bins = window[0]["bins"]
    # Stack per-chunk ASDs into (n_chunks, n_bins).
    asd_stack = np.stack([c["asd"] for c in window])

    # Work in log space; the +1e-12 floor avoids log10(0) for empty bins.
    X_clean = np.log10(asd_stack + 1e-12).astype(np.float32)

    # Oldest chunks train the model; the newest chunk is the one under test.
    X_train = X_clean[:-1]
    X_test = X_clean[-1]

    # Fit PCA on current window
    pca, scaler, X_test_s = pca_fit(X_train, X_test)
    short = pca_reconstruct(pca, X_test_s)

    reduced = {
        "residual": short["residual"],
        "bins": bins,
        "asd": X_test,
    }

    if first_pca is not None and first_scaler is not None:
        X_test_s_long = first_scaler.transform(X_test.reshape(1, -1))
        reduced["long_term_residual"] = pca_reconstruct(first_pca, X_test_s_long)["residual"]

    return reduced, pca, scaler