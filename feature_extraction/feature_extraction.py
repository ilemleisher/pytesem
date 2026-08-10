"""
PCA-based dimensionality reduction for sliding windows of preprocessed
noise spectra.

Fits a PCA model on the older chunks in a window (log10-ASD, mean-centered
only), reconstructs the newest chunk from that model, and reports the
reconstruction residual as the anomaly signal. Optionally also scores the
newest chunk against a "first" (long-term reference) PCA model fit earlier
in the run.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Minimum number of rows required in the window before PCA is attempted.
# This is separate from the window size; just a fail-safe.
# NOTE: PCA actually trains on len(window) - 1 rows (the last row is held
# out as the test sample), so the effective training count is one less than
# the window length. See reduce_window.
MIN_TRAIN_ROWS = 15


def pca_fit(X_train, X_test):
    """
    Mean-center the training/test rows and fit a PCA model on the training set.

    Parameters:
    - X_train: (n_train, n_bins) array of log10-ASD rows used to fit PCA.
    - X_test: (n_bins,) array, the single row to be reconstructed later.

    Returns:
    - pca: fitted sklearn PCA object, retaining components covering 99%
      of the training variance.
    - X_test_s: (1, n_bins) mean-centered test row.
    - scaler: fitted StandardScaler (mean-only) used for the centering,
      returned so the same centering can be reapplied to other test rows.
    """
    # Mean-center only (no std-division). Residuals stay in log10(ASD)
    # units, so they're directly interpretable as fractional amplitude
    # deviations, e.g. a residual of 0.3 ~ roughly a factor of 2 (10**0.3),
    # and 1.0 ~ a factor of 10 — no per-bin variance normalization needed.
    scaler = StandardScaler(with_std=False).fit(X_train)
    X_test = X_test.reshape(1, -1)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Keep enough principal components to explain 99% of training variance.
    pca = PCA(n_components=0.99, svd_solver="full")
    pca.fit(X_train_s)

    return pca, X_test_s, scaler


def pca_reconstruct(pca, X_test_s):
    """
    Project a mean-centered test row into PCA space and back out, then
    compute the reconstruction residual.

    Parameters:
    - pca: fitted PCA object.
    - X_test_s: (1, n_bins) mean-centered test row to reconstruct.

    Returns:
    - dict with key "residual": (n_bins,) array, the per-bin difference
      between the original and PCA-reconstructed test row (in log10-ASD
      units / dex).
    """
    # Forward projection (encode) then inverse projection (decode).
    Z = pca.transform(X_test_s)
    Xhat_s = pca.inverse_transform(Z)
    # Residual = what PCA couldn't explain from the training set's
    # dominant modes; large values indicate a bin behaving anomalously.
    residual = (X_test_s - Xhat_s).reshape(-1)

    return {"residual": residual}


def reduce_window(window, first_pca=None, first_scaler=None):
    """
    Reduce a window of processed chunks into a PCA-based anomaly summary.

    Treats the oldest chunks in the window as the training set and the most
    recent chunk as the test sample. Stacks the per-chunk ASDs, converts to
    log space, and hands off to PCA. The frequency axis (bins) is carried
    through into the result so downstream saving/plotting can align the
    residual to real frequencies.

    Residuals are reported directly in log10(ASD) units (dex) rather than
    as standardized z-scores: PCA is fit on mean-centered (not variance-
    scaled) log-ASD, so the reconstruction residual itself is already a
    fractional-amplitude-deviation measure, comparable across bins without
    a separate per-bin variance normalization.

    Parameters:
    - window: sequence of chunk dicts, each with keys "bins" and "asd"
              (as produced by process_chunk). Length must be >= MIN_TRAIN_ROWS.

    Returns:
    - dict: containing "residual", "bins" (1D frequency axis), "asd", and
      (if a long-term model is supplied) "long_term_residual". All values
      are numpy arrays, suitable for np.savez(**result).

    Raises:
    - ValueError: if the window has fewer than MIN_TRAIN_ROWS entries.
    """
    if len(window) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Window too small for PCA: got {len(window)}, "
            f"need >= {MIN_TRAIN_ROWS}"
        )

    print(f"Reducing window...")

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
    pca, X_test_s, scaler = pca_fit(X_train, X_test)
    short = pca_reconstruct(pca, X_test_s)

    reduced = {
        "residual": short["residual"],
        "bins": bins,
        "asd": X_test,
    }

    # If a long-term reference model has been established, also score the
    # newest chunk against it (using its own frozen centering), so drift
    # relative to the start of the run can be tracked alongside short-term
    # window-to-window anomalies.
    if first_pca is not None and first_scaler is not None:
        X_test_s_long = first_scaler.transform(X_test.reshape(1, -1))
        long_term = pca_reconstruct(first_pca, X_test_s_long)
        reduced["long_term_residual"] = long_term["residual"]

    return reduced, pca, scaler