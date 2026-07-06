import numpy as np
from sklearn.decomposition import PCA
from utils import get_files, filter_files,  stitch_files
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
import optuna

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


def evaluate_params(X_train_clean, X_val, y_val, params):
    # 1) fit PCA on clean train
    pca = PCA(n_components=params["n_components"], svd_solver="full", random_state=0)
    pca.fit(X_train_clean)

    # 2) score/flag
    flags, idx, residual, scores = pca_recon_flags_binlocal(
        X_val,
        pca,
        thr_global=params["thr_global"],
        thr_local=params["thr_local"],
        mode=params["mode"],
        topk=params.get("topk", 5),
        band_idx=params.get("band_idx", None),
        alpha=params.get("alpha", 0.5),
        thr_combined=params.get("thr_combined", None),
    )

    # 3) metrics
    f1 = f1_score(y_val, flags, zero_division=0)
    p = precision_score(y_val, flags, zero_division=0)
    r = recall_score(y_val, flags, zero_division=0)

    return {"f1": f1, "precision": p, "recall": r, "n_flagged": int(flags.sum())}, pca


def run_detector(X_train, X_val, params):
    pca = PCA(
        n_components=params["n_components"],
        svd_solver="full",
        random_state=0
    )
    pca.fit(X_train)

    flags, _, _, _ = pca_recon_flags_binlocal(
        X_val, pca,
        thr_global=params["thr_global"],
        thr_local=params["thr_local"],
        mode=params["mode"],
        topk=params.get("topk", 5),
        band_idx=params.get("band_idx", None),
        alpha=params.get("alpha", 0.5),
        thr_combined=params.get("thr_combined", None),
    )
    return flags

def objective(trial, X_train_clean, X_val, y_val, band_idx):

    mode = trial.suggest_categorical("mode", ["max_z", "topk_z_mean", "band_max_z"])

    params = {
        "n_components": trial.suggest_int("n_components", 5, 60),
        "thr_global": trial.suggest_float("thr_global", 1e-5, 5e-3, log=True),
        "thr_local": trial.suggest_float("thr_local", 2.5, 8.0),
        "mode": mode,
        "topk": trial.suggest_int("topk", 3, 15),
        "alpha": trial.suggest_float("alpha", 0.0, 1.0),
        "thr_combined": trial.suggest_float("thr_combined", 1.0, 8.0)}

    # only define band_idx when needed
    if mode == "band_max_z":
        lo, hi = band_idx  # e.g. (k0, k1) allowed bin range
        start = trial.suggest_int("band_start", lo, hi - 2)
        stop  = trial.suggest_int("band_stop", start + 1, hi)
        params["band_idx"] = (start, stop)
    else:
        params["band_idx"] = None

    # Fit PCA on CLEAN train only
    pca = PCA(n_components=params["n_components"], svd_solver="full", random_state=0)
    pca.fit(X_train_clean)

    # Evaluate on labeled mixed validation set
    flags, _, _, scores = pca_recon_flags_binlocal(
        X_val, pca,
        thr_global=params["thr_global"],
        thr_local=params["thr_local"],
        mode=params["mode"],
        topk=params["topk"],
        band_idx=params["band_idx"],
        alpha=params["alpha"],
        thr_combined=params["thr_combined"],
    )

    ap = average_precision_score(y_val, scores["combined_score"])
    return ap



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
target = 'I4_D20250102_T225835'

# Load continuous data from preprocessed files following naming format from preprocess.py
filenames = filter_files(get_files(path),target)
print(f"Found {len(filenames)} files")

# Stitch together continuous dataset
containers = stitch_files(path, filenames, 'freqs_list','asd_list')
freqs_total = containers['freqs_list']
asd_total = containers['asd_list']

X_train_clean = np.log10(asd_total + 1e-12).astype(np.float32)
print(f"Found {len(X_train_clean)} chunks")

k0, k1 = 800,1200


# # -------------------
# # MANUAL PARAM SPACE 
# # -------------------
# param_list = []
# for n_components in [5, 10, 20, 40]:
#     for thr_global in [5e-4, 1e-3, 1.5e-3, 2e-3]:
#         for mode in ["max_z", "topk_z_mean", "band_max_z"]:
#             for thr_local in [3.5, 4.0, 5.0, 6.0]:
#                 cfg = {
#                     "n_components": n_components,
#                     "thr_global": thr_global,
#                     "mode": mode,
#                     "thr_local": thr_local,
#                     "alpha": 0.4,
#                     "thr_combined": None,   # start simple
#                 }
#                 if mode == "topk_z_mean":
#                     cfg["topk"] = 5
#                 if mode == "band_max_z":
#                     cfg["band_idx"] = (k0, k1)  # <-- set your kHz bin range
#                 param_list.append(cfg)

# # -----------------------------
# # RUN MANUAL SEARCH
# # -----------------------------
# results = []
# best = None
# best_cfg = None
# best_pca = None

# for i, cfg in enumerate(param_list, 1):
#     metrics, pca_model = evaluate_params(X_train_clean, X_val, y_val, cfg)
#     row = {**cfg, **metrics}
#     results.append(row)

#     # choose objective (F1 here)
#     if (best is None) or (metrics["f1"] > best["f1"]):
#         best = metrics
#         best_cfg = cfg
#         best_pca = pca_model

#     print(f"[{i:03d}/{len(param_list)}] cfg={cfg} -> {metrics}")

# print("\nBest config:")
# print(best_cfg)
# print("Best metrics:")
# print(best)




# Optuna tuning
# ---- run search ----
# X, y are labeled development data (not final test)
# band_idx = (k0, k1) for your kHz bins
study = optuna.create_study(direction="maximize")
study.optimize(lambda t: objective(t, X_train_clean, X_val, y_val, band_idx=(k0, k1)), n_trials=50)

print("Best AP:", study.best_value)
print("Best params:", study.best_params)