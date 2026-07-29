import numpy as np
from analysis import unpack

# To add new evaluation metrics to either the summary statistics (used in live figures showing noise per PSD) or the per bin statistics 
# (used in live figures showing noise per frequency bin), simply add the calculation in the corresponding section (e.g., rss, bin_energy, ...)
# and add a corresponding key to the metrics dictionary. 


def get_metrics(reduced_data, mode):

    if mode not in ["sum_stat", "per_bin_stat"]:
        raise ValueError(f"{mode} is not a valid mode. Must be sum_stat or per_bin_stat.")

    residual, mean, scale = unpack(reduced_data, ["residual", "mean", "scale"])
    
    # -----------------------------------------
    # --- CONFIGURE HERE TO ADD NEW METRICS ---
    # ------- ADD KEYS TO DICT AS WELL --------
    # -----------------------------------------

    if mode == "sum_stat":
    # ----------- SUMMARY STATISTICS ----------
        rss                 = float(np.sum(residual**2))
        mean_scale          = float(np.mean(scale))
        mean_level_centered = float(np.mean(mean))     

        metrics = {
            'rss': rss,
            'mean_scale': mean_scale,
            'mean_level_centered': mean_level_centered
        }

        return metrics                                      
    # -----------------------------------------



    if mode == "per_bin_stat":
    # ---------- PER BIN STATISTICS -----------
        bin_energy = np.sum(residual**2, axis=0)

        metrics = {
            'bin_energy': bin_energy
        }

        return metrics
    # -----------------------------------------