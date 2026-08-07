import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from analysis.calculations import get_metrics
from analysis.utils import unpack


def get_worst_bins(residual_or_reduced_data, n_top=3, residual_key="residual"):
    """
    Return the n_top bin indices whose residuals attain the largest
    single abs(residual) value across all chunks in the input history.
    """
    if isinstance(residual_or_reduced_data, dict):
        residual = residual_or_reduced_data[residual_key]
    else:
        residual = residual_or_reduced_data

    residual = np.abs(np.asarray(residual))

    if residual.ndim == 1:
        values = np.nan_to_num(residual, nan=-np.inf)
        return np.argsort(values)[::-1][:n_top]

    elif residual.ndim == 2:
        max_per_bin = np.nanmax(residual, axis=0)
        max_per_bin = np.nan_to_num(max_per_bin, nan=-np.inf)
        return np.argsort(max_per_bin)[::-1][:n_top]

    else:
        raise ValueError(f"residual must be 1D or 2D, got shape {residual.shape}")


def live_figure_sum_stats(reduced_data, timestamp,
                 state={"fig": None, "axes": None, "start_time": None,
                        "x": [], "series": {}}):
    """
    Update a live, persistent plot of summary metrics from one reduced window.

    Called once per full window as new chunks arrive. Extracts whatever scalar
    metrics get_metrics(reduced_data) returns, appends each to its own running
    time series, and redraws one subplot per metric. The same Figure is reused
    and updated across calls. The set of metrics (keys of eval_metrics) is
    expected to stay the same across the life of the process, since the number
    of subplots is fixed on the first call.

    Parameters:
    - reduced_data: dict from reduce_window, passed to get_metrics.
    - timestamp: datetime of the chunk under test; used as the x-axis value.
    - state: persistent accumulator, implemented as a mutable default argument.
      NOTE: ties all calls in a process to one shared plot; fine for a single
      stream, not for driving two independent streams concurrently.

    Returns:
    - matplotlib.figure.Figure: the live figure, so the caller can save it.
    """
    eval_metrics = get_metrics(reduced_data, "sum_stat")

    if state["start_time"] is None:
        state["start_time"] = timestamp
    state["x"].append(timestamp)

    for key, value in eval_metrics.items():
        state["series"].setdefault(key, []).append(value)

    if state["fig"] is None:
        plt.ion()
        n = len(eval_metrics)
        fig, axes = plt.subplots(
            n, 1, figsize=(10, 2.2 * n), sharex=True, constrained_layout=True)
        axes = np.atleast_1d(axes)
        state["fig"] = fig
        state["axes"] = dict(zip(eval_metrics.keys(), axes))
        state["fig"].show()

    x = state["x"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (key, y) in enumerate(state["series"].items()):
        ax = state["axes"][key]
        ax.clear()
        ax.plot(x, y, color=colors[i % len(colors)], lw=1)
        ax.set_ylabel(key.replace("_", " "), fontsize=9)
        ax.grid(alpha=0.3)

    keys = list(state["axes"].keys())
    last_ax = state["axes"][keys[-1]]
    last_ax.ticklabel_format(useOffset=False, axis="y")
    last_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    last_ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    last_ax.set_xlabel("Time (HH:MM)")

    state["axes"][keys[0]].set_title(
        f"PCA sliding-window summary metrics  "
        f"(start {state['start_time']:%Y-%m-%d %H:%M:%S})")
    state["fig"].autofmt_xdate()

    state["fig"].canvas.draw_idle()
    state["fig"].canvas.flush_events()
    plt.pause(0.001)

    return state["fig"]


def _select_bins_above_threshold(residuals_hist_2d, threshold, bin_mask=None, max_bins=5):
    """
    Select up to max_bins bin indices (restricted to bin_mask, if given)
    whose peak residual across the stored history has surpassed
    `threshold` at least once. `residuals_hist_2d` is assumed to already
    be abs-valued. Returned indices are sorted by peak residual,
    descending (both for legend ordering and so truncation to max_bins
    keeps the most severe offenders).

    Parameters:
    - bin_mask: boolean array of length n_bins restricting the candidate
      pool (e.g. to a frequency band). If None, all bins are eligible.
    - max_bins: hard cap on how many bins are returned, regardless of
      how many exceed threshold.
    """
    n_bins = residuals_hist_2d.shape[1]
    if bin_mask is None:
        bin_mask = np.ones(n_bins, dtype=bool)

    max_per_bin = np.nanmax(residuals_hist_2d, axis=0)
    max_per_bin = np.nan_to_num(max_per_bin, nan=-np.inf)

    candidate_mask = (max_per_bin > threshold) & bin_mask
    if not np.any(candidate_mask):
        return np.array([], dtype=int)

    candidate_indices = np.where(candidate_mask)[0]
    candidate_values = max_per_bin[candidate_indices]
    order = np.argsort(candidate_values)[::-1]
    return candidate_indices[order][:max_bins]


def _autoscale_from_bins(ax, residuals_all_times, bins):
    """Autoscale y based on the currently plotted bin histories; ignore NaNs."""
    if bins is None or len(bins) == 0:
        return
    vmin = np.inf
    vmax = -np.inf
    for b in bins:
        y = residuals_all_times[:, b]
        y = y[np.isfinite(y)]
        if y.size == 0:
            continue
        vmin = min(vmin, float(y.min()))
        vmax = max(vmax, float(y.max()))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return

    if vmin == vmax:
        pad = max(abs(vmin) * 0.1, 1e-12)
        ax.set_ylim(vmin - pad, vmax + pad)
    else:
        pad = (vmax - vmin) * 0.1
        ax.set_ylim(vmin - pad, vmax + pad)


def _color_for_bin(b, n_bins, cmap_name="gist_rainbow"):
    """
    Deterministic, unique-ish color for a given bin index, based on its
    position within the full range of bins (0..n_bins-1). Using a
    continuous colormap instead of a fixed discrete cycle avoids color
    collisions when the number of flagged bins exceeds the cycle length.
    """
    cmap = plt.colormaps.get_cmap(cmap_name)
    denom = max(n_bins - 1, 1)
    return cmap(b / denom)


def _plot_residual_panel(ax, x, hist_2d, top_bins, threshold, bins_axis,
                          term_label, band_label, length, active=True):
    """
    Draw one (term x band) panel of the worst-bins figure: a residual
    history line per flagged bin, capped at len(top_bins) <= max_bins.

    Parameters:
    - active: False means "no data available yet" (e.g. long-term before
      the first model is frozen) rather than "data available but nothing
      crossed threshold" — these get different titles.
    """
    ax.clear()
    if active and len(top_bins) > 0:
        for b in top_bins:
            freq = bins_axis[b]
            ax.plot(
                x, hist_2d[:, b],
                lw=2, color=_color_for_bin(b, length),
                label=f"{freq:.3g} Hz"
            )
        ax.legend(fontsize=8, ncol=max(1, len(top_bins)), loc="best")
        title = f"{len(top_bins)} bin(s) exceeded threshold"
    elif active:
        title = "No bins have exceeded threshold"
    else:
        title = "Waiting for first long-term model..."

    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    if active:
        ax.axhline(threshold, color="gray", lw=1, ls="--", alpha=0.7)
        ax.set_title(f"{term_label}, {band_label} — {title} "
                     f"(threshold={threshold:g})", fontsize=9)
        _autoscale_from_bins(ax, hist_2d, top_bins)
    else:
        ax.set_title(f"{term_label}, {band_label} — {title}", fontsize=9)

    ax.set_ylabel(f"Abs residual\n({term_label})", fontsize=9)


def live_figure_worst_bins(
    reduced_data, timestamp, threshold_low, threshold_high,
    freq_cutoff=1000.0, max_bins=5,
    state=None
):
    """
    Live 2x2 grid of residual history: rows are short-term / long-term,
    columns are low-frequency (< freq_cutoff) / high-frequency
    (>= freq_cutoff). Residuals are always treated as abs(residual).

    Each column has its own threshold, since "interesting" residual
    magnitudes can differ substantially between low- and high-frequency
    bins. Each panel plots at most `max_bins` bins — the ones with the
    largest peak residual in that band's history — even if more bins
    have crossed threshold.

    Bin selection is computed from the full residual history so that
    live_figure_psd() can mark the exact same bins (it simply reads the
    selection back out of the shared state).

    Parameters:
    - threshold_low: float. Threshold (in residual units) for bins below
      freq_cutoff.
    - threshold_high: float. Threshold for bins at/above freq_cutoff.
    - freq_cutoff: float. Frequency (Hz) separating the low/high bands.
    - max_bins: int. Max number of bins plotted/legended per panel.
    """

    if state is None:
        raise ValueError("state must be provided (shared with live_figure_psd).")
    if threshold_low is None or threshold_high is None:
        raise ValueError("threshold_low and threshold_high must be provided.")

    # ---- Extract residuals + bin frequencies (always abs) ----
    residual_short, bins_axis = unpack(reduced_data, ['residual', 'bins'])
    residual_short = np.abs(residual_short)

    if "long_term_residual" in reduced_data:
        residual_long = np.abs(reduced_data["long_term_residual"])
        has_long = True
    else:
        residual_long = np.full_like(residual_short, np.nan, dtype=float)
        has_long = False

    # ---- Persistent state init ----
    if state.get("start_time", None) is None:
        state["start_time"] = timestamp

    state["x"].append(timestamp)
    state["residual_history_short"].append(residual_short)
    state["residual_history_long"].append(residual_long)

    x = state["x"]
    residuals_short = np.asarray(state["residual_history_short"])  # (T, n_bins)
    residuals_long = np.asarray(state["residual_history_long"])    # (T, n_bins)

    low_mask = bins_axis < freq_cutoff
    high_mask = ~low_mask

    # Compute bin selections from history (this is what PSD must match)
    state["top_bins_short_low"] = _select_bins_above_threshold(
        residuals_short, threshold_low, bin_mask=low_mask, max_bins=max_bins)
    state["top_bins_short_high"] = _select_bins_above_threshold(
        residuals_short, threshold_high, bin_mask=high_mask, max_bins=max_bins)

    if has_long:
        state["top_bins_long_low"] = _select_bins_above_threshold(
            residuals_long, threshold_low, bin_mask=low_mask, max_bins=max_bins)
        state["top_bins_long_high"] = _select_bins_above_threshold(
            residuals_long, threshold_high, bin_mask=high_mask, max_bins=max_bins)
    else:
        state["top_bins_long_low"] = np.array([], dtype=int)
        state["top_bins_long_high"] = np.array([], dtype=int)

    # ---- Create figure once (2x2 grid) ----
    if state["worst_fig"] is None:
        plt.ion()
        fig, axarr = plt.subplots(
            2, 2, figsize=(16, 9), sharex=True, constrained_layout=True
        )
        state["worst_fig"] = fig
        state["worst_axes"] = {
            "short_low": axarr[0, 0],
            "short_high": axarr[0, 1],
            "long_low": axarr[1, 0],
            "long_high": axarr[1, 1],
        }
        state["worst_fig"].show()

    axes = state["worst_axes"]
    low_label = f"< {freq_cutoff:g} Hz"
    high_label = f"\u2265 {freq_cutoff:g} Hz"
    
    _plot_residual_panel(
        axes["short_low"], x, residuals_short, state["top_bins_short_low"],
        threshold_low, bins_axis, "short-term", low_label, length=len(np.where(low_mask == True)[0]), active=True
    )
    _plot_residual_panel(
        axes["short_high"], x, residuals_short, state["top_bins_short_high"],
        threshold_high, bins_axis, "short-term", high_label, length=len(np.where(high_mask == True)[0]), active=True
    )
    _plot_residual_panel(
        axes["long_low"], x, residuals_long, state["top_bins_long_low"],
        threshold_low, bins_axis, "long-term", low_label, length=len(np.where(low_mask == True)[0]), active=has_long
    )
    _plot_residual_panel(
        axes["long_high"], x, residuals_long, state["top_bins_long_high"],
        threshold_high, bins_axis, "long-term", high_label, length=len(np.where(high_mask == True)[0]), active=has_long
    )

    # ---- Shared x-axis formatting (bottom row) ----
    for key in ("long_low", "long_high"):
        axes[key].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axes[key].xaxis.set_major_locator(mdates.AutoDateLocator())
        axes[key].set_xlabel("Time (HH:MM)")

    state["worst_fig"].suptitle(
        f"PCA sliding-window residual anomalies "
        f"(start {state['start_time']:%Y-%m-%d %H:%M:%S})"
    )

    if not state.get("did_autofmt_xdate", False):
        state["worst_fig"].autofmt_xdate()
        state["did_autofmt_xdate"] = True
    state["worst_fig"].canvas.draw_idle()
    state["worst_fig"].canvas.flush_events()
    plt.pause(0.001)

    return state["worst_fig"]


def live_figure_psd(
    reduced_data, timestamp,
    state=None
):
    """
    PSD plot with marker lines.

    Uses the exact bin indices selected by live_figure_worst_bins (stored
    in state), across all four bands (short/long x low/high). Since
    live_figure_worst_bins only keeps bins whose residual history has
    crossed its band's threshold (capped at max_bins per band), the
    markers drawn here automatically reflect the same set — no separate
    threshold or frequency-band logic is needed in this function.
    """

    if state is None:
        raise ValueError("state must be provided (shared with live_figure_worst_bins).")

    bins, asd = unpack(reduced_data, ["bins", "asd"])

    # Read the *already computed* bin selections, merging low+high bands.
    top_bins_short = np.concatenate([
        np.asarray(state.get("top_bins_short_low", []), dtype=int),
        np.asarray(state.get("top_bins_short_high", []), dtype=int),
    ])
    top_bins_long = np.concatenate([
        np.asarray(state.get("top_bins_long_low", []), dtype=int),
        np.asarray(state.get("top_bins_long_high", []), dtype=int),
    ])

    if state["psd_fig"] is None:
        plt.ion()
        state["psd_fig"], state["psd_ax"] = plt.subplots(1, 1, figsize=(7, 7))
        state["psd_fig"].show()

    ax = state["psd_ax"]
    ax.clear()

    # IMPORTANT: your reduce_window stores "asd" as log10(ASD)
    ax.loglog(bins, 10 ** asd, color='green')

    short_color = "red"
    long_color = "blue"

    for i, b in enumerate(top_bins_short):
        ax.axvline(
            bins[b],
            color=short_color,
            alpha=0.6,
            lw=2,
            ls="-",
            ymin=0.0, ymax=0.1,   # bottom fifth of the axis
            label="Short-term flagged bins" if i == 0 else None,
            zorder=2,
        )

    for i, b in enumerate(top_bins_long):
        ax.axvline(
            bins[b],
            color=long_color,
            alpha=0.6,
            lw=2,
            ls="-",
            ymin=0.9, ymax=1.0,   # top fifth of the axis
            label="Long-term flagged bins" if i == 0 else None,
            zorder=3,
        )

    ax.set_title(f"PSD ({timestamp:%H:%M:%S})")
    ax.set_xlabel("Frequency [Hz]", fontsize=14)
    ax.set_ylabel(r"[ADC/$\sqrt{\mathrm{Hz}}$]", fontsize=14)
    ax.set_xlim(0.01, 10**5)
    ax.set_ylim(10**(-5), 100)
    ax.grid(True, ls='--')
    ax.legend(fontsize=12)

    state["psd_fig"].canvas.draw_idle()
    state["psd_fig"].canvas.flush_events()
    plt.pause(0.001)

    return state["psd_fig"]


def analyze(reduced_data, timestamp, threshold_low, threshold_high,
            freq_cutoff=1000.0, max_bins=5):
    """
    Parameters:
    - threshold_low: residual threshold for bins below freq_cutoff.
    - threshold_high: residual threshold for bins at/above freq_cutoff.
    - freq_cutoff: frequency (Hz) separating the low/high bands.
    - max_bins: max number of bins plotted per band, per term.
    """
    # Shared state object so both live figures use identical bin selection.
    # (Persistent across repeated calls to analyze)
    if not hasattr(analyze, "_shared_state"):
        analyze._shared_state = {
            "worst_fig": None,
            "worst_axes": None,

            "psd_fig": None,
            "psd_ax": None,

            "start_time": None,
            "x": [],
            "residual_history_short": [],
            "residual_history_long": [],

            "top_bins_short_low": np.array([], dtype=int),
            "top_bins_short_high": np.array([], dtype=int),
            "top_bins_long_low": np.array([], dtype=int),
            "top_bins_long_high": np.array([], dtype=int),

            "did_autofmt_xdate": False,
        }

    shared_state = analyze._shared_state

    fig1 = live_figure_worst_bins(
        reduced_data, timestamp, threshold_low, threshold_high,
        freq_cutoff=freq_cutoff, max_bins=max_bins, state=shared_state
    )
    fig2 = live_figure_psd(reduced_data, timestamp, state=shared_state)

    return fig1, fig2