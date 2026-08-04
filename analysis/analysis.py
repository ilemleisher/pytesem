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


def _select_bins_above_threshold(residuals_hist_2d, threshold):
    """
    Select all bin indices whose peak residual across the stored history
    has surpassed `threshold` at least once. `residuals_hist_2d` is
    assumed to already be abs-valued. Returned indices are sorted by
    peak residual, descending (purely for consistent legend ordering —
    every qualifying bin is included, with no cap on count).
    """
    max_per_bin = np.nanmax(residuals_hist_2d, axis=0)
    max_per_bin = np.nan_to_num(max_per_bin, nan=-np.inf)

    candidate_mask = max_per_bin > threshold
    if not np.any(candidate_mask):
        return np.array([], dtype=int)

    candidate_indices = np.where(candidate_mask)[0]
    candidate_values = max_per_bin[candidate_indices]
    order = np.argsort(candidate_values)[::-1]
    return candidate_indices[order]


def live_figure_worst_bins(
    reduced_data, timestamp, threshold,
    state=None
):
    """
    Live 2-panel plot of residual history (short + long). Residuals are
    always treated as abs(residual).

    A bin's full residual history is plotted if and only if it has
    exceeded `threshold` at some point in the accumulated history.
    There is no cap on how many bins can qualify — every bin that has
    ever crossed the threshold gets its own line.

    Bin selection is computed from the full residual history so that
    live_figure_psd() can mark the exact same bins (it simply reads the
    selection back out of the shared state).

    Parameters:
    - threshold: float. Minimum residual a bin must reach at some point
      in its history to be selected and plotted.
    """

    if state is None:
        raise ValueError("state must be provided (shared with live_figure_psd).")
    if threshold is None:
        raise ValueError("threshold must be provided.")

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

    # Compute bin selections from history (this is what PSD must match)
    state["top_bins_short"] = _select_bins_above_threshold(residuals_short, threshold)

    if has_long:
        residuals_long = np.asarray(state["residual_history_long"])  # (T, n_bins)
        state["top_bins_long"] = _select_bins_above_threshold(residuals_long, threshold)
    else:
        residuals_long = None
        state["top_bins_long"] = np.array([], dtype=int)

    top_bins_short = state["top_bins_short"]
    top_bins_long = state["top_bins_long"]

    # ---- Create figure once ----
    if state["worst_fig"] is None:
        plt.ion()
        fig, (ax_short, ax_long) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True, constrained_layout=True
        )
        state["worst_fig"] = fig
        state["worst_axes"] = {"short": ax_short, "long": ax_long}
        state["worst_fig"].show()

    axes = state["worst_axes"]

    def autoscale_from_bins(ax, residuals_all_times, bins):
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

    threshold_note = f" (threshold={threshold:g})"

    # ----------------------
    # SHORT-term subplot
    # ----------------------
    ax = axes["short"]
    ax.clear()

    if len(top_bins_short) > 0:
        for i, b in enumerate(top_bins_short):
            freq = bins_axis[b]
            ax.plot(
                x, residuals_short[:, b],
                lw=2, color=_color_for_bin(b, len(bins_axis)),
                label=f"{freq:.3g} Hz"
            )
        ax.legend(fontsize=4, ncol=max(1, len(top_bins_short)), loc="best")
        title = f"{len(top_bins_short)} bins exceeded threshold (short-term)"
    else:
        title = f"No bins have exceeded threshold (short-term){threshold_note}"

    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.axhline(threshold, color="gray", lw=1, ls="--", alpha=0.7)
    ax.set_ylabel("Abs residual\n(short-term)", fontsize=10)
    ax.set_title(f"{title}{threshold_note if top_bins_short.size else ''} "
                 f"start {state['start_time']:%Y-%m-%d %H:%M:%S}")

    autoscale_from_bins(ax, residuals_short, top_bins_short)

    # ----------------------
    # LONG-term subplot
    # ----------------------
    ax = axes["long"]
    ax.clear()

    if has_long and len(top_bins_long) > 0:
        for i, b in enumerate(top_bins_long):
            freq = bins_axis[b]
            ax.plot(
                x, residuals_long[:, b],
                lw=2, color=_color_for_bin(b, len(bins_axis)),
                label=f"{freq:.3g} Hz"
            )
        ax.legend(fontsize=4, ncol=max(1, len(top_bins_long)), loc="best")
        title = f"{len(top_bins_long)} bins exceeded threshold (long-term)"
    elif has_long:
        title = f"No bins have exceeded threshold (long-term){threshold_note}"
    else:
        title = "Anomaly bins (long-term) — waiting for first long-term model..."

    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    if has_long:
        ax.axhline(threshold, color="gray", lw=1, ls="--", alpha=0.7)
    ax.set_ylabel("Abs residual\n(long-term)", fontsize=10)
    ax.set_title(f"{title} start {state['start_time']:%Y-%m-%d %H:%M:%S}")

    if has_long:
        autoscale_from_bins(ax, residuals_long, top_bins_long)

    # ---- Shared x-axis formatting ----
    axes["long"].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes["long"].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes["long"].set_xlabel("Time (HH:MM)")

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
    in state). Since live_figure_worst_bins only keeps bins whose residual
    history has crossed the configured threshold, the markers drawn here
    automatically reflect the same threshold-filtered bins — no separate
    threshold logic is needed in this function.
    """

    if state is None:
        raise ValueError("state must be provided (shared with live_figure_worst_bins).")

    bins, asd = unpack(reduced_data, ["bins", "asd"])

    # Read the *already computed* bin selections
    top_bins_short = state.get("top_bins_short", [])
    top_bins_long = state.get("top_bins_long", [])

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
            alpha=0.3,
            lw=2,
            label="Short-term flagged bins" if i == 0 else None
        )

    if top_bins_long is not None and len(top_bins_long) > 0:
        for i, b in enumerate(top_bins_long):
            ax.axvline(
                bins[b],
                color=long_color,
                alpha=0.6,
                lw=2.5,
                label="Long-term flagged bins" if i == 0 else None
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


def analyze(reduced_data, timestamp, threshold):
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

            "top_bins_short": np.array([], dtype=int),
            "top_bins_long": np.array([], dtype=int),

            "did_autofmt_xdate": False,
        }

    shared_state = analyze._shared_state

    fig1 = live_figure_worst_bins(
        reduced_data, timestamp, threshold, state=shared_state
    )
    fig2 = live_figure_psd(reduced_data, timestamp, state=shared_state)

    return fig1, fig2