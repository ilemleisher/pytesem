import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from analysis.calculations import get_metrics
from analysis.utils import unpack

    
def get_worst_bins(reduced_data, n_top=5, residual_key="residual"):
    residual = reduced_data[residual_key]
    # "energy" in the standardized residual space
    energies = np.square(residual)
    top_bins = np.argsort(energies)[::-1][:n_top]
    return top_bins


def live_figure(reduced_data, timestamp,
                 state={"fig": None, "axes": None, "start_time": None,
                        "x": [], "series": {}, "residual_history": []}):
    """
    Update a single, persistent live figure combining summary stats, worst-bin
    residual traces, and the current chunk's PSD.

    Called once per full window. Left column: one subplot per summary metric
    from get_metrics, plus one subplot showing the per-bin residual history
    for the n_top currently-worst bins (all sharing a time x-axis). Right
    column: PSD of the current chunk, with the n_top worst bins marked.

    Parameters:
    - reduced_data: dict from reduce_window, containing at least "residual",
      "bins", and "asd", plus whatever get_metrics needs.
    - timestamp: datetime of the chunk under test.
    - n_top: number of worst bins to highlight/track.
    - state: persistent accumulator, implemented as a mutable default
      argument. Ties all calls in a process to one shared figure.

    Returns:
    - matplotlib.figure.Figure: the live figure, so the caller can save it.
    """

    residual, bins, asd = unpack(reduced_data, ["residual", "bins", "asd"])
    eval_metrics = get_metrics(reduced_data,"sum_stat")
    top_bins = get_worst_bins(reduced_data)

    if state["start_time"] is None:
        state["start_time"] = timestamp
    state["x"].append(timestamp)
    for key, value in eval_metrics.items():
        state["series"].setdefault(key, []).append(value)
    state["residual_history"].append(residual)

    residuals = np.array(state["residual_history"])
    x = state["x"]

    # --- Create the figure once ---
    if state["fig"] is None:
        plt.ion()
        n_left = len(eval_metrics) + 1  # + 1 for worst-bins panel
        fig = plt.figure(figsize=(14, 2.2 * n_left), constrained_layout=True)
        gs = fig.add_gridspec(n_left, 2, width_ratios=[1.3, 1])

        left_axes = [fig.add_subplot(gs[i, 0]) for i in range(n_left)]
        for ax in left_axes[1:]:
            ax.sharex(left_axes[0])

        psd_ax = fig.add_subplot(gs[:, 1])

        state["fig"] = fig
        state["axes"] = {
            "metrics": dict(zip(eval_metrics.keys(), left_axes[:-1])),
            "worst_bins": left_axes[-1],
            "psd": psd_ax,
        }
        state["fig"].show()

    axes = state["axes"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # --- Summary metric subplots ---
    for i, (key, y) in enumerate(state["series"].items()):
        ax = axes["metrics"][key]
        ax.clear()
        ax.plot(x, y, color=colors[i % len(colors)], lw=1)
        ax.set_ylabel(key.replace("_", " "), fontsize=9)
        ax.grid(alpha=0.3)

    # --- Worst-bins subplot ---
    ax = axes["worst_bins"]
    ax.clear()
    for b in top_bins:
        ax.plot(x, residuals[:, b], lw=2, label=f"Bin {b}")
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_ylabel("Standardized\nresidual", fontsize=9)
    ax.legend(fontsize=7, ncol=n_top)
    ax.grid(alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.set_xlabel("Time (HH:MM)")

    metric_keys = list(axes["metrics"].keys())
    axes["metrics"][metric_keys[0]].set_title(
        f"Summary metrics  (start {state['start_time']:%Y-%m-%d %H:%M:%S})")
    state["fig"].autofmt_xdate()

    # --- PSD subplot ---
    psd_ax = axes["psd"]
    psd_ax.clear()
    psd_ax.loglog(bins, asd)
    for i, b in enumerate(top_bins):
        psd_ax.axvline(bins[b], color="red", alpha=0.5, lw=2,
                        label="Problem Bins" if i == 0 else None)
    psd_ax.set_title(f"PSD (chunk starting {timestamp:%H:%M:%S})")
    psd_ax.set_xlabel("Frequency [Hz]", fontsize=12)
    psd_ax.set_ylabel(r"[ADC/$\sqrt{\mathrm{Hz}}$]", fontsize=12)
    psd_ax.set_xlim(0.01, 10**5)
    psd_ax.set_ylim(10**(-5), 100)
    psd_ax.grid(True, ls='--')
    psd_ax.legend(fontsize=10)

    state["fig"].canvas.draw_idle()
    state["fig"].canvas.flush_events()
    plt.pause(0.001)

    return state["fig"]


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

    # --- Create the figure once, sized to however many metrics we got ---
    if state["fig"] is None:
        plt.ion()
        n = len(eval_metrics)
        fig, axes = plt.subplots(
            n, 1, figsize=(10, 2.2 * n), sharex=True, constrained_layout=True)
        axes = np.atleast_1d(axes)  # so single-metric case still indexes cleanly
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


def live_figure_worst_bins(reduced_data, timestamp, n_top=5,
                          state={"fig": None, "axes": None, "start_time": None,
                                 "x": [],
                                 "residual_history_short": [],
                                 "residual_history_long": []}):
    '''
    Updates a live figure with two axes, showing the top n 
    most anomalous bins in both the long term residual history 
    and the short term residual history. These bins are consistent
    with the bins shown on the PSD in live_figure_psd(). 
    '''
    
    # ---- Extract residuals + bin frequencies ----
    residual_short, bins_axis = unpack(reduced_data, ['residual','bins'])

    if "long_term_residual" in reduced_data:
        residual_long = reduced_data["long_term_residual"]
        has_long = True
    else:
        residual_long = np.full_like(residual_short, np.nan, dtype=float)
        has_long = False

    # ---- Persistent state update ----
    if state["start_time"] is None:
        state["start_time"] = timestamp

    state["x"].append(timestamp)
    state["residual_history_short"].append(residual_short)
    state["residual_history_long"].append(residual_long)

    x = state["x"]
    residuals_short = np.array(state["residual_history_short"])
    residuals_long = np.array(state["residual_history_long"])

    # ---- Create figure once ----
    if state["fig"] is None:
        plt.ion()
        fig, (ax_short, ax_long) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True, constrained_layout=True
        )
        state["fig"] = fig
        state["axes"] = {"short": ax_short, "long": ax_long}
        state["fig"].show()

    axes = state["axes"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # ---- Select worst bins for this update ----
    top_bins_short = get_worst_bins(reduced_data, n_top=n_top, residual_key="residual")

    top_bins_long = []
    if has_long:
        top_bins_long = get_worst_bins(
            reduced_data, n_top=n_top, residual_key="long_term_residual"
        )

    def autoscale_from_bins(ax, residuals_all_times, bins):
        """Autoscale y based on the currently plotted bin histories; ignore NaNs."""
        if len(bins) == 0:
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
            pad = (vmax - vmin) * 0.1  # 10% padding
            ax.set_ylim(vmin - pad, vmax + pad)

    # ----------------------
    # SHORT-term subplot
    # ----------------------
    ax = axes["short"]
    ax.clear()

    for i, b in enumerate(top_bins_short):
        freq = bins_axis[b]
        ax.plot(
            x, residuals_short[:, b],
            lw=2, color=colors[i % len(colors)],
            label=f"Bin {b} ({freq:.3g} Hz)"
        )

    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_ylabel("Std residual\n(short-term)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=max(1, n_top), loc="best")
    ax.set_title(
        f"Top {n_top} anomalous frequency bins (short-term) "
        f"start {state['start_time']:%Y-%m-%d %H:%M:%S}"
    )

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
                lw=2, color=colors[i % len(colors)],
                label=f"Bin {b} ({freq:.3g} Hz)"
            )
        ax.legend(fontsize=8, ncol=max(1, n_top), loc="best")
        title = f"Top {n_top} anomalous frequency bins (long-term)"
    else:
        title = "Top anomaly bins (long-term) — waiting for first long-term model..."

    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_ylabel("Std residual\n(long-term)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_title(f"{title} start {state['start_time']:%Y-%m-%d %H:%M:%S}")

    autoscale_from_bins(ax, residuals_long, top_bins_long)

    # ---- Shared x-axis formatting ----
    axes["long"].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes["long"].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes["long"].set_xlabel("Time (HH:MM)")

    state["fig"].autofmt_xdate()
    state["fig"].canvas.draw_idle()
    state["fig"].canvas.flush_events()
    plt.pause(0.001)

    return state["fig"]


def live_figure_psd(reduced_data, timestamp, n_top=5,
                     state={"fig": None, "ax": None}):
    """
    Update a live, persistent plot of the PSD for the current chunk.

    Marks BOTH:
      - top-n short-term anomalous bins (red) using residual
      - top-n long-term anomalous bins (blue) using long_term_residual (if available)
    """

    bins, asd = unpack(reduced_data, ["bins", "asd"])

    # Short-term top bins
    top_bins_short = get_worst_bins(
        reduced_data, n_top=n_top, residual_key="residual"
    )

    # Long-term top bins (optional)
    has_long = "long_term_residual" in reduced_data
    top_bins_long = []
    if has_long:
        top_bins_long = get_worst_bins(
            reduced_data, n_top=n_top, residual_key="long_term_residual"
        )

    if state["fig"] is None:
        plt.ion()
        state["fig"], state["ax"] = plt.subplots(1, 1, figsize=(7, 7))
        state["fig"].show()

    ax = state["ax"]
    ax.clear()

    ax.loglog(bins, asd)

    short_color = "red"
    long_color = "blue"

    # Plot short-term marker lines
    for i, b in enumerate(top_bins_short):
        ax.axvline(
            bins[b],
            color=short_color,
            alpha=0.6,
            lw=2.5,
            label="Short-term top bins" if i == 0 else None
        )

    # Plot long-term marker lines (if available)
    if has_long:
        for i, b in enumerate(top_bins_long):
            ax.axvline(
                bins[b],
                color=long_color,
                alpha=0.6,
                lw=2.5,
                label="Long-term top bins" if i == 0 else None
            )

    ax.set_title(f"PSD ({timestamp:%H:%M:%S})")
    ax.set_xlabel("Frequency [Hz]", fontsize=14)
    ax.set_ylabel(r"[ADC/$\sqrt{\mathrm{Hz}}$]", fontsize=14)
    ax.set_xlim(0.01, 10**5)
    ax.set_ylim(10**(-5), 100)
    ax.grid(True, ls='--')
    ax.legend(fontsize=12)

    state["fig"].canvas.draw_idle()
    state["fig"].canvas.flush_events()
    plt.pause(0.001)

    return state["fig"]


def analyze(reduced_data, timestamp):

    fig = live_figure_worst_bins(reduced_data, timestamp) # Create fig for saving
    live_figure_psd(reduced_data, timestamp)
    
    return fig
