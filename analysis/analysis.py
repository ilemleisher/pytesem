import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from calculations import get_metrics
from operator import itemgetter
from datetime import timedelta


def unpack(reduced_data, keys):
    try:
        return itemgetter(*keys)(reduced_data)
    except KeyError as e:
        raise KeyError(f"Key {e} not found in reduced_data.") from e
    

def get_worst_bins(reduced_data, n_top=5):

    bin_energy = get_metrics(reduced_data, "per_bin_stat")
    top_bins = np.argsort(bin_energy)[::-1][:n_top]

    return top_bins


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
                          state={"fig": None, "ax": None, "start_time": None,
                                 "x": [], "residual_history": []}):
    """
    Update a live, persistent plot of per-bin residuals for the worst bins.

    Called once per full window as new chunks arrive. Tracks the full history
    of per-bin residuals, then on each call re-selects the n_top bins with the
    highest current bin energy and plots their residual trace over time. The
    set of "worst" bins can change from call to call as new anomalies appear;
    this always reflects the most recent ranking, redrawing the full history
    for whichever bins currently qualify.

    Parameters:
    - reduced_data: dict from reduce_window, containing at least "residual"
      (array of per-bin standardized residuals for the current chunk).
    - timestamp: datetime of the chunk under test; used as the x-axis value.
    - n_top: number of worst bins to plot.
    - state: persistent accumulator, implemented as a mutable default argument.
      NOTE: ties all calls in a process to one shared plot; fine for a single
      stream, not for driving two independent streams concurrently.

    Returns:
    - matplotlib.figure.Figure: the live figure, so the caller can save it.
    """
    top_bins = get_worst_bins(reduced_data)

    residual = unpack(reduced_data, ["residual"])

    if state["start_time"] is None:
        state["start_time"] = timestamp
    state["x"].append(timestamp)
    state["residual_history"].append(residual)

    residuals = np.array(state["residual_history"])  # shape (T, n_bins)
    x = state["x"]

    if state["fig"] is None:
        plt.ion()
        state["fig"], state["ax"] = plt.subplots(
            1, 1, figsize=(10, 4), constrained_layout=True)
        state["fig"].show()

    ax = state["ax"]
    ax.clear()

    for b in top_bins:
        ax.plot(x, residuals[:, b], lw=3, label=f"Bin {b}")

    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_ylabel("Standardized residual")
    ax.set_xlabel("Time (HH:MM)")
    ax.set_title(
        f"Top {n_top} most anomalous frequency bins over time  "
        f"(start {state['start_time']:%Y-%m-%d %H:%M:%S})")
    ax.legend(fontsize=8, ncol=n_top)
    ax.grid(alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    state["fig"].autofmt_xdate()

    state["fig"].canvas.draw_idle()
    state["fig"].canvas.flush_events()
    plt.pause(0.001)

    return state["fig"]


def live_figure_psd(reduced_data, timestamp, n_top=5,
                     state={"fig": None, "ax": None}):
    """
    Update a live, persistent plot of the PSD for the current chunk.

    Called once per chunk. Unlike the other live_figure* functions, this does
    not accumulate a time series — each call redraws the full PSD for the
    current chunk only, with the n_top most anomalous bins (by residual energy)
    marked as vertical lines. The same Figure is reused and updated across
    calls.

    Parameters:
    - reduced_data: dict from reduce_window, containing at least "bins"
      (frequency values), "asd" (amplitude spectral density for the current
      chunk), and "residual" (per-bin standardized residual, used to rank
      bins by anomalousness).
    - timestamp: datetime marking the start of the chunk under test.
    - chunk_seconds: duration of the chunk, used to compute the end time
      shown in the title.
    - n_top: number of worst bins to highlight.
    - state: persistent accumulator, implemented as a mutable default
      argument. NOTE: ties all calls in a process to one shared plot; fine
      for a single stream, not for driving two independent streams
      concurrently.

    Returns:
    - matplotlib.figure.Figure: the live figure, so the caller can save it.
    """
    bins, asd = unpack(reduced_data, ["bins", "asd"])

    top_bins = get_worst_bins(reduced_data)

    if state["fig"] is None:
        plt.ion()
        state["fig"], state["ax"] = plt.subplots(1, 1, figsize=(7, 7))
        state["fig"].show()

    ax = state["ax"]
    ax.clear()

    ax.loglog(bins, asd)

    for i, b in enumerate(top_bins):
        ax.axvline(bins[b], color="red", alpha=0.5, lw=2,
                   label="Problem Bins" if i == 0 else None)

    ax.set_title(f"PSD ({timestamp:%H:%M:%S})")
    ax.set_xlabel("Frequency [Hz]", fontsize=14)
    ax.set_ylabel(r"[ADC/$\sqrt{\mathrm{Hz}}$]", fontsize=14)
    ax.set_xlim(0.01, 10**5)
    ax.set_ylim(10**(-5), 100)
    ax.grid(True, ls='--')
    ax.legend(fontsize=14)

    state["fig"].canvas.draw_idle()
    state["fig"].canvas.flush_events()
    plt.pause(0.001)

    return state["fig"]