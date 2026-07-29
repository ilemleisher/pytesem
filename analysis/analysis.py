import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def analyze(reduced_data, timestamp,
              state={"fig": None, "axes": None, "start_time": None,
                     "x": [], "rss": [], "mean_scale": [], "mean_level": []}):
    """
    Update a live, persistent plot of summary metrics from one reduced window.

    Called once per full window as new chunks arrive. Each call extracts three
    scalar summary metrics from the current window's reduced data, appends them
    to running time series, and redraws a stacked set of live plots. The same
    Figure is reused and updated across calls.

    Metrics plotted (all vs. time):
    - RSS: total squared PCA reconstruction residual of the test chunk. This is
      the sharp anomaly indicator — it spikes when the newest chunk is spectrally
      unlike the training window.
    - Mean per-bin variability: mean of the training scaler's per-bin std. Changes
      only slowly as the window slides, so this trace is smooth by construction.
    - Mean per-bin level: mean of the training scaler's per-bin mean. Also slow-moving.

    Parameters:
    - reduced_data: dict from reduce_window, containing at least "residual",
      "mean", and "scale" (arrays). "bins" and "pca_components" are present but
      not used by this function.
    - timestamp: datetime of the chunk under test; used as the x-axis value.
    - state: persistent accumulator. Implemented as a mutable default argument so
      it survives across calls without the caller having to thread it through.
      NOTE: this ties all calls in a process to a single shared plot/series, so
      analyze() cannot drive two independent streams at once. Fine for the
      single-stream pipeline in run.py.

    Returns:
    - matplotlib.figure.Figure: the live figure, so the caller can save it.
    """
    print("Analyzing data...")
    # --- Load the fields we need from the reduced data ---
    # NOTE: key is "residual" (singular), matching reduce_window's output.
    residual = reduced_data["residual"]
    mean     = reduced_data["mean"]
    scale    = reduced_data["scale"]

    # --- Metric calculations ---
    # RSS collapses the (1, n_bins) residual to a single scalar.
    rss                 = float(np.sum(residual**2))
    # scale and mean are per-bin arrays (length n_bins); summarize each by its
    # mean across bins to get one scalar per window.
    mean_scale          = float(np.mean(scale))
    mean_level_centered = float(np.mean(mean))

    # --- Accumulate into the running time series ---
    if state["start_time"] is None:
        state["start_time"] = timestamp          # remember first timestamp for the title
    state["x"].append(timestamp)
    state["rss"].append(rss)
    state["mean_scale"].append(mean_scale)
    state["mean_level"].append(mean_level_centered)

    # Pair each series with its axis label and color, one entry per subplot.
    metrics = [
        (state["rss"],        "Reconstruction\nresidual (RSS)",            "tab:blue"),
        (state["mean_scale"], "Mean per-bin\nvariability (std)",           "tab:purple"),
        (state["mean_level"], "Mean per-bin level\n(deviation from mean)", "tab:orange"),
    ]

    # --- Create the figure once, on the first call ---
    if state["fig"] is None:
        plt.ion()                                 # interactive mode: non-blocking draws
        state["fig"], state["axes"] = plt.subplots(
            len(metrics), 1, figsize=(10, 2.2 * len(metrics)),
            sharex=True, constrained_layout=True)
        state["fig"].show()

    axes = state["axes"]
    x = state["x"]

    # --- Redraw every subplot from scratch each call ---
    # (clear + replot is simple and robust for modest point counts.)
    for ax, (y, label, color) in zip(axes, metrics):
        ax.clear()
        ax.plot(x, y, color=color, lw=1)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.3)

    # --- Format the shared x-axis (time) on the bottom subplot ---
    axes[-1].ticklabel_format(useOffset=False, axis='y')   # no offset notation on y
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].set_xlabel("Time (HH:MM)")
    axes[0].set_title(
        f"PCA sliding-window summary metrics  "
        f"(start {state['start_time']:%Y-%m-%d %H:%M:%S})")
    state["fig"].autofmt_xdate()                  # rotate/space date labels nicely

    # --- Push the update to the live canvas ---
    state["fig"].canvas.draw_idle()
    state["fig"].canvas.flush_events()
    plt.pause(0.001)                              # yield briefly so the GUI repaints
    print("Data analyzed.")
    return state["fig"]
