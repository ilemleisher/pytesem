"""
Live plotting utilities for PCA residual anomalies and raw PSDs.

Maintains persistent state across repeated calls (via analyze's shared
state dict) to build up rolling time-history plots of the worst-offending
frequency bins (short-term and long-term residuals, split into low/high
frequency bands), plus a live PSD plot marking which bins are currently
flagged.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from operator import itemgetter


def unpack(reduced_data, keys):
    """
    Pull multiple named values out of the reduced_data dict at once.

    Parameters:
    - reduced_data: dict produced by reduce_window (or loaded from disk).
    - keys: sequence of key names to extract, in order.

    Returns:
    - tuple of values corresponding to `keys`, in the same order.

    Raises:
    - KeyError: if any key is missing from reduced_data.
    """
    try:
        return itemgetter(*keys)(reduced_data)
    except KeyError as e:
        raise KeyError(f"Key {e} not found in reduced_data.") from e


def _select_bins_above_threshold(residual_hist_2d, threshold, bin_mask=None, max_bins=5):
    """
    Select up to max_bins bin indices (restricted to bin_mask, if given)
    whose peak residual across the stored history has surpassed
    `threshold` at least once. `residual_hist_2d` is assumed to already
    be abs-valued. Returned indices are sorted by peak residual,
    descending (both for legend ordering and so truncation to max_bins
    keeps the most severe offenders).

    Parameters:
    - bin_mask: boolean array of length n_bins restricting the candidate
      pool (e.g. to a frequency band). If None, all bins are eligible.
    - max_bins: hard cap on how many bins are returned, regardless of
      how many exceed threshold.
    """
    n_bins = residual_hist_2d.shape[1]
    if bin_mask is None:
        bin_mask = np.ones(n_bins, dtype=bool)

    # Peak (over time) residual for every bin; NaNs (e.g. before a
    # long-term model exists) are treated as never exceeding threshold.
    max_per_bin = np.nanmax(residual_hist_2d, axis=0)
    max_per_bin = np.nan_to_num(max_per_bin, nan=-np.inf)

    candidate_mask = (max_per_bin > threshold) & bin_mask
    if not np.any(candidate_mask):
        return np.array([], dtype=int)

    # Rank surviving candidates by peak severity, descending, then truncate.
    candidate_indices = np.where(candidate_mask)[0]
    candidate_values = max_per_bin[candidate_indices]
    order = np.argsort(candidate_values)[::-1]
    return candidate_indices[order][:max_bins]


def _autoscale_from_bins(ax, zs_all_times, bins):
    """Autoscale y based on the currently plotted bin histories; ignore NaNs."""
    if bins is None or len(bins) == 0:
        return
    vmin = np.inf
    vmax = -np.inf
    # Scan only the flagged bins' histories to find the tightest y-range
    # that still shows all of their data.
    for b in bins:
        y = zs_all_times[:, b]
        y = y[np.isfinite(y)]
        if y.size == 0:
            continue
        vmin = min(vmin, float(y.min()))
        vmax = max(vmax, float(y.max()))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return

    # Pad the range a bit so lines aren't flush against the axes edges.
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
                          term_label, band_label, length, active=True,
                          band_width=2.0):
    """
    Draw one residual-vs-time panel (one term/band combination) onto `ax`.

    Clears and redraws the axes each call with the current set of flagged
    bins' time histories, a dashed line at `threshold`, and shaded
    caution/red zones above it. If `active` is False (e.g. no long-term
    model exists yet), only a placeholder title is drawn.

    Parameters:
    - ax: matplotlib Axes to draw into.
    - x: shared x-axis values (timestamps) for all bin histories.
    - hist_2d: (n_times, n_bins) residual history array for this term.
    - top_bins: bin indices to plot (already selected/ranked by severity).
    - threshold: dex threshold marking the start of the caution band.
    - bins_axis: frequency (Hz) for each bin index, used in legend labels.
    - term_label: "short-term" or "long-term", used in titles/labels.
    - band_label: frequency-band description (e.g. "< 1000 Hz"), used in titles.
    - length: number of bins eligible in this band, used for color scaling.
    - active: whether this panel currently has valid data to plot.
    - band_width: dex-width of the yellow caution band drawn just above
      threshold, spanning [threshold, threshold + band_width]. The red
      band above it spans from there to the top of the axes, regardless
      of the current y-limits (clipped visually, not a real data bound).
    """
    ax.clear()
    if active and len(top_bins) > 0:
        # One line per flagged bin, colored uniquely and labeled by frequency.
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

    # Zero reference line, always drawn.
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    if active:
        # Threshold line plus yellow (caution) and red (severe) shaded bands.
        ax.axhline(threshold, color="gray", lw=1, ls="--", alpha=0.7)
        ax.axhspan(threshold, threshold + band_width,
                   color="yellow", alpha=0.1, zorder=0)
        ax.axhspan(threshold + band_width, 1e6,
                   color="red", alpha=0.1, zorder=0)
        ax.set_title(f"{term_label}, {band_label} — {title} "
                     f"(threshold={threshold:g} dex)", fontsize=9)
        _autoscale_from_bins(ax, hist_2d, top_bins)  # sets explicit ylim,
                                                       # overriding any
                                                       # autoscale expansion
                                                       # from the tall patch
    else:
        ax.set_title(f"{term_label}, {band_label} — {title}", fontsize=9)

    ax.set_ylabel(f"|Δlog₁₀(ASD)|\n({term_label})", fontsize=9)

def live_figure_worst_bins(
    reduced_data, timestamp, threshold_low, threshold_high, band_width=2.0,
    freq_cutoff=1000.0, max_bins=5,
    state=None
):
    """
    Update (or create) the 2x2 grid of residual-history panels: short-term
    vs long-term residuals, each split into low-frequency and
    high-frequency bands relative to freq_cutoff.

    Appends the current timestamp/residuals to the persistent `state`,
    recomputes which bins are currently flagged in each of the four
    term/band combinations, and redraws each panel via _plot_residual_panel.
    The flagged-bin selections are also stashed back into `state` so
    live_figure_psd can mark the same bins on the PSD plot.

    Residuals are log10(ASD) reconstruction residuals (dex), always
    plotted as abs(residual). threshold_low/threshold_high are therefore
    in dex units too — e.g. 0.3 ~ factor of ~2, 1.0 ~ factor of 10 — not
    sigma/z-score units.

    Parameters:
    - reduced_data: dict from reduce_window, must contain "residual" and
      "bins", and optionally "long_term_residual".
    - timestamp: datetime for this window, used as the new x-axis point.
    - threshold_low / threshold_high: dex thresholds for bins below/at-or-
      above freq_cutoff, respectively.
    - freq_cutoff: frequency (Hz) separating the low/high bands.
    - max_bins: max number of bins plotted per band, per term.
    - state: shared mutable dict (see analyze) holding figure handles and
      rolling history; required.

    Returns:
    - matplotlib Figure: the 2x2 residual-history figure.
    """

    if state is None:
        raise ValueError("state must be provided (shared with live_figure_psd).")
    if threshold_low is None or threshold_high is None:
        raise ValueError("threshold_low and threshold_high must be provided.")

    # ---- Extract residuals + bin frequencies (always abs) ----
    residual_short, bins_axis = unpack(reduced_data, ['residual', 'bins'])
    residual_short = np.abs(residual_short)

    # Long-term residual only exists once a reference PCA model has been
    # frozen; fill with NaNs until then so history arrays stay rectangular.
    if "long_term_residual" in reduced_data:
        residual_long = np.abs(reduced_data["long_term_residual"])
        has_long = True
    else:
        residual_long = np.full_like(residual_short, np.nan, dtype=float)
        has_long = False

    # ---- Persistent state init ----
    if state.get("start_time", None) is None:
        state["start_time"] = timestamp

    # Append this window's data point to the rolling history.
    state["x"].append(timestamp)
    state["residual_history_short"].append(residual_short)
    state["residual_history_long"].append(residual_long)

    x = state["x"]
    res_short = np.asarray(state["residual_history_short"])  # (T, n_bins)
    res_long = np.asarray(state["residual_history_long"])    # (T, n_bins)

    # Split bins into low/high frequency bands relative to freq_cutoff.
    low_mask = bins_axis < freq_cutoff
    high_mask = ~low_mask

    # Compute bin selections from history (this is what PSD must match)
    state["top_bins_short_low"] = _select_bins_above_threshold(
        res_short, threshold_low, bin_mask=low_mask, max_bins=max_bins)
    state["top_bins_short_high"] = _select_bins_above_threshold(
        res_short, threshold_high, bin_mask=high_mask, max_bins=max_bins)

    if has_long:
        state["top_bins_long_low"] = _select_bins_above_threshold(
            res_long, threshold_low, bin_mask=low_mask, max_bins=max_bins)
        state["top_bins_long_high"] = _select_bins_above_threshold(
            res_long, threshold_high, bin_mask=high_mask, max_bins=max_bins)
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

    # Redraw all four panels with the latest history and bin selections.
    _plot_residual_panel(
        axes["short_low"], x, res_short, state["top_bins_short_low"],
        threshold_low, bins_axis, "short-term", low_label, band_width=band_width, length=len(np.where(low_mask == True)[0]), active=True
    )
    _plot_residual_panel(
        axes["short_high"], x, res_short, state["top_bins_short_high"],
        threshold_high, bins_axis, "short-term", high_label, band_width=band_width, length=len(np.where(high_mask == True)[0]), active=True
    )
    _plot_residual_panel(
        axes["long_low"], x, res_long, state["top_bins_long_low"],
        threshold_low, bins_axis, "long-term", low_label, band_width=band_width, length=len(np.where(low_mask == True)[0]), active=has_long
    )
    _plot_residual_panel(
        axes["long_high"], x, res_long, state["top_bins_long_high"],
        threshold_high, bins_axis, "long-term", high_label, band_width=band_width, length=len(np.where(high_mask == True)[0]), active=has_long
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

    # Only auto-rotate/format the date labels once; repeating it every call
    # is unnecessary and can accumulate redundant layout adjustments.
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

    Parameters:
    - reduced_data: dict from reduce_window, must contain "bins" and "asd".
    - timestamp: datetime for this window, shown in the plot title.
    - state: shared mutable dict (see analyze) holding figure handles and
      the current flagged-bin selections; required.

    Returns:
    - matplotlib Figure: the live PSD figure.
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

    # Mark short-term-flagged bins with short vertical ticks at the bottom.
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

    # Mark long-term-flagged bins with short vertical ticks at the top.
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


def analyze(reduced_data, timestamp, threshold_low, threshold_high, figures, band_width=2.0,
            freq_cutoff=1000.0, max_bins=5):
    """
    Entry point called once per processed window: updates whichever live
    figures are requested (worst-bin residual histories and/or the PSD)
    using a shared, function-level persistent state so bin selections
    stay consistent between the two figures across repeated calls.

    Parameters:
    - reduced_data: dict from reduce_window for the current window.
    - timestamp: datetime for this window.
    - threshold_low: residual threshold for bins below freq_cutoff.
    - threshold_high: residual threshold for bins at/above freq_cutoff.
    - figures: dict with boolean keys "worst_bins" and "psd" controlling
      which live figures are generated/updated this call.
    - freq_cutoff: frequency (Hz) separating the low/high bands.
    - max_bins: max number of bins plotted per band, per term.

    Returns:
    - (worst_bins, psd): the residual-history figure and the PSD figure,
      or None for either one that wasn't requested in `figures`.
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

    # Only compute/update the worst-bins panel if explicitly requested;
    # note this also drives the bin selections used by the PSD panel below.
    if figures["worst_bins"]:
        worst_bins = live_figure_worst_bins(
            reduced_data, timestamp, threshold_low, threshold_high, band_width=band_width,
            freq_cutoff=freq_cutoff, max_bins=max_bins, state=shared_state
        )
    else:
        worst_bins = None

    # Only compute/update the PSD panel if explicitly requested.
    if figures["psd"]:
        psd = live_figure_psd(
            reduced_data, timestamp, state=shared_state
        )
    else:
        psd = None

    return worst_bins, psd