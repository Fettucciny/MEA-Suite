#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 MEA-NAP pipeline -- STAGE 2a: neuronal-activity analysis + plots
================================================================================
Reads the per-H5 spike files via mea_nap_io.iter_batch_units() and, for each
Batch.csv unit (file x window x region), computes neuronal-activity metrics and
writes the MEA-NAP-style activity plot set into PLOT_DIR/<stem>_<region>/:

    firing_rate_by_electrode   (half-violin + scatter of per-electrode FR)
    firing_rate_heatmap        (8x8 electrode grid)
    raster
    burst_rate_heatmap
    burst_duration_heatmap
    frac_spikes_in_burst_heatmap
    isi_within_burst_heatmap
    isi_outside_burst_heatmap
    burst_detection_info       (population rate + network bursts + per-elec counts)

Per-electrode bursts use the max-interval (ISI-threshold) detector -- a standard,
well-defined method that yields all five burst metrics directly. (MEA-NAP also
offers a Bakkum single-channel variant; this detector is swappable via
BURST_PARAMS and can be matched to it if strict parity is needed.)

Network analysis (STTC, edge-threshold check, graph/node metrics) and the
node-by-age/group comparisons are Stage 2b/2c.
================================================================================
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # Stage-2 plots are SAVED, not popups
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import mea_nap_io as io
import mea_nap_progress as progress_gui
import mea_nap_original_plots as orig     # faithful MEA-NAP figure set (parallel tree)
import mea_nap_spectral_propagation as spprop   # PSD + signal-propagation figures
import mea_nap_digital_events as digital         # laser timing from MCS Digital Event Recorder

# Also emit the SAND-Lab MEA-NAP MATLAB figure set into a parallel
# "MEA-NAP Original Output" folder beside PLOT_DIR (the pipeline's own Plots/ are
# untouched). Set False to skip. See mea_nap_original_plots.py.
MEANAP_ORIGINAL_PLOTS = True                     # superseded by STAGE2_SECTIONS["meanap"]

# Coarse per-section on/off for Stage 2, driven by the GUI "Analyses to run" panel
# so a quick single-section test is possible. The per-recording activity COMPUTE
# always runs when Stage 2 runs (it feeds network / spectral / comparisons); these
# flags gate the OUTPUTS and the heavy computes. Turning all of them off (together
# with the SpKit / Phase-Lock switches) makes a run stop right after detection.
STAGE2_SECTIONS = {
    "activity":    True,   # per-recording activity + burst plots
    "network":     True,   # STTC connectivity + graph metrics + network plots
    "spectral":    True,   # PSD + signal propagation + CV field
    "meanap":      True,   # MEA-NAP "original output" plot tree (2A/4A/2B/4B)
    "comparisons": True,   # group / age comparison figures (Stage 2c)
}

# --- SAND-Lab MEA-NAP Python port (vendored under ./meanap) --------------------
# These are the lab's OWN validated modules (github.com/SAND-Lab/MEA-NAP,
# src/meanap/pipeline). We call them directly for network math so results match
# their port exactly rather than a re-implementation.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from meanap.pipeline import network_metrics as _nm
from meanap.pipeline import modularity as _mod
from meanap.pipeline import sttc as _sttc
from meanap.pipeline import probabilistic_threshold as _pthr
from meanap.pipeline.step4_metrics import compute_network_metrics


# ==============================================================================
# PARAMETERS
# ==============================================================================
# FIRING RATE / ACTIVE ELECTRODES
#   active_min_rate_hz / active_min_spikes : an electrode counts as "active" (and
#       is included in network-level summaries later) only above both floors.
FR_PARAMS = {
    "active_min_rate_hz": 0.10,
    "active_min_spikes":  5,
}

# PER-ELECTRODE BURST DETECTION
#   method : "isi_n"  -> Bakkum ISI_N, ADAPTIVE and MEA-NAP-aligned (default).
#                        A burst is a run of spikes where the time to span `isi_n_N`
#                        consecutive spikes stays <= isi_n_thr_s. Because it keys off
#                        N-spike windows rather than a single fixed gap, it catches
#                        bursts across electrodes with very different baseline rates
#                        -- which is why it fixes the "shows poor bursting when there
#                        clearly is bursting" problem the fixed-gap method had.
#            "maxinterval" -> classic fixed ISI-threshold method (legacy).
#   isi_n_N       : spikes per ISI_N window (5 is typical; lower = more sensitive).
#   isi_n_thr_s   : ISI_N threshold. Raise if real bursts are being missed, lower
#                   if noise is being called a burst. 0.1 s means N spikes within
#                   100 ms. If a lot of true bursts are missed, try 0.15-0.2.
#   (max_isi_*    : only used by the legacy "maxinterval" method.)
#   min_ibi_s     : merge bursts separated by less than this.
#   min_burst_dur_s / min_spikes : discard bursts below these.
#   rate_unit     : "per_min" or "per_sec" for the burst-rate heatmap.
BURST_PARAMS = {
    "method":          "isi_n",
    "isi_n_N":         5,
    "isi_n_thr_s":     0.10,
    "max_isi_start_s": 0.10,
    "max_isi_end_s":   0.20,
    "min_ibi_s":       0.20,
    "min_burst_dur_s": 0.02,
    "min_spikes":      5,
    "rate_unit":       "per_min",
}

# NETWORK BURSTS (Bakkum ISI_N over pooled spikes) -- used by burst_detection_info
#   net_N          : spikes spanned by the ISI_N window.
#   net_isi_thr_s  : ISI_N threshold.
#   net_min_chan   : distinct electrodes required to call a network burst.
NETBURST_PARAMS = {
    "net_N":        30,
    "net_isi_thr_s": 0.10,
    "net_min_chan": 3,
}

GRID_PARAMS = {"n_rows": 8, "n_cols": 8}     # MCS 60-MEA grid; labels 'CR', 1..8
PLOT_PARAMS = {
    "dpi": 600,                              # high-res output (>=600 as requested)
    # rasters carry up to millions of 0.3-pt ticks; 600 dpi there is the single
    # biggest render cost and looks identical to 300. Rendered at raster_dpi (raise
    # to 600 to match the rest exactly). Range 150 - 600.
    "raster_dpi": 300,
    "heatmap_cmap": "viridis",
    "raster_max_spikes_per_ch": 12000,   # cap PER VISIBLE ROW-WINDOW (not global)
    # robust color limits for heatmaps: clip the color scale to these percentiles
    # of the data so a few outlier electrodes don't wash out the whole map.
    "heatmap_clip_pct": (2, 98),
    # raster time scaling: a fixed seconds-per-inch so a 3x longer recording is
    # drawn 3x longer; wrapped into stacked rows once a single row would exceed
    # raster_row_seconds, keeping the physical time scale identical across files.
    "raster_row_seconds": 300.0,             # max seconds per raster row
    "raster_sec_per_inch": 30.0,             # time compression (s per inch)
    # network graph node/edge sizing (normalized -> no giant dots / hairball)
    "graph_node_size": (20, 260),            # min/max node area by degree
    "graph_edge_width": (0.3, 2.4),          # min/max edge width by weight
    "graph_edge_alpha": 0.35,
}


# ==============================================================================
# BLOCK A  --  per-electrode burst detection + metrics
# ==============================================================================
def _bursts_maxinterval(s, bp):
    n = s.size
    if n < bp["min_spikes"]:
        return []
    isi = np.diff(s)
    bursts, i = [], 0
    while i < n - 1:
        if isi[i] <= bp["max_isi_start_s"]:
            j = i
            while j < n - 1 and isi[j] <= bp["max_isi_end_s"]:
                j += 1
            bursts.append([s[i], s[j], i, j + 1])
            i = j + 1
        else:
            i += 1
    return bursts


def _runs_of_true(mask):
    """Half-open [i, j) runs of True in a 1-D bool array, as an (n_runs, 2)
    int array. Vectorised replacement for the per-element while-loops below;
    the runs it returns are exactly the (i, j) pairs those loops visited."""
    if mask.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    m = np.concatenate(([False], mask, [False]))
    d = np.diff(m.astype(np.int8))
    return np.column_stack([np.where(d == 1)[0], np.where(d == -1)[0]])


def _bursts_isi_n(s, bp):
    """Bakkum ISI_N: burst = run of spikes whose N-spike span stays <= threshold."""
    n = s.size
    N = bp["isi_n_N"]
    if n < max(N, bp["min_spikes"]):
        return []
    isi_n = s[N - 1:] - s[:n - N + 1]              # ISI_N[i] spans s[i .. i+N-1]
    in_b = isi_n <= bp["isi_n_thr_s"]
    bursts = []
    for i, j in _runs_of_true(in_b):               # same runs the old loop found
        a = int(i)
        b = int(min(j - 1 + N - 1, n - 1))
        bursts.append([s[a], s[b], a, b + 1])
    return bursts


def detect_electrode_bursts(spikes, bp=BURST_PARAMS):
    """Per-electrode burst detection. Returns list of (start, end, i0, i1) with
    i0:i1 the spike-index span (i1 exclusive). Method set by BURST_PARAMS['method']."""
    s = np.sort(np.asarray(spikes))
    bursts = (_bursts_isi_n(s, bp) if bp.get("method", "isi_n") == "isi_n"
              else _bursts_maxinterval(s, bp))
    if not bursts:
        return []
    merged = [list(bursts[0])]                     # merge bursts closer than min_ibi
    for b in bursts[1:]:
        if b[0] - merged[-1][1] < bp["min_ibi_s"]:
            merged[-1][1] = b[1]; merged[-1][3] = b[3]
        else:
            merged.append(list(b))
    return [tuple(b) for b in merged               # filter by duration + spike count
            if (b[1] - b[0]) >= bp["min_burst_dur_s"]
            and (b[3] - b[2]) >= bp["min_spikes"]]


def electrode_burst_metrics(spikes, duration_s, bp=BURST_PARAMS):
    """Return the five per-electrode burst metrics used by the heatmaps."""
    s = np.sort(np.asarray(spikes))
    m = {"n_bursts": 0, "burst_rate": 0.0, "mean_burst_dur_s": np.nan,
         "frac_spikes_in_burst": 0.0, "isi_within_burst_s": np.nan,
         "isi_outside_burst_s": np.nan}
    if s.size < 2 or duration_s <= 0:
        return m
    bursts = detect_electrode_bursts(s, bp)
    isi = np.diff(s)
    in_burst_isi = np.zeros(isi.size, bool)
    spikes_in_burst = 0
    durs = []
    for (t0, t1, i0, i1) in bursts:
        in_burst_isi[i0:i1 - 1] = True              # ISIs internal to the burst
        spikes_in_burst += (i1 - i0)
        durs.append(t1 - t0)

    m["n_bursts"] = len(bursts)
    rate = len(bursts) / duration_s
    m["burst_rate"] = rate * 60.0 if bp["rate_unit"] == "per_min" else rate
    if durs:
        m["mean_burst_dur_s"] = float(np.mean(durs))
    m["frac_spikes_in_burst"] = spikes_in_burst / s.size
    if in_burst_isi.any():
        m["isi_within_burst_s"] = float(np.mean(isi[in_burst_isi]))
    if (~in_burst_isi).any():
        m["isi_outside_burst_s"] = float(np.mean(isi[~in_burst_isi]))
    return m


def detect_network_bursts(spikes_by_ch, np_=NETBURST_PARAMS):
    """Bakkum ISI_N network bursts over pooled spikes. Returns list of dicts."""
    times, chans = [], []
    for ci, s in enumerate(spikes_by_ch):
        if len(s):
            times.append(np.asarray(s)); chans.append(np.full(len(s), ci))
    if not times:
        return []
    t = np.concatenate(times); c = np.concatenate(chans)
    order = np.argsort(t); t, c = t[order], c[order]
    N = np_["net_N"]; n = t.size
    if n < N:
        return []
    isi_n = t[N - 1:] - t[:n - N + 1]
    in_b = isi_n <= np_["net_isi_thr_s"]
    out = []
    for i, j in _runs_of_true(in_b):               # same runs the old loop found
        a, b = int(i), int(min(j - 1 + N - 1, n - 1))
        uniq = np.unique(c[a:b + 1])
        if uniq.size >= np_["net_min_chan"]:
            out.append({"start": float(t[a]), "end": float(t[b]),
                        "duration": float(t[b] - t[a]), "n_channels": int(uniq.size)})
    return out


# ==============================================================================
# BLOCK B  --  unit analysis
# ==============================================================================
def analyze_activity(unit):
    """Compute per-electrode + summary activity metrics for one batch unit."""
    labels = unit["labels"]; spikes = unit["spikes"]; dur = unit["duration"]
    n = len(labels)
    fr = np.array([len(s) / dur if dur > 0 else 0.0 for s in spikes])
    counts = np.array([len(s) for s in spikes])
    active = (fr >= FR_PARAMS["active_min_rate_hz"]) & \
             (counts >= FR_PARAMS["active_min_spikes"])

    per = {k: np.full(n, np.nan) for k in
           ["burst_rate", "mean_burst_dur_s", "frac_spikes_in_burst",
            "isi_within_burst_s", "isi_outside_burst_s"]}
    per["n_bursts"] = np.zeros(n)
    for i, s in enumerate(spikes):
        m = electrode_burst_metrics(s, dur)
        for k in per:
            per[k][i] = m[k]

    net_bursts = detect_network_bursts(spikes)
    # recording-level network-burst metrics (MEA-NAP Step 2B)
    all_sp = np.sort(np.concatenate([np.asarray(s) for s in spikes if len(s)])) \
        if any(len(s) for s in spikes) else np.empty(0)
    nb_len = [b["end"] - b["start"] for b in net_bursts]
    nb_nch = [b["n_channels"] for b in net_bursts]
    in_nb = np.zeros(all_sp.size, bool)
    # all_sp is sorted, so each burst is one contiguous slice: searchsorted gives
    # the identical mask to `(all_sp >= start) & (all_sp <= end)` without a pass
    # over every spike per burst. Searched as float64 ONCE: the spike arrays are
    # float32 and searchsorted with a Python float would otherwise upcast the
    # whole array on every call (4.5 ms x thousands of bursts); burst bounds are
    # float32 values widened to float64, so the comparison is exact either way.
    all64 = np.asarray(all_sp, dtype=np.float64)
    for b in net_bursts:
        lo = np.searchsorted(all64, float(b["start"]), side="left")
        hi = np.searchsorted(all64, float(b["end"]), side="right")
        in_nb[lo:hi] = True
    isi_all = np.diff(all_sp) if all_sp.size > 1 else np.array([])
    isi_in = isi_all[in_nb[:-1] & in_nb[1:]] if isi_all.size else isi_all
    isi_out = isi_all[~(in_nb[:-1] & in_nb[1:])] if isi_all.size else isi_all
    ibi = np.diff([b["start"] for b in net_bursts]) if len(net_bursts) > 1 else np.array([])

    summary = {
        "stem": unit["stem"], "region": unit["region"], "group": unit["group"],
        "age": unit["age"], "duration_s": dur,
        "n_electrodes": n, "n_active": int(active.sum()),
        "mean_FR_hz": float(fr.mean()),
        "mean_FR_active_hz": float(fr[active].mean()) if active.any() else 0.0,
        "median_FR_hz": float(np.median(fr)),
        "total_spikes": int(counts.sum()),
        "n_network_bursts": len(net_bursts),
        "network_burst_rate_per_min": len(net_bursts) / dur * 60 if dur > 0 else 0.0,
        "mean_electrodes_in_nb": float(np.mean(nb_nch)) if nb_nch else np.nan,
        "mean_nb_length_s": float(np.mean(nb_len)) if nb_len else np.nan,
        "isi_within_nb_ms": float(np.mean(isi_in) * 1000) if isi_in.size else np.nan,
        "isi_outside_nb_ms": float(np.mean(isi_out) * 1000) if isi_out.size else np.nan,
        "cov_ibi": float(np.std(ibi) / np.mean(ibi)) if ibi.size and np.mean(ibi) > 0 else np.nan,
        "frac_spikes_in_nb": float(in_nb.mean()) if in_nb.size else 0.0,
        "mean_elec_burst_rate": float(np.nanmean(per["burst_rate"])),
        "mean_frac_spikes_in_burst": float(np.nanmean(per["frac_spikes_in_burst"])),
    }
    return {"fr": fr, "active": active, "per": per,
            "net_bursts": net_bursts, "summary": summary}


# ==============================================================================
# BLOCK C  --  plotting helpers + activity plots
# ==============================================================================
def _grid_xy(label):
    if len(label) == 2 and label.isdigit():
        c, r = int(label[0]), int(label[1])
        if 1 <= c <= GRID_PARAMS["n_cols"] and 1 <= r <= GRID_PARAMS["n_rows"]:
            return c, r
    return None


REGION_ORDER = ["pre", "stim", "post"]      # panel order within a file


def _sorted_regions(units):
    """Units ordered pre -> stim -> post (unknown regions appended, sorted)."""
    known = [u for u in units if u["region"] in REGION_ORDER]
    other = sorted([u for u in units if u["region"] not in REGION_ORDER],
                   key=lambda u: u["region"])
    known.sort(key=lambda u: REGION_ORDER.index(u["region"]))
    return known + other


def _grid_heatmap_ax(fig, ax, values, labels, title, cbar_label, cmap=None,
                     vlim=None, cbar=True):
    """One 8x8 electrode-grid heatmap onto `ax`. `vlim` forces a shared colour
    scale so panels from different regions are directly comparable."""
    nr, nc = GRID_PARAMS["n_rows"], GRID_PARAMS["n_cols"]
    grid = np.full((nr, nc), np.nan)
    for v, lab in zip(values, labels):
        xy = _grid_xy(lab)
        if xy:
            grid[xy[1] - 1, xy[0] - 1] = v
    if vlim is not None:
        vmin, vmax = vlim
    else:
        finite = grid[np.isfinite(grid)]
        lo, hi = PLOT_PARAMS["heatmap_clip_pct"]
        vmin, vmax = (np.nanpercentile(finite, [lo, hi]) if finite.size else (0, 1))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = (0.0, 1.0) if not np.isfinite(vmin) else (vmin - 0.5, vmax + 0.5)
    im = ax.imshow(grid, cmap=cmap or PLOT_PARAMS["heatmap_cmap"], origin="upper",
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(nc)); ax.set_xticklabels(range(1, nc + 1), fontsize=7)
    ax.set_yticks(range(nr)); ax.set_yticklabels(range(1, nr + 1), fontsize=7)
    ax.set_xticks(np.arange(-.5, nc, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nr, 1), minor=True)
    ax.grid(which="minor", color="w", lw=0.6); ax.tick_params(which="minor", length=0)
    ax.set(title=title, xlabel="Electrode column", ylabel="Electrode row")
    if cbar:
        fig.colorbar(im, ax=ax, label=cbar_label, extend="both")
    return im


def _shared_vlim(values_list):
    """Robust colour limits pooled across regions (comparable panels)."""
    pooled = np.concatenate([np.asarray(v, float).ravel() for v in values_list]) \
        if values_list else np.array([])
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return None
    lo, hi = PLOT_PARAMS["heatmap_clip_pct"]
    vmin, vmax = np.nanpercentile(pooled, [lo, hi])
    return (float(vmin), float(vmax))


def _fr_by_electrode_ax(ax, fr, title):
    v = np.asarray(fr, float)
    v = v[np.isfinite(v)]
    if v.size and v.max() > 0:
        parts = ax.violinplot([v], showextrema=False, widths=0.9)
        for b in parts["bodies"]:
            b.set_facecolor("0.8"); b.set_alpha(0.6)
            m = np.mean(b.get_paths()[0].vertices[:, 0])   # keep only right half
            b.get_paths()[0].vertices[:, 0] = np.clip(
                b.get_paths()[0].vertices[:, 0], m, np.inf)
    ax.scatter(np.random.uniform(0.75, 0.98, v.size), v, s=10, color="tab:blue",
               alpha=0.6, edgecolors="none")
    ax.set(xticks=[], ylabel="Firing rate (Hz)", title=title)
    ax.margins(y=0.12)


def _burst_info_ax(ax1, ax2, spikes, net_bursts, per, labels, dur, title):
    allsp = np.concatenate([np.asarray(s) for s in spikes if len(s)]) \
        if any(len(s) for s in spikes) else np.empty(0)
    if allsp.size and dur > 0:
        bw = 0.05
        edges = np.arange(0, dur + bw, bw)
        cnt, _ = np.histogram(allsp, bins=edges)
        ax1.plot(edges[:-1], cnt / bw, lw=0.6, color="k")
    for b in net_bursts:
        ax1.axvspan(b["start"], b["end"], color="tab:red", alpha=0.25, lw=0)
    ax1.set(xlim=(0, max(dur, 1e-6)), ylabel="Pop. rate (Hz)",
            title=f"{title}  |  {len(net_bursts)} network bursts")
    ax2.bar(np.arange(len(labels)), per["n_bursts"], color="tab:purple")
    ax2.set(xlabel="Electrode #", ylabel="# bursts", xlim=(-1, len(labels)))


# ---- single-figure wrappers (kept for ad-hoc use) ----------------------------
def _grid_heatmap(values, labels, title, cbar_label, out_png, cmap=None):
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    _grid_heatmap_ax(fig, ax, values, labels, title, cbar_label, cmap)
    fig.tight_layout(); fig.savefig(out_png, dpi=PLOT_PARAMS["dpi"]); plt.close(fig)


def plot_fr_by_electrode(fr, title, out_png):
    fig, ax = plt.subplots(figsize=(4.6, 5.4))
    _fr_by_electrode_ax(ax, fr, title)
    fig.tight_layout(); fig.savefig(out_png, dpi=PLOT_PARAMS["dpi"]); plt.close(fig)


# ==============================================================================
# MULTI-REGION FIGURES  (one file -> one folder; regions are PANELS)
# ==============================================================================
def _panels_fig(n, w=4.6, h=5.0):
    fig, axes = plt.subplots(1, max(n, 1), figsize=(w * max(n, 1), h),
                             squeeze=False, layout="constrained")
    return fig, axes[0]


def plot_fr_by_electrode_multi(units, res_by_region, title, out_png):
    us = _sorted_regions(units)
    fig, axes = _panels_fig(len(us), 3.6, 5.0)
    ymax = max([np.nanmax(res_by_region[u["region"]]["fr"]) for u in us
                if np.isfinite(res_by_region[u["region"]]["fr"]).any()] or [1.0])
    for ax, u in zip(axes, us):
        _fr_by_electrode_ax(ax, res_by_region[u["region"]]["fr"], u["region"])
        ax.set_ylim(0, ymax * 1.1)                  # shared scale across regions
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_png, dpi=PLOT_PARAMS["dpi"]); plt.close(fig)


def plot_heatmap_multi(units, values_by_region, labels_by_region, title,
                       cbar_label, out_png, cmap=None):
    us = _sorted_regions(units)
    vlim = _shared_vlim([values_by_region[u["region"]] for u in us])
    fig, axes = _panels_fig(len(us), 4.4, 4.6)
    im = None
    for ax, u in zip(axes, us):
        im = _grid_heatmap_ax(fig, ax, values_by_region[u["region"]],
                              labels_by_region[u["region"]], u["region"],
                              cbar_label, cmap, vlim=vlim, cbar=False)
    if im is not None:
        fig.colorbar(im, ax=list(axes), label=cbar_label, extend="both",
                     shrink=0.85)
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_png, dpi=PLOT_PARAMS["dpi"]); plt.close(fig)


def plot_raster_multi(units, title, out_png):
    """All regions in ONE raster figure. Every row spans the same number of
    seconds (constant time scale), regions are stacked and labelled, and long
    regions wrap onto extra rows -- so pre/stim/post stay directly comparable."""
    us = _sorted_regions(units)
    spi = PLOT_PARAMS["raster_sec_per_inch"]
    row_s = PLOT_PARAMS["raster_row_seconds"]
    max_rows = PLOT_PARAMS.get("raster_max_rows", 12)
    longest = max([u["duration"] for u in us] or [1.0])
    if longest > row_s * max_rows:                  # keep total height sane
        row_s = float(np.ceil(longest / max_rows))
    span = min(row_s, max(longest, 1e-6))

    layout = []                                     # (unit, row_index_within_region)
    for u in us:
        n_rows = max(1, int(np.ceil(u["duration"] / span)))
        for r in range(n_rows):
            layout.append((u, r))
    n_ch = max([len(u["labels"]) for u in us] or [1])
    row_h = max(1.3, 0.028 * n_ch)
    fig, axes = plt.subplots(len(layout), 1, squeeze=False,
                             figsize=(span / spi + 1.4, row_h * len(layout) + 0.8),
                             layout="constrained")
    axes = axes[:, 0]
    for ax, (u, r) in zip(axes, layout):
        t0 = r * span
        t_end = min(t0 + span, u["duration"])
        cap = PLOT_PARAMS["raster_max_spikes_per_ch"]
        # Build ONE LineCollection for the whole panel instead of a vlines() call
        # per channel: same picture, ~2.4x faster on dense rasters (these panels
        # can carry millions of ticks).
        segs = []
        for i, s in enumerate(u["spikes"]):
            s = np.asarray(s)
            seg = s[(s >= t0) & (s < t_end)]
            if seg.size > cap:
                seg = seg[np.linspace(0, seg.size - 1, cap).astype(int)]
            if seg.size:
                blk = np.empty((seg.size, 2, 2))
                blk[:, 0, 0] = seg; blk[:, 1, 0] = seg
                blk[:, 0, 1] = i + 0.5; blk[:, 1, 1] = i + 1.5
                segs.append(blk)
        if segs:
            ax.add_collection(LineCollection(np.concatenate(segs), linewidths=0.3,
                                             colors="k"))
        ax.set_xlim(t0, t0 + span)
        ax.set_ylim(0.5, len(u["labels"]) + 0.5)
        ax.set_ylabel(f"{u['region']}\nelectrode #" if r == 0 else "electrode #",
                      fontsize=7)
        if t_end < t0 + span:
            ax.axvspan(t_end, t0 + span, color="0.94", lw=0)
    axes[-1].set_xlabel("Time within region (s)")
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_png, dpi=PLOT_PARAMS.get("raster_dpi", PLOT_PARAMS["dpi"]))
    plt.close(fig)


def plot_burst_info_multi(units, res_by_region, title, out_png):
    us = _sorted_regions(units)
    fig, axes = plt.subplots(2, len(us), squeeze=False,
                             figsize=(5.6 * len(us), 5.6),
                             gridspec_kw={"height_ratios": [2, 1]},
                             layout="constrained")
    for j, u in enumerate(us):
        res = res_by_region[u["region"]]
        _burst_info_ax(axes[0, j], axes[1, j], u["spikes"], res["net_bursts"],
                       res["per"], u["labels"], u["duration"], u["region"])
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_png, dpi=PLOT_PARAMS["dpi"]); plt.close(fig)


def make_activity_plots_multi(stem, units, res_by_region, out_dir):
    """All activity plots for ONE FILE, each with one panel per region."""
    os.makedirs(out_dir, exist_ok=True)
    us = _sorted_regions(units)
    lab = {u["region"]: u["labels"] for u in us}
    ru = "bursts/min" if BURST_PARAMS["rate_unit"] == "per_min" else "bursts/s"
    P = lambda f: os.path.join(out_dir, f)

    en = orig.PLOT_ENABLE.get                        # compute-level: skip disabled plots
    if en("firing_rate_by_electrode", True):
        plot_fr_by_electrode_multi(us, res_by_region, f"FR by electrode — {stem}",
                                   P("firing_rate_by_electrode.png"))
    if en("electrode_heatmap", True):
        plot_heatmap_multi(us, {u["region"]: res_by_region[u["region"]]["fr"] for u in us},
                           lab, f"Firing rate — {stem}", "Hz", P("firing_rate_heatmap.png"))
    if en("raster", True):
        plot_raster_multi(us, f"Raster — {stem}", P("raster.png"))
    for key, nice, unit_lbl, cm in [
            ("burst_rate", "Burst rate", ru, None),
            ("mean_burst_dur_s", "Burst duration", "s", None),
            ("frac_spikes_in_burst", "Frac. spikes in bursts", "fraction", None),
            ("isi_within_burst_s", "ISI within bursts", "s", "magma"),
            ("isi_outside_burst_s", "ISI outside bursts", "s", "magma")]:
        vals = {u["region"]: res_by_region[u["region"]]["per"][key] for u in us}
        fname = {"burst_rate": "burst_rate_heatmap.png",
                 "mean_burst_dur_s": "burst_duration_heatmap.png",
                 "frac_spikes_in_burst": "frac_spikes_in_burst_heatmap.png",
                 "isi_within_burst_s": "isi_within_burst_heatmap.png",
                 "isi_outside_burst_s": "isi_outside_burst_heatmap.png"}[key]
        plot_heatmap_multi(us, vals, lab, f"{nice} — {stem}", unit_lbl, P(fname), cm)
    plot_burst_info_multi(us, res_by_region, f"Burst detection — {stem}",
                          P("burst_detection_info.png"))


# ==============================================================================
# BLOCK D  --  STTC functional connectivity  (Cutts & Eglen 2014)
# ------------------------------------------------------------------------------
#   dt_s : tiling window (+/- dt). MEA-NAP commonly probes ~10-50 ms.
# STTC in [-1,1]; 0 = chance, robust to firing-rate differences.
# ==============================================================================
STTC_PARAMS = {
    # MEA-NAP FuncConLagval default = [10, 25, 50] ms.
    "lags_ms": [10, 25, 50],
    "dt_s": 0.05,
}

# PARITY: how closely to follow the SAND-Lab port vs our fast GPU path.
#   sttc_backend : "gpu"    -> our binned-matmul STTC surrogates (DEFAULT; ~1000x
#                             faster, matches exact STTC within ~0.006). REQUIRED
#                             at this dataset's scale: the "exact" path costs ~6s
#                             per STTC matrix at 80k spikes/channel, x201
#                             (real+200 surrogates) x3 lags = ~1 hour PER WINDOW
#                             (~4 days for the batch). Use "exact" only for small
#                             windows or a final figure.
#                  "exact"  -> the lab's get_sttc + adjm_thr (bit-for-bit with
#                             their port; CPU; impractically slow on long windows)
#   pc_mode      : "norm" -> participation_coef_norm (what MEA-NAP saves as PC;
#                             100 null-model iters/recording/lag, ~15-35s at 60
#                             nodes) | "raw" -> participation_coef (fast, exact,
#                             but not what MEA-NAP reports)
#   auto_cartography_boundaries : MEA-NAP default (autoSetCartographyBoudariesPerLag=1)
#                             derives the 5 boundaries per lag from the pooled
#                             PC/Z density landscape (k-means). False = fixed
#                             boundaries below.
PARITY_PARAMS = {
    "sttc_backend": "gpu",
    "pc_mode":      "norm",
    "pc_norm_iter": 100,        # MEA-NAP participation_coef_norm n_iter
    "auto_cartography_boundaries": False,   # MEA-NAP ships this =1 (per-lag k-means
                                # via the port's trial_landscape_density, which IS
                                # vendored). Kept False for consistent fixed
                                # boundaries; see the differences note.
    "fs_hz":        20000.0,    # sampling rate for the port's frame-based shift
    "min_nodes":    25,         # minNumberOfNodesToCalNetMet (MEA-NAP default)
}


def _run_T(spikes, dt, start, end):
    n = len(spikes)
    if n == 0:
        return 0.0
    t = 2.0 * n * dt
    if n == 1:
        if (spikes[0] - start) < dt:
            t += -start + spikes[0] - dt
        if (end - spikes[0]) < dt:
            t += -spikes[0] - dt + end
        return t
    diffs = np.diff(spikes)
    ov = diffs[diffs < 2 * dt]
    t -= np.sum(2 * dt - ov)
    if (spikes[0] - start) < dt:
        t += -start + spikes[0] - dt
    if (end - spikes[-1]) < dt:
        t += -spikes[-1] - dt + end
    return t


def _run_P(spikes1, spikes2, dt):
    if len(spikes1) == 0 or len(spikes2) == 0:
        return 0
    s2 = np.sort(spikes2)
    idx = np.searchsorted(s2, spikes1)
    left = np.clip(idx - 1, 0, len(s2) - 1)
    right = np.clip(idx, 0, len(s2) - 1)
    dist = np.minimum(np.abs(spikes1 - s2[left]), np.abs(spikes1 - s2[right]))
    return int(np.sum(dist <= dt))


def sttc_pair(s1, s2, dt, start, end):
    n1, n2 = len(s1), len(s2)
    if n1 == 0 or n2 == 0:
        return np.nan
    T = end - start
    Ta = _run_T(np.sort(s1), dt, start, end) / T
    Tb = _run_T(np.sort(s2), dt, start, end) / T
    Pa = _run_P(np.sort(s1), s2, dt) / n1
    Pb = _run_P(np.sort(s2), s1, dt) / n2
    dA, dB = 1 - Pa * Tb, 1 - Pb * Ta
    a = (Pa - Tb) / dA if dA != 0 else 0.0
    b = (Pb - Ta) / dB if dB != 0 else 0.0
    return 0.5 * (a + b)


def sttc_matrix(spikes_by_ch, dt, start, end):
    n = len(spikes_by_ch)
    M = np.full((n, n), np.nan)
    srt = [np.sort(np.asarray(s)) for s in spikes_by_ch]
    for i in range(n):
        M[i, i] = 1.0
        for j in range(i + 1, n):
            v = sttc_pair(srt[i], srt[j], dt, start, end)
            M[i, j] = M[j, i] = v
    return M


# ==============================================================================
# BLOCK E  --  probabilistic (surrogate) thresholding
# ------------------------------------------------------------------------------
# Significant edges are those whose real STTC beats `sig_threshold` fraction of a
# circular-shift surrogate null (shift preserves ISI structure, destroys pairwise
# timing). NOTE ON RUNTIME: cost ~ n_surrogates * n_pairs * spikes; long
# high-firing stim windows are the expensive case -- lower n_surrogates (or use a
# shorter network window) if it's slow. A GPU surrogate path can be added later.
# ==============================================================================
PROB_PARAMS = {
    "n_surrogates":  200,       # MEA-NAP ProbThreshRepNum (recommend >=180)
    "tail":          0.05,      # MEA-NAP ProbThreshTail (upper-tail significance)
    "sig_threshold": 0.95,      # = 1 - tail; used by the GPU backend
    "keep_negative": False,
    "random_seed":   0,
    "keep_surrogates_for_plot": True,
    "use_gpu":       True,      # engaged when PARITY_PARAMS["sttc_backend"]=="gpu"
                                # (must be True or the gpu backend silently falls
                                # back to the slow exact CPU loop)
    "strict_gpu":    True,      # True -> a GPU failure RAISES instead of quietly
                                # dropping to the ~1000x slower exact CPU path
                                # (that fallback is what turns a 2 h run into days)
    "bins_per_dt":   25,
                               # exact, more VRAM; 25 -> ~2 ms bins at dt=50 ms)
}


# ==============================================================================
# PERFORMANCE  --  numba enforcement + parallel network metrics
# ------------------------------------------------------------------------------
# The vendored MEA-NAP port (compute_network_metrics = ExtractNetMet.m) runs the
# heavy null-model normalisations MEA-NAP does -- small-worldness (latmio 10000 /
# randmio 5000 rewires), the normalised participation coefficient (100 degree-
# preserving randomisations) and consensus modularity (50 Louvain reps) -- ONCE
# per (region x lag). Those swap loops use numba when it is importable and fall
# back to a pure-NumPy path ~10-30x slower otherwise (measured ~40 s vs ~2 s per
# region x lag on a 60-node network). Two levers:
#   require_numba : "warn" (default) prints a loud banner if numba is inactive so
#                   the 10-30x fallback can never run unnoticed -- the analogue of
#                   PROB_PARAMS['strict_gpu']. "raise" refuses to run; "off" mutes.
#                   (Enabling numba changes ONLY the normalised participation
#                   coefficient PC, within its stochastic range; SW/Q/Ci/Z/CC/BC/
#                   efficiency/degree/strength are unaffected.)
#   parallel_network_metrics : the (region x lag) compute_network_metrics calls
#                   are independent, so fan them out across processes. STTC and
#                   its surrogates stay on the single GPU (serial); only the CPU
#                   metrics parallelise. Auto-falls back to serial if the pool
#                   cannot start or sttc_backend != 'gpu'. Each worker reseeds
#                   default_rng(PROB_PARAMS['random_seed']) so results are
#                   identical to (and independent of the ordering of) the serial
#                   GPU-backend path. max_workers=None -> (CPU count - 1).
# ==============================================================================
PERF_PARAMS = {
    "require_numba":            "warn",     # options: "warn" | "raise" | "off"
    "parallel_network_metrics": True,       # options: True | False
    "max_workers":              None,       # None -> (CPU count - 1); or an int in
                                            # range 1 .. os.cpu_count() (capped to
                                            # the per-file region x lag job count)
}


def _numba_active():
    """True iff the port's null-model loops are numba-compiled (reads the port's
    own flag, so it reflects what actually runs, not just importability)."""
    try:
        from meanap.pipeline import null_models as _null
        return bool(getattr(_null, "_HAVE_NUMBA", False))
    except Exception:
        return False


def _check_numba():
    """Loud guard (mirrors PROB_PARAMS['strict_gpu']) so the ~10-30x pure-NumPy
    null-model fallback can never run unnoticed."""
    mode = PERF_PARAMS.get("require_numba", "warn")
    if mode == "off" or _numba_active():
        return
    msg = ("numba is NOT active -> the MEA-NAP null-model loops "
           "(participation_coef_norm's degree-preserving randomisations, etc.) "
           "run a pure-NumPy path ~10-30x slower (measured ~40 s vs ~2 s per "
           "region x lag on a 60-node network). Fix:  pip install numba  "
           "-- results are unchanged except the normalised participation "
           "coefficient PC, which shifts within its stochastic range.")
    if mode == "raise":
        raise RuntimeError(msg)
    print("\n" + "!" * 78 + f"\n  [numba] {msg}\n" + "!" * 78 + "\n", flush=True)


def _circular_shift(spikes, start, end, rng):
    T = end - start
    if T <= 0 or len(spikes) == 0:
        return spikes
    return np.sort(start + np.mod((spikes - start) + rng.uniform(0, T), T))


def _sttc_gpu_surrogates(spikes_by_ch, dt, dur, keep_surr=True):
    """GPU (binned-matmul) STTC + circular-shift surrogate thresholding.

    Time is binned at dt/bins_per_dt. Per-channel spike-count (SC) and coverage
    (COV, bins within +/-dt of a spike) become (n, K) matrices; then
        P[i,j] = (SC_i . COV_j) / N_i            -> one matmul  SC @ COV^T
        T[i]   = mean(COV_i)                       (roll-invariant)
    A circular shift is just a per-row roll of SC and COV, so every surrogate is
    another matmul. Runs on cuda when available (else cpu-torch). Returns
    (W, frac, surr_stack) matching the CPU path."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(spikes_by_ch)
    w = dt / PROB_PARAMS["bins_per_dt"]
    K = int(np.ceil(dur / w))
    R = max(1, int(round(dt / w)))

    SC = torch.zeros((n, K), dtype=torch.float32, device=dev)
    for i, s in enumerate(spikes_by_ch):
        s = np.asarray(s); s = s[(s >= 0) & (s < dur)]
        if s.size:
            b = torch.as_tensor(np.clip((s / w).astype(np.int64), 0, K - 1),
                                device=dev)
            SC[i].index_add_(0, b, torch.ones(b.shape[0], device=dev))
    N = SC.sum(1).clamp_min(1.0)
    cov = torch.nn.functional.max_pool1d((SC > 0).float().unsqueeze(1),
                                         2 * R + 1, 1, R).squeeze(1)
    T = cov.mean(1)
    Tcol, Trow = T[None, :], T[:, None]

    def sttc_from(SCx, COVx):
        P = (SCx @ COVx.t()) / N[:, None]
        with torch.no_grad():
            eps = 1e-6
            da = 1 - P * Tcol
            db = 1 - P.t() * Trow
            a = torch.where(da.abs() < eps, torch.zeros_like(da), (P - Tcol) / da)
            b = torch.where(db.abs() < eps, torch.zeros_like(db), (P.t() - Trow) / db)
            M = 0.5 * (a + b)
            M = torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
        M.fill_diagonal_(1.0)
        return M

    real = sttc_from(SC, cov)
    gen = torch.Generator(device=dev).manual_seed(PROB_PARAMS["random_seed"])
    ge = torch.zeros((n, n), device=dev)
    surr_all = torch.empty((PROB_PARAMS["n_surrogates"], n, n), device=dev)
    SCs = torch.empty_like(SC); COVs = torch.empty_like(cov)
    from tqdm import tqdm
    for r in tqdm(range(PROB_PARAMS["n_surrogates"]), desc="      surrogates(gpu)",
                  leave=False, unit="surr"):
        # Circular shift per channel via per-row torch.roll. (A materialized
        # (n x K) gather index would be int64 and multi-GB on hour-long windows
        # -- this keeps peak VRAM at ~2x the SC/cov matrices.)
        off = torch.randint(0, K, (n,), device=dev, generator=gen)
        for i in range(n):
            o = int(off[i])
            SCs[i] = torch.roll(SC[i], o)
            COVs[i] = torch.roll(cov[i], o)
        Ms = sttc_from(SCs, COVs)
        Ms.fill_diagonal_(0.0)
        ge += (real >= Ms).float()
        surr_all[r] = Ms

    frac = (ge / PROB_PARAMS["n_surrogates"]).cpu().numpy()
    M = real.cpu().numpy()
    # MEA-NAP threshold rule (adjm_thr): keep an edge only if the real STTC beats
    # the ceil((1-tail)*rep)-th smallest surrogate. Same rule as the exact path;
    # the only remaining difference is this backend's time binning.
    import math
    rep = PROB_PARAMS["n_surrogates"]
    cutoff = min(max(int(math.ceil((1 - PROB_PARAMS["tail"]) * rep)) - 1, 0), rep - 1)
    thr = torch.sort(surr_all, dim=0).values[cutoff].cpu().numpy()
    W = M.copy()
    W[thr > M] = 0.0
    np.fill_diagonal(W, 0.0)
    W = np.nan_to_num(W, nan=0.0)
    if not PROB_PARAMS["keep_negative"]:
        W[W < 0] = 0.0
    stack = surr_all.cpu().numpy().astype(np.float32) if keep_surr else None
    return M, W, frac, stack


def probabilistic_threshold(spikes_by_ch, dt, dur, keep_surr=True):
    """Compute the STTC matrix, surrogate-threshold it, and return
    (M, W, frac, surr_stack). GPU path (binned matmul) when enabled+available;
    otherwise the exact CPU STTC + circular-shift loop."""
    if PROB_PARAMS["use_gpu"]:
        try:
            import torch  # noqa
            return _sttc_gpu_surrogates(spikes_by_ch, dt, dur, keep_surr)
        except Exception as e:
            # The CPU fallback is ~1000x slower (~6 s PER STTC matrix at this
            # dataset's scale, x201 matrices x3 lags = ~1 hour PER WINDOW). Silently
            # degrading here turns a 2-hour run into a multi-day one, so shout.
            msg = (f"GPU STTC failed ({e!r}). The exact CPU fallback is ~1000x "
                   f"slower and will take DAYS on hour-long windows.")
            if PROB_PARAMS["strict_gpu"]:
                raise RuntimeError(
                    msg + " Set PROB_PARAMS['strict_gpu']=False to allow the slow "
                          "CPU path, or lower PROB_PARAMS['bins_per_dt'] if this is "
                          "a VRAM error.") from e
            print("\n" + "!" * 78 + f"\n  {msg}\n" + "!" * 78 + "\n", flush=True)

    real_M = sttc_matrix(spikes_by_ch, dt, 0.0, dur)     # exact CPU
    rng = np.random.default_rng(PROB_PARAMS["random_seed"])
    n = len(spikes_by_ch)
    srt = [np.sort(np.asarray(s)) for s in spikes_by_ch]
    ge = np.zeros((n, n))
    stack = [] if keep_surr else None
    from tqdm import tqdm
    for _ in tqdm(range(PROB_PARAMS["n_surrogates"]), desc="      surrogates",
                  leave=False, unit="surr"):
        shifted = [_circular_shift(s, 0.0, dur, rng) for s in srt]
        surr = sttc_matrix(shifted, dt, 0.0, dur)
        with np.errstate(invalid="ignore"):
            ge += (real_M >= surr).astype(float)
        if keep_surr:
            stack.append(surr.astype(np.float32))
    frac = ge / PROB_PARAMS["n_surrogates"]
    sig = frac >= PROB_PARAMS["sig_threshold"]
    np.fill_diagonal(sig, False)
    W = np.nan_to_num(np.where(sig, real_M, 0.0), nan=0.0)
    if not PROB_PARAMS["keep_negative"]:
        W[W < 0] = 0.0
    return real_M, W, frac, (np.asarray(stack) if keep_surr else None)


# ==============================================================================
# BLOCK F  --  network + node metrics
# ------------------------------------------------------------------------------
# Graph-theory summary of the thresholded functional network, plus per-NODE
# metrics (degree, strength, betweenness, clustering, local efficiency,
# participation coefficient, within-module z-score) that feed the Stage 2c
# node-by-age/group comparisons. Community detection (for participation /
# within-module z) uses Louvain when available.
# ==============================================================================
NET_PARAMS = {"binary_edge_min": 1e-9, "community_resolution": 0.4}

try:
    import networkx as nx
    _HAVE_NX = True
except Exception:
    _HAVE_NX = False


def _safe_mean(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else np.nan


def compute_graph(W, labels):
    """Compute per-node metrics AND recording-level graph metrics in one pass
    (Louvain communities computed once). Returns (net_dict, node_rows)."""
    n = W.shape[0]
    A = (W > NET_PARAMS["binary_edge_min"]).astype(float)
    np.fill_diagonal(A, 0)
    deg = A.sum(1); strength = W.sum(1)
    n_edges = int(A.sum() / 2)
    possible = n * (n - 1) / 2
    ew = W[np.triu(A, 1) > 0]                          # significant edge weights

    node_rows = [{"label": labels[i], "degree": float(deg[i]),
                  "strength": float(strength[i])} for i in range(n)]
    net = {"network_size": int(np.sum(deg > 0)), "n_edges": n_edges,
           "edge_density": n_edges / possible if possible else 0.0,
           "mean_degree": float(deg.mean()) if n else 0.0,
           "mean_strength": float(strength.mean()) if n else 0.0,
           "mean_edge_weight": float(ew.mean()) if ew.size else 0.0}

    # top-k summaries (MEA-NAP: top 25% degree, top 10% edge weight)
    if deg.max() > 0:
        k = max(1, int(np.ceil(0.25 * np.sum(deg > 0))))
        net["top25_mean_degree"] = float(np.sort(deg[deg > 0])[-k:].mean())
    else:
        net["top25_mean_degree"] = 0.0
    net["top10_mean_edge_weight"] = (float(np.sort(ew)[-max(1, int(np.ceil(0.1 * ew.size))):].mean())
                                     if ew.size else 0.0)

    if _HAVE_NX and n_edges > 0:
        Wg = np.where(A > 0, W, 0.0)
        G = nx.from_numpy_array(Wg)
        bc = nx.betweenness_centrality(G, weight=None)
        cl = nx.clustering(G, weight="weight")
        le = _local_efficiency_per_node(G)
        # MEA-NAP uses consensus clustering at resolution gamma=0.4 (coarser, less
        # fragmented modules). Weighted Louvain at resolution=0.4 matches that.
        try:
            comms = nx.community.louvain_communities(
                G, weight="weight", resolution=NET_PARAMS["community_resolution"],
                seed=0)
        except Exception:
            comms = list(nx.community.greedy_modularity_communities(G))
        node_comm = {u: ci for ci, c in enumerate(comms) for u in c}
        part = _participation_coeff(Wg, node_comm)     # WEIGHTED (MEA-NAP)
        wmz = _within_module_z(Wg, node_comm)          # WEIGHTED (MEA-NAP)
        for i in range(n):
            node_rows[i].update(betweenness=float(bc.get(i, 0.0)),
                                clustering=float(cl.get(i, 0.0)),
                                local_efficiency=float(le.get(i, 0.0)),
                                participation=float(part[i]),
                                within_module_z=float(wmz[i]))
        parts = np.array([nd["participation"] for nd in node_rows])

        # --- recording-level graph metrics ---
        net["clustering_coef"] = _safe_mean([nd["clustering"] for nd in node_rows])
        net["global_efficiency"] = float(nx.global_efficiency(G))
        net["mean_local_efficiency"] = _safe_mean([nd["local_efficiency"] for nd in node_rows])
        net["mean_betweenness"] = _safe_mean(list(bc.values()))
        net["n_modules"] = len(comms)
        net["modularity"] = float(nx.community.modularity(G, comms))
        net["mean_participation"] = _safe_mean(parts)
        nz = parts[deg > 0]
        net["participation_top10"] = (float(np.sort(nz)[-max(1, int(np.ceil(0.1 * nz.size))):].mean())
                                      if nz.size else np.nan)
        net["participation_bottom10"] = (float(np.sort(nz)[:max(1, int(np.ceil(0.1 * nz.size)))].mean())
                                         if nz.size else np.nan)
        z = wmz[deg > 0]
        net["pct_within_z_gt0"] = float(np.mean(z > 0) * 100) if z.size else np.nan
        net["pct_within_z_lt0"] = float(np.mean(z < 0) * 100) if z.size else np.nan
        # mean path length + small-worldness (analytic random-graph normalization)
        comp = max(nx.connected_components(G), key=len)
        Gc = G.subgraph(comp)
        try:
            L = nx.average_shortest_path_length(Gc)
        except Exception:
            L = np.nan
        net["mean_path_length"] = float(L)
        k_mean = deg[deg > 0].mean() if np.any(deg > 0) else 0
        Nn = net["network_size"]
        if k_mean > 1 and Nn > 1 and np.isfinite(L) and L > 0:
            C_rand = k_mean / (Nn - 1)
            L_rand = np.log(Nn) / np.log(k_mean)
            net["small_worldness_sigma"] = float((net["clustering_coef"] / C_rand)
                                                 / (L / L_rand)) if C_rand > 0 else np.nan
        else:
            net["small_worldness_sigma"] = np.nan
    else:
        for r in node_rows:
            r.update(betweenness=0.0, clustering=0.0, local_efficiency=0.0,
                     participation=0.0, within_module_z=0.0)
        for kkey in ["clustering_coef", "global_efficiency", "mean_local_efficiency",
                     "mean_betweenness", "modularity", "mean_participation",
                     "participation_top10", "participation_bottom10",
                     "pct_within_z_gt0", "pct_within_z_lt0", "mean_path_length",
                     "small_worldness_sigma"]:
            net[kkey] = np.nan
        net["n_modules"] = 0
    return net, node_rows


# thin back-compat wrappers
def network_metrics(W):
    return compute_graph(W, [str(i) for i in range(W.shape[0])])[0]


def node_metrics(W, labels):
    return compute_graph(W, labels)[1]


def _local_efficiency_per_node(G):
    eff = {}
    for u in G.nodes():
        nb = list(G.neighbors(u))
        if len(nb) < 2:
            eff[u] = 0.0; continue
        sub = G.subgraph(nb)
        eff[u] = nx.global_efficiency(sub)
    return eff


def _participation_coeff(W, node_comm):
    """Weighted participation coefficient (MEA-NAP participation_coef_norm base):
    PC_i = 1 - sum_c (S_ic / S_i)^2, using edge WEIGHTS (strength), not binary."""
    n = W.shape[0]
    S = W.sum(1)                                    # weighted degree (strength)
    ncomm = max(node_comm.values()) + 1 if node_comm else 1
    P = np.zeros(n)
    for i in range(n):
        if S[i] <= 0:
            continue
        sc = np.zeros(ncomm)
        for j in range(n):
            if W[i, j] > 0:
                sc[node_comm.get(j, 0)] += W[i, j]
        P[i] = 1.0 - np.sum((sc / S[i]) ** 2)
    return P


def _within_module_z(W, node_comm):
    """Weighted within-module degree z-score (BCT module_degree_zscore): for each
    module, z_i = (Kin_i - mean Kin) / std Kin, Kin = within-module strength."""
    n = W.shape[0]
    z = np.zeros(n)
    for c in set(node_comm.values()):
        members = [i for i in range(n) if node_comm.get(i) == c]
        kin = np.array([sum(W[i, j] for j in members) for i in members])
        mu, sd = kin.mean(), kin.std()
        for idx, i in enumerate(members):
            z[i] = (kin[idx] - mu) / sd if sd > 0 else 0.0
    return z


# ==============================================================================
# BLOCK G  --  network plots
# ==============================================================================
def plot_sttc_matrix(M, title, out_png):
    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(np.nan_to_num(M), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set(title=title, xlabel="Electrode #", ylabel="Electrode #")
    fig.colorbar(im, ax=ax, label="STTC")
    fig.tight_layout(); fig.savefig(out_png, dpi=PLOT_PARAMS["dpi"]); plt.close(fig)


def plot_edge_threshold_check(real_M, frac, surr_stack, W, title, out_png):
    n = real_M.shape[0]
    iu = np.triu_indices(n, k=1)
    real = np.nan_to_num(real_M[iu], nan=0.0); fr = frac[iu]
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9))
    a = ax[0, 0]
    if surr_stack is not None:
        pooled = surr_stack[:, iu[0], iu[1]].ravel()
        pooled = pooled[np.isfinite(pooled)]
        a.hist(pooled, bins=60, density=True, alpha=0.55, color="grey",
               label=f"surrogate null (n={PROB_PARAMS['n_surrogates']})")
    a.hist(real[np.isfinite(real)], bins=60, density=True, alpha=0.65,
           color="firebrick", label="real edges")
    a.set(xlabel="STTC", ylabel="density", title="A. Real vs surrogate STTC")
    a.legend(fontsize=8)
    a = ax[0, 1]
    im = a.imshow(frac, cmap="magma", vmin=0, vmax=1)
    a.set(title="B. Significance fraction", xlabel="Electrode #", ylabel="Electrode #")
    fig.colorbar(im, ax=a, label="frac. surrogates beaten")
    a = ax[1, 0]
    sweep = np.linspace(0.5, 1.0, 51)
    a.plot(sweep, [(fr >= s).sum() for s in sweep], "-o", ms=2.5, color="steelblue")
    a.axvline(PROB_PARAMS["sig_threshold"], color="crimson", ls="--",
              label=f"chosen = {PROB_PARAMS['sig_threshold']}")
    a.set(xlabel="significance threshold", ylabel="# edges retained",
          title="C. Threshold sensitivity"); a.legend(fontsize=8)
    a = ax[1, 1]
    order = np.argsort(real); x = np.arange(len(order))
    if surr_stack is not None:
        sm = np.nanmean(surr_stack[:, iu[0], iu[1]], 0)[order]
        ss = np.nanstd(surr_stack[:, iu[0], iu[1]], 0)[order]
        a.fill_between(x, sm - ss, sm + ss, color="grey", alpha=0.4,
                       label="surrogate mean +/- SD")
    km = (fr >= PROB_PARAMS["sig_threshold"])[order]
    a.plot(x[~km], real[order][~km], ".", ms=3, color="lightsteelblue", label="rejected")
    a.plot(x[km], real[order][km], ".", ms=4, color="firebrick", label="retained")
    a.set(xlabel="edge (sorted by real STTC)", ylabel="STTC",
          title=f"D. Edge retention ({int(km.sum())}/{len(real)} kept)")
    a.legend(fontsize=8)
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(out_png, dpi=PLOT_PARAMS["dpi"]); plt.close(fig)


def _add_node_size_legend(ax, deg, node_sz, loc="lower left"):
    """Legend explaining what the DOT SIZE means (node degree)."""
    d = deg[deg > 0]
    if d.size == 0:
        return None
    lo, hi = int(d.min()), int(d.max())
    ticks = sorted(set([lo, int(round((lo + hi) / 2)), hi]))
    smin, smax = PLOT_PARAMS["graph_node_size"]
    dmax = deg.max() if deg.max() > 0 else 1
    handles = [plt.Line2D([0], [0], marker="o", ls="", markerfacecolor="0.55",
                          markeredgecolor="0.2",
                          markersize=np.sqrt(smin + (smax - smin) * (t / dmax)),
                          label=str(t)) for t in ticks]
    leg = ax.legend(handles=handles, title="node degree\n(dot size)", fontsize=6,
                    title_fontsize=6, loc=loc, framealpha=0.85,
                    labelspacing=1.1, borderpad=0.7, handletextpad=1.0)
    return leg


def _network_graph_ax(fig, ax, W, labels, title, node_color=None,
                      color_label=None, cmap="viridis", role_mode=False,
                      size_legend=True):
    """Draw one network graph onto `ax`. Node SIZE = degree (explained by a
    legend); node COLOUR = a metric or cartography role."""
    if not _HAVE_NX:
        return
    A = (W > NET_PARAMS["binary_edge_min"])
    G = nx.from_numpy_array(np.where(A, W, 0.0))
    order = list(G.nodes())
    pos = {}
    for i, l in enumerate(labels):
        xy = _grid_xy(l)
        pos[i] = (xy[0], -xy[1]) if xy else (np.cos(i), np.sin(i))

    deg = np.array([G.degree(u) for u in order], float)
    active = deg > 0                                    # MEA-NAP plots only these
    smin, smax = PLOT_PARAMS["graph_node_size"]
    node_sz = smin + (smax - smin) * (deg / deg.max() if deg.max() > 0 else deg)

    ews = np.array([G[u][v].get("weight", 0.0) for u, v in G.edges()], float)
    wmin, wmax = PLOT_PARAMS["graph_edge_width"]
    ew = ((wmin + (wmax - wmin) * (ews / ews.max())).tolist()
          if ews.size and ews.max() > 0 else [wmin] * len(ews))

    nx.draw_networkx_edges(G, pos, ax=ax, width=ew, edge_color="0.45",
                           alpha=PLOT_PARAMS["graph_edge_alpha"])
    drawn = [u for u in order if active[u]]             # degree>0 node list
    kw = dict(nodelist=drawn, node_size=node_sz[active],
              linewidths=0.4, edgecolors="w")
    colour_leg = None
    if role_mode and node_color is not None:            # DISCRETE role colours
        arr = np.asarray(node_color)
        cols = [_ROLE_COLORS[int(arr[u])] for u in drawn]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=cols, **kw)
        present = sorted(set(int(arr[u]) for u in drawn) - {0})
        colour_leg = ax.legend(
            handles=[plt.Line2D([0], [0], marker="o", ls="", markersize=7,
                                markerfacecolor=_ROLE_COLORS[r],
                                markeredgecolor="w", label=_ROLE_NAMES[r])
                     for r in present],
            fontsize=6, loc="upper right", framealpha=0.85, title="node role\n(colour)",
            title_fontsize=6)
    elif node_color is not None:                        # CONTINUOUS metric colouring
        vals_all = np.asarray(node_color, float)
        vals = vals_all[active]
        vmax = np.nanmax(vals) if vals.size else 1.0
        vmin = np.nanmin(vals) if vals.size else 0.0
        if not np.isfinite(vmin) or vmax == vmin:       # MEA-NAP edge-case guard
            vmin = vmax - 0.001
        nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_color=vals, cmap=cmap,
                                       vmin=vmin, vmax=vmax, **kw)
        fig.colorbar(nodes, ax=ax, label=color_label, shrink=0.7)
    else:
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color="firebrick", **kw)
    if size_legend:
        sl = _add_node_size_legend(ax, deg, node_sz)
        if colour_leg is not None and sl is not None:
            ax.add_artist(colour_leg)                   # keep BOTH legends
    ax.set_title(title, fontsize=10); ax.axis("off"); ax.margins(0.10)


def plot_network_graph(W, labels, title, out_png, node_color=None,
                       color_label=None, cmap="viridis", role_mode=False):
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    _network_graph_ax(fig, ax, W, labels, title, node_color, color_label, cmap,
                      role_mode)
    fig.tight_layout(); fig.savefig(out_png, dpi=PLOT_PARAMS["dpi"]); plt.close(fig)


# ==============================================================================
# BLOCK G2  --  node cartography (Guimera-Amaral roles) + colored network plots
# ------------------------------------------------------------------------------
# Each node gets a cartography role from its within-module degree z-score and
# participation coefficient using the classic Guimera & Amaral (2005) boundaries
# (MEA-NAP auto-tunes these from the dataset density landscape; these fixed
# boundaries are the standard default, exposed here for tuning).
# ==============================================================================
# ==============================================================================
# BLOCK G2  --  node cartography (matches MEA-NAP NodeCartography.m exactly)
# ------------------------------------------------------------------------------
# Six roles from within-module degree z-score (Z) and participation coefficient
# (PC), with MEA-NAP's default boundaries and colours. Nodes are assigned:
#   1 Peripheral        : Z<=2.5, PC<=0.625
#   2 Non-hub connector : Z<=2.5, 0.625<PC<=0.8
#   3 Non-hub kinless   : Z<=2.5, PC>0.8
#   4 Provincial hub    : Z>2.5,  PC<=0.3
#   5 Connector hub     : Z>2.5,  0.3<PC<=0.75
#   6 Kinless hub       : Z>2.5,  PC>0.75
# (inactive nodes -> role 0, drawn grey, matching MEA-NAP's "group 7").
# ==============================================================================
CARTOGRAPHY_PARAMS = {
    "hubBoundaryWMdDeg":       2.5,     # hub vs non-hub (within-module z)
    "periPartCoef":            0.625,   # peripheral | non-hub connector
    "nonHubconnectorPartCoef": 0.8,     # non-hub connector | non-hub kinless
    "proHubpartCoef":          0.3,     # provincial hub | connector hub
    "connectorHubPartCoef":    0.75,    # connector hub | kinless hub
}
# roles 1..6 (0 reserved for inactive). MEA-NAP colours (green non-hubs, blue hubs)
_ROLE_NAMES = ["inactive", "Peripheral", "Non-hub connector", "Non-hub kinless",
               "Provincial hub", "Connector hub", "Kinless hub"]
_ROLE_COLORS = ["#bfbfbf",                       # 0 inactive (grey)
                (0.800, 0.902, 0.310),           # 1 light green
                (0.580, 0.706, 0.278),           # 2 medium green
                (0.369, 0.435, 0.122),           # 3 dark green
                (0.200, 0.729, 0.949),           # 4 light blue
                (0.078, 0.424, 0.835),           # 5 medium blue
                (0.016, 0.235, 0.498)]           # 6 dark blue


def cartography_role(z, P):
    """MEA-NAP NodeCartography role (1..6). Inactive (no edges) is handled by the
    caller as role 0."""
    c = CARTOGRAPHY_PARAMS
    if z <= c["hubBoundaryWMdDeg"]:
        if P <= c["periPartCoef"]:
            return 1
        if P <= c["nonHubconnectorPartCoef"]:
            return 2
        return 3
    if P <= c["proHubpartCoef"]:
        return 4
    if P <= c["connectorHubPartCoef"]:
        return 5
    return 6


def _cartography_scatter_ax(ax, zs, ps, roles, title):
    """Node cartography diagram exactly as MEA-NAP draws it (NodeCartography.m):
    participation coefficient (x) vs within-module degree z-score (y), dashed
    black role boundaries (no shading), coloured scatter points, data-driven axes."""
    zs, ps, roles = np.asarray(zs, float), np.asarray(ps, float), np.asarray(roles)
    c = CARTOGRAPHY_PARAMS
    hub = c["hubBoundaryWMdDeg"]
    act = roles > 0                                  # only active nodes on the plot
    zsa, psa, ra = zs[act], ps[act], roles[act]
    if psa.size and np.nanmin(psa) != np.nanmax(psa):
        xr = [float(np.nanmin(psa)), float(np.nanmax(psa))]
    else:
        xr = [0.0, 1.0]
    if zsa.size and np.nanmin(zsa) != np.nanmax(zsa):
        zmin, zmax = float(np.nanmin(zsa)), float(np.nanmax(zsa))
        yr = [zmin * 1.1 if zmin < 0 else zmin * 0.9,
              zmax * 1.1 if zmax > 0 else zmax * 0.9]
    else:
        yr = [-2.0, 4.0]
    if yr[0] == yr[1]:
        yr = [yr[0] - 1.0, yr[1] + 1.0]

    ax.plot(xr, [hub, hub], "--k", lw=0.9)           # hub boundary (horizontal)
    ax.plot([c["periPartCoef"]] * 2, [yr[0], hub], "--k", lw=0.9)
    ax.plot([c["nonHubconnectorPartCoef"]] * 2, [yr[0], hub], "--k", lw=0.9)
    ax.plot([c["proHubpartCoef"]] * 2, [hub, yr[1]], "--k", lw=0.9)
    ax.plot([c["connectorHubPartCoef"]] * 2, [hub, yr[1]], "--k", lw=0.9)
    for r in range(1, 7):
        m = ra == r
        if m.any():
            ax.scatter(psa[m], zsa[m], s=18, color=_ROLE_COLORS[r],
                       edgecolors=_ROLE_COLORS[r], label=_ROLE_NAMES[r], zorder=4)
    ax.set(xlim=xr, ylim=yr, xlabel="participation coefficient",
           ylabel="within-module degree z-score", title=title)
    ax.tick_params(direction="out")
    ax.legend(fontsize=5.5, loc="best", framealpha=0.85)


def plot_cartography_scatter(zs, ps, roles, title, out_png):
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    _cartography_scatter_ax(ax, zs, ps, roles, title)
    fig.tight_layout(); fig.savefig(out_png, dpi=PLOT_PARAMS["dpi"]); plt.close(fig)


def plot_role_proportions(roles, title, out_png):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    roles = np.asarray(roles)
    counts = np.array([(roles == r).sum() for r in range(1, 7)], float)
    prop = counts / counts.sum() if counts.sum() else counts
    ax.bar(range(6), prop, color=[_ROLE_COLORS[r] for r in range(1, 7)])
    ax.set_xticks(range(6))
    ax.set_xticklabels(_ROLE_NAMES[1:], fontsize=7, rotation=30, ha="right")
    ax.set(ylabel="Proportion of nodes", title=title)
    fig.tight_layout(); fig.savefig(out_png, dpi=PLOT_PARAMS["dpi"]); plt.close(fig)


# ==============================================================================
# BLOCK H  --  network unit analysis
# ==============================================================================
def _build_params():
    """Params object for the port, populated from CARTOGRAPHY_PARAMS + PARITY_PARAMS."""
    from meanap.params import Params
    p = Params()
    c = CARTOGRAPHY_PARAMS
    p.hub_boundary_wm_d_deg = c["hubBoundaryWMdDeg"]
    p.peri_part_coef = c["periPartCoef"]
    p.non_hub_connector_part_coef = c["nonHubconnectorPartCoef"]
    p.pro_hub_part_coef = c["proHubpartCoef"]
    p.connector_hub_part_coef = c["connectorHubPartCoef"]
    p.min_activity_level = 0.0
    return p


def _sttc_exact_threshold(spike_dict, n_ch, lag_ms, dur, rng):
    """Exact SAND-port STTC + circular-shift surrogate thresholding, but keeping
    the surrogate stack so our edge-threshold-check plot still works. Threshold
    rule matches probabilistic_threshold.adjm_thr (upper tail = ProbThreshTail)."""
    fs = PARITY_PARAMS["fs_hz"]
    rep = PROB_PARAMS["n_surrogates"]; tail = PROB_PARAMS["tail"]
    M = _sttc.get_sttc(spike_dict, n_ch, lag_ms, dur)
    surr = np.empty((rep, n_ch, n_ch), np.float32)
    for r in range(rep):
        synth = _pthr.circular_shift_spikes(spike_dict, n_ch, fs, dur, rng)
        s = _sttc.get_sttc(synth, n_ch, lag_ms, dur)
        np.fill_diagonal(s, 0.0)
        surr[r] = s
    import math
    cutoff = min(max(int(math.ceil((1 - tail) * rep)) - 1, 0), rep - 1)
    thr = np.sort(surr, axis=0)[cutoff]
    W = M.copy(); W[thr > M] = 0.0
    with np.errstate(invalid="ignore"):
        frac = (M[None, ...] >= surr).mean(0)          # for the plot only
    return M, W, frac, surr


def _map_result(res, labels):
    """Map the port's compute_network_metrics result dict to our (net, node_rows,
    active_labels) structure so downstream plots/CSVs keep working."""
    idx = res["activeChannelIndex"]
    alab = [labels[i] for i in idx]
    def arr(k):
        v = res.get(k)
        return np.asarray(v) if v is not None else np.full(len(idx), np.nan)
    ND, NS, BC = arr("ND"), arr("NS"), arr("BC")
    CC, Eloc = arr("CC_raw"), arr("Eloc")
    PC, Z = arr("PC"), arr("Z")
    Ci = arr("Ci"); roles = arr("NdCartDiv")
    node_rows = []
    for j in range(len(idx)):
        node_rows.append({
            "label": alab[j], "degree": float(ND[j]), "strength": float(NS[j]),
            "betweenness": float(BC[j]), "clustering": float(CC[j]),
            "local_efficiency": float(Eloc[j]), "participation": float(PC[j]),
            "within_module_z": float(Z[j]),
            "module": int(Ci[j]) if np.isfinite(Ci[j]) else 0,
            "role": int(roles[j]) if (roles.size and np.isfinite(roles[j])) else 0})
    g = lambda k, d=np.nan: float(res[k]) if k in res else d
    net = {
        "network_size": int(res.get("aN", 0)), "n_edges": int((res.get("adjMsub",
              np.zeros((1, 1))) > 0).sum() / 2),
        "edge_density": g("Dens"), "mean_degree": g("NDmean"),
        "mean_strength": g("NSmean"), "mean_edge_weight": g("sigEdgesMean"),
        "top25_mean_degree": g("NDtop25"), "top10_mean_edge_weight": g("sigEdgesTop10"),
        "clustering_coef": g("CC_rawMean"), "global_efficiency": g("Eglob"),
        "mean_local_efficiency": g("ElocMean"),
        "mean_betweenness": float(np.nanmean(BC)) if BC.size else np.nan,
        "n_modules": int(res.get("nMod", 0)), "modularity": g("Q"),
        "mean_participation": g("PCmean"), "participation_top10": g("PCmeanTop10"),
        "participation_bottom10": g("PCmeanBottom10"),
        "pct_within_z_gt0": g("percentZscoreGreaterThanZero"),
        "pct_within_z_lt0": g("percentZscoreLessThanZero"),
        "mean_path_length": g("PL_raw"), "small_worldness_sigma": g("SW"),
        "small_worldness_omega": g("SWw"),
        "ave_controllability": g("aveControlMean"),
        "modal_controllability": g("modalControlMean")}
    return net, node_rows, alab


def analyze_network(unit, res_activity, lag_ms):
    """One (recording, lag): exact SAND-port STTC + surrogate thresholding, then
    the port's compute_network_metrics (= ExtractNetMet.m) for all network + node
    metrics, modules (consensus clustering), normalized PC/Z, and cartography."""
    labels, spikes, dur = unit["labels"], unit["spikes"], unit["duration"]
    n_ch = len(labels)
    spike_dict = {i: np.asarray(spikes[i], float) for i in range(n_ch)}
    spike_counts = np.array([len(spikes[i]) for i in range(n_ch)], float)
    rng = np.random.default_rng(PROB_PARAMS["random_seed"])

    if PARITY_PARAMS["sttc_backend"] == "gpu":
        asp = [spike_dict[i] for i in range(n_ch)]
        M, W, frac, surr = probabilistic_threshold(
            asp, lag_ms / 1000.0, dur,
            keep_surr=PROB_PARAMS["keep_surrogates_for_plot"])
    else:
        M, W, frac, surr = _sttc_exact_threshold(spike_dict, n_ch, lag_ms, dur, rng)

    res = compute_network_metrics(
        W, spike_counts, dur, min_activity_level=0.0,
        min_nodes=PARITY_PARAMS["min_nodes"], params=_build_params(), rng=rng)
    if res.get("aN", 0) < 3:
        return None
    net, nodes, alab = _map_result(res, labels)
    # restrict the plotted matrix/surrogates to the active-node subset
    aidx = res["activeChannelIndex"]
    Msub = M[np.ix_(aidx, aidx)]; Wsub = W[np.ix_(aidx, aidx)]
    fracsub = frac[np.ix_(aidx, aidx)]
    surrsub = surr[:, aidx][:, :, aidx] if surr is not None else None
    return {"labels": alab, "M": Msub, "W": Wsub, "frac": fracsub, "surr": surrsub,
            "net": net, "nodes": nodes, "lag_ms": lag_ms,
            "pc": np.array([nd["participation"] for nd in nodes]),
            "z": np.array([nd["within_module_z"] for nd in nodes])}


# ------------------------------------------------------------------------------
# analyze_network(), split for parallelism
# ------------------------------------------------------------------------------
# _network_connectivity() is the GPU/STTC + surrogate part (kept serial: one GPU);
# _metrics_worker() is the CPU-heavy compute_network_metrics part (fanned out over
# processes across the region x lag grid); _finalize_network() stitches a worker's
# result back onto the connectivity matrices. Together they reproduce exactly what
# analyze_network() returns for the GPU backend -- see PERF_PARAMS.
# ------------------------------------------------------------------------------
def _network_connectivity(unit, lag_ms):
    """STTC + surrogate thresholding for ONE (recording, lag). Returns the
    connectivity matrices plus the inputs compute_network_metrics needs."""
    labels, spikes, dur = unit["labels"], unit["spikes"], unit["duration"]
    n_ch = len(labels)
    spike_dict = {i: np.asarray(spikes[i], float) for i in range(n_ch)}
    spike_counts = np.array([len(spikes[i]) for i in range(n_ch)], float)
    if PARITY_PARAMS["sttc_backend"] == "gpu":
        asp = [spike_dict[i] for i in range(n_ch)]
        M, W, frac, surr = probabilistic_threshold(
            asp, lag_ms / 1000.0, dur,
            keep_surr=PROB_PARAMS["keep_surrogates_for_plot"])
    else:
        rng = np.random.default_rng(PROB_PARAMS["random_seed"])
        M, W, frac, surr = _sttc_exact_threshold(spike_dict, n_ch, lag_ms, dur, rng)
    return {"labels": labels, "dur": dur, "spike_counts": spike_counts,
            "M": M, "W": W, "frac": frac, "surr": surr, "lag_ms": lag_ms}


def _metrics_worker(payload):
    """Picklable process-pool worker: the port's compute_network_metrics for ONE
    (region x lag). A fresh default_rng(seed) is created here so the result is
    identical to the serial GPU-backend path and independent of scheduling order.
    Must stay importable at module scope for spawn-based pools (Windows)."""
    W, spike_counts, dur, min_nodes, seed = payload
    return compute_network_metrics(
        W, spike_counts, dur, min_activity_level=0.0, min_nodes=min_nodes,
        params=_build_params(), rng=np.random.default_rng(seed))


def _finalize_network(conn, res):
    """Map a compute_network_metrics result onto the connectivity matrices
    (active-node subsetting) -> the same nr dict analyze_network returns, or None
    when fewer than 3 nodes are active."""
    if res.get("aN", 0) < 3:
        return None
    net, nodes, alab = _map_result(res, conn["labels"])
    aidx = res["activeChannelIndex"]
    Msub = conn["M"][np.ix_(aidx, aidx)]
    Wsub = conn["W"][np.ix_(aidx, aidx)]
    fracsub = conn["frac"][np.ix_(aidx, aidx)]
    surr = conn["surr"]
    surrsub = surr[:, aidx][:, :, aidx] if surr is not None else None
    return {"labels": alab, "M": Msub, "W": Wsub, "frac": fracsub, "surr": surrsub,
            "net": net, "nodes": nodes, "lag_ms": conn["lag_ms"],
            "pc": np.array([nd["participation"] for nd in nodes]),
            "z": np.array([nd["within_module_z"] for nd in nodes])}


def make_network_plots_multi(stem, units, netres_by_region, lag, out_dir):
    """All network plots for ONE FILE at ONE lag, one panel per region."""
    os.makedirs(out_dir, exist_ok=True)
    us = [u for u in _sorted_regions(units)
          if netres_by_region.get(u["region"]) is not None]
    if not us:
        return
    P = lambda f: os.path.join(out_dir, f)
    sub = f"{stem} | lag {lag} ms"
    R = lambda u: netres_by_region[u["region"]]

    # STTC matrices
    fig, axes = _panels_fig(len(us), 4.4, 4.4)
    im = None
    for ax, u in zip(axes, us):
        im = ax.imshow(np.nan_to_num(R(u)["M"]), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set(title=u["region"], xlabel="Electrode #", ylabel="Electrode #")
    fig.colorbar(im, ax=list(axes), label="STTC", shrink=0.85)
    fig.suptitle(f"STTC — {sub}", fontsize=12)
    fig.savefig(P("sttc_matrix.png"), dpi=PLOT_PARAMS["dpi"]); plt.close(fig)

    # network graphs (plain + node-metric coloured + cartography)
    for key, lbl, cm, fname in [
            (None, None, None, "network_graph.png"),
            ("betweenness", "betweenness centrality", "plasma",
             "network_betweenness.png"),
            ("participation", "participation coefficient", "viridis",
             "network_participation.png"),
            ("local_efficiency", "local efficiency", "cividis",
             "network_local_efficiency.png"),
            ("role", None, None, "network_cartography.png")]:
        fig, axes = _panels_fig(len(us), 5.6, 5.4)
        for ax, u in zip(axes, us):
            nr = R(u)
            nodes = nr["nodes"]
            if key is None:
                _network_graph_ax(fig, ax, nr["W"], nr["labels"], u["region"])
            elif key == "role":
                _network_graph_ax(fig, ax, nr["W"], nr["labels"], u["region"],
                                  node_color=np.array([nd["role"] for nd in nodes]),
                                  role_mode=True)
            else:
                _network_graph_ax(fig, ax, nr["W"], nr["labels"], u["region"],
                                  node_color=[nd[key] for nd in nodes],
                                  color_label=lbl, cmap=cm)
        ttl = ("Network" if key is None else
               "Node cartography" if key == "role" else lbl.capitalize())
        fig.suptitle(f"{ttl} — {sub}", fontsize=12)
        fig.savefig(P(fname), dpi=PLOT_PARAMS["dpi"]); plt.close(fig)

    # cartography scatter
    fig, axes = _panels_fig(len(us), 5.0, 4.6)
    for ax, u in zip(axes, us):
        nodes = R(u)["nodes"]
        _cartography_scatter_ax(ax, [nd["within_module_z"] for nd in nodes],
                                [nd["participation"] for nd in nodes],
                                np.array([nd["role"] for nd in nodes]), u["region"])
    fig.suptitle(f"Node cartography — {sub}", fontsize=12)
    fig.savefig(P("cartography_scatter.png"), dpi=PLOT_PARAMS["dpi"]); plt.close(fig)

    # cartography role proportions
    fig, axes = _panels_fig(len(us), 4.0, 4.0)
    for ax, u in zip(axes, us):
        roles = np.asarray([nd["role"] for nd in R(u)["nodes"]])
        counts = np.array([(roles == r).sum() for r in range(1, 7)], float)
        prop = counts / counts.sum() if counts.sum() else counts
        ax.bar(range(6), prop, color=[_ROLE_COLORS[r] for r in range(1, 7)])
        ax.set_xticks(range(6))
        ax.set_xticklabels(_ROLE_NAMES[1:], fontsize=6, rotation=30, ha="right")
        ax.set(ylabel="Proportion of nodes", title=u["region"], ylim=(0, 1))
    fig.suptitle(f"Cartography role proportions — {sub}", fontsize=12)
    fig.savefig(P("cartography_proportions.png"), dpi=PLOT_PARAMS["dpi"])
    plt.close(fig)

    # edge-threshold diagnostic stays per-region (it is already a 4-panel figure)
    for u in us:
        nr = R(u)
        plot_edge_threshold_check(nr["M"], nr["frac"], nr["surr"], nr["W"],
                                  f"Edge threshold check — {stem} {u['region']} "
                                  f"| lag {lag} ms",
                                  P(f"edge_threshold_check_{u['region']}.png"))


# ==============================================================================
# BLOCK I  --  Stage 2c: node-by-age / node-by-group / across-region comparisons
# ------------------------------------------------------------------------------
# Reads the accumulated per-node metrics and produces grouped half-violin
# comparisons into PLOT_DIR/comparisons/:
#   node_by_age_<region>.png    x = age (DIV), violins split by GROUP
#   node_by_group_<region>.png  x = group,     violins split by AGE
#   node_across_region.png      x = region (pre/stim/post), split by GROUP
# Each is a multi-panel figure, one panel per node metric.
# ==============================================================================
COMPARE_PARAMS = {
    "node_metrics": ["degree", "strength", "betweenness", "clustering",
                     "local_efficiency", "participation", "within_module_z"],
    # Step 2B recording-level (activity / network-burst) metrics -- lag-free
    "activity_recording_metrics": [
        "n_active", "mean_FR_hz", "median_FR_hz", "network_burst_rate_per_min",
        "mean_electrodes_in_nb", "mean_nb_length_s", "isi_within_nb_ms",
        "isi_outside_nb_ms", "cov_ibi", "frac_spikes_in_nb"],
    # Step 4B recording-level graph metrics -- per STTC lag
    "network_recording_metrics": [
        "network_size", "edge_density", "mean_degree", "top25_mean_degree",
        "mean_edge_weight", "top10_mean_edge_weight", "mean_strength",
        "mean_local_efficiency", "clustering_coef", "n_modules", "modularity",
        "pct_within_z_gt0", "pct_within_z_lt0", "mean_path_length",
        "mean_participation", "participation_top10", "participation_bottom10",
        "global_efficiency", "small_worldness_sigma", "small_worldness_omega",
        "ave_controllability", "modal_controllability"],
    "region_order": ["pre", "stim", "post"],
    "out_subdir":   "Comparisons",
}


def _grouped_violin(ax, get_vals, x_cats, hues, ylabel, title, hue_title):
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(hues), 1)))
    nh = len(hues); width = 0.8 / max(nh, 1)
    for hi, h in enumerate(hues):
        for xi, xc in enumerate(x_cats):
            v = np.asarray(get_vals(xc, h), float)
            v = v[np.isfinite(v)]
            pos = xi + (hi - (nh - 1) / 2) * width
            if v.size >= 2:
                parts = ax.violinplot([v], positions=[pos], widths=width * 0.9,
                                      showmeans=True, showextrema=False)
                for b in parts["bodies"]:
                    b.set_facecolor(colors[hi]); b.set_alpha(0.6)
                    b.set_edgecolor("none")
            elif v.size == 1:
                ax.scatter([pos], v, s=14, color=colors[hi])
    ax.set_xticks(range(len(x_cats)))
    ax.set_xticklabels([str(x) for x in x_cats], fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9); ax.set_title(title, fontsize=10)
    ax.margins(x=0.06, y=0.14)          # headroom so violin tails aren't clipped
    handles = [plt.Line2D([0], [0], marker="s", ls="", markerfacecolor=colors[hi],
                          markeredgecolor="none", label=str(h), markersize=7)
               for hi, h in enumerate(hues)]
    ax.legend(handles=handles, fontsize=6, title=hue_title, title_fontsize=7,
              loc="best", framealpha=0.6)


def _panel_grid(nmetrics):
    ncol = 3
    nrow = int(np.ceil(nmetrics / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow),
                             squeeze=False, layout="constrained")
    return fig, axes.ravel()


def _node_comparison(df, xcol, huecol, title_prefix, out_png):
    metrics = [m for m in COMPARE_PARAMS["node_metrics"] if m in df.columns]
    x_cats = orig._cat_sorted(df[xcol])
    hues = orig._cat_sorted(df[huecol])
    fig, axes = _panel_grid(len(metrics))
    for ax, met in zip(axes, metrics):
        _grouped_violin(
            ax, lambda xc, h: df[(df[xcol] == xc) & (df[huecol] == h)][met].values,
            x_cats, hues, met, met, huecol)
        ax.set_xlabel(xcol, fontsize=9)
    for ax in axes[len(metrics):]:
        ax.axis("off")
    fig.suptitle(title_prefix, fontsize=12)
    fig.savefig(out_png, dpi=PLOT_PARAMS["dpi"]); plt.close(fig)


def _grouped_points_mean(ax, df, xcol, ycol, huecol, x_cats, hues):
    """MEA-NAP-style: jittered per-recording scatter + black mean +/- s.e.m."""
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(hues), 1)))
    nh = len(hues); width = 0.8 / max(nh, 1)
    for hi, h in enumerate(hues):
        for xi, xc in enumerate(x_cats):
            v = df[(df[xcol] == xc) & (df[huecol] == h)][ycol].values
            v = v[np.isfinite(v)]
            pos = xi + (hi - (nh - 1) / 2) * width
            if v.size:
                ax.scatter(np.random.normal(pos, width * 0.09, v.size), v, s=11,
                           color=colors[hi], alpha=0.5, edgecolors="none")
                se = v.std() / np.sqrt(v.size) if v.size > 1 else 0.0
                ax.errorbar([pos], [v.mean()], yerr=[se], fmt="o", color="k",
                            ms=4, capsize=3, zorder=5)
    ax.set_xticks(range(len(x_cats)))
    ax.set_xticklabels([str(x) for x in x_cats], fontsize=8)
    ax.margins(x=0.06, y=0.15)
    handles = [plt.Line2D([0], [0], marker="s", ls="", markerfacecolor=colors[hi],
                          markeredgecolor="none", label=str(h), markersize=7)
               for hi, h in enumerate(hues)]
    ax.legend(handles=handles, fontsize=6, title=huecol, title_fontsize=7,
              loc="best", framealpha=0.6)


def _dist_comparison(df, metrics, xcol, huecol, title, out_png):
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        return
    x_cats = orig._cat_sorted(df[xcol])
    hues = orig._cat_sorted(df[huecol])
    fig, axes = _panel_grid(len(metrics))
    for ax, met in zip(axes, metrics):
        _grouped_points_mean(ax, df, xcol, met, huecol, x_cats, hues)
        ax.set_title(met, fontsize=9); ax.set_xlabel(xcol, fontsize=8)
    for ax in axes[len(metrics):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_png, dpi=PLOT_PARAMS["dpi"]); plt.close(fig)


def _metrics_by_lag(df, metrics, huecol, title, out_png):
    """Line graphs of each metric vs STTC lag, mean +/- s.e.m. split by hue."""
    metrics = [m for m in metrics if m in df.columns]
    lags = sorted(df["lag_ms"].unique())
    hues = orig._cat_sorted(df[huecol])
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(hues), 1)))
    fig, axes = _panel_grid(len(metrics))
    for ax, met in zip(axes, metrics):
        for hi, h in enumerate(hues):
            mu, se = [], []
            for lg in lags:
                v = df[(df["lag_ms"] == lg) & (df[huecol] == h)][met].values
                v = v[np.isfinite(v)]
                mu.append(v.mean() if v.size else np.nan)
                se.append(v.std() / np.sqrt(v.size) if v.size > 1 else 0.0)
            mu, se = np.array(mu), np.array(se)
            ax.plot(lags, mu, "-o", color=colors[hi], ms=3, label=str(h))
            ax.fill_between(lags, mu - se, mu + se, color=colors[hi], alpha=0.2)
        ax.set_title(met, fontsize=9); ax.set_xlabel("STTC lag (ms)", fontsize=8)
        ax.margins(y=0.12)
    axes[0].legend(fontsize=6, title=huecol)
    for ax in axes[len(metrics):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_png, dpi=PLOT_PARAMS["dpi"]); plt.close(fig)


def _roles_by_lag(node_df, title, out_png):
    lags = sorted(node_df["lag_ms"].unique())
    act = node_df[node_df["role"] > 0]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for role in range(1, 7):
        prop = [np.mean(act[act["lag_ms"] == lg]["role"].values == role)
                if (act["lag_ms"] == lg).any() else np.nan for lg in lags]
        ax.plot(lags, prop, "-o", color=_ROLE_COLORS[role], ms=3,
                label=_ROLE_NAMES[role])
    ax.set(xlabel="STTC lag (ms)", ylabel="Proportion of active nodes", title=title)
    ax.legend(fontsize=6, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout(); fig.savefig(out_png, dpi=PLOT_PARAMS["dpi"]); plt.close(fig)


def _cartography_proportions_by(df, xcol, title, out_png):
    """Stacked-bar cartography role proportions (active nodes) across x (group/age)."""
    df = df[df["role"] > 0]
    x_cats = orig._cat_sorted(df[xcol])
    props = np.zeros((len(x_cats), 6))
    for xi, xc in enumerate(x_cats):
        r = df[df[xcol] == xc]["role"].values
        for k, role in enumerate(range(1, 7)):
            props[xi, k] = np.mean(r == role) if len(r) else 0.0
    fig, ax = plt.subplots(figsize=(2.0 + 1.1 * len(x_cats), 4.6))
    bottom = np.zeros(len(x_cats))
    for k, role in enumerate(range(1, 7)):
        ax.bar(range(len(x_cats)), props[:, k], bottom=bottom,
               color=_ROLE_COLORS[role], label=_ROLE_NAMES[role])
        bottom += props[:, k]
    ax.set_xticks(range(len(x_cats)))
    ax.set_xticklabels([str(x) for x in x_cats])
    ax.set(xlabel=xcol, ylabel="Proportion of active nodes", title=title, ylim=(0, 1))
    ax.legend(fontsize=6, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout(); fig.savefig(out_png, dpi=PLOT_PARAMS["dpi"]); plt.close(fig)


def _present_regions(df):
    return [r for r in COMPARE_PARAMS["region_order"] if r in df["region"].unique()]


def run_stage2c(node_df, net_df=None, activity_df=None):
    """Stage 2c comparisons. Node + network + cartography comparisons are produced
    PER STTC LAG (comparisons/lag<X>ms/); recording-level activity/network-burst
    comparisons (Step 2B) are lag-free; graph-metrics-by-lag and cartography-by-lag
    summarize across lags."""
    import pandas as pd
    base = os.path.join(io.CONFIG["PLOT_DIR"], COMPARE_PARAMS["out_subdir"])
    os.makedirs(base, exist_ok=True)

    # --- Step 2B: recording-level activity / network-burst comparisons (lag-free)
    if activity_df is not None and len(activity_df):
        for region in _present_regions(activity_df):
            sub = activity_df[activity_df["region"] == region]
            _dist_comparison(sub, COMPARE_PARAMS["activity_recording_metrics"],
                             "age", "group", f"Recordings by group (activity) — {region}",
                             os.path.join(base, f"recordings_by_group_activity_{region}.png"))
            _dist_comparison(sub, COMPARE_PARAMS["activity_recording_metrics"],
                             "group", "age", f"Recordings by age (activity) — {region}",
                             os.path.join(base, f"recordings_by_age_activity_{region}.png"))

    # --- per-lag node/cartography comparisons
    if node_df is not None and len(node_df):
        for lag in sorted(node_df["lag_ms"].unique()):
            ldir = os.path.join(base, f"lag{lag}ms"); os.makedirs(ldir, exist_ok=True)
            ndf = node_df[node_df["lag_ms"] == lag]
            for region in _present_regions(ndf):
                sub = ndf[ndf["region"] == region]
                _node_comparison(sub, "age", "group",
                                 f"Node by age — {region} lag{lag}ms",
                                 os.path.join(ldir, f"node_by_age_{region}.png"))
                _node_comparison(sub, "group", "age",
                                 f"Node by group — {region} lag{lag}ms",
                                 os.path.join(ldir, f"node_by_group_{region}.png"))
                plot_cartography_scatter(sub["within_module_z"].values,
                    sub["participation"].values, sub["role"].values,
                    f"Density landscape — {region} lag{lag}ms",
                    os.path.join(ldir, f"density_landscape_{region}.png"))
                _cartography_proportions_by(sub, "group",
                    f"Cartography by group — {region} lag{lag}ms",
                    os.path.join(ldir, f"cartography_by_group_{region}.png"))
                _cartography_proportions_by(sub, "age",
                    f"Cartography by age — {region} lag{lag}ms",
                    os.path.join(ldir, f"cartography_by_age_{region}.png"))
            present = _present_regions(ndf)
            if len(present) >= 2:
                dfx = ndf.copy()
                dfx["region"] = pd.Categorical(dfx["region"], categories=present,
                                               ordered=True)
                _node_comparison(dfx, "region", "group",
                                 f"Node across regions — lag{lag}ms",
                                 os.path.join(ldir, "node_across_region.png"))

    # --- Step 4B: recording-level graph metrics per lag + by-lag summaries
    if net_df is not None and len(net_df):
        for lag in sorted(net_df["lag_ms"].unique()):
            ldir = os.path.join(base, f"lag{lag}ms"); os.makedirs(ldir, exist_ok=True)
            ndf = net_df[net_df["lag_ms"] == lag]
            for region in _present_regions(ndf):
                sub = ndf[ndf["region"] == region]
                _dist_comparison(sub, COMPARE_PARAMS["network_recording_metrics"],
                    "age", "group", f"Recordings by group (network) — {region} lag{lag}ms",
                    os.path.join(ldir, f"recordings_by_group_network_{region}.png"))
                _dist_comparison(sub, COMPARE_PARAMS["network_recording_metrics"],
                    "group", "age", f"Recordings by age (network) — {region} lag{lag}ms",
                    os.path.join(ldir, f"recordings_by_age_network_{region}.png"))
        for region in _present_regions(net_df):
            sub = net_df[net_df["region"] == region]
            _metrics_by_lag(sub, COMPARE_PARAMS["network_recording_metrics"], "age",
                            f"Graph metrics by lag — {region}",
                            os.path.join(base, f"graph_metrics_by_lag_{region}.png"))

    if node_df is not None and len(node_df):
        for region in _present_regions(node_df):
            _roles_by_lag(node_df[node_df["region"] == region],
                          f"Node cartography by lag — {region}",
                          os.path.join(base, f"cartography_by_lag_{region}.png"))


# ==============================================================================
# COMPUTE-LEVEL GATING
# ------------------------------------------------------------------------------
# A figure's checkbox (orig.PLOT_ENABLE) now skips the WORK, not just the PNG. The
# expensive network stage (STTC + surrogates + graph metrics + cartography, the
# dominant Stage-2 cost) is skipped ENTIRELY when no figure/output that needs it is
# enabled — so an activity/PSD-only run no longer pays for STTC.
_NETWORK_FIG_KEYS = (
    "edge_threshold_check", "connectivity_stats", "network_plot",
    "network_NodeDegree", "network_NodeStrength", "network_BetweennessCentrality",
    "network_ParticipationCoefficient", "network_WithinModuleZ",
    "network_LocalEfficiency", "node_cartography",
    "network_by_group", "network_by_age", "node_by_group", "node_by_age",
    "graph_metrics_by_lag", "cartography_by_lag", "density_landscape",
)


def _need_network():
    """True if any enabled figure/output requires the STTC + network-metrics stage."""
    return any(orig.PLOT_ENABLE.get(k, True) for k in _NETWORK_FIG_KEYS)


# ==============================================================================
# DRIVER  --  Stage 2 (activity + multi-lag network) over all batch units
# ==============================================================================
def run_stage2():
    import pandas as pd
    import time
    from collections import OrderedDict
    csv_path = os.path.join(io.CONFIG["DATA_DIR"], io.CONFIG["BATCH_CSV_NAME"])
    rows = io.parse_batch(csv_path)
    lags = STTC_PARAMS["lags_ms"]
    activity_rows, node_rows, net_rows = [], [], []
    sweep_rows = []                                   # per-file measured sweep speed

    # Group the windowed units BY FILE so each file gets ONE output folder and
    # every plot carries one panel per region (pre/stim/post).
    by_stem = OrderedDict()
    for unit in io.iter_batch_units(rows):
        by_stem.setdefault(unit["stem"], []).append(unit)

    need_net = bool(STAGE2_SECTIONS["network"])      # section-level gate (Outputs panel)
    if not need_net:
        io._log("[compute] network stage SKIPPED — no network/node figures enabled; "
                "STTC + graph metrics and node/network CSVs will not be produced. "
                "Enable any Network/Node figure to turn it back on.")

    _check_numba()                                   # loud if the 10-30x fallback
    import concurrent.futures as _cf                  # would run unnoticed
    pool = None
    if (need_net and PERF_PARAMS["parallel_network_metrics"]
            and PARITY_PARAMS["sttc_backend"] == "gpu" and by_stem):
        try:
            # Only len(lags) x regions metric jobs run per file, and the pool
            # barriers per file, so never spawn more workers than that -- a big
            # max_workers just pays a slow spawn/import storm for idle processes.
            n_jobs_max = max((len(lags) * len(u) for u in by_stem.values()),
                             default=1)
            mw = PERF_PARAMS["max_workers"] or min(
                max(1, (os.cpu_count() or 2) - 1), n_jobs_max, 12)
            pool = _cf.ProcessPoolExecutor(max_workers=mw)
            io._log(f"  [network metrics] parallel across {mw} process(es)")
        except Exception as e:
            io._log(f"  [network metrics] process pool unavailable ({e!r}); serial")
            pool = None

    meanap_root, meanap_group_max, meanap_elec_rows = None, {}, []
    if STAGE2_SECTIONS["meanap"]:
        try:
            meanap_root = orig.output_root(io.CONFIG["PLOT_DIR"])
            groups = sorted({u["group"] for us in by_stem.values() for u in us})
            orig.create_output_folders(meanap_root, groups)
            meanap_group_max = orig.group_max_firing_rate(by_stem)   # per-group maxFR
            io._log(f"  [MEA-NAP original plots] writing to {meanap_root}")
        except Exception as e:
            io._log(f"  [MEA-NAP original plots] setup failed ({e!r}); skipping")
            meanap_root = None

    gui = progress_gui.make(list(by_stem.keys()), "MEA Analyzer — Stage 2 analysis",
                            f"Activity + network ({len(lags)} STTC lags)")
    file_times = []                                  # (stem, seconds) per-file analysis
    t_loop0 = time.perf_counter()
    try:
        for stem, units in by_stem.items():
            io.check_abort()
            units = _sorted_regions(units)
            n_steps = len(units) + (len(units) * len(lags) if need_net else 0)
            step = 0
            gui.start(stem, "activity analysis...")
            _t_file = time.perf_counter()
            base_dir = os.path.join(io.CONFIG["PLOT_DIR"], "Recordings", stem)

            res_by_region = {}
            for u in units:
                res_by_region[u["region"]] = analyze_activity(u)
                activity_rows.append(res_by_region[u["region"]]["summary"])
                if meanap_root:                       # 2A neuronal-activity figures
                    try:
                        gmax = meanap_group_max.get(u["group"], 0.1)
                        orig.render_activity_recording(
                            u, res_by_region[u["region"]], meanap_root, gmax)
                        meanap_elec_rows.extend(orig.electrode_activity_row(
                            u, res_by_region[u["region"]]))   # per-electrode -> 2B
                    except Exception as e:
                        io._log(f"  [MEA-NAP 2A warn] {stem} {u['region']}: {e!r}")
                step += 1
                gui.set(stem, step / n_steps, note=f"activity: {u['region']}")
            if STAGE2_SECTIONS["activity"]:
                gui.set(stem, step / n_steps, note="activity plots…")
                make_activity_plots_multi(stem, units, res_by_region, base_dir)

            # ---- PSD + signal-propagation figures into Plots/<stem>/ ----
            # Optionally read the laser ON times from the MCS Digital Event Recorder
            # and hand Stage-2 propagation a stimulus-locked event source.
            laser_by_region = None
            if digital.DIGITAL_PARAMS.get("use_in_stage2", True):
                try:
                    fpath = os.path.join(io.CONFIG["DATA_DIR"], stem + ".h5")
                    if os.path.exists(fpath):
                        ev = digital.read_digital_events(
                            fpath, io.CONFIG.get("STREAM_INDEX"))
                        # sweep speed = trigger rate (same pattern across groups)
                        sm = digital.sweep_metrics(ev)
                        sweep_rows.append({
                            "stem": stem, "group": units[0]["group"],
                            "n_triggers": sm["n_triggers"],
                            "trigger_hz": sm["trigger_hz"],
                            "median_isi_ms": (sm["median_isi_s"] * 1000.0
                                              if sm["median_isi_s"] == sm["median_isi_s"]
                                              else float("nan")),
                            "sweep_mm_s": digital.sweep_speed_mm_s(sm["trigger_hz"]),
                            "source": ev.get("source")})
                        if ev.get("source"):
                            laser_by_region = {
                                u["region"]: digital.events_in_window(
                                    ev, u["win"][0], u["win"][1]) for u in units}
                            io._log(f"  [digital] {stem}: laser events from "
                                    f"{ev['source']} ({ev['on_s'].size} ON, "
                                    f"{sm['trigger_hz']:.3f} Hz)")
                except Exception as e:
                    io._log(f"  [digital] {stem}: read failed ({e!r})")
            if STAGE2_SECTIONS["spectral"]:
                gui.set(stem, step / n_steps, note="PSD + propagation…")
                try:
                    spprop.render_spectral_propagation(
                        stem, units, res_by_region, base_dir, laser_by_region)
                except Exception as e:
                    io._log(f"  [spectral/propagation warn] {stem}: {e!r}")

            # ---- network: STTC/surrogates (serial, GPU) then metrics ----
            # Rows are accumulated locally and only committed once the whole file
            # succeeds, so a mid-file parallel failure can cleanly retry serially
            # without double-counting.
            region_unit = {u["region"]: u for u in units}
            netres_by_lag = {lag: {} for lag in lags}
            file_net_rows, file_node_rows = [], []
            step_before = step
            ok_par = False
            if pool is not None:
                try:
                    keys = [(lag, u["region"]) for lag in lags for u in units]
                    conns = {}
                    for lag, region in keys:                 # GPU/STTC, serial --
                        gui.set(stem, step / n_steps,        # show live progress so
                                note=f"connectivity: {region} @ {lag} ms")  # this
                        conns[(lag, region)] = _network_connectivity(  # phase never
                            region_unit[region], lag)                   # looks hung
                    payloads = [(conns[k]["W"], conns[k]["spike_counts"],
                                 conns[k]["dur"], PARITY_PARAMS["min_nodes"],
                                 PROB_PARAMS["random_seed"]) for k in keys]
                    for k, res in zip(keys, pool.map(_metrics_worker, payloads)):
                        lag, region = k
                        nr = _finalize_network(conns[k], res)
                        netres_by_lag[lag][region] = nr
                        step += 1
                        gui.set(stem, step / n_steps,
                                note=f"network: {region} @ {lag} ms")
                        if nr is None:
                            continue
                        u = region_unit[region]
                        meta = {"stem": stem, "region": region, "group": u["group"],
                                "age": u["age"], "lag_ms": lag}
                        file_net_rows.append({**meta, **nr["net"]})
                        for nd in nr["nodes"]:
                            file_node_rows.append({**meta, **nd})
                        if meanap_root:               # 4A individual-network figures
                            try:
                                orig.render_network_recording(
                                    nr, u["group"], stem, region, meanap_root, lag)
                            except Exception as e:
                                io._log(f"  [MEA-NAP 4A warn] {stem} {region} "
                                        f"{lag}ms: {e!r}")
                    ok_par = True
                except Exception as e:
                    io._log(f"  [network parallel path failed on {stem}: {e!r}; "
                            f"falling back to serial]")
                    netres_by_lag = {lag: {} for lag in lags}
                    file_net_rows, file_node_rows = [], []
                    step = step_before

            if not ok_par and need_net:                        # serial fallback
                for lag in lags:
                    for u in units:
                        nr = analyze_network(u, res_by_region[u["region"]], lag)
                        netres_by_lag[lag][u["region"]] = nr
                        step += 1
                        gui.set(stem, step / n_steps,
                                note=f"network: {u['region']} @ {lag} ms")
                        if nr is None:
                            continue
                        meta = {"stem": stem, "region": u["region"],
                                "group": u["group"], "age": u["age"], "lag_ms": lag}
                        file_net_rows.append({**meta, **nr["net"]})
                        for nd in nr["nodes"]:
                            file_node_rows.append({**meta, **nd})
                        if meanap_root:               # 4A individual-network figures
                            try:
                                orig.render_network_recording(
                                    nr, u["group"], stem, u["region"], meanap_root, lag)
                            except Exception as e:
                                io._log(f"  [MEA-NAP 4A warn] {stem} {u['region']} "
                                        f"{lag}ms: {e!r}")

            net_rows.extend(file_net_rows)
            node_rows.extend(file_node_rows)
            for lag in lags:
                if any(v is not None for v in netres_by_lag[lag].values()):
                    make_network_plots_multi(stem, units, netres_by_lag[lag], lag,
                                             os.path.join(base_dir, f"lag{lag}ms"))
            gui.done(stem, f"{len(units)} region(s), {len(lags)} lag(s)")
            file_times.append((stem, time.perf_counter() - _t_file))
    finally:
        gui.close()
        if pool is not None:
            pool.shutdown(wait=True)
    t_loop1 = time.perf_counter()                     # end of the per-file analysis

    P = os.path.join(io.CONFIG["PLOT_DIR"], "Tables")   # summary CSVs bucket
    os.makedirs(P, exist_ok=True)
    activity_df = pd.DataFrame(activity_rows) if activity_rows else None
    node_df = pd.DataFrame(node_rows) if node_rows else None
    net_df = pd.DataFrame(net_rows) if net_rows else None
    if activity_df is not None:
        activity_df.to_csv(os.path.join(P, "activity_summary.csv"), index=False)
    if node_df is not None:
        node_df.to_csv(os.path.join(P, "node_metrics.csv"), index=False)
    if net_df is not None:
        net_df.to_csv(os.path.join(P, "network_metrics.csv"), index=False)
        # diagnostic: flag sparse/degenerate networks (cause of uniform node plots)
        try:
            g = net_df.groupby(["region", "lag_ms"]).agg(
                density=("edge_density", "mean"),
                size=("network_size", "mean"),
                modules=("n_modules", "mean")).reset_index()
            print("\n[network density diagnostic] mean per region x lag:")
            for _, r in g.iterrows():
                flag = "  <-- SPARSE (node metrics will be near-uniform)" \
                    if r["density"] < 0.03 else ""
                print(f"  {r['region']:>5} lag{int(r['lag_ms'])}ms: "
                      f"density={r['density']:.3f} size={r['size']:.1f} "
                      f"modules={r['modules']:.1f}{flag}")
        except Exception:
            pass
    # --- measured sweep speed (trigger rate) per file + per group; order groups
    #     by speed so the by-group figures read slow -> fast (Control first) ---
    group_order = None
    if sweep_rows:
        try:
            sweep_df = pd.DataFrame(sweep_rows)
            sweep_df.to_csv(os.path.join(P, "sweep_speed_summary.csv"), index=False)
            gspeed = sweep_df.groupby("group")["trigger_hz"].median()
            laser_g = gspeed.dropna().sort_values().index.tolist()
            nolaser = [g for g in gspeed.index if g not in laser_g]
            group_order = nolaser + laser_g          # Control/no-laser first, then slow->fast
            io._log("[sweep speed] measured trigger rate per group:")
            for g in group_order:
                v = gspeed.get(g, float("nan"))
                io._log(f"    {g}: {v:.3f} Hz" if v == v
                        else f"    {g}: (no laser / Control)")
            for _, r in sweep_df.iterrows():          # flag likely mislabels
                gm = gspeed.get(r["group"], float("nan"))
                if gm == gm and r["trigger_hz"] == r["trigger_hz"] and gm > 0 \
                        and abs(r["trigger_hz"] - gm) / gm > 0.15:
                    io._log(f"    [!] {r['stem']} ({r['group']}): "
                            f"{r['trigger_hz']:.3f} Hz differs >15% from group "
                            f"median {gm:.3f} Hz — check the group label")
        except Exception as e:
            io._log(f"  [sweep speed] summary failed ({e!r})")

    def _order_groups(df):                            # ordered Categorical -> sorted() by speed
        if group_order and df is not None and "group" in df.columns:
            try:
                present = [g for g in group_order if g in set(df["group"])]
                extra = [g for g in df["group"].dropna().unique() if g not in present]
                df["group"] = pd.Categorical(df["group"], categories=present + extra,
                                             ordered=True)
            except Exception:
                pass
        return df

    activity_df = _order_groups(activity_df)
    node_df = _order_groups(node_df)
    net_df = _order_groups(net_df)

    if STAGE2_SECTIONS["comparisons"]:
        if meanap_root:                               # 2B + 4B group comparisons
            try:
                elec_df = _order_groups(pd.DataFrame(meanap_elec_rows)
                                        if meanap_elec_rows else None)
                orig.render_group_comparisons(activity_df, node_df, net_df, meanap_root,
                                              electrode_df=elec_df)
                io._log(f"  [MEA-NAP original plots] complete -> {meanap_root}")
            except Exception as e:
                io._log(f"  [MEA-NAP 4B warn] group comparisons: {e!r}")
        run_stage2c(node_df, net_df, activity_df)
    else:
        io._log("[compute] group comparisons SKIPPED (section off).")

    # ---- timing breakdown (per-file analysis vs comparison) ------------------
    analysis_s = t_loop1 - t_loop0
    comparison_s = time.perf_counter() - t_loop1
    io._log("[timing] Stage 2 breakdown:")
    if file_times:
        io._log(f"    per-file analysis: {io._fmt_dur(analysis_s)} over "
                f"{len(file_times)} file(s)  (mean "
                f"{io._fmt_dur(analysis_s / len(file_times))}/file)")
        sl = max(file_times, key=lambda t: t[1])
        fa = min(file_times, key=lambda t: t[1])
        io._log(f"      slowest: {sl[0]} {io._fmt_dur(sl[1])}   "
                f"fastest: {fa[0]} {io._fmt_dur(fa[1])}")
    else:
        io._log(f"    per-file analysis: {io._fmt_dur(analysis_s)}")
    io._log(f"    comparison stage:  {io._fmt_dur(comparison_s)}")
    tot = analysis_s + comparison_s
    if tot > 0:
        io._log(f"    -> per-file analysis = {100 * analysis_s / tot:.0f}% of Stage 2 "
                f"(the part that parallelizes across files)")
    return activity_rows, node_rows, net_rows


run_activity = run_stage2                                  # backward-compatible alias


if __name__ == "__main__":
    run_stage2()
