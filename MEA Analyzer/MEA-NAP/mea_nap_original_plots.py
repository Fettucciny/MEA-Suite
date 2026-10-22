#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 MEA-NAP ORIGINAL-STYLE PLOTS  --  faithful matplotlib recreation of the SAND-Lab
 MEA-NAP MATLAB figure set, written into a parallel "MEA-NAP Original Output" tree
================================================================================
The lab's Python port (the vendored meanap/ package) ships NO plotting code -- the
canonical MEA-NAP figures live only in the MATLAB repo (Functions/*.m). This
module reproduces those figures from the data Stage 2 already computes, and writes
them into the MEA-NAP output-folder layout, ALONGSIDE the pipeline's own Plots/:

    <Output>/
      Plots/                        <- the pipeline's existing custom figures
      MEA-NAP Original Output/      <- this module (mirrors OutputData<date>/)
        2_NeuronalActivity/
          2A_IndividualNeuronalAnalysis/<stem>_<region>/
              1_FiringRateByElectrode.png     (HalfViolinPlot.m)
              2_Heatmap.png                    (electrodeHeatMaps.m)
              3_Raster.png                     (rasterPlot.m)
        3_EdgeThresholdingCheck/               (installment 2)
        4_NetworkActivity/                     (installment 2)
          4A_IndividualNetworkAnalysis/<group>/<stem>_<region>/ ...
          4B_GroupComparisons/ ...

Fidelity notes (matched to the MATLAB sources, R2014b+ defaults):
  * white background, ticks out, Arial, no grid            (aesthetics.m)
  * PNG at 300 dpi                                         (pipelineSaveFig.m)
  * raster: per-second firing-rate matrix, imagesc, parula truncated to 85%,
    two tiles (scaled to this recording / to the whole batch), x in minutes,
    upper colour limit = 99th percentile                   (rasterPlot.m,
                                                             rasterPlotUpperPercentile=99)
  * heatmap: electrode circles at MCS60 coords, parula, two tiles (this
    recording @ 99th pct / batch max), grounded electrodes grey
                                                            (electrodeHeatMaps.m)
  * firing-rate half-violin: grey [0.5 0.5 0.5], improvedSJ-style KDE, jittered
    points, black mean dot + SEM bar                        (HalfViolinPlot.m,
                                                             kdeHeight=0.3)

This module is import-safe (no torch/pywt); run it directly for a synthetic
self-test that renders the 2A set into ./_meanap_selftest/.
================================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ==============================================================================
# PARAMS (mirrors the relevant MEA-NAP AdvancedSettings.m defaults)
# ==============================================================================
MEANAP_PLOT_PARAMS = {
    # output_folder_name : name of the parallel tree beside Plots/. Any string.
    "output_folder_name":   "MEA-NAP Original Output",
    # dpi : output resolution. Range 150 (draft) - 600 (publication); MEA-NAP = 300.
    "dpi":                  300,
    # raster_upper_pct : raster colour-cap percentile (rasterPlotUpperPercentile).
    #   Range 90 - 100; lower saturates more of the map. MEA-NAP default 99.
    "raster_upper_pct":     99,
    # raster_cmap_trunc : fraction of parula used for the raster. Range 0.7 - 1.0
    #   (1.0 = full parula incl. brightest yellow). MEA-NAP uses 0.85.
    "raster_cmap_trunc":    0.85,
    # kde_height : half-violin plotting width (kdeHeight; visual only, does NOT
    #   change the KDE). Range 0.1 - 0.5; wider = fatter violins. MEA-NAP = 0.3.
    "kde_height":           0.3,
    # node_scale_f : electrode-circle diameter in grid units. Range 0.4 - 0.9;
    #   larger overlaps neighbours. MEA-NAP electrodeHeatMaps uses 2/3.
    "node_scale_f":         2.0 / 3.0,
    # heatmap_pct : recording-scaled heatmap colour-cap percentile. Range 90 - 100.
    "heatmap_pct":          99,
    # min_activity_level : active-electrode firing-rate floor in Hz (minActivityLevel).
    #   Range 0.0 - 1.0. Electrodes below this are treated as inactive.
    "min_activity_level":   0.1,
    # font_family : any installed font family name (Windows has 'Arial').
    "font_family":          "Arial",
    # ---- colour schemes (see CMAP_OPTIONS / PALETTE_OPTIONS) ------------------
    # raster_cmap : colormap for the raster. parula | viridis | magma | inferno |
    #   plasma | cividis | gray | turbo. MEA-NAP uses parula.
    "raster_cmap":          "parula",
    # heatmap_cmap : colormap for the electrode heatmap + density landscape.
    "heatmap_cmap":         "parula",
    # network_node_cmap : colormap for metric-coloured network nodes.
    "network_node_cmap":    "viridis",
    # violin_palette : colour scheme for the pre/stim/post half-violins.
    #   region (use REGION_COLORS) | tab10 | Set1 | Set2 | Dark2 | Paired | Accent.
    "violin_palette":       "region",
    # ---- marker / node / edge sizes -----------------------------------------
    # violin_point_size : jittered data-point size in half-violins. Range 2 - 20.
    "violin_point_size":    6.0,
    # violin_mean_size : black mean-dot size in half-violins. Range 8 - 80.
    "violin_mean_size":     28.0,
    # node_min_size / node_max_size : network node diameter (grid units) for the
    #   lowest / highest degree. Range 0.05 - 0.4 / 0.3 - 0.9.
    "node_min_size":        0.12,
    "node_max_size":        0.55,
    # edge_max_width : thickest network edge (line width) at the strongest weight.
    #   Range 1.0 - 5.0.
    "edge_max_width":       3.0,
}

# Colormaps offered in the GUI (parula is our embedded MATLAB map; the rest are
# matplotlib built-ins). Sequential maps suit firing-rate/weight data.
CMAP_OPTIONS = ["parula", "viridis", "magma", "inferno", "plasma", "cividis",
                "gray", "turbo"]
# Qualitative palettes for the pre/stim/post half-violins ("region" = the named
# REGION_COLORS below).
PALETTE_OPTIONS = ["region", "tab10", "Set1", "Set2", "Dark2", "Paired", "Accent"]

# ------------------------------------------------------------------------------
# SAMPLE / REGION MODEL  (the two extra Batch.csv columns MEA-NAP does not have)
# ------------------------------------------------------------------------------
# A "sample" is one recording = one filename stem. The Batch.csv `region` column
# (pre / stim / post) marks time windows WITHIN that same sample, so all rows that
# share a stem are the same sample measured pre / during (stim) / after. The group
# comparisons below therefore treat `region` as a first-class factor:
#   * NodeByAge / RecordingsByAge : x = age (day), one violin per region, one
#     subplot per group -> compare pre/stim/post across the days, per group.
#   * NodeByGroup / RecordingsByGroup : x = group, one violin per region, pooled
#     across samples/days -> compare pre/stim/post across samples by group.
# REGION_ORDER : plotting order of the window labels. Edit to match your Batch.csv
#   column-5 labels (any labels work; unknown ones are appended alphabetically).
REGION_ORDER = ["pre", "stim", "post"]
# REGION_COLORS : colour per region in the comparison plots. Any matplotlib RGB
#   (0-1) tuples; add entries if you use more/other region labels.
REGION_COLORS = {
    "pre":  (0.20, 0.49, 0.72),     # blue
    "stim": (0.89, 0.42, 0.13),     # orange
    "post": (0.30, 0.62, 0.35),     # green
}

# ------------------------------------------------------------------------------
# MATLAB parula colormap (R2014b+), 64x3 -- MEA-NAP's raster/heatmap colormap.
# ------------------------------------------------------------------------------
_PARULA = [
    [0.2081, 0.1663, 0.5292], [0.2116, 0.1898, 0.5777], [0.2123, 0.2138, 0.6270],
    [0.2081, 0.2386, 0.6771], [0.1959, 0.2645, 0.7279], [0.1707, 0.2919, 0.7792],
    [0.1253, 0.3242, 0.8303], [0.0591, 0.3598, 0.8683], [0.0117, 0.3875, 0.8820],
    [0.0060, 0.4086, 0.8828], [0.0165, 0.4266, 0.8786], [0.0329, 0.4430, 0.8720],
    [0.0498, 0.4586, 0.8641], [0.0629, 0.4737, 0.8554], [0.0723, 0.4887, 0.8467],
    [0.0779, 0.5040, 0.8384], [0.0793, 0.5200, 0.8312], [0.0749, 0.5375, 0.8263],
    [0.0641, 0.5570, 0.8240], [0.0488, 0.5772, 0.8228], [0.0343, 0.5966, 0.8199],
    [0.0265, 0.6137, 0.8135], [0.0239, 0.6287, 0.8038], [0.0231, 0.6418, 0.7913],
    [0.0228, 0.6535, 0.7768], [0.0267, 0.6642, 0.7607], [0.0384, 0.6743, 0.7436],
    [0.0590, 0.6838, 0.7254], [0.0843, 0.6928, 0.7062], [0.1133, 0.7015, 0.6859],
    [0.1453, 0.7098, 0.6646], [0.1801, 0.7177, 0.6424], [0.2178, 0.7250, 0.6193],
    [0.2586, 0.7317, 0.5954], [0.3022, 0.7376, 0.5712], [0.3482, 0.7424, 0.5473],
    [0.3953, 0.7459, 0.5244], [0.4420, 0.7481, 0.5033], [0.4871, 0.7491, 0.4840],
    [0.5300, 0.7491, 0.4661], [0.5709, 0.7485, 0.4494], [0.6100, 0.7473, 0.4337],
    [0.6473, 0.7456, 0.4188], [0.6834, 0.7435, 0.4044], [0.7184, 0.7411, 0.3905],
    [0.7525, 0.7384, 0.3768], [0.7858, 0.7356, 0.3633], [0.8185, 0.7327, 0.3498],
    [0.8507, 0.7299, 0.3360], [0.8824, 0.7274, 0.3217], [0.9139, 0.7258, 0.3063],
    [0.9450, 0.7261, 0.2886], [0.9739, 0.7314, 0.2666], [0.9938, 0.7455, 0.2403],
    [0.9990, 0.7653, 0.2164], [0.9955, 0.7861, 0.1967], [0.9880, 0.8066, 0.1794],
    [0.9789, 0.8271, 0.1633], [0.9697, 0.8481, 0.1475], [0.9626, 0.8705, 0.1309],
    [0.9589, 0.8949, 0.1132], [0.9598, 0.9218, 0.0948], [0.9661, 0.9514, 0.0755],
    [0.9763, 0.9831, 0.0538],
]
PARULA = LinearSegmentedColormap.from_list("parula", _PARULA, N=256)


def _cmap(name):
    """Resolve a colormap name to an object ('parula' -> our embedded map)."""
    if name in ("parula", None):
        return PARULA
    try:
        return plt.get_cmap(name)
    except Exception:
        return PARULA


def _cmap_trunc(name, frac):
    """`name` colormap truncated to its first `frac` (MEA-NAP raster: parula@0.85)."""
    base = _cmap(name)
    n = max(2, int(round(256 * frac)))
    return LinearSegmentedColormap.from_list(f"{name}_{frac}", base(np.linspace(0, 1, n)))


# ==============================================================================
# PLOT ENABLE REGISTRY  --  per-figure on/off toggles (GUI "Plot outputs" tab)
# ------------------------------------------------------------------------------
# Set any value False to skip that figure. Keys map 1:1 to the figures below;
# the render drivers check _on(key) before drawing. "check all" in the GUI just
# sets every value True.
# ==============================================================================
PLOT_ENABLE = {
    # 2A neuronal activity (per recording)
    "firing_rate_by_electrode": True,
    "electrode_heatmap":        True,
    "raster":                   True,
    # 3 edge-thresholding check (per recording x lag)
    "edge_threshold_check":     True,
    # 4A individual network (per recording x lag)
    "connectivity_stats":       True,
    "network_plot":             True,     # plain node-size-by-degree graph
    "network_NodeDegree":       True,
    "network_NodeStrength":     True,
    "network_BetweennessCentrality": True,
    "network_ParticipationCoefficient": True,
    "network_WithinModuleZ":    True,
    "network_LocalEfficiency":  True,
    "node_cartography":         True,
    # 2B / 4B group comparisons
    "activity_by_group":        True,
    "activity_by_age":          True,
    "electrode_by_group":       True,
    "electrode_by_age":         True,
    "network_by_group":         True,
    "network_by_age":           True,
    "node_by_group":            True,
    "node_by_age":              True,
    "graph_metrics_by_lag":     True,
    "cartography_by_lag":       True,
    "density_landscape":        True,
}


def _on(key):
    return bool(PLOT_ENABLE.get(key, True))


def _region_palette():
    """Region -> RGB mapping from MEANAP_PLOT_PARAMS['violin_palette']."""
    scheme = MEANAP_PLOT_PARAMS.get("violin_palette", "region")
    if scheme == "region":
        return dict(REGION_COLORS)
    cm = _cmap(scheme)
    return {r: tuple(cm(i % 10)[:3]) for i, r in enumerate(REGION_ORDER)}


# ==============================================================================
# AESTHETICS  (aesthetics.m: white bg, box off, ticks out, no grid, Arial)
# ==============================================================================
def _aesthetics(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(direction="out")
    ax.grid(False)


def _finish(fig, path):
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        for lbl in (ax.get_xticklabels() + ax.get_yticklabels()
                    + [ax.title, ax.xaxis.label, ax.yaxis.label]):
            lbl.set_fontname(MEANAP_PLOT_PARAMS["font_family"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=MEANAP_PLOT_PARAMS["dpi"], facecolor="white",
                bbox_inches="tight")
    plt.close(fig)


plt.rcParams["font.family"] = MEANAP_PLOT_PARAMS["font_family"]
plt.rcParams["font.sans-serif"] = [MEANAP_PLOT_PARAMS["font_family"], "DejaVu Sans"]

# ==============================================================================
# MCS60 electrode coordinates from a channel label ("CR" -> column C, row R)
# ------------------------------------------------------------------------------
# MEA-NAP getCoordsFromLayout('MCS60') lays the 8x8 grid (corners removed) with
# x = column and y = row, row 1 at the TOP. We derive the same from the pipeline's
# string labels so heatmaps/graphs share MEA-NAP's spatial layout.
# ==============================================================================
def mcs_coord(label):
    """(x, y) grid position for an MCS label like '47' (col 4, row 7), y flipped
    so row 1 is at the top; None for non-grid labels (e.g. 'Ref')."""
    s = str(label).strip()
    if len(s) == 2 and s.isdigit():
        col, row = int(s[0]), int(s[1])
        if 1 <= col <= 8 and 1 <= row <= 8:
            return float(col), float(9 - row)     # y flipped (row 1 on top)
    return None


_MCS60_CORNERS = {"11", "18", "81", "88"}


def _mcs60_labels():
    """All 60 MCS 8x8-grid electrode labels (four corners removed)."""
    return [f"{c}{r}" for c in range(1, 9) for r in range(1, 9)
            if f"{c}{r}" not in _MCS60_CORNERS]


# ==============================================================================
# HalfViolinPlot.m  --  right-half KDE violin + jittered points + mean/SEM
# ==============================================================================
def _half_violin(ax, data, pos, colour, width=MEANAP_PLOT_PARAMS["kde_height"]):
    data = np.asarray(data, float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return
    if data.size > 1 and np.std(data) > 1e-8:
        from scipy.stats import gaussian_kde
        # MEA-NAP uses an improved Sheather-Jones bandwidth with a floor of
        # 0.1*range; Scott's rule (gaussian_kde default) with the same floor is a
        # close, dependency-free match for the plotted violin shape.
        std = data.std(ddof=1)
        kde = gaussian_kde(data)
        bw_floor = (data.max() - data.min()) * 0.1
        if std > 0 and kde.factor * std < bw_floor:
            kde.set_bandwidth(bw_floor / std)
        # Evaluate over a grid EXTENDED past the data (~3 bandwidths each side) so
        # the KDE tapers to ~0 at both ends, matching MATLAB ksdensity. Draw with
        # fill_betweenx against a straight spine at x = pos + 0.1 so the violin's
        # left edge is exactly vertical (a grid clipped to [min, max] leaves a
        # non-zero density at the ends -> a tilted closing edge on sparse panels).
        bw = kde.factor * std                          # effective bandwidth
        pad = 3.0 * bw
        xi = np.linspace(data.min() - pad, data.max() + pad, 256)
        f = kde(xi)
        f = f / f.max() * width
        ax.fill_betweenx(xi, pos + 0.1, f + pos + 0.1, facecolor=colour,
                         edgecolor=colour, lw=1.0)
    # jittered points to the LEFT of the violin. Marker sizes scale with the
    # violin width (relative to the single-violin kde_height) so the narrow
    # grouped region panels get proportionally smaller dots.
    scale = np.clip(width / MEANAP_PLOT_PARAMS["kde_height"], 0.35, 1.0)
    s_pt = MEANAP_PLOT_PARAMS["violin_point_size"] * scale
    s_mean = MEANAP_PLOT_PARAMS["violin_mean_size"] * scale
    jitter = np.random.rand(data.size) * width
    ax.scatter(jitter + pos - (width + 0.1), data, s=s_pt, c=[colour],
               edgecolors="none")
    # mean (black dot) + SEM bar
    mean = float(np.mean(data))
    sem = float(np.std(data, ddof=1) / np.sqrt(data.size)) if data.size > 1 else 0.0
    ax.plot([pos, pos], [mean - sem, mean + sem], color="k",
            lw=max(1.2, 2.4 * scale), solid_capstyle="butt")
    ax.scatter([pos], [mean], s=s_mean, c="k", zorder=5)


# ==============================================================================
# 2A FIGURE 1 -- 1_FiringRateByElectrode  (firingRateElectrodeDistribution.m)
# ==============================================================================
def fig_firing_rate_by_electrode(unit, res, out_dir):
    fr = np.asarray(res["fr"], float)
    fig, ax = plt.subplots(figsize=(5.0, 6.0))
    _half_violin(ax, fr, 1.0, (0.5, 0.5, 0.5))
    ax.set_xlim(0.5, 1.5)
    ax.set_xticks([])
    ax.set_xlabel(f"age{_age_str(unit)}")
    ax.set_ylabel("mean firing rate per electrode (Hz)")
    mx = max(float(np.nanmax(fr)) if fr.size else 0.1, 0.1)
    ax.set_ylim(0, mx + mx * 0.15)
    ax.set_title(_title(unit), fontsize=10, loc="center")
    _aesthetics(ax)
    _finish(fig, os.path.join(out_dir, "1_FiringRateByElectrode.png"))


# ==============================================================================
# 2A FIGURE 2 -- 2_Heatmap  (electrodeHeatMaps.m): electrode circles on the grid
# ==============================================================================
def fig_electrode_heatmap(unit, res, out_dir, group_max_fr):
    labels = unit["labels"]
    fr = np.asarray(res["fr"], float)
    cmap = _cmap(MEANAP_PLOT_PARAMS["heatmap_cmap"])
    kept = {str(l): float(v) for l, v in zip(labels, fr) if mcs_coord(l) is not None}
    if not kept:
        return
    # grounded / excluded electrodes drawn as grey circles (electrodeHeatMaps.m
    # groundColor = [0.8 0.8 0.8]). Prefer an explicit list if the unit carries
    # one; otherwise every MCS60 grid position that wasn't analysed (grounded +
    # reference + auto-grounded) is greyed.
    if unit.get("grounded") is not None:
        grounded = [str(g) for g in unit["grounded"]
                    if mcs_coord(g) is not None and str(g) not in kept]
    else:
        grounded = [g for g in _mcs60_labels() if g not in kept]
    GREY = (0.8, 0.8, 0.8)
    vals = np.array(list(kept.values()))
    allpts = np.array([mcs_coord(l) for l in list(kept) + grounded])
    xc, yc = allpts[:, 0], allpts[:, 1]
    d = MEANAP_PLOT_PARAMS["node_scale_f"]
    pct = float(np.percentile(vals, MEANAP_PLOT_PARAMS["heatmap_pct"])) if vals.size else 1.0
    pct = max(pct, 1e-9)
    group_max_fr = max(float(group_max_fr), 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.7))
    for ax, vmax, sub in ((axes[0], pct, "scaled to recording"),
                          (axes[1], group_max_fr, "scaled to entire group")):
        for g in grounded:                                   # grounded first (grey)
            x, y = mcs_coord(g)
            ax.add_patch(plt.Circle((x, y), d / 2, facecolor=GREY,
                                    edgecolor="w", lw=0.3, zorder=1))
        for lab, v in kept.items():
            x, y = mcs_coord(lab)
            frac = np.clip(v / vmax, 0.0, 1.0)
            ax.add_patch(plt.Circle((x, y), d / 2, facecolor=cmap(frac),
                                    edgecolor="w", lw=0.3, zorder=2))
        ax.set_xlim(xc.min() - 1, xc.max() + 1)
        ax.set_ylim(yc.min() - 1, yc.max() + 1)
        ax.set_aspect("equal"); ax.axis("off")
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=0, vmax=vmax))
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("mean firing rate (Hz)")
        cb.outline.set_visible(False); cb.ax.tick_params(direction="out")
        ax.set_title(f"{_title(unit)}\n{sub}", fontsize=10)
    _finish(fig, os.path.join(out_dir, "2_Heatmap.png"))


# ==============================================================================
# 2A FIGURE 3 -- 3_Raster  (rasterPlot.m): per-second FR matrix, imagesc, parula
# ==============================================================================
def _persec_matrix(unit):
    """Downsample spikes to a (n_seconds x n_channels) per-second firing-rate
    matrix (MEA-NAP downSampleSum -> spikes per 1 s bin = Hz)."""
    dur = float(unit["duration"])
    nsec = max(1, int(np.floor(dur)))
    edges = np.arange(nsec + 1, dtype=float)
    chans = unit["spikes"]
    M = np.zeros((nsec, len(chans)))
    for j, s in enumerate(chans):
        s = np.asarray(s, float)
        if s.size:
            M[:, j] = np.histogram(s[(s >= 0) & (s < nsec)], bins=edges)[0]
    return M                                        # rows = seconds, cols = channels


def fig_raster(unit, res, out_dir, group_max_fr):
    M = _persec_matrix(unit)                        # (nsec, nch), values = Hz
    nsec, nch = M.shape
    cmap = _cmap_trunc(MEANAP_PLOT_PARAMS["raster_cmap"],
                       MEANAP_PLOT_PARAMS["raster_cmap_trunc"])
    top_lim = max(float(np.percentile(M, MEANAP_PLOT_PARAMS["raster_upper_pct"])), 1.0)
    group_lim = max(float(group_max_fr), 1.0)

    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    for ax, vmax, sub in ((axes[0], top_lim, "raster scaled to recording"),
                          (axes[1], group_lim, "raster scaled to entire group")):
        im = ax.imshow(M.T, aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                       origin="upper", interpolation="nearest",
                       extent=[0, nsec, nch + 0.5, 0.5])
        # x ticks every 60 s, labelled in minutes
        mins = int(nsec // 60)
        if mins >= 1:
            ax.set_xticks(np.arange(1, mins + 1) * 60)
            ax.set_xticklabels(np.arange(1, mins + 1))
        ny = min(max(nch, 1), 7)
        ax.set_yticks(np.round(np.linspace(1, max(nch, ny), ny)).astype(int))
        ax.set_xlabel("Time (min)"); ax.set_ylabel("Electrode")
        ax.set_title(f"{_title(unit)} {sub}", fontsize=11)
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label("Firing Rate (Hz)")
        cb.outline.set_visible(False); cb.ax.tick_params(direction="out")
        _aesthetics(ax)
    fig.tight_layout()
    _finish(fig, os.path.join(out_dir, "3_Raster.png"))


# ==============================================================================
# helpers + per-recording driver
# ==============================================================================
def _age_str(unit):
    a = unit.get("age")
    try:
        return str(int(a)) if float(a) == int(a) else str(a)
    except Exception:
        return str(a)


def _title(unit):
    # MEA-NAP strips underscores from the filename in titles (regexprep _ '')
    return f"{unit['stem']}{unit.get('region', '')}".replace("_", "")


def output_root(plot_dir):
    """The 'MEA-NAP Original Output' folder, a subfolder UNDER the selected output
    path (PLOT_DIR). Previously this was written to the PARENT of PLOT_DIR, which
    put it outside the folder chosen in the GUI and made it look 'missing'."""
    return os.path.join(os.path.normpath(plot_dir),
                        MEANAP_PLOT_PARAMS["output_folder_name"])


def create_output_folders(root, groups):
    """Recreate MEA-NAP's CreateOutputFolders.m tree under `root`."""
    subs = [
        "ExperimentMatFiles", "1_SpikeDetection/1A_SpikeDetectedData",
        "1_SpikeDetection/1B_SpikeDetectionChecks",
        "2_NeuronalActivity/2A_IndividualNeuronalAnalysis",
        "2_NeuronalActivity/2B_GroupComparisons/1_NodeByGroup",
        "2_NeuronalActivity/2B_GroupComparisons/2_NodeByAge",
        "2_NeuronalActivity/2B_GroupComparisons/3_RecordingsByGroup/HalfViolinPlots",
        "2_NeuronalActivity/2B_GroupComparisons/4_RecordingsByAge/HalfViolinPlots",
        "3_EdgeThresholdingCheck",
        "4_NetworkActivity/4A_IndividualNetworkAnalysis",
        "4_NetworkActivity/4B_GroupComparisons/1_NodeByGroup",
        "4_NetworkActivity/4B_GroupComparisons/2_NodeByAge",
        "4_NetworkActivity/4B_GroupComparisons/3_RecordingsByGroup/HalfViolinPlots",
        "4_NetworkActivity/4B_GroupComparisons/4_RecordingsByAge/HalfViolinPlots",
        "4_NetworkActivity/4B_GroupComparisons/5_GraphMetricsByLag",
        "4_NetworkActivity/4B_GroupComparisons/6_NodeCartographyByLag",
        "4_NetworkActivity/4B_GroupComparisons/7_DensityLandscape",
    ]
    for g in groups:
        subs.append(f"4_NetworkActivity/4A_IndividualNetworkAnalysis/{g}")
    for s in subs:
        os.makedirs(os.path.join(root, *s.split("/")), exist_ok=True)


def render_activity_recording(unit, res, root, group_max_fr):
    """Write the 2A neuronal-activity figure set for ONE (stem, region) unit.
    `group_max_fr` scales the second heatmap/raster panel to this unit's GROUP."""
    rec = f"{unit['stem']}_{unit['region']}"
    out_dir = os.path.join(root, "2_NeuronalActivity",
                           "2A_IndividualNeuronalAnalysis", rec)
    os.makedirs(out_dir, exist_ok=True)
    if _on("firing_rate_by_electrode"):
        fig_firing_rate_by_electrode(unit, res, out_dir)
    if _on("electrode_heatmap"):
        fig_electrode_heatmap(unit, res, out_dir, group_max_fr)
    if _on("raster"):
        fig_raster(unit, res, out_dir, group_max_fr)
    return out_dir


# ==============================================================================
# 4A INDIVIDUAL NETWORK ANALYSIS + 3 EDGE-THRESHOLD CHECK
# ------------------------------------------------------------------------------
# Faithful to StandardisedNetworkPlot.m (MEA layout: node size by a metric, cyan
# or metric-coloured nodes, weight-scaled grey edges, node-degree + edge-weight
# legends), plotConnectivityProperties.m (adjM imagesc + corr bars + ND/NS/EW
# histograms) and NodeCartography.m (PC-vs-Z scatter with 6 Guimera-Amaral roles).
# ==============================================================================
_CYAN = (0.020, 0.729, 0.859)                 # StandardisedNetworkPlot node colour
_ROLE_COLORS = [(0.800, 0.902, 0.310), (0.580, 0.706, 0.278), (0.369, 0.435, 0.122),
                (0.200, 0.729, 0.949), (0.078, 0.424, 0.835), (0.016, 0.235, 0.498)]
_ROLE_NAMES = ["Peripheral", "Non-hub connector", "Non-hub kinless",
               "Provincial hub", "Connector hub", "Kinless hub"]
# Guimera-Amaral cartography boundaries (keep in sync with mea_nap_activity
# CARTOGRAPHY_PARAMS). Each is adjustable within the range noted:
#   hub    : within-module z hub / non-hub split.  Range 2.0 - 3.0  (MEA-NAP 2.5)
#   peri   : participation coef, peripheral | non-hub connector. Range 0.5 - 0.7 (0.625)
#   nonhub : non-hub connector | non-hub kinless.  Range 0.7 - 0.9  (0.8)
#   prohub : provincial hub | connector hub.       Range 0.2 - 0.4  (0.3)
#   conhub : connector hub | kinless hub.          Range 0.6 - 0.8  (0.75)
_CART = dict(hub=2.5, peri=0.625, nonhub=0.8, prohub=0.3, conhub=0.75)

# node-colour-map metrics MEA-NAP plots on the network (label -> node key, cmap)
NET_COLORMAP_METRICS = [
    ("NodeDegree", "degree", "ND", None),
    ("NodeStrength", "strength", "NS", "viridis"),
    ("BetweennessCentrality", "betweenness", "BC", "viridis"),
    ("ParticipationCoefficient", "participation", "PC", "viridis"),
    ("WithinModuleZ", "within_module_z", "Z", "viridis"),
    ("LocalEfficiency", "local_efficiency", "Eloc", "viridis"),
]


def _node_size(z, zmax, base=None, top=None):
    """Node diameter in grid units, scaled by metric z (StandardisedNetworkPlot
    getNodeSize: min node + fraction of the metric's max)."""
    base = MEANAP_PLOT_PARAMS["node_min_size"] if base is None else base
    top = MEANAP_PLOT_PARAMS["node_max_size"] if top is None else top
    zmax = max(float(zmax), 1e-9)
    return base + (top - base) * np.clip(float(z) / zmax, 0.0, 1.0)


def _edge_weight_range(W):
    """Min / max of the positive upper-triangle edge weights (for widths+legend)."""
    iu = np.triu_indices(W.shape[0], 1)
    w = W[iu]; w = w[w > 0]
    if w.size == 0:
        return None
    return float(w.min()), float(w.max())


def _draw_mea_edges(ax, W, xy):
    """Weight-scaled grey edges (light->dark AND thin->thick by weight), darker
    drawn on top. Line width scales to MEANAP_PLOT_PARAMS['edge_max_width']."""
    rng = _edge_weight_range(W)
    if rng is None:
        return
    wmin, wmax = rng
    span = (wmax - wmin) if wmax > wmin else max(wmax, 1e-9)
    ew_max = MEANAP_PLOT_PARAMS["edge_max_width"]
    iu = np.triu_indices(W.shape[0], 1)
    segs = []
    for a, b in zip(*iu):
        wv = W[a, b]
        if wv <= 0:
            continue
        frac = (wv - wmin) / span
        lw = 0.15 + (ew_max - 0.15) * frac
        shade = 0.82 - 0.72 * frac               # light grey -> near-black by weight
        segs.append((shade, [xy[a], xy[b]], lw))
    for shade, seg, lw in sorted(segs, key=lambda s: -s[0]):   # dark last/on top
        ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], lw=lw,
                color=(shade, shade, shade), solid_capstyle="round", zorder=1)


def _network_axis(ax, nr, key, zmax, node_colors=None, cmap=None, cbar_label=None,
                  fig=None):
    labels = nr["labels"]; W = np.asarray(nr["W"], float)
    xy = [mcs_coord(l) for l in labels]
    idx = [i for i, c in enumerate(xy) if c is not None]
    if not idx:
        return None
    xy = {i: xy[i] for i in idx}
    zvals = np.array([nr["nodes"][i][key] for i in range(len(labels))], float)
    _draw_mea_edges(ax, W[np.ix_(idx, idx)],
                    {k: xy[i] for k, i in enumerate(idx)})
    sm = None
    for i in idx:
        z = zvals[i]
        if z <= 0:
            continue
        d = _node_size(z, zmax)
        if node_colors is not None:
            fc = node_colors(i)
        else:
            fc = _CYAN
        ax.add_patch(plt.Circle(xy[i], d / 2, facecolor=fc, edgecolor="w",
                                lw=0.3, zorder=3))
    xs = [c[0] for c in xy.values()]; ys = [c[1] for c in xy.values()]
    lo, hi = min(xs + ys) - 1, max(xs + ys) + 1
    ax.set_xlim(lo, hi + 2.8); ax.set_ylim(lo, hi)
    ax.set_aspect("equal"); ax.axis("off")
    return zvals


def fig_network_plot(nr, out_dir, lag, stem, region, pnum, metric=None):
    """StandardisedNetworkPlot (MEA): node size = degree; nodes cyan (plain) or
    coloured by `metric` with a colorbar. Weight-scaled grey edges + node-degree
    AND edge-weight legends."""
    deg = np.array([nd["degree"] for nd in nr["nodes"]], float)
    zmax = max(deg.max() if deg.size else 1.0, 3.0)     # ND legend divisor >= 3
    # active-node subset weight range (for the edge-weight legend)
    labels = nr["labels"]; W = np.asarray(nr["W"], float)
    idx = [i for i, l in enumerate(labels) if mcs_coord(l) is not None]
    ewr = _edge_weight_range(W[np.ix_(idx, idx)]) if idx else None
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    if metric is None:
        _network_axis(ax, nr, "degree", zmax)
    else:
        nice, key, short, _ = metric
        vals = np.array([nd[key] for nd in nr["nodes"]], float)
        vmin = float(np.nanmin(vals)) if vals.size else 0.0
        vmax = float(np.nanmax(vals)) if vals.size else 1.0
        if not np.isfinite(vmin) or vmax == vmin:
            vmin, vmax = (vmin - 1e-3, vmax)
        norm = plt.Normalize(vmin, vmax)
        cm = _cmap(MEANAP_PLOT_PARAMS["network_node_cmap"])   # user-selectable
        _network_axis(ax, nr, "degree", zmax,
                      node_colors=lambda i: cm(norm(vals[i])))
        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm); sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label(nice); cb.outline.set_visible(False)
        cb.ax.tick_params(direction="out")
    _network_node_edge_legend(ax, zmax, ewr)
    ax.set_title(f"{_title_lag(stem, region, lag)}", fontsize=10)
    name = f"{pnum}_MEA_NetworkPlot" + (f"_{metric[0]}" if metric else "") + ".png"
    _finish(fig, os.path.join(out_dir, name))


def _network_node_edge_legend(ax, zmax, ewr=None):
    """Node-degree size legend (3 circle sizes) + edge-weight legend (3 line
    widths) to the right of the graph, matching StandardisedNetworkPlot.m."""
    x0 = ax.get_xlim()[1] - 2.4
    ytop = ax.get_ylim()[1] - 0.7
    ax.text(x0 - 0.3, ytop + 0.5, "node degree:", fontsize=8)
    for k, frac in enumerate((1 / 3, 2 / 3, 1.0)):
        z = max(int(round(zmax * frac)), 1)
        d = _node_size(z, zmax)
        y = ytop - k * 0.9
        ax.add_patch(plt.Circle((x0, y), d / 2, facecolor=_CYAN, edgecolor="w",
                                lw=0.3, zorder=3))
        ax.text(x0 + 0.85, y, f"{z:02d}", fontsize=8, va="center")
    # edge-weight legend (3 widths from thin/light to thick/dark)
    if ewr is not None:
        wmin, wmax = ewr
        ew_max = MEANAP_PLOT_PARAMS["edge_max_width"]
        yb = ytop - 3 * 0.9 - 0.6
        ax.text(x0 - 0.3, yb + 0.5, "edge weight:", fontsize=8)
        for k, frac in enumerate((1 / 3, 2 / 3, 1.0)):
            wv = wmin + (wmax - wmin) * frac
            lw = 0.15 + (ew_max - 0.15) * frac
            shade = 0.82 - 0.72 * frac
            y = yb - k * 0.55
            ax.plot([x0 - 0.35, x0 + 0.35], [y, y], lw=lw,
                    color=(shade, shade, shade), solid_capstyle="round")
            ax.text(x0 + 0.85, y, f"{wv:.3f}", fontsize=8, va="center")


def fig_connectivity_stats(nr, out_dir, lag, stem, region):
    """plotConnectivityProperties.m: adjM imagesc + max/mean corr bars + ND/NS/EW
    histograms (saved as 1_adjM<lag>msConnectivityStats)."""
    M = np.asarray(nr["M"], float); W = np.asarray(nr["W"], float)
    deg = np.array([nd["degree"] for nd in nr["nodes"]], float)
    strg = np.array([nd["strength"] for nd in nr["nodes"]], float)
    ew = W[np.triu(np.ones_like(W), 1) > 0]; ew = ew[ew > 0]
    maxc = float(np.nanmax(M)) if M.size else 0.0
    meanc = float(np.nanmean(M[np.triu_indices(M.shape[0], 1)])) if M.size else 0.0
    ylimc = max(float(np.nanmax(W)) if W.size else 0.1, 1e-4) * 1.15

    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(2, 4)
    axA = fig.add_subplot(gs[:, 0:2])
    im = axA.imshow(np.nan_to_num(M), cmap="parula" if False else PARULA, aspect="auto")
    axA.set_xlabel("nodes"); axA.set_ylabel("nodes"); axA.set_title("adjacency matrix")
    cb = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04)
    cb.set_label("correlation coefficient"); cb.outline.set_visible(False)
    _aesthetics(axA)
    axm = fig.add_subplot(gs[0, 2])
    axm.bar([0], [maxc], color=(0, 0.447, 0.741)); axm.set_ylim(0, ylimc)
    axm.set_title("max corr. value"); axm.set_xticks([]); _aesthetics(axm)
    axmn = fig.add_subplot(gs[0, 3])
    axmn.bar([0], [meanc], color=(0, 0.447, 0.741)); axmn.set_ylim(0, ylimc)
    axmn.set_title("mean corr. value"); axmn.set_xticks([]); _aesthetics(axmn)
    axd = fig.add_subplot(gs[1, 2])
    axd.hist(deg, bins=min(50, max(5, deg.size)), color=(0.301, 0.745, 0.933))
    axd.set_xlabel("node degree"); axd.set_ylabel("frequency"); _aesthetics(axd)
    axs = fig.add_subplot(gs[1, 3])
    axs.hist(ew if ew.size else [0], bins=min(50, max(5, ew.size)),
             color=(0.301, 0.745, 0.933))
    axs.set_xlabel("edge weight"); axs.set_ylabel("frequency"); _aesthetics(axs)
    fig.suptitle(_title_lag(stem, region, lag), fontsize=11)
    fig.tight_layout()
    _finish(fig, os.path.join(out_dir, f"1_adjM{lag}msConnectivityStats.png"))


def fig_node_cartography(nr, out_dir, lag, stem, region):
    """NodeCartography.m: PC (x) vs within-module-z (y) scatter, 6 Guimera-Amaral
    role colours, dashed boundaries + a role-region schematic (saved as
    9_adjM<lag>msNodeCartography)."""
    pc = np.array([nd["participation"] for nd in nr["nodes"]], float)
    z = np.array([nd["within_module_z"] for nd in nr["nodes"]], float)
    roles = np.array([nd["role"] for nd in nr["nodes"]], int)
    xr = [float(np.nanmin(pc)), float(np.nanmax(pc))] if pc.size and np.nanmin(pc) != np.nanmax(pc) else [0, 1]
    if z.size and np.nanmin(z) != np.nanmax(z):
        zlo, zhi = float(np.nanmin(z)), float(np.nanmax(z))
        yr = [zlo * 1.1 if zlo < 0 else zlo * 0.9, zhi * 1.1 if zhi > 0 else zhi * 0.9]
    else:
        yr = [-2, 4]
    fig, (ax, axd) = plt.subplots(2, 1, figsize=(6, 7),
                                  gridspec_kw={"height_ratios": [3, 2]})
    hub = _CART["hub"]
    ax.plot(xr, [hub, hub], "--k", lw=0.9)
    ax.plot([_CART["peri"]] * 2, [yr[0], hub], "--k", lw=0.9)
    ax.plot([_CART["nonhub"]] * 2, [yr[0], hub], "--k", lw=0.9)
    ax.plot([_CART["prohub"]] * 2, [hub, yr[1]], "--k", lw=0.9)
    ax.plot([_CART["conhub"]] * 2, [hub, yr[1]], "--k", lw=0.9)
    for r in range(1, 7):
        m = roles == r
        if m.any():
            ax.scatter(pc[m], z[m], s=18, color=_ROLE_COLORS[r - 1],
                       edgecolors=_ROLE_COLORS[r - 1], label=_ROLE_NAMES[r - 1])
    ax.set_xlim(xr); ax.set_ylim(yr)
    ax.set_xlabel("participation coefficient")
    ax.set_ylabel("within-module degree z-score")
    ax.set_title("node cartography"); _aesthetics(ax)
    _cartography_diagram(axd)
    fig.suptitle(_title_lag(stem, region, lag), fontsize=11)
    fig.tight_layout()
    _finish(fig, os.path.join(out_dir, f"9_adjM{lag}msNodeCartography.png"))


def _cartography_diagram(ax):
    """Schematic of the 6 Guimera-Amaral role regions (stand-in for MEA-NAP's
    NodeCartographyDiagram.jpg)."""
    hub = _CART["hub"]
    rects = [  # (x0,x1,y0,y1,role)
        (0, _CART["peri"], 0, hub, 1), (_CART["peri"], _CART["nonhub"], 0, hub, 2),
        (_CART["nonhub"], 1, 0, hub, 3), (0, _CART["prohub"], hub, 4, 4),
        (_CART["prohub"], _CART["conhub"], hub, 4, 5), (_CART["conhub"], 1, hub, 4, 6)]
    for x0, x1, y0, y1, r in rects:
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   facecolor=_ROLE_COLORS[r - 1], edgecolor="w"))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, str(r), ha="center", va="center",
                color="w", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 4)
    ax.set_xlabel("participation coefficient")
    ax.set_ylabel("within-module z")
    handles = [plt.Line2D([0], [0], marker="s", ls="", markerfacecolor=_ROLE_COLORS[i],
                          markeredgecolor="none", label=f"{i+1} {_ROLE_NAMES[i]}",
                          markersize=7) for i in range(6)]
    ax.legend(handles=handles, fontsize=6, loc="center left",
              bbox_to_anchor=(1.01, 0.5), framealpha=0.9)
    _aesthetics(ax)


def fig_edge_threshold_check(nr, out_dir, lag, stem, region):
    """3_EdgeThresholdingCheck: real STTC vs the circular-shift surrogate null,
    the per-edge significance fraction, threshold sensitivity, and which edges are
    retained (plotConnectivityProperties / MEA-NAP edge-thresholding diagnostic)."""
    M = np.asarray(nr["M"], float)
    frac = np.asarray(nr["frac"], float) if nr.get("frac") is not None else None
    surr = nr.get("surr")
    n = M.shape[0]
    iu = np.triu_indices(n, 1)
    real = np.nan_to_num(M[iu])
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    # A: real vs surrogate STTC distributions
    a = ax[0, 0]
    if surr is not None:
        pooled = np.asarray(surr)[:, iu[0], iu[1]].ravel()
        pooled = pooled[np.isfinite(pooled)]
        a.hist(pooled, bins=60, density=True, alpha=0.55, color="0.6",
               label="surrogate null")
    a.hist(real[np.isfinite(real)], bins=60, density=True, alpha=0.65,
           color="firebrick", label="real edges")
    a.set(xlabel="STTC", ylabel="density", title="A. Real vs surrogate STTC")
    a.legend(fontsize=8); _aesthetics(a)
    # B: significance fraction heatmap
    b = ax[0, 1]
    if frac is not None:
        im = b.imshow(frac, cmap="magma", vmin=0, vmax=1, aspect="auto")
        fig.colorbar(im, ax=b, label="frac. surrogates beaten")
    b.set(title="B. Significance fraction", xlabel="node", ylabel="node")
    _aesthetics(b)
    # C: threshold sensitivity (edges retained vs significance threshold)
    c = ax[1, 0]
    if frac is not None:
        fr = frac[iu]
        sweep = np.linspace(0.5, 1.0, 51)
        c.plot(sweep, [(fr >= s).sum() for s in sweep], "-o", ms=2.5,
               color="steelblue")
        c.axvline(MEANAP_PLOT_PARAMS.get("min_activity_level", 0) * 0 + 0.95,
                  color="crimson", ls="--", label="0.95")
        c.legend(fontsize=8)
    c.set(xlabel="significance threshold", ylabel="# edges retained",
          title="C. Threshold sensitivity"); _aesthetics(c)
    # D: edge retention (sorted by real STTC)
    d = ax[1, 1]
    if surr is not None and frac is not None:
        order = np.argsort(real); x = np.arange(len(order))
        sm = np.nanmean(np.asarray(surr)[:, iu[0], iu[1]], 0)[order]
        ss = np.nanstd(np.asarray(surr)[:, iu[0], iu[1]], 0)[order]
        d.fill_between(x, sm - ss, sm + ss, color="0.7", alpha=0.4,
                       label="surrogate mean ± SD")
        km = (frac[iu] >= 0.95)[order]
        d.plot(x[~km], real[order][~km], ".", ms=3, color="lightsteelblue",
               label="rejected")
        d.plot(x[km], real[order][km], ".", ms=4, color="firebrick",
               label="retained")
        d.legend(fontsize=8)
        d.set(title=f"D. Edge retention ({int(km.sum())}/{len(real)} kept)")
    else:
        d.text(0.5, 0.5, "surrogates not retained\n(PROB_PARAMS.keep_surrogates_for_plot)",
               ha="center", va="center", transform=d.transAxes, color="0.5")
        d.set(title="D. Edge retention")
    d.set(xlabel="edge (sorted by real STTC)", ylabel="STTC"); _aesthetics(d)
    fig.suptitle(f"Edge thresholding — {_title_lag(stem, region, lag)}", fontsize=12)
    fig.tight_layout()
    _finish(fig, os.path.join(out_dir, f"{stem}_{region}_adjM{lag}ms_EdgeThreshold.png"))


def _title_lag(stem, region, lag):
    return f"{stem}{region}".replace("_", "") + f" {lag} ms lag"


def render_network_recording(nr, group, stem, region, root, lag):
    """Write the 4A individual-network figure set + the 3_EdgeThresholdingCheck
    figure for ONE (stem, region, lag). Each figure honours PLOT_ENABLE."""
    if nr is None:
        return None
    rec = f"{stem}_{region}"
    out_dir = os.path.join(root, "4_NetworkActivity", "4A_IndividualNetworkAnalysis",
                           group, rec, f"{lag}mslag")
    os.makedirs(out_dir, exist_ok=True)
    if _on("connectivity_stats"):
        fig_connectivity_stats(nr, out_dir, lag, stem, region)
    if _on("network_plot"):
        fig_network_plot(nr, out_dir, lag, stem, region, pnum="2")   # plain
    for i, metric in enumerate(NET_COLORMAP_METRICS):
        if _on("network_" + metric[0]):
            fig_network_plot(nr, out_dir, lag, stem, region, pnum=str(3 + i),
                             metric=metric)
    if _on("node_cartography"):
        fig_node_cartography(nr, out_dir, lag, stem, region)
    if _on("edge_threshold_check"):
        et_dir = os.path.join(root, "3_EdgeThresholdingCheck", rec)
        os.makedirs(et_dir, exist_ok=True)
        fig_edge_threshold_check(nr, et_dir, lag, stem, region)
    return out_dir


# ==============================================================================
# 2B / 4B GROUP COMPARISONS
# ------------------------------------------------------------------------------
# plotHalfViolinByX.m (half-violin per group [x=age] or per age [x=group]),
# plotGraphMetricsByLag.m (metric vs lag), node-cartography-by-lag (role
# proportions vs lag) and the PC-vs-z density landscape (TrialLandscapeDensity.m).
# MEA-NAP pools by group x DIV with each recording a data point; here each
# (stem, region) window is one recording, so pre/stim/post are pooled the way
# MEA-NAP pools recordings (the pipeline's own Plots/ keep regions separate).
# ==============================================================================
NETWORK_METRIC_LABELS = [
    ("network_size", "Network size"), ("edge_density", "Density"),
    ("mean_degree", "Mean node degree"), ("mean_strength", "Mean node strength"),
    ("clustering_coef", "Clustering coefficient"),
    ("mean_betweenness", "Betweenness centrality"),
    ("global_efficiency", "Global efficiency"),
    ("mean_local_efficiency", "Local efficiency"),
    ("modularity", "Modularity Q"), ("n_modules", "Number of modules"),
    ("mean_participation", "Participation coefficient"),
    ("small_worldness_sigma", "Small-worldness sigma"),
    ("mean_path_length", "Characteristic path length"),
]
NODE_METRIC_LABELS = [
    ("degree", "Node degree"), ("strength", "Node strength"),
    ("betweenness", "Betweenness centrality"), ("clustering", "Clustering coefficient"),
    ("local_efficiency", "Local efficiency"), ("participation", "Participation coefficient"),
    ("within_module_z", "Within-module z-score"),
]
ACTIVITY_METRIC_LABELS = [
    ("mean_FR_hz", "Mean firing rate (Hz)"), ("median_FR_hz", "Median firing rate (Hz)"),
    ("n_active", "Active electrodes"),
    ("network_burst_rate_per_min", "Network burst rate (per min)"),
    ("mean_nb_length_s", "Network burst length (s)"),
    ("frac_spikes_in_nb", "Fraction of spikes in network burst"),
]


def _cat_sorted(series):
    """Unique values of a Series in Categorical order when it is an ORDERED
    Categorical (e.g. groups pre-ordered by measured sweep speed); otherwise a
    plain sorted() — so numeric columns like lag_ms are unaffected."""
    try:
        if str(getattr(series, "dtype", "")) == "category" and series.cat.ordered:
            present = set(series.dropna().tolist())
            return [c for c in series.cat.categories if c in present]
    except Exception:
        pass
    return sorted(series.dropna().unique())


def _region_list(df):
    """Region labels present in df, ordered by REGION_ORDER then alphabetically."""
    regs = list(df["region"].dropna().unique()) if "region" in df.columns else []
    ordered = [r for r in REGION_ORDER if r in regs]
    ordered += sorted(r for r in regs if r not in REGION_ORDER)
    return ordered


_REGION_FALLBACK = plt.cm.tab10(np.linspace(0, 1, 10))


def _region_color(reg, i):
    pal = _region_palette()
    if reg in pal:
        return pal[reg]
    return tuple(_REGION_FALLBACK[i % 10][:3])


def fig_region_by_x(df, metric, label, x_col, out_path, panel_col=None, xlabel=None):
    """Grouped half-violins with `region` (pre/stim/post) as the compared factor.
    x_col categories run along the x-axis; at each, one half-violin per region.
    panel_col (optional) makes one subplot per its value (e.g. per group), so all
    three of {sample identity=stem, x_col, region} are preserved. This is the
    region-aware replacement for MEA-NAP's plotHalfViolinByX.m."""
    if df is None or df.empty or metric not in df.columns:
        return
    regions = _region_list(df)
    x_cats = _cat_sorted(df[x_col])
    if not regions or not x_cats:
        return
    panels = (_cat_sorted(df[panel_col])
              if panel_col and panel_col in df.columns else [None])
    # Spacing so a region's jittered points (left) never overlap the neighbouring
    # region's violin (right). One half-violin spans ~[pos-foot, pos+foot]; regions
    # are placed region_step apart (> 2*foot -> a clear gap), and x-categories are
    # cat_step apart. The figure width scales with the total x-extent so the
    # violins keep a good visual size instead of being squeezed together.
    vw = 0.13                                       # per-region half-violin width
    nreg = len(regions)
    foot = vw + 0.12                                # half-footprint (points + violin)
    region_step = 2 * foot + 0.14                   # clear gap between regions
    half_span = (nreg - 1) / 2 * region_step + foot
    cat_step = 2 * half_span + 0.7                  # clear gap between x categories
    centers = [(xi + 1) * cat_step for xi in range(len(x_cats))]
    extent = (centers[-1] - centers[0]) + 2 * half_span + 1.0
    pw = max(4.6, 1.4 * extent)                     # per-panel width (inches)
    fig, axes = plt.subplots(1, len(panels), figsize=(pw * len(panels), 5),
                             squeeze=False, sharey=True)
    axes = axes[0]
    for pi, pv in enumerate(panels):
        ax = axes[pi]
        d = df if pv is None else df[df[panel_col] == pv]
        for xi, xc in enumerate(x_cats):
            base = centers[xi]
            for ri, reg in enumerate(regions):
                vals = d[(d[x_col] == xc) & (d["region"] == reg)][metric].values
                off = (ri - (nreg - 1) / 2) * region_step
                _half_violin(ax, np.asarray(vals, float), base + off,
                             _region_color(reg, ri), width=vw)
        ax.set_xticks(centers)
        ax.set_xticklabels([str(x) for x in x_cats], fontsize=8)
        ax.set_xlabel(xlabel or x_col.capitalize())
        ax.set_xlim(centers[0] - half_span - 0.5, centers[-1] + half_span + 0.5)
        if pv is not None:
            ax.set_title(str(pv), fontsize=10)
        if pi == 0:
            ax.set_ylabel(label)
        _aesthetics(ax)
    handles = [plt.Line2D([0], [0], marker="s", ls="", markersize=8,
                          markerfacecolor=_region_color(r, i), markeredgecolor="none",
                          label=r) for i, r in enumerate(regions)]
    axes[-1].legend(handles=handles, fontsize=7, title="region", loc="best",
                    framealpha=0.9)
    _finish(fig, out_path)


def fig_metric_by_lag(df, metric, label, out_path, hue="region"):
    """plotGraphMetricsByLag.m: metric vs STTC lag, mean +/- SEM, one line per
    region (pre/stim/post) so each window's lag profile is compared."""
    if metric not in df.columns or df.empty:
        return
    lags = sorted(df["lag_ms"].dropna().unique())
    hues = _region_list(df) if hue == "region" else _cat_sorted(df[hue])
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for hi, hv in enumerate(hues):
        col = _region_color(hv, hi) if hue == "region" else plt.cm.tab10(hi % 10)[:3]
        mu, se = [], []
        for lg in lags:
            v = df[(df["lag_ms"] == lg) & (df[hue] == hv)][metric].values
            v = v[np.isfinite(v)]
            mu.append(np.mean(v) if v.size else np.nan)
            se.append(np.std(v, ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0)
        mu, se = np.array(mu), np.array(se)
        ax.plot(lags, mu, "-o", color=col, ms=4, label=str(hv))
        ax.fill_between(lags, mu - se, mu + se, color=col, alpha=0.2)
    ax.set_xlabel("STTC lag (ms)"); ax.set_ylabel(label)
    ax.legend(fontsize=7, title=("region" if hue == "region" else hue))
    _aesthetics(fig.axes[0])
    _finish(fig, out_path)


def fig_cartography_by_lag(node_df, out_path):
    """Node-cartography role proportions across STTC lag (active nodes)."""
    if node_df.empty or "role" not in node_df.columns:
        return
    act = node_df[node_df["role"] > 0]
    lags = sorted(node_df["lag_ms"].dropna().unique())
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for r in range(1, 7):
        prop = [np.mean(act[act["lag_ms"] == lg]["role"].values == r)
                if (act["lag_ms"] == lg).any() else np.nan for lg in lags]
        ax.plot(lags, prop, "-o", color=_ROLE_COLORS[r - 1], ms=4,
                label=f"{r} {_ROLE_NAMES[r - 1]}")
    ax.set_xlabel("STTC lag (ms)"); ax.set_ylabel("Proportion of active nodes")
    ax.legend(fontsize=6, bbox_to_anchor=(1.01, 1), loc="upper left")
    _aesthetics(ax); _finish(fig, out_path)


def fig_density_landscape(node_df, lag, out_path):
    """TrialLandscapeDensity.m: 2D density of participation coefficient (x) vs
    within-module z (y), pooled across nodes at one lag, with role boundaries."""
    d = node_df[node_df["lag_ms"] == lag]
    pc = d["participation"].values.astype(float)
    z = d["within_module_z"].values.astype(float)
    ok = np.isfinite(pc) & np.isfinite(z)
    pc, z = pc[ok], z[ok]
    if pc.size < 3:
        return
    fig, ax = plt.subplots(figsize=(6, 5.4))
    hb = ax.hexbin(pc, z, gridsize=25, cmap=_cmap(MEANAP_PLOT_PARAMS["heatmap_cmap"]),
                   mincnt=1)
    cb = fig.colorbar(hb, ax=ax); cb.set_label("node count")
    cb.outline.set_visible(False)
    hub = _CART["hub"]
    for xline in (_CART["peri"], _CART["nonhub"], _CART["prohub"], _CART["conhub"]):
        ax.axvline(xline, ls="--", color="w", lw=0.8)
    ax.axhline(hub, ls="--", color="w", lw=0.8)
    ax.set_xlabel("participation coefficient")
    ax.set_ylabel("within-module degree z-score")
    ax.set_title(f"Density landscape — {lag} ms lag")
    _aesthetics(ax); _finish(fig, out_path)


def _P(root, *parts):
    p = os.path.join(root, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


# per-electrode neuronal-activity metrics (fill the 2B NodeByGroup / NodeByAge)
ELECTRODE_METRIC_LABELS = [
    ("fr", "Firing rate (Hz)"), ("burst_rate", "Electrode burst rate"),
    ("frac_spikes_in_burst", "Fraction of spikes in burst"),
    ("mean_burst_dur_s", "Burst duration (s)"),
    ("isi_within_burst_s", "ISI within burst (s)"),
    ("isi_outside_burst_s", "ISI outside burst (s)"),
]


def _safe(fn, *a, **k):
    """Run a render call, logging (never raising) so one figure can't abort the
    rest of the group-comparison set."""
    try:
        fn(*a, **k)
    except Exception as e:
        print(f"  [MEA-NAP plot warn] {getattr(fn, '__name__', fn)}: {e!r}")


def render_group_comparisons(activity_df, node_df, net_df, root, electrode_df=None):
    """Write the full 2B (activity) + 4B (network) group-comparison set, with
    region (pre/stim/post) as the compared factor within each plot:
      byGroup : x = group,  one violin per region, pooled across samples/days.
      byAge   : x = age,    one violin per region, one subplot per group.
    Each sub-section is isolated so a failure in one still lets the others run."""
    NA = "4_NetworkActivity/4B_GroupComparisons"
    AB = "2_NeuronalActivity/2B_GroupComparisons"
    if net_df is not None and len(net_df):
        for lag in sorted(net_df["lag_ms"].dropna().unique()):
            nd = net_df[net_df["lag_ms"] == lag]
            for i, (m, lab) in enumerate(NETWORK_METRIC_LABELS, 1):
                if _on("network_by_group"):
                    _safe(fig_region_by_x, nd, m, lab, "group",
                        _P(root, *f"{NA}/3_RecordingsByGroup/HalfViolinPlots/{lag}mslag".split("/"),
                           f"{i}_{lab}_byGroup.png"), xlabel="Group")
                if _on("network_by_age"):
                    _safe(fig_region_by_x, nd, m, lab, "age",
                        _P(root, *f"{NA}/4_RecordingsByAge/HalfViolinPlots/{lag}mslag".split("/"),
                           f"{i}_{lab}_byAge.png"), panel_col="group", xlabel="Age")
        if _on("graph_metrics_by_lag"):
            for i, (m, lab) in enumerate(NETWORK_METRIC_LABELS, 1):
                _safe(fig_metric_by_lag, net_df, m, lab,
                    _P(root, *f"{NA}/5_GraphMetricsByLag".split("/"), f"{i}_{lab}.png"))
    if node_df is not None and len(node_df):
        for lag in sorted(node_df["lag_ms"].dropna().unique()):
            nd = node_df[node_df["lag_ms"] == lag]
            for i, (m, lab) in enumerate(NODE_METRIC_LABELS, 1):
                if _on("node_by_group"):
                    _safe(fig_region_by_x, nd, m, lab, "group",
                        _P(root, *f"{NA}/1_NodeByGroup/{lag}mslag".split("/"),
                           f"{i}_{lab}_byGroup.png"), xlabel="Group")
                if _on("node_by_age"):
                    _safe(fig_region_by_x, nd, m, lab, "age",
                        _P(root, *f"{NA}/2_NodeByAge/{lag}mslag".split("/"),
                           f"{i}_{lab}_byAge.png"), panel_col="group", xlabel="Age")
            if _on("density_landscape"):
                _safe(fig_density_landscape, node_df, lag,
                    _P(root, *f"{NA}/7_DensityLandscape".split("/"),
                       f"DensityLandscape_{lag}mslag.png"))
        if _on("cartography_by_lag"):
            _safe(fig_cartography_by_lag, node_df,
                _P(root, *f"{NA}/6_NodeCartographyByLag".split("/"),
                   "NodeCartographyProportionsByLag.png"))
    if activity_df is not None and len(activity_df):
        for i, (m, lab) in enumerate(ACTIVITY_METRIC_LABELS, 1):
            if _on("activity_by_group"):
                _safe(fig_region_by_x, activity_df, m, lab, "group",
                    _P(root, *f"{AB}/3_RecordingsByGroup/HalfViolinPlots".split("/"),
                       f"{i}_{lab}_byGroup.png"), xlabel="Group")
            if _on("activity_by_age"):
                _safe(fig_region_by_x, activity_df, m, lab, "age",
                    _P(root, *f"{AB}/4_RecordingsByAge/HalfViolinPlots".split("/"),
                       f"{i}_{lab}_byAge.png"), panel_col="group", xlabel="Age")
    # 2B node-level (per-electrode) activity -> fills the previously empty
    # 2B/1_NodeByGroup and 2B/2_NodeByAge folders.
    if electrode_df is not None and len(electrode_df):
        for i, (m, lab) in enumerate(ELECTRODE_METRIC_LABELS, 1):
            if m not in electrode_df.columns:
                continue
            if _on("electrode_by_group"):
                _safe(fig_region_by_x, electrode_df, m, lab, "group",
                    _P(root, *f"{AB}/1_NodeByGroup".split("/"),
                       f"{i}_{lab}_byGroup.png"), xlabel="Group")
            if _on("electrode_by_age"):
                _safe(fig_region_by_x, electrode_df, m, lab, "age",
                    _P(root, *f"{AB}/2_NodeByAge".split("/"),
                       f"{i}_{lab}_byAge.png"), panel_col="group", xlabel="Age")


# ==============================================================================
# BATCH DRIVER  --  called by mea_nap_activity.run_stage2 (installment 4 wire-in)
# ==============================================================================
def batch_max_firing_rate(units_by_stem):
    """Dataset-wide max per-electrode firing rate (spikeFreqMax)."""
    mx = 0.1
    for units in units_by_stem.values():
        for u in units:
            dur = float(u["duration"])
            if dur <= 0:
                continue
            for s in u["spikes"]:
                mx = max(mx, len(s) / dur)
    return mx


def group_max_firing_rate(units_by_stem):
    """Per-GROUP max per-electrode firing rate -> {group: maxFR}, used to scale the
    second heatmap/raster panel to the recording's group (not the whole dataset)."""
    gmax = {}
    for units in units_by_stem.values():
        for u in units:
            dur = float(u["duration"])
            if dur <= 0:
                continue
            g = u.get("group", "?")
            m = max((len(s) / dur for s in u["spikes"]), default=0.0)
            gmax[g] = max(gmax.get(g, 0.1), m)
    return gmax


def electrode_activity_row(unit, res):
    """One row per ELECTRODE for the 2B node-level (per-electrode) comparisons:
    stem/region/group/age + per-electrode firing rate & burst metrics."""
    fr = np.asarray(res["fr"], float)
    per = res.get("per", {}) or {}
    rows = []
    for i in range(len(fr)):
        row = {"stem": unit["stem"], "region": unit["region"],
               "group": unit["group"], "age": unit["age"], "fr": float(fr[i])}
        for k in ("burst_rate", "frac_spikes_in_burst", "mean_burst_dur_s",
                  "isi_within_burst_s", "isi_outside_burst_s"):
            v = per.get(k)
            row[k] = float(v[i]) if v is not None and i < len(v) else np.nan
        rows.append(row)
    return rows


# ==============================================================================
# SELF-TEST -- synthetic units -> renders the full set into ./_meanap_selftest/
# ==============================================================================
def _synthetic_unit(stem, region, age, dur=300.0, seed=0):
    rng = np.random.default_rng(seed)
    grounded = ["15", "26", "63"]                    # reference + two grounded
    labels, spikes = [], []
    for col in range(1, 9):
        for row in range(1, 9):
            if (col, row) in [(1, 1), (1, 8), (8, 1), (8, 8)]:
                continue
            lab = f"{col}{row}"
            if lab in grounded:                      # excluded from analysis
                continue
            rate = rng.uniform(0.2, 8.0) * (1.6 if region == "stim" else 1.0)
            n = rng.poisson(rate * dur)
            spikes.append(np.sort(rng.uniform(0, dur, n)).astype(np.float32))
            labels.append(lab)
    fr = np.array([len(s) / dur for s in spikes])
    unit = {"stem": stem, "region": region, "age": age, "group": "Control",
            "duration": dur, "labels": labels, "spikes": spikes,
            "grounded": grounded}
    res = {"fr": fr, "active": fr >= 0.1, "per": {}, "net_bursts": []}
    return unit, res


def _synthetic_nr(n=40, density=0.15, seed=3, lag=25):
    rng = np.random.default_rng(seed)
    A = rng.random((n, n)); A = (A + A.T) / 2
    thr = np.quantile(A[np.triu_indices(n, 1)], 1 - density)
    W = np.where(A >= thr, A, 0.0); np.fill_diagonal(W, 0.0)
    deg = (W > 0).sum(1).astype(float)
    labs = [l for l in _mcs60_labels()][:n]
    nodes = []
    for i in range(n):
        z = rng.normal(0, 1.3)
        pc = np.clip(rng.random(), 0, 1)
        role = (1 if z <= 2.5 and pc <= 0.625 else 2 if z <= 2.5 and pc <= 0.8
                else 3 if z <= 2.5 else 4 if pc <= 0.3 else 5 if pc <= 0.75 else 6)
        nodes.append({"label": labs[i], "degree": float(deg[i]),
                      "strength": float(W[i].sum()), "betweenness": rng.random(),
                      "clustering": rng.random(), "local_efficiency": rng.random(),
                      "participation": pc, "within_module_z": z,
                      "module": int(rng.integers(1, 5)), "role": role})
    return {"labels": labs, "M": W.copy(), "W": W, "frac": np.zeros((n, n)),
            "surr": None, "net": {}, "nodes": nodes, "lag_ms": lag,
            "pc": np.array([nd["participation"] for nd in nodes]),
            "z": np.array([nd["within_module_z"] for nd in nodes])}


if __name__ == "__main__":
    root = os.path.join(os.getcwd(), "_meanap_selftest",
                        MEANAP_PLOT_PARAMS["output_folder_name"])
    create_output_folders(root, ["Control", "Laser"])
    units = [_synthetic_unit("MEA_D14_A2", r, 14, seed=i)
             for i, r in enumerate(["pre", "stim", "post"])]
    gmax = {"Control": max(float(np.nanmax(res["fr"])) for _, res in units)}
    for unit, res in units:
        render_activity_recording(unit, res, root, gmax.get(unit["group"], 1.0))
    nr = _synthetic_nr()
    nr["frac"] = np.random.default_rng(1).random((nr["W"].shape[0],) * 2)
    nr["surr"] = np.random.default_rng(2).random((50,) + nr["W"].shape)
    render_network_recording(nr, "Control", "MEA_D14_A2", "stim", root, 25)

    # synthetic dataframes for the group-comparison set
    import pandas as pd
    rng = np.random.default_rng(7)
    net_rows, node_rows, act_rows = [], [], []
    for grp in ["Control", "Laser"]:
        for age in [14, 21]:
            for rep in range(4):                       # 4 samples (stems) per grp x age
                stem = f"MEA_{grp}_D{age}_A{rep}"
                for region in ["pre", "stim", "post"]:
                    bump = 1.5 if region == "stim" else 1.0
                    for lag in [10, 25, 50]:
                        net_rows.append({"stem": stem, "region": region, "group": grp,
                            "age": age, "lag_ms": lag,
                            **{m: rng.random() * bump * (2 if grp == "Laser" else 1)
                               for m, _ in NETWORK_METRIC_LABELS}})
                        for _ in range(30):
                            node_rows.append({"stem": stem, "region": region,
                                "group": grp, "age": age, "lag_ms": lag,
                                "role": int(rng.integers(1, 7)),
                                **{m: rng.random() * bump for m, _ in NODE_METRIC_LABELS}})
                    act_rows.append({"stem": stem, "region": region, "group": grp,
                        "age": age,
                        **{m: rng.random() * 10 * bump for m, _ in ACTIVITY_METRIC_LABELS}})
    elec_rows = []
    for grp in ["Control", "Laser"]:
        for age in [14, 21]:
            for rep in range(4):
                for region in ["pre", "stim", "post"]:
                    bump = 1.5 if region == "stim" else 1.0
                    for _ in range(50):
                        elec_rows.append({"stem": f"MEA_{grp}_D{age}_A{rep}",
                            "region": region, "group": grp, "age": age,
                            "fr": rng.random() * 8 * bump,
                            "burst_rate": rng.random() * 5 * bump,
                            "frac_spikes_in_burst": rng.random()})
    render_group_comparisons(pd.DataFrame(act_rows), pd.DataFrame(node_rows),
                             pd.DataFrame(net_rows), root,
                             electrode_df=pd.DataFrame(elec_rows))
    print("SELFTEST full set done ->", root)
