#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 mea_nap_gui.py  --  one-window launcher for the whole pipeline  (PyQt6)
================================================================================
Fills in the paths, sets any parameters in the tabbed panel, then runs Stage 1
(spike detection, mea_nap_io) and Stage 2 (activity + network, mea_nap_activity)
SEQUENTIALLY in a background thread -- start it, leave it overnight, and Stage 2
begins automatically the moment Stage 1 finishes.

This is the PyQt6 port of the original Tkinter GUI. The pipeline back-end
(mea_nap_io / mea_nap_activity / mea_nap_wavelet_detect / mea_nap_original_plots /
mea_nap_progress and the meanap/ package) is UNCHANGED -- only the window layer
was rebuilt on Qt, which gives smooth live-resize, crisp high-DPI rendering, and
the shared MEA-Suite look via mea_theme.apply_qt().

The worker still runs on a plain background thread that posts commands to a
queue.Queue; a QTimer drains that queue on the GUI thread (Qt widgets are only
ever touched on the main thread), so the pipeline's EmbeddedProgress drop-in for
mea_nap_progress.BarProgress works exactly as before.

Features
--------
* Tabbed parameter panel exposing EVERY tunable parameter across all modules,
  grouped by section, each row showing its adjustment range.
* A large overall-progress bar (percentage) above the per-sample rings.
* Per-sample progress drawn INSIDE this window (no popups) + a live log panel.
* High-DPI aware (crisp on high-resolution Windows displays; automatic on Qt6).
* Entered settings persist to mea_nap_gui_settings.json (beside this script).

Run (use the venv python that has numba + CUDA torch + PyQt6):
  & "...\\.venv\\Scripts\\python.exe" "...\\Scripts\\MEA-NAP\\mea_nap_gui.py"
Must sit beside mea_nap_io / mea_nap_activity / mea_nap_wavelet_detect /
mea_nap_original_plots / mea_nap_progress, mea_theme.py, and the meanap/ package.
================================================================================
"""

import os
import re
import sys
import json
import queue
import threading
import traceback

import matplotlib
matplotlib.use("Agg")          # pipeline figures are SAVED; this GUI uses Qt

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

import mea_nap_io as io
import mea_nap_wavelet_detect as det
import mea_nap_activity as activity
import mea_nap_original_plots as orig
import mea_nap_spectral_propagation as spprop
import mea_nap_spkit_egm as spkit_egm
import mea_nap_phase_lockin as phase_lockin
import mea_nap_burst_latency as burst_latency
import mea_nap_psd_trend as psd_trend
import mea_nap_digital_events as digital
import mea_nap_progress as progress_gui
import mea_theme

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "mea_nap_gui_settings.json")
POLL_MS = 100

# per-sample circular progress ring geometry + reflowing grid cell width (px)
RING, RING_TH, CELL_W = 68, 8, 150
_STATE_COLOR = {
    "pending": "#4a5160", "running": "#4f9dff", "done": "#3ddc84",
    "skipped": "#f0a63a", "failed": "#ef6b62",
}
_TRACK = "#39404e"                              # ring track on the dark background

# ---- palette (single source of truth = the shared MEA-Suite theme) ----------
# Colours the custom-painted bits (rings, banner, env dots) need directly; the
# rest of the look comes from mea_theme.apply_qt()'s stylesheet.
ACCENT = mea_theme.accent("analyzer")           # MEA Analyzer blue (#3B82F6)
PALETTE = {
    "bg":       mea_theme.WINDOW_TOP,
    "panel":    mea_theme.PANEL,
    "banner":   mea_theme.WINDOW_TOP,
    "banner2":  mea_theme.TEAL,
    "accent":   ACCENT,
    "text":     mea_theme.TEXT,
    "muted":    mea_theme.TEXT_MUTED,
    "ok":       "#34D399",
    "warn":     "#FBBF24",
    "bad":      "#EF5350",
    "line":     mea_theme.LINE,
    "log_bg":   mea_theme.PLOT_BG,
}


# ==============================================================================
# stdout capture + the make()-compatible embedded progress object
# ==============================================================================
class _QueueWriter:
    def __init__(self, q):
        self.q = q

    def write(self, s):
        if s and "\r" not in s:
            self.q.put(("log", s))

    def flush(self):
        pass


class _Sink:
    def write(self, *a):
        pass

    def flush(self):
        pass


class EmbeddedProgress:
    """Drop-in for mea_nap_progress.BarProgress: enqueues commands the GUI thread
    renders into its embedded ring area (Qt touched only on the main thread)."""
    def __init__(self, q, names, title="Progress", subtitle=""):
        self.q = q
        self.ok = True
        q.put(("newrun", list(names), title, subtitle))

    def start(self, name, note="running"):
        self.q.put(("update", name, 0.0, "running", note))

    def set(self, name, frac, state=None, note=None):
        self.q.put(("update", name, frac, state, note))

    def done(self, name, note="done"):
        self.q.put(("update", name, 1.0, "done", note))

    def skipped(self, name, note="skipped"):
        self.q.put(("update", name, 1.0, "skipped", note))

    def failed(self, name, note="failed"):
        self.q.put(("update", name, 1.0, "failed", note))

    def close(self):
        pass


def _P(obj, dictname, key, label, typ, rng="", options=None):
    return {"obj": obj, "key": key, "skey": f"{dictname}.{key}", "label": label,
            "type": typ, "rng": rng, "options": options}


def _MA(skey, label, typ, getter, setter, rng="", options=None):
    return {"obj": None, "getter": getter, "setter": setter, "skey": skey,
            "label": label, "type": typ, "rng": rng, "options": options}


# ==============================================================================
# HELP TEXT (from the MEA-NAP paper, Sit et al. 2024 Cell Reports Methods, and the
# MEA-NAP docs). Shown on hovering the "?" next to each parameter / plot option.
# ==============================================================================
PARAM_HELP = {
    "DETECT_PARAMS.bp_low_hz": "Lower cut-off (high-pass) of the pre-detection band-pass filter; removes low-frequency drift/LFP so only the spike band remains.",
    "DETECT_PARAMS.bp_high_hz": "Upper cut-off (low-pass) of the pre-detection band-pass filter; removes high-frequency noise above the spike band. With the low cut-off it isolates the action-potential frequency range.",
    "DETECT_PARAMS.wid_ms": "Expected spike-width window (ms) the continuous-wavelet detector searches, defining the wavelet scales matched to action-potential shape. Bracket the true spike duration (fractions of a ms to ~1 ms).",
    "DETECT_PARAMS.ns": "Number of wavelet scales sampled within the spike-width range. More scales sample the range more finely at extra computational cost.",
    "DETECT_PARAMS.L": "Bayesian cost of the wavelet detector, trading false positives vs false negatives (~ -2 to 2). More negative is more conservative; MEA-NAP recommends starting at L = -0.12.",
    "DETECT_PARAMS.option": "How the detection cost L is applied — 'l' (liberal, fixed user cost) vs 'c' (conservative). Use liberal for reproducible tuned sensitivity across recordings.",
    "QUALITY_PARAMS.auto_ground": "Automatically exclude dead/saturated/noisy electrodes before analysis (in addition to the Batch.csv grounded column).",
    "QUALITY_PARAMS.flat_uv": "Voltage-amplitude floor (µV): electrodes with less variation than this are treated as flat/dead and removed.",
    "QUALITY_PARAMS.noisy_mad_ratio": "Median-absolute-deviation ratio above which an electrode is judged excessively noisy and rejected from spike/network analysis.",
    "QUALITY_PARAMS.silent_max_rate_hz": "Firing-rate ceiling (spikes/s) below which an electrode is treated as silent and excluded, so near-quiescent channels don't add noise to connectivity.",
    "CONFIG.WINDOWS_FROM_TRIGGERS": "Snap each Batch.csv window edge to the laser-trigger block recorded in the file (50 triggers/s while sweeping, silence at baseline). Fixes recordings whose stimulation started late: the recorded block drifts up to 44 s from the nominal 300-3600 s rows, which otherwise leaks stimulation into the start of 'post'. Edges further than 60 s from a block edge, and files without triggers (controls), are untouched; every change is logged.",
    "FR_PARAMS.active_min_rate_hz": "Minimum firing rate (spikes/s) for an electrode to count as an active node included in the analysis.",
    "BURST_PARAMS.method": "Single-electrode burst detector. ISI_N (Bakkum) is adaptive and MEA-NAP-aligned; maxinterval is the classic fixed-gap method.",
    "BURST_PARAMS.isi_n_N": "The N of the Bakkum ISI_N burst detector — minimum number of consecutive close spikes to call a burst. Larger N needs longer spike sequences (MEA-NAP default 10).",
    "BURST_PARAMS.isi_n_thr_s": "Inter-spike-interval threshold (s) for ISI_N: spikes closer than this are grouped into the same burst.",
    "NETBURST_PARAMS.net_N": "Minimum number of spikes to define an array-wide (network) burst across electrodes.",
    "NETBURST_PARAMS.net_isi_thr_s": "Inter-spike-interval threshold (s) grouping spikes into a network burst across electrodes.",
    "NETBURST_PARAMS.net_min_chan": "Minimum electrodes that must participate for an event to count as a network burst rather than local activity.",
    "STTC_PARAMS.lags_ms": "STTC time-lag(s) Δt (ms) — the window within which spikes on two electrodes count as coincident (MEA-NAP default 10, 25, 50). Shorter lags emphasise direct connections; longer lags capture multi-synaptic activity.",
    "STTC_PARAMS.dt_s": "STTC coincidence window Δt in seconds (same synchrony tolerance as the lag). Smaller windows detect only tightly synchronised firing.",
    "PROB_PARAMS.n_surrogates": "Circular-shift surrogate iterations per electrode pair for probabilistic edge thresholding (ProbThreshRepNum, default 200). More iterations give a more stable significance threshold.",
    "PROB_PARAMS.tail": "Significance tail (ProbThreshTail): 0.05 keeps edges above the 95th surrogate percentile, 0.01 the 99th. Smaller = stricter, fewer edges.",
    "PROB_PARAMS.sig_threshold": "The surrogate percentile a real STTC must exceed to keep its edge (= 1 − tail). Edges failing this are set to zero.",
    "PROB_PARAMS.bins_per_dt": "Time bins per STTC lag in the GPU surrogate path; more bins = finer/exact but more VRAM. Lower it if you hit an out-of-memory error.",
    "PROB_PARAMS.strict_gpu": "If on, a GPU STTC failure raises loudly instead of silently dropping to the ~1000× slower exact CPU path.",
    "PARITY_PARAMS.sttc_backend": "STTC implementation: 'gpu' (binned-matmul, ~1000× faster, matches exact within ~0.006) vs 'exact' (the lab's bit-for-bit CPU routine, impractically slow on long windows).",
    "PARITY_PARAMS.pc_mode": "How the participation coefficient is computed — 'norm' (null-model normalised, what MEA-NAP reports) vs 'raw' (fast, deterministic).",
    "PARITY_PARAMS.min_nodes": "Minimum active nodes a recording must have before graph metrics are computed (minNumberOfNodesToCalNetMet; ~25 for 60/64-electrode MEAs).",
    "NET_PARAMS.community_resolution": "Louvain modularity resolution γ. Above 1 gives more, smaller modules; below 1 gives fewer, larger modules.",
    "CARTOGRAPHY_PARAMS.hubBoundaryWMdDeg": "Within-module degree z-score threshold separating hub from non-hub nodes in node cartography (classically ~2.5).",
    "CARTOGRAPHY_PARAMS.periPartCoef": "Participation-coefficient boundary between peripheral non-hubs (edges within their module) and non-hub connectors (edges reaching other modules).",
    "CARTOGRAPHY_PARAMS.nonHubconnectorPartCoef": "Upper participation boundary for non-hub connectors, above which a non-hub is 'kinless' (edges spread across many modules).",
    "CARTOGRAPHY_PARAMS.proHubpartCoef": "Participation boundary below which a hub is 'provincial' — highly connected mainly within its own module.",
    "CARTOGRAPHY_PARAMS.connectorHubPartCoef": "Participation boundary above which a hub is a connector/kinless hub linking many different modules.",
    "PLOT_PARAMS.raster_dpi": "Resolution for the (dense) spike-raster PNGs only. Rasters carry up to millions of ticks and are the biggest render cost; 300 looks identical to 600 but renders faster and files are smaller. Raise to 600 to match the other figures exactly.",
    "MEANAP_PLOT_PARAMS.raster_cmap": "Colormap for the raster. parula is MEA-NAP's default; sequential maps (viridis, magma, …) suit firing-rate data.",
    "MEANAP_PLOT_PARAMS.heatmap_cmap": "Colormap for the electrode heatmap and the density landscape.",
    "MEANAP_PLOT_PARAMS.network_node_cmap": "Colormap used when network nodes are coloured by a graph metric (degree, participation, …).",
    "MEANAP_PLOT_PARAMS.violin_palette": "Colour scheme for the pre/stim/post half-violins. 'region' uses the fixed REGION_COLORS; the rest are matplotlib qualitative palettes.",
    "MEANAP_PLOT_PARAMS.violin_point_size": "Size of the individual jittered data points in half-violin plots.",
    "MEANAP_PLOT_PARAMS.violin_mean_size": "Size of the black mean dot in half-violin plots.",
    "MEANAP_PLOT_PARAMS.node_min_size": "Network node diameter (grid units) for the lowest-degree node.",
    "MEANAP_PLOT_PARAMS.node_max_size": "Network node diameter (grid units) for the highest-degree node — lower it if nodes overlap.",
    "MEANAP_PLOT_PARAMS.edge_max_width": "Thickest network edge (line width) at the strongest STTC weight; thinner/lighter edges are weaker.",
    "SPECTRAL_PARAMS.psd_bin_s": "Bin width (s) used to turn spikes into the population firing-rate time series before the Welch PSD. Smaller bins resolve higher frequencies but add noise (0.05 s ≈ up to 10 Hz).",
    "SPECTRAL_PARAMS.psd_fmax_hz": "Upper frequency (Hz) shown on the firing/bursting spectra. Spontaneous MEA rhythms are slow, so 10 Hz usually captures the band of interest.",
    "SPECTRAL_PARAMS.psd_welch_nperseg": "Welch segment length (samples). Longer segments give finer frequency resolution but fewer averages (noisier). Must not exceed the number of bins in the shortest region.",
    "SPECTRAL_PARAMS.psd_logy": "Plot spectral power on a log axis. On is usually clearer because PSD power spans several orders of magnitude.",
    "PROP_PARAMS.electrode_pitch_um": "Centre-to-centre electrode spacing (µm) of the MEA. Sets the physical distance scale, so it directly scales the conduction-velocity estimate (MCS 60-MEA = 200 µm).",
    "PROP_PARAMS.min_electrodes": "Minimum electrodes that must fire within an event before a propagation direction/velocity is fitted. Higher = more reliable plane fits, fewer events.",
    "PROP_PARAMS.min_fit_r2": "Minimum R² of the local-activation-time plane fit for an event to count as a coherent travelling wave. Raise it to keep only clean, planar propagation.",
    "PROP_PARAMS.window_ms": "Sliding-window length (ms) used by the window-based propagation method to snapshot travelling waves independent of burst detection.",
    "PROP_PARAMS.window_step_ms": "Step (ms) between successive sliding windows; smaller steps oversample the recording at more compute cost.",
    "PROP_PARAMS.window_min_active_frac": "Fraction of electrodes that must be active inside a sliding window for it to be fitted, so quiet windows are skipped.",
    "PROP_PARAMS.sweep_angle_deg": "Direction of the stimulation laser sweep in electrode coordinates (135° = bottom-right → top-left). Drawn as the red dashed axis and used for the alignment score.",
    "PROP_PARAMS.max_maps": "How many example activation-time maps to draw per region (top row of the propagation figure). 1 keeps the figure compact.",
    "PROP_PARAMS.max_events_per_region": "Cap on propagation events fitted per region (sliding windows or laser triggers). A 1-hour or high-trigger-rate recording can present 100k+ events; they're uniformly subsampled to this many, which gives the same direction/CV statistics without the compute blowing up. Raise for more precision, lower for speed.",
    "CVFIELD_PARAMS.cv_field_mode": "Which events feed the averaged activation map for the local CV flow-field: 'bursts' (detected network bursts, cleaner) or 'windows' (fixed sliding windows).",
    "CVFIELD_PARAMS.cv_neighbor_factor": "Radius of the local plane fit, as a multiple of the electrode pitch. ~1.6 gathers the 8 nearest neighbours; larger = smoother field, fewer distinct vectors.",
    "CVFIELD_PARAMS.cv_min_neighbors": "Minimum electrodes inside the local radius needed to fit a velocity vector there. Higher = smoother, fewer vectors.",
    "CVFIELD_PARAMS.cv_local_min_r2": "Minimum local plane-fit R² to keep a vector; drops incoherent local patches so the flow field shows only reliable propagation.",
    "SPKIT_EGM.enable": "Master switch for the spkit continuous-EGM conduction-velocity pipeline (runs on the RAW voltage, independent of our spike detection). When on, it runs after Stage 2 and writes to the SpKit/ folder — good for an overnight cross-check.",
    "SPKIT_EGM.evt_bp_low_hz": "Low edge (Hz) of the band used to build the population activity envelope for detecting network-event onsets in the continuous signal.",
    "SPKIT_EGM.evt_bp_high_hz": "High edge (Hz) of that event-detection band.",
    "SPKIT_EGM.evt_k_mad": "Network-event threshold = median + k×MAD of the population envelope. Higher k = fewer, stronger events averaged into each electrode's representative EGM.",
    "SPKIT_EGM.evt_min_isi_ms": "Minimum spacing (ms) between detected events, so one burst isn't counted many times.",
    "SPKIT_EGM.evt_max_events": "Cap on how many events are averaged (strongest kept). Limits compute on very active recordings.",
    "SPKIT_EGM.evt_min_events": "Minimum events required to analyse a region window; below this the region is skipped (too little data for a stable EGM).",
    "SPKIT_EGM.egm_pre_ms": "Milliseconds of signal kept before each event onset when extracting the per-electrode EGM segment.",
    "SPKIT_EGM.egm_post_ms": "Milliseconds kept after each event onset; should be long enough to contain the propagating activation across the whole array.",
    "SPKIT_EGM.at_method": "spkit activation-time criterion: 'min_dvdt' (max negative slope, standard for extracellular EGM), 'max_dvdt', or 'max_abs_dvdt'.",
    "SPKIT_EGM.at_gradient": "spkit gradient/derivative method used before finding the activation slope: 'fdiff' (finite difference) is the default and needs no smoothing window; 'gauss_diff' and the 'sg*' variants add smoothing (good for noisy Control recordings) and REQUIRE the smoothing windows below to be non-zero.",
    "SPKIT_EGM.gauss_window": "Length (samples, forced odd) of the Gaussian smoothing kernel used by the 'gauss_diff' gradient method. Must be >= 3 — spkit's default of 0 builds an empty kernel and crashes. Ignored by fdiff/fgrad.",
    "SPKIT_EGM.sg_window": "Length (samples, forced odd) of the Savitzky-Golay window used by the 'sg*_diff' gradient methods. Must be >= 5. Ignored by fdiff/fgrad/gauss_diff.",
    "SPKIT_EGM.eD_um": "Inter-electrode distance (µm) passed to spkit.compute_cv; sets the physical scale of the conduction-velocity estimate (MCS 60-MEA = 200 µm).",
    "SPKIT_EGM.cv_thr_cm_s": "spkit conduction-velocity ceiling (cm/s); local CV above this is treated as an interpolation artefact and dropped. Neuronal CV is ~10–50 cm/s.",
    "SPKIT_EGM.sweep_angle_deg": "Nominal stimulation-sweep axis (135° = bottom-right → top-left), used for the alignment score in the SpKit figures.",
    "SPKIT_EGM.use_digital_events": "Use the laser ON times from the MCS Digital Event Recorder as the stimulus-locked event onsets whenever a window contains them; fall back to population-based event detection for windows without laser (e.g. pre/post).",
    "SPKIT_EGM.digital_lag_ms": "Shift (ms) applied to each laser ON time before cutting the EGM segment. 0 aligns on the trigger; increase if the tissue response lags the laser pulse.",
    "PHASE_LOCKIN.enable": "Master switch for the sweep phase lock-in pipeline. Folds every spike onto the sweep period and recovers the sweep's direction/speed as a spatial phase gradient — the method purpose-built to test whether the network locks to (and, post-stim, 'remembers') the SLOW periodic laser sweep, unlike the fast intra-burst LAT/EGM methods. Runs after Stage 2 (and SpKit) and writes to the 'Phase Lock' folder.",
    "PHASE_LOCKIN.use_digital_events": "Seed the sweep period from the laser trigger rate (period ≈ frames_per_sweep / trigger_hz) when the recording has digital events; otherwise the period is seeded from population-rate autocorrelation.",
    "PHASE_LOCKIN.frames_per_sweep": "Polygon frames in one full cycle of the projected pattern. The trigger advances one frame, so the pattern repeats every frames / trigger_hz seconds. Sept2021_SweepBRtoTL.seq = 160, Oct2026_RadialCenterOut.seq = 133. This is the DEFAULT only: a recording whose Batch.csv row fills the optional 7th 'frames' column uses that value instead, so a batch mixing sweep and radial files seeds each one correctly.",
    "PHASE_LOCKIN.period_s": "Explicit sweep-period override in seconds. Leave blank for automatic (trigger-derived, or autocorrelation for windows without triggers).",
    "PHASE_LOCKIN.refine": "Search around the seed period for the true cycle period that best explains the data. A period off by even a few percent smears the phase fit, so refinement is recommended.",
    "PHASE_LOCKIN.refine_search_frac": "Fractional half-width of the period search around the seed (0.30 = ±30%).",
    "PHASE_LOCKIN.refine_objective": "'pooled' = fast non-spatial locking strength; 'spatial' = weight each candidate period by plane-wave coherence (rejects non-spatial period aliases, ~15× slower).",
    "PHASE_LOCKIN.min_spikes": "An electrode needs at least this many spikes in the window to contribute a phase-lock vector; fewer makes its resultant too noisy.",
    "PHASE_LOCKIN.peak_ratio_ok": "Period-search peak/median ratio below which a panel is flagged low-confidence (no sharply tuned period was found).",
    "PHASE_LOCKIN.coherence_floor": "Plane-wave coherence below which direction/speed are flagged as near the noise floor (needs more cycles or less background, not a different period).",
    "PHASE_LOCKIN.do_pvalue": "Compute a phase-rotation permutation p-value per window (tests that the coherent spatial gradient exceeds chance). Slower.",
    "PHASE_LOCKIN.n_perm": "Number of permutations for the p-value null.",
    "PHASE_LOCKIN.radial_model": "'quadrant' (default): plane-wave fit per quadrant about the ring centre, scored against each quadrant's outward direction. 'spiral': electrodes ordered by distance from the ring centre and the phase read against that 1-D coordinate exactly like the linear sweep (matched filter over signed radial wavenumber); the outward contrast (+1 = fully outward, -1 = inward) replaces 'alignment' and a phase-vs-radius figure is added. Needs a run.",
    "PHASE_LOCKIN.exclude_fast_bursts": "Remove spontaneous network bursts before the lock-in: events where the pooled 10 ms firing rate exceeds the threshold and that last less than the max duration. A synchronous burst packs the array's firing into tens of ms; the laser-driven wave spreads over the whole sweep cycle and is never removed. Cleans the stim-period lock (bursts dilute it); needs a run, not a replot.",
    "PHASE_LOCKIN.burst_rate_factor": "Pooled-rate threshold, as a multiple of the median 10 ms-binned rate, at which a fast-burst event starts (it extends until the rate falls back below 1.5× median).",
    "PHASE_LOCKIN.burst_max_dur_s": "Only events shorter than this count as spontaneous bursts and are removed; longer excursions are kept as potentially stimulus-driven.",
    "PSD_TREND.bin_s": "Bin width for the pooled array firing rate; 0.05 s gives a 20 Hz sampling rate (sees rhythms up to 10 Hz).",
    "PSD_TREND.welch_nperseg": "Welch segment length in rate samples; longer = finer frequency resolution, needs a longer window (auto-shortened for short windows).",
    "PSD_TREND.fmax_hz": "Highest frequency drawn in the per-day PSD overlay.",
    "PSD_TREND.peak_lo_hz": "Lower edge of the band searched for the dominant slow (bursting) PSD peak.",
    "PSD_TREND.peak_hi_hz": "Upper edge of the peak-search band.",
    "PSD_TREND.burst_rate_factor": "Fast-burst event threshold (× median pooled 10 ms rate); same definition as the phase lock-in filter and Burst Latency.",
    "PSD_TREND.burst_max_dur_s": "Events longer than this are not counted as network bursts.",
    "BURST_LATENCY.direction_window_s": "Bin width for the burst-travel convergence curve (burst_direction_<group>.png): each bin averages the per-burst convergences of the travelling bursts inside it, like the phase lock-in's convergence-over-time plot.",
    "BURST_LATENCY.min_bursts_per_bin": "Bins with fewer travelling bursts than this are drawn FADED (low confidence) instead of full opacity; bins with no travelling burst at all stay gaps.",
    "BURST_LATENCY.burst_rate_factor": "Pooled-rate threshold (× the median 10 ms-binned rate) at which a fast-burst event starts; same definition as the phase lock-in's burst filter.",
    "BURST_LATENCY.burst_max_dur_s": "Events longer than this are not network bursts and are skipped.",
    "BURST_LATENCY.tail_s": "The first-spike latency window extends this far past the event end so trailing-edge electrodes still count.",
    "BURST_LATENCY.min_lat_electrodes": "Electrodes that must fire within a burst for its latency plane to be fitted.",
    "BURST_LATENCY.min_fit_r2": "Minimum R² of the latency plane for a burst to count as travelling; incoherent bursts are dropped.",
    "BURST_LATENCY.min_fits": "A window needs at least this many fitted (travelling) bursts before its mean direction appears in the summary figure; all windows reach the CSV.",
    "BURST_LATENCY.pitch_um": "Inter-electrode spacing (µm); sets the speed scale and the corner-to-corner delay (MCS 60-MEA = 200).",
    "BURST_LATENCY.sweep_angle_deg": "The trained axis the burst directions are scored against (BR→TL = 135).",
    "PHASE_LOCKIN.null_model": "'rotation' (original) randomises each electrode's preferred phase: it rejects whenever phases are not independent, which synchronous bursting alone causes (every pre/post/control window hit the p floor on the Oct-2026 batch). 'position' shuffles electrode positions while each keeps its own resultant: the common-phase (k=0) part is unchanged, so only a spatially organised (travelling) phase pattern is significant.",
    "PHASE_LOCKIN.pitch_um": "Inter-electrode spacing (µm); sets the spatial Nyquist limit and the speed scale (MCS 60-MEA = 200 µm).",
    "PHASE_LOCKIN.sweep_angle_deg": "Nominal sweep axis (135° = bottom-right → top-left), used for the per-panel alignment score (cos of the angle between recovered heading and the sweep axis).",
    "DIGITAL_PARAMS.use_in_stage2": "If on, Stage 2 reads the laser ON times and adds the stimulus-locked propagation figure (and enables the 'digital' CV-field mode). Costs one extra single-channel read per file; turn off if your recordings have no digital data.",
    "DIGITAL_PARAMS.digital_bit": "Which digital-input line carries the laser trigger. Leave as None to treat ANY non-zero digital value as 'on'; set 0–15 to isolate one bit if several lines are wired.",
    "DIGITAL_PARAMS.min_pulse_ms": "Debounce: laser ON/OFF pulses shorter than this (ms) are ignored, so electrical glitches aren't counted as stimulation.",
    "DIGITAL_PARAMS.invert": "Treat the LOW state as 'laser on' instead of HIGH. Only needed if your digital line is wired active-low.",
    "DIGITAL_PARAMS.analog_label_hint": "Case-insensitive substring used to find the digital square-wave stream by its label/DataSubType (MCS usually tags it 'Digital'). The reader also falls back to an EventStream or a low-channel stream automatically.",
    "SWEEP_CALIB.sample_diameter_mm": "Diameter (mm) of the sample the light bar sweeps across — used with the frame-crossing count to convert the measured trigger rate into an absolute sweep speed in mm/s.",
    "SWEEP_CALIB.frames_per_crossing": "How many Polygon frames the bar takes to cross the sample, from your DMD→MEA registration. Leave None until calibrated (the pipeline still reports the exact relative sweep speed as the trigger rate in Hz); set it to get an absolute mm/s.",
}
PLOT_HELP = {
    "firing_rate_by_electrode": "Per-electrode mean firing rate (spikes/s) as a half-violin with the group mean ± s.e.m. and a density curve; shows how active the culture is and how firing is distributed across electrodes.",
    "electrode_heatmap": "Spatial map of the MEA with each electrode coloured by its mean firing rate; reveals where activity concentrates. Panels scale colour to this recording and to the whole group.",
    "raster": "Time-by-electrode plot of binned firing (mean rate per 1-s bin). Vertical bands across many electrodes indicate network-wide synchronous bursts.",
    "edge_threshold_check": "Diagnostic comparing each pair's real STTC against circular-shifted surrogate spike trains, and how significance stabilises with surrogate count; confirms retained edges are genuinely above chance.",
    "connectivity_stats": "The thresholded STTC adjacency matrix plus histograms of node degree, node strength and edge weight; summarises how densely and strongly electrodes are functionally connected.",
    "network_plot": "Functional network on the physical MEA grid: node position = electrode, node size = degree, node colour = a metric, edge thickness = STTC weight. Shows where the highly connected nodes lie.",
    "network_NodeDegree": "Network graph with nodes coloured by node degree (number of significant connections) — highlights candidate hubs.",
    "network_NodeStrength": "Network graph with nodes coloured by node strength (summed STTC weights) — weights connectivity by correlation magnitude.",
    "network_BetweennessCentrality": "Network graph coloured by betweenness centrality — marks bridging nodes on many shortest paths.",
    "network_ParticipationCoefficient": "Network graph coloured by participation coefficient — how evenly a node's links spread across modules (0 = within-module, 1 = integrator).",
    "network_WithinModuleZ": "Network graph coloured by within-module degree z-score — how strongly a node connects inside its own module (hubs are high).",
    "network_LocalEfficiency": "Network graph coloured by local efficiency — how efficiently a node's neighbours intercommunicate (fault-tolerant local connectivity).",
    "node_cartography": "Scatter of within-module degree z-score vs participation coefficient with Guimerà–Amaral role regions; classifies each node as peripheral, connector, or a provincial/connector/kinless hub.",
    "activity_by_group": "Half-violins of a recording-level activity metric compared across groups, split by region (pre/stim/post), pooled across samples.",
    "activity_by_age": "Half-violins of a recording-level activity metric across ages/days, split by region, one panel per group.",
    "electrode_by_group": "Half-violins of a per-electrode activity metric (firing/burst) across groups, split by region.",
    "electrode_by_age": "Half-violins of a per-electrode activity metric across ages/days, split by region, one panel per group.",
    "network_by_group": "Half-violins of a recording-level network metric across groups, split by region.",
    "network_by_age": "Half-violins of a recording-level network metric across ages/days, split by region, one panel per group.",
    "node_by_group": "Half-violins of a node-level network metric across groups, split by region.",
    "node_by_age": "Half-violins of a node-level network metric across ages/days, split by region, one panel per group.",
    "graph_metrics_by_lag": "Line plots (mean ± s.e.m.) of network metrics vs the STTC lag Δt; shows how sensitive results are to lag choice.",
    "cartography_by_lag": "Proportion of nodes in each Guimerà–Amaral role as a function of STTC lag; an at-a-glance view of hub/module architecture.",
    "density_landscape": "2-D node density over the participation-coefficient / within-module-z plane; MEA-NAP uses this surface to place the cartography role boundaries.",
    "psd_firing_bursting": "Welch power spectra of the population firing rate and of the network-burst envelope, overlaid for pre/stim/post; shows whether light stimulation shifted the dominant spontaneous firing/bursting frequency.",
    "propagation_bursts": "Signal-propagation analysis computed on detected network bursts: an example activation-time map, a rose of wave directions, and a conduction-velocity histogram per region, with the sweep axis and mean-direction/alignment marked — to test for a memorised sweep direction (STDP).",
    "propagation_windows": "The same propagation analysis but over fixed sliding windows instead of detected bursts, so travelling waves are sampled independently of burst detection.",
    "cv_field": "Local conduction-velocity flow-field per region: a velocity vector at each electrode (from a local plane fit of the averaged activation-time surface), coloured by local CV. Reveals spatial heterogeneity — e.g. a memory effect confined to one corridor — that the single global propagation fit averages away.",
    "propagation_digital": "Stimulus-locked propagation: the same direction rose / CV / alignment analysis but with each event window anchored on a laser ON time from the MCS Digital Event Recorder. Only drawn for regions that contain laser events (i.e. the stim region), so it measures the directly evoked wave rather than spontaneous activity.",
}
# friendly labels for the plot-outputs tab
PLOT_LABELS = {
    "firing_rate_by_electrode": "Firing rate by electrode (2A)",
    "electrode_heatmap": "Electrode heatmap (2A)",
    "raster": "Raster (2A)",
    "edge_threshold_check": "Edge-thresholding check (3)",
    "connectivity_stats": "Connectivity stats / adjacency (4A)",
    "network_plot": "Network plot — plain (4A)",
    "network_NodeDegree": "Network — node degree (4A)",
    "network_NodeStrength": "Network — node strength (4A)",
    "network_BetweennessCentrality": "Network — betweenness (4A)",
    "network_ParticipationCoefficient": "Network — participation (4A)",
    "network_WithinModuleZ": "Network — within-module z (4A)",
    "network_LocalEfficiency": "Network — local efficiency (4A)",
    "node_cartography": "Node cartography (4A)",
    "activity_by_group": "Activity by group (2B)",
    "activity_by_age": "Activity by age (2B)",
    "electrode_by_group": "Electrode activity by group (2B)",
    "electrode_by_age": "Electrode activity by age (2B)",
    "network_by_group": "Network metrics by group (4B)",
    "network_by_age": "Network metrics by age (4B)",
    "node_by_group": "Node metrics by group (4B)",
    "node_by_age": "Node metrics by age (4B)",
    "graph_metrics_by_lag": "Graph metrics by lag (4B)",
    "cartography_by_lag": "Cartography by lag (4B)",
    "density_landscape": "Density landscape (4B)",
    "psd_firing_bursting": "PSD firing / bursting by region",
    "propagation_bursts": "Signal propagation — network bursts",
    "propagation_windows": "Signal propagation — sliding windows",
    "propagation_digital": "Signal propagation — laser-locked",
    "cv_field": "Local CV flow-field by region",
}


def build_schema():
    D, G = det.DETECT_PARAMS, det.GPU_PARAMS
    C, QU, QC = io.CONFIG, io.QUALITY_PARAMS, io.QC_PARAMS
    FR, BU, NB = activity.FR_PARAMS, activity.BURST_PARAMS, activity.NETBURST_PARAMS
    ST, PB, PA = activity.STTC_PARAMS, activity.PROB_PARAMS, activity.PARITY_PARAMS
    PE, CA, NE = activity.PERF_PARAMS, activity.CARTOGRAPHY_PARAMS, activity.NET_PARAMS
    PL, MP = activity.PLOT_PARAMS, orig.MEANAP_PLOT_PARAMS
    SP, PR = spprop.SPECTRAL_PARAMS, spprop.PROP_PARAMS
    CVF = spprop.CVFIELD_PARAMS
    SE = spkit_egm.SPKIT_EGM
    PLK = phase_lockin.PHASE_LOCKIN
    BL = burst_latency.BURST_LATENCY
    PT = psd_trend.PSD_TREND
    DE = digital.DIGITAL_PARAMS
    SC = digital.SWEEP_CALIB
    return [
        ("I/O", [
            _P(C, "CONFIG", "RECOMPUTE_SPIKES", "Recompute spikes", "bool", "True | False"),
            _P(C, "CONFIG", "BATCH_HAS_HEADER", "Batch.csv has header", "bool3",
               "auto | True | False", ["auto", "True", "False"]),
            _P(C, "CONFIG", "STREAM_INDEX", "Stream index", "intn", "None = auto, or int >= 0"),
            _P(C, "CONFIG", "VERBOSE", "Verbose logging", "bool", "True | False"),
            _P(C, "CONFIG", "MIN_WINDOW_S", "Min usable window (s)", "float", "0.5 – 5.0"),
            _P(C, "CONFIG", "WINDOWS_FROM_TRIGGERS", "Windows from laser triggers",
               "bool", "True | False"),
            _P(C, "CONFIG", "REFERENCE_ELECTRODE", "Reference electrodes", "list_str",
               "labels, comma-sep (e.g. Ref,15)"),
            _MA("MAX_ATTEMPTS", "Detection retries", "int",
                lambda: io.MAX_ATTEMPTS, lambda v: setattr(io, "MAX_ATTEMPTS", v), "1 – 5"),
        ]),
        ("Detection", [
            _P(D, "DETECT_PARAMS", "bp_low_hz", "Bandpass low (Hz)", "float", "300 – 800"),
            _P(D, "DETECT_PARAMS", "bp_high_hz", "Bandpass high (Hz)", "float",
               "3000 – 8000  (< fs/2)"),
            _P(D, "DETECT_PARAMS", "bp_order", "Butterworth order", "int", "2 – 4"),
            _P(D, "DETECT_PARAMS", "wname", "Wavelet", "str", "bior1.5 only"),
            _P(D, "DETECT_PARAMS", "wid_ms", "Spike width (ms)", "tuple2",
               "(0.3,0.7) – (0.5,1.0)"),
            _P(D, "DETECT_PARAMS", "ns", "CWT scales", "int", "3 – 6"),
            _P(D, "DETECT_PARAMS", "option", "Cost option", "enum", "l | c", ["l", "c"]),
            _P(D, "DETECT_PARAMS", "L", "Bayesian cost L", "float",
               "-0.2 (more) .. +0.1 (fewer)"),
            _P(D, "DETECT_PARAMS", "Lmax", "Cost scaling Lmax", "readonly", "DO NOT CHANGE"),
            _P(G, "GPU_PARAMS", "mem_budget_gb", "GPU mem budget (GB)", "float", "2 – 20"),
            _P(G, "GPU_PARAMS", "ch_batch", "Channels / GPU batch", "intn",
               "None = auto, or 1..n_channels"),
            _P(G, "GPU_PARAMS", "fp64", "float64 CWT", "bool", "True | False (False = fast)"),
        ]),
        ("Quality", [
            _P(QU, "QUALITY_PARAMS", "auto_ground", "Auto-ground bad electrodes", "bool",
               "True | False"),
            _P(QU, "QUALITY_PARAMS", "flat_uv", "Dead-electrode MAD floor (uV)", "float",
               "0.2 – 1.0"),
            _P(QU, "QUALITY_PARAMS", "noisy_mad_ratio", "Noisy: x median MAD", "float",
               "2.0 – 5.0"),
            _P(QU, "QUALITY_PARAMS", "noisy_max_rate_frac", "Noisy: rate < frac of median",
               "float", "0.1 – 0.5"),
            _P(QU, "QUALITY_PARAMS", "silent_max_rate_hz", "Silent rate ceiling (Hz)", "float",
               "0.0 – 0.1"),
            _P(QU, "QUALITY_PARAMS", "noise_snippets", "Noise snippets", "int", "2 – 6"),
            _P(QU, "QUALITY_PARAMS", "noise_snippet_s", "Noise snippet len (s)", "float",
               "5 – 60"),
        ]),
        ("QC", [
            _P(QC, "QC_PARAMS", "mode", "QC mode", "enum", "save | off | popup",
               ["save", "off", "popup"]),
            _P(QC, "QC_PARAMS", "popup_backend", "Popup backend", "enum",
               "viewer | matplotlib", ["viewer", "matplotlib"]),
            _P(QC, "QC_PARAMS", "block", "Block on popup", "bool", "True | False"),
            _P(QC, "QC_PARAMS", "snippet_len_s", "Snippet length (s)", "float", "30 – 120"),
            _P(QC, "QC_PARAMS", "short_max_min", "Short-recording cutoff (min)", "float",
               "5 – 20"),
            _P(QC, "QC_PARAMS", "long_min_min", "Long-recording cutoff (min)", "float",
               "30 – 90"),
            _P(QC, "QC_PARAMS", "edge_min", "Edge band width (min)", "float", "1 – 5"),
            _P(QC, "QC_PARAMS", "n_short", "Snippets (short)", "int", "1 – 4"),
            _P(QC, "QC_PARAMS", "n_long", "Snippets (long)", "int", "4 – 12"),
            _P(QC, "QC_PARAMS", "n_mid", "Snippets (mid)", "int", "2 – 8"),
            _P(QC, "QC_PARAMS", "prefer_active", "Prefer active channels", "bool",
               "True | False"),
            _P(QC, "QC_PARAMS", "seed", "RNG seed", "intn", "None or int"),
            _P(QC, "QC_PARAMS", "display_bins", "Display decimation bins", "int",
               "2000 – 20000"),
            _P(QC, "QC_PARAMS", "save_dir_name", "QC subfolder", "str", "any name"),
        ]),
        ("Firing", [
            _P(FR, "FR_PARAMS", "active_min_rate_hz", "Active min rate (Hz)", "float",
               "0.01 – 1.0"),
            _P(FR, "FR_PARAMS", "active_min_spikes", "Active min spikes", "int", "1 – 20"),
        ]),
        ("Bursts", [
            _P(BU, "BURST_PARAMS", "method", "Method", "enum", "isi_n | maxinterval",
               ["isi_n", "maxinterval"]),
            _P(BU, "BURST_PARAMS", "isi_n_N", "ISI_N spikes (N)", "int", "3 – 10"),
            _P(BU, "BURST_PARAMS", "isi_n_thr_s", "ISI_N threshold (s)", "float", "0.05 – 0.2"),
            _P(BU, "BURST_PARAMS", "max_isi_start_s", "Max ISI start (s)", "float",
               "0.05 – 0.2"),
            _P(BU, "BURST_PARAMS", "max_isi_end_s", "Max ISI end (s)", "float", "0.1 – 0.3"),
            _P(BU, "BURST_PARAMS", "min_ibi_s", "Min inter-burst (s)", "float", "0.1 – 0.5"),
            _P(BU, "BURST_PARAMS", "min_burst_dur_s", "Min burst duration (s)", "float",
               "0.01 – 0.1"),
            _P(BU, "BURST_PARAMS", "min_spikes", "Min spikes / burst", "int", "3 – 10"),
            _P(BU, "BURST_PARAMS", "rate_unit", "Rate unit", "enum", "per_min | per_sec",
               ["per_min", "per_sec"]),
        ]),
        ("Net-bursts", [
            _P(NB, "NETBURST_PARAMS", "net_N", "ISI_N spikes (N)", "int", "10 – 50"),
            _P(NB, "NETBURST_PARAMS", "net_isi_thr_s", "ISI_N threshold (s)", "float",
               "0.05 – 0.2"),
            _P(NB, "NETBURST_PARAMS", "net_min_chan", "Min channels", "int", "3 – 10"),
        ]),
        ("STTC", [
            _P(ST, "STTC_PARAMS", "lags_ms", "STTC lags (ms)", "list_int", "e.g. 10,25,50"),
            _P(ST, "STTC_PARAMS", "dt_s", "Tiling window dt (s)", "float", "0.01 – 0.1"),
        ]),
        ("Surrogate", [
            _P(PB, "PROB_PARAMS", "n_surrogates", "Surrogates", "int", ">= 180 (e.g. 200)"),
            _P(PB, "PROB_PARAMS", "tail", "Upper-tail alpha", "float", "0.01 – 0.1"),
            _P(PB, "PROB_PARAMS", "sig_threshold", "Significance (=1-tail)", "float",
               "0.90 – 0.99"),
            _P(PB, "PROB_PARAMS", "keep_negative", "Keep negative edges", "bool",
               "True | False"),
            _P(PB, "PROB_PARAMS", "random_seed", "RNG seed", "int", "any int"),
            _P(PB, "PROB_PARAMS", "keep_surrogates_for_plot", "Keep surrogates (plots)",
               "bool", "True | False"),
            _P(PB, "PROB_PARAMS", "use_gpu", "Use GPU STTC", "bool", "True | False"),
            _P(PB, "PROB_PARAMS", "strict_gpu", "Strict GPU (raise on fail)", "bool",
               "True | False"),
            _P(PB, "PROB_PARAMS", "bins_per_dt", "Bins per dt", "int", "10 – 25 (lower if VRAM)"),
        ]),
        ("Network", [
            _P(PA, "PARITY_PARAMS", "sttc_backend", "STTC backend", "enum", "gpu | exact",
               ["gpu", "exact"]),
            _P(PA, "PARITY_PARAMS", "pc_mode", "Participation mode", "enum", "norm | raw",
               ["norm", "raw"]),
            _P(PA, "PARITY_PARAMS", "pc_norm_iter", "PC-norm iterations", "int", "25 – 100"),
            _P(PA, "PARITY_PARAMS", "auto_cartography_boundaries", "Auto cartography bounds",
               "bool", "True | False"),
            _P(PA, "PARITY_PARAMS", "fs_hz", "Sampling rate (Hz)", "float", "e.g. 20000"),
            _P(PA, "PARITY_PARAMS", "min_nodes", "Min network nodes", "int", "10 – 30"),
            _P(NE, "NET_PARAMS", "binary_edge_min", "Binary edge min", "float", "small > 0"),
            _P(NE, "NET_PARAMS", "community_resolution", "Community resolution", "float",
               "0.2 – 1.0"),
        ]),
        ("Cartography", [
            _P(CA, "CARTOGRAPHY_PARAMS", "hubBoundaryWMdDeg", "Hub boundary (z)", "float",
               "2.0 – 3.0"),
            _P(CA, "CARTOGRAPHY_PARAMS", "periPartCoef", "Peripheral | connector", "float",
               "0.5 – 0.7"),
            _P(CA, "CARTOGRAPHY_PARAMS", "nonHubconnectorPartCoef", "Connector | kinless",
               "float", "0.7 – 0.9"),
            _P(CA, "CARTOGRAPHY_PARAMS", "proHubpartCoef", "Provincial | connector hub",
               "float", "0.2 – 0.4"),
            _P(CA, "CARTOGRAPHY_PARAMS", "connectorHubPartCoef", "Connector | kinless hub",
               "float", "0.6 – 0.8"),
        ]),
        ("Perf", [
            _P(PE, "PERF_PARAMS", "require_numba", "Require numba", "enum",
               "warn | raise | off", ["warn", "raise", "off"]),
            _P(PE, "PERF_PARAMS", "parallel_network_metrics", "Parallel network metrics",
               "bool", "True | False"),
            _P(PE, "PERF_PARAMS", "max_workers", "Max workers", "intn",
               "None = CPU-1, or 1..CPU"),
        ]),
        ("Plots", [
            _P(PL, "PLOT_PARAMS", "dpi", "Pipeline plot DPI", "int", "150 – 600"),
            _P(PL, "PLOT_PARAMS", "raster_dpi", "Raster DPI (speed)", "int", "150 – 600"),
            _P(PL, "PLOT_PARAMS", "heatmap_cmap", "Heatmap colormap", "str", "matplotlib cmap"),
            _P(PL, "PLOT_PARAMS", "raster_row_seconds", "Raster row seconds", "float",
               "60 – 600"),
            _P(PL, "PLOT_PARAMS", "raster_sec_per_inch", "Raster sec / inch", "float",
               "10 – 60"),
            _P(MP, "MEANAP_PLOT_PARAMS", "dpi", "MEA-NAP output DPI", "int", "150 – 600"),
            _P(MP, "MEANAP_PLOT_PARAMS", "raster_upper_pct", "Raster upper pct", "int",
               "90 – 100"),
            _P(MP, "MEANAP_PLOT_PARAMS", "raster_cmap_trunc", "Parula truncation", "float",
               "0.7 – 1.0"),
            _P(MP, "MEANAP_PLOT_PARAMS", "kde_height", "Half-violin width", "float",
               "0.1 – 0.5"),
            _P(MP, "MEANAP_PLOT_PARAMS", "node_scale_f", "Electrode circle size", "float",
               "0.4 – 0.9"),
            _P(MP, "MEANAP_PLOT_PARAMS", "heatmap_pct", "Heatmap upper pct", "int", "90 – 100"),
            _P(MP, "MEANAP_PLOT_PARAMS", "min_activity_level", "Active FR floor (Hz)", "float",
               "0.0 – 1.0"),
        ]),
        ("Colours", [
            _P(MP, "MEANAP_PLOT_PARAMS", "raster_cmap", "Raster colormap", "enum",
               " | ".join(orig.CMAP_OPTIONS), list(orig.CMAP_OPTIONS)),
            _P(MP, "MEANAP_PLOT_PARAMS", "heatmap_cmap", "Heatmap / density colormap", "enum",
               " | ".join(orig.CMAP_OPTIONS), list(orig.CMAP_OPTIONS)),
            _P(MP, "MEANAP_PLOT_PARAMS", "network_node_cmap", "Network node colormap", "enum",
               " | ".join(orig.CMAP_OPTIONS), list(orig.CMAP_OPTIONS)),
            _P(MP, "MEANAP_PLOT_PARAMS", "violin_palette", "Violin palette (pre/stim/post)",
               "enum", " | ".join(orig.PALETTE_OPTIONS), list(orig.PALETTE_OPTIONS)),
            _P(MP, "MEANAP_PLOT_PARAMS", "violin_point_size", "Violin point size", "float",
               "2 – 20"),
            _P(MP, "MEANAP_PLOT_PARAMS", "violin_mean_size", "Violin mean-dot size", "float",
               "8 – 80"),
            _P(MP, "MEANAP_PLOT_PARAMS", "node_min_size", "Network node min size", "float",
               "0.05 – 0.4"),
            _P(MP, "MEANAP_PLOT_PARAMS", "node_max_size", "Network node max size", "float",
               "0.3 – 0.9"),
            _P(MP, "MEANAP_PLOT_PARAMS", "edge_max_width", "Network edge max width", "float",
               "1.0 – 5.0"),
        ]),
        ("Spectral / Propagation", [
            _P(SP, "SPECTRAL_PARAMS", "psd_bin_s", "PSD bin width (s)", "float",
               "0.02 – 0.1"),
            _P(SP, "SPECTRAL_PARAMS", "psd_fmax_hz", "PSD max frequency (Hz)", "float",
               "5 – 20"),
            _P(SP, "SPECTRAL_PARAMS", "psd_welch_nperseg", "Welch segment (samples)", "int",
               "256 – 4096"),
            _P(SP, "SPECTRAL_PARAMS", "psd_logy", "PSD log y-axis", "bool", "True | False"),
            _P(PR, "PROP_PARAMS", "electrode_pitch_um", "Electrode pitch (um)", "float",
               "100 – 300"),
            _P(PR, "PROP_PARAMS", "min_electrodes", "Min electrodes / wave", "int", "4 – 20"),
            _P(PR, "PROP_PARAMS", "min_fit_r2", "Min plane-fit R^2", "float", "0.1 – 0.6"),
            _P(PR, "PROP_PARAMS", "window_ms", "Sliding window (ms)", "float", "50 – 250"),
            _P(PR, "PROP_PARAMS", "window_step_ms", "Window step (ms)", "float", "25 – 150"),
            _P(PR, "PROP_PARAMS", "window_min_active_frac", "Window min active frac",
               "float", "0.1 – 0.6"),
            _P(PR, "PROP_PARAMS", "sweep_angle_deg", "Sweep axis (deg, BR->TL)", "float",
               "0 – 360"),
            _P(PR, "PROP_PARAMS", "max_maps", "Example maps / region", "int", "1 – 4"),
            _P(PR, "PROP_PARAMS", "max_events_per_region", "Max events / region", "int",
               "500 – 20000"),
            _P(CVF, "CVFIELD_PARAMS", "cv_field_mode", "CV-field events", "enum",
               "bursts | windows | digital", ["bursts", "windows", "digital"]),
            _P(CVF, "CVFIELD_PARAMS", "cv_neighbor_factor", "CV-field radius (×pitch)",
               "float", "1.2 – 2.5"),
            _P(CVF, "CVFIELD_PARAMS", "cv_min_neighbors", "CV-field min neighbours", "int",
               "3 – 6"),
            _P(CVF, "CVFIELD_PARAMS", "cv_local_min_r2", "CV-field min local R^2", "float",
               "0.0 – 0.6"),
            _P(DE, "DIGITAL_PARAMS", "use_in_stage2", "Use laser events in Stage 2",
               "bool", "True | False"),
            _P(DE, "DIGITAL_PARAMS", "digital_bit", "Laser digital bit", "intn",
               "None = any bit, or 0 – 15"),
            _P(DE, "DIGITAL_PARAMS", "min_pulse_ms", "Min laser pulse (ms)", "float",
               "0 – 50"),
            _P(DE, "DIGITAL_PARAMS", "invert", "Invert digital (low=on)", "bool",
               "True | False"),
            _P(DE, "DIGITAL_PARAMS", "analog_label_hint", "Digital stream hint", "str",
               "substring, e.g. 'digital'"),
            _P(SC, "SWEEP_CALIB", "sample_diameter_mm", "Sample diameter (mm)", "float",
               "1 – 10"),
            _P(SC, "SWEEP_CALIB", "frames_per_crossing", "Frames to cross sample",
               "intn", "None, or int (from DMD→MEA calibration)"),
        ]),
        ("SpKit EGM", [
            _P(SE, "SPKIT_EGM", "use_digital_events", "Use laser events (stim-locked)",
               "bool", "True | False"),
            _P(SE, "SPKIT_EGM", "digital_lag_ms", "Laser lag (ms)", "float", "-20 – 50"),
            _P(SE, "SPKIT_EGM", "evt_bp_low_hz", "Event band low (Hz)", "float",
               "100 – 500"),
            _P(SE, "SPKIT_EGM", "evt_bp_high_hz", "Event band high (Hz)", "float",
               "1000 – 3000"),
            _P(SE, "SPKIT_EGM", "evt_k_mad", "Event threshold (k·MAD)", "float", "3 – 8"),
            _P(SE, "SPKIT_EGM", "evt_min_isi_ms", "Event min ISI (ms)", "float",
               "100 – 1000"),
            _P(SE, "SPKIT_EGM", "evt_max_events", "Max events averaged", "int", "20 – 1000"),
            _P(SE, "SPKIT_EGM", "evt_min_events", "Min events to analyse", "int", "2 – 20"),
            _P(SE, "SPKIT_EGM", "egm_pre_ms", "EGM pre-onset (ms)", "float", "5 – 40"),
            _P(SE, "SPKIT_EGM", "egm_post_ms", "EGM post-onset (ms)", "float", "20 – 120"),
            _P(SE, "SPKIT_EGM", "at_method", "Activation method", "enum",
               "min_dvdt | max_dvdt | max_abs_dvdt",
               ["min_dvdt", "max_dvdt", "max_abs_dvdt"]),
            _P(SE, "SPKIT_EGM", "at_gradient", "Gradient method", "enum",
               "fdiff | fgrad | sgdiff | gauss_diff",
               ["fdiff", "fgrad", "sgdiff", "gauss_diff"]),
            _P(SE, "SPKIT_EGM", "gauss_window", "Gauss smooth window", "int", "3 – 51 (odd)"),
            _P(SE, "SPKIT_EGM", "sg_window", "SG smooth window", "int", "5 – 51 (odd)"),
            _P(SE, "SPKIT_EGM", "eD_um", "Electrode distance (um)", "float", "100 – 700"),
            _P(SE, "SPKIT_EGM", "cv_thr_cm_s", "CV ceiling (cm/s)", "float", "50 – 500"),
            _P(SE, "SPKIT_EGM", "sweep_angle_deg", "Sweep axis (deg, BR->TL)", "float",
               "0 – 360"),
            _P(SE, "SPKIT_EGM", "run_egm_cv", "Run fast EGM-CV", "bool",
               "True | False"),
            _P(SE, "SPKIT_EGM", "run_sweep_latency", "Run slow-sweep latency", "bool",
               "True | False  (matched to the laser sweep)"),
            _P(SE, "SPKIT_EGM", "frames_per_sweep", "Frames per sweep (default)", "float",
               "1 – 100000  (Batch.csv col 7 overrides)"),
            _P(SE, "SPKIT_EGM", "sweep_box_mm", "Sweep field size (mm)", "float",
               "1 – 10  (square side, centred on pad)"),
            _P(SE, "SPKIT_EGM", "lat_bp_low_hz", "Latency band low (Hz)", "float",
               "100 – 1000"),
            _P(SE, "SPKIT_EGM", "lat_bp_high_hz", "Latency band high (Hz)", "float",
               "1000 – 6000"),
            _P(SE, "SPKIT_EGM", "lat_min_coh", "Min folding coherence", "float",
               "0 – 0.5"),
            _P(SE, "SPKIT_EGM", "lat_min_electrodes", "Min electrodes", "int", "6 – 40"),
        ]),
        ("Phase Lock-In", [
            _P(PLK, "PHASE_LOCKIN", "use_digital_events", "Seed period from laser triggers",
               "bool", "True | False"),
            _P(PLK, "PHASE_LOCKIN", "frames_per_sweep", "Frames per sweep (default)", "float",
               "1 – 100000  (sweep 160, radial 133; Batch.csv col 7 overrides)"),
            _P(PLK, "PHASE_LOCKIN", "period_s", "Period override (s)", "floatn",
               "None = auto, or seconds"),
            _P(PLK, "PHASE_LOCKIN", "refine", "Refine period around seed", "bool",
               "True | False"),
            _P(PLK, "PHASE_LOCKIN", "refine_search_frac", "Period search ± frac", "float",
               "0.05 – 0.50"),
            _P(PLK, "PHASE_LOCKIN", "refine_objective", "Refine objective", "enum",
               "pooled | spatial", ["pooled", "spatial"]),
            _P(PLK, "PHASE_LOCKIN", "min_spikes", "Min spikes / electrode", "int", "5 – 60"),
            _P(PLK, "PHASE_LOCKIN", "min_cycles", "Min sweeps in window", "float",
               "1.5 – 10  (rejects period ≈ window)"),
            _P(PLK, "PHASE_LOCKIN", "max_trigger_period_s", "Max trigger period (s)",
               "float", "60 – 600  (>this → autocorrelation; flood control)"),
            _P(PLK, "PHASE_LOCKIN", "min_phase_span_rad", "Min phase span (rad)",
               "float", "0.2 – 1.5  (below → 'no propagation', flood)"),
            _P(PLK, "PHASE_LOCKIN", "peak_ratio_ok", "Peak-ratio confidence", "float",
               "1.2 – 4.0"),
            _P(PLK, "PHASE_LOCKIN", "coherence_floor", "Coherence floor", "float",
               "0.0 – 0.5"),
            _P(PLK, "PHASE_LOCKIN", "do_pvalue", "Permutation p-value", "bool",
               "True | False  (slow)"),
            _P(PLK, "PHASE_LOCKIN", "n_perm", "Permutations", "int", "100 – 1000"),
            _P(PLK, "PHASE_LOCKIN", "null_model", "P-value null model", "enum",
               "rotation | position", ["rotation", "position"]),
            _P(PLK, "PHASE_LOCKIN", "pitch_um", "Electrode pitch (um)", "float", "100 – 700"),
            _P(PLK, "PHASE_LOCKIN", "sweep_angle_deg", "Sweep axis (deg, BR->TL)", "float",
               "0 – 360"),
            _P(PLK, "PHASE_LOCKIN", "radial_model", "Radial spatial model", "enum",
               "quadrant | spiral", ["quadrant", "spiral"]),
            _P(PLK, "PHASE_LOCKIN", "exclude_fast_bursts", "Exclude fast network bursts",
               "bool", "True | False"),
            _P(PLK, "PHASE_LOCKIN", "burst_rate_factor", "Burst rate threshold (× median)",
               "float", "2 – 10"),
            _P(PLK, "PHASE_LOCKIN", "burst_max_dur_s", "Burst max duration (s)", "float",
               "0.1 – 1.0"),
            _P(PLK, "PHASE_LOCKIN", "direction_window_s", "Direction bin (s)", "float",
               "10 – 600  (heading-over-time curve only)"),
            _P(PLK, "PHASE_LOCKIN", "n_workers", "Parallel workers", "int",
               "0 = auto (cores-1), 1 = serial, 2 – 16"),
            _P(PLK, "PHASE_LOCKIN", "post_map_window_s", "Post map window (s)", "float",
               "0 = full post; 30 – 1800  (needs a run)"),
            _P(PLK, "PHASE_LOCKIN", "direction_smooth_bins", "Direction smoothing (bins)",
               "float", "0.5 – 10  (plot-only: Replot applies it)"),
        ]),
        ("Burst Latency", [
            _P(BL, "BURST_LATENCY", "burst_rate_factor", "Burst rate threshold (× median)",
               "float", "2 – 10"),
            _P(BL, "BURST_LATENCY", "burst_max_dur_s", "Burst max duration (s)", "float",
               "0.1 – 1.0"),
            _P(BL, "BURST_LATENCY", "tail_s", "Latency tail after event (s)", "float",
               "0 – 0.2"),
            _P(BL, "BURST_LATENCY", "min_lat_electrodes", "Min electrodes per burst",
               "int", "6 – 40"),
            _P(BL, "BURST_LATENCY", "min_fit_r2", "Min plane-fit R²", "float", "0 – 0.9"),
            _P(BL, "BURST_LATENCY", "min_fits", "Min fitted bursts (summary)", "int",
               "1 – 100"),
            _P(BL, "BURST_LATENCY", "pitch_um", "Electrode pitch (um)", "float",
               "100 – 700"),
            _P(BL, "BURST_LATENCY", "sweep_angle_deg", "Sweep axis (deg, BR->TL)",
               "float", "0 – 360"),
            _P(BL, "BURST_LATENCY", "direction_window_s", "Direction bin (s)", "float",
               "30 – 600  (burst-travel-over-time curve)"),
            _P(BL, "BURST_LATENCY", "min_bursts_per_bin", "Min bursts per bin", "int",
               "1 – 20"),
        ]),
        ("PSD Trends", [
            _P(PT, "PSD_TREND", "bin_s", "Population-rate bin (s)", "float",
               "0.02 – 0.2"),
            _P(PT, "PSD_TREND", "welch_nperseg", "Welch segment (samples)", "int",
               "256 – 8192"),
            _P(PT, "PSD_TREND", "fmax_hz", "Overlay max frequency (Hz)", "float",
               "0.5 – 10"),
            _P(PT, "PSD_TREND", "peak_lo_hz", "Peak search low (Hz)", "float",
               "0.005 – 1"),
            _P(PT, "PSD_TREND", "peak_hi_hz", "Peak search high (Hz)", "float",
               "0.05 – 5"),
            _P(PT, "PSD_TREND", "burst_rate_factor", "Burst rate threshold (× median)",
               "float", "2 – 10"),
            _P(PT, "PSD_TREND", "burst_max_dur_s", "Burst max duration (s)", "float",
               "0.1 – 1.0"),
        ]),
    ]


# ---- value <-> string conversion --------------------------------------------
def _cur(spec):
    return spec["getter"]() if spec.get("getter") else spec["obj"][spec["key"]]


def _fmt(spec):
    v = _cur(spec)
    t = spec["type"]
    if t == "bool3":
        return "auto" if v == "auto" else str(bool(v))
    if t in ("intn", "floatn"):
        return "" if v is None else str(v)
    if t == "tuple2":
        return f"{v[0]},{v[1]}"
    if t in ("list_int", "list_str"):
        return ",".join(str(x) for x in v)
    return "" if v is None else str(v)


def _parse(spec, var):
    t = spec["type"]
    if t == "bool":
        return bool(var.get())
    if t == "readonly":
        return _cur(spec)
    s = str(var.get()).strip()
    if t == "bool3":
        return {"auto": "auto", "True": True, "False": False}.get(s, "auto")
    if t in ("enum", "str"):
        return s
    if t == "float":
        return float(s)
    if t == "int":
        return int(float(s))
    if t == "intn":
        return None if s == "" else int(float(s))
    if t == "floatn":
        return None if s == "" else float(s)
    if t == "tuple2":
        a, b = [float(x) for x in s.replace(";", ",").split(",")[:2]]
        return (a, b)
    if t == "list_int":
        return [int(float(x)) for x in s.replace(";", ",").split(",") if x.strip()]
    if t == "list_str":
        return [x.strip() for x in s.replace(";", ",").split(",") if x.strip()]
    return s


def parse_int(text, default=None):
    t = str(text).strip()
    if t == "":
        return default
    try:
        return int(float(t))
    except Exception:
        return default
_REQUIRED_PKGS = [
    ("numpy", "numpy", True),
    ("scipy", "scipy", True),
    ("matplotlib", "matplotlib", True),
    ("pywt", "PyWavelets", True),
    ("h5py", "h5py", True),
    ("pandas", "pandas", True),
    ("networkx", "networkx", True),
    ("torch", "torch", True),
    ("numba", "numba", False),
    ("meanap", "meanap (port)", True),
    ("spkit", "spkit (EGM)", False),
]


def _probe_environment():
    """Return {'gpu': (state, label, detail), 'pkgs': [(friendly, state, detail)]}.
    state in {'ok','warn','bad'}. Import-heavy; call off the UI thread."""
    import importlib
    pkgs = []
    for mod, friendly, required in _REQUIRED_PKGS:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "")
            pkgs.append((friendly, "ok", str(ver)))
        except Exception as e:
            pkgs.append((friendly, "bad" if required else "warn",
                         type(e).__name__))
    # GPU
    gpu = ("bad", "GPU: unavailable", "torch not importable")
    try:
        import torch
        if torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:
                name = "CUDA device"
            try:
                cc = torch.version.cuda or "?"
            except Exception:
                cc = "?"
            gpu = ("ok", "GPU: enabled", f"{name}  ·  CUDA {cc}")
        else:
            gpu = ("warn", "GPU: CPU-only", "torch present, CUDA not available")
    except Exception as e:
        gpu = ("bad", "GPU: unavailable", type(e).__name__)
    return {"gpu": gpu, "pkgs": pkgs}


# ==============================================================================
# Qt helpers: a Tk-Variable-like adapter + custom widgets
# ==============================================================================
class QtVar:
    """Gives a Qt editor a Tk-Variable-like .get()/.set() so the original
    _fmt/_parse/_collect_config/_apply_config logic works unchanged."""
    def __init__(self, kind, widget):
        self.kind = kind                      # 'bool' | 'text'
        self.w = widget

    def get(self):
        if self.kind == "bool":
            return self.w.isChecked()
        if isinstance(self.w, QtWidgets.QComboBox):
            return self.w.currentText()
        return self.w.text()

    def set(self, v):
        if self.kind == "bool":
            self.w.setChecked(bool(v))
        elif isinstance(self.w, QtWidgets.QComboBox):
            i = self.w.findText(str(v))
            if i >= 0:
                self.w.setCurrentIndex(i)
            elif self.w.isEditable():
                self.w.setEditText(str(v))
        else:
            self.w.setText("" if v is None else str(v))


class LabelVar:
    """Wrap a QLabel so status/percentage lines keep the original .set()/.get()."""
    def __init__(self, label):
        self.lbl = label

    def set(self, text):
        self.lbl.setText("" if text is None else str(text))

    def get(self):
        return self.lbl.text()


class RingWidget(QtWidgets.QWidget):
    """Per-sample circular progress ring, painted with QPainter (crisp, DPI-scaled)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frac = 0.0
        self.state = "pending"
        self.setFixedSize(RING, RING)

    def set_state(self, frac, state):
        if frac is not None:
            self.frac = max(0.0, min(float(frac), 1.0))
        if state:
            self.state = state
        self.update()

    def paintEvent(self, _e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        pad = RING_TH // 2 + 3
        rect = QtCore.QRectF(pad, pad, RING - 2 * pad, RING - 2 * pad)
        col = QtGui.QColor(_STATE_COLOR.get(self.state, PALETTE["accent"]))
        track = QtGui.QPen(QtGui.QColor(_TRACK), RING_TH)
        track.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(track)
        p.drawArc(rect, 0, 360 * 16)
        arc = QtGui.QPen(col, RING_TH)
        arc.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(arc)
        if self.state in ("done", "skipped", "failed"):
            p.drawArc(rect, 0, 360 * 16)
        elif self.frac > 0.001:
            p.drawArc(rect, 90 * 16, int(-360 * 16 * self.frac))
        glyph = {"done": "✓", "failed": "×", "skipped": "»"}.get(self.state)
        if glyph:
            p.setPen(col)
            p.setFont(QtGui.QFont(mea_theme.FONT_FAMILY, 16, QtGui.QFont.Weight.Bold))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, glyph)
        elif self.state == "running":
            p.setPen(QtGui.QColor(PALETTE["text"]))
            p.setFont(QtGui.QFont(mea_theme.FONT_FAMILY, 10, QtGui.QFont.Weight.Bold))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       f"{int(round(self.frac * 100))}%")
        p.end()


class FlowLayout(QtWidgets.QLayout):
    """Left-to-right layout that wraps to the next line — reflows the sample rings
    into as many columns as the panel width allows (replaces the old canvas grid)."""
    def __init__(self, parent=None, margin=0, spacing=8):
        super().__init__(parent)
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)
        self._items = []

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def clear(self):
        while self._items:
            it = self._items.pop()
            w = it.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        return self._do(QtCore.QRect(0, 0, w, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        s = QtCore.QSize()
        for it in self._items:
            s = s.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return s + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    def _do(self, rect, test):
        m = self.contentsMargins()
        x = rect.x() + m.left()
        y = rect.y() + m.top()
        right = rect.right() - m.right()
        line_h = 0
        sp = self.spacing()
        for it in self._items:
            sz = it.sizeHint()
            w, h = sz.width(), sz.height()
            if x + w > right and line_h > 0:
                x = rect.x() + m.left()
                y += line_h + sp
                line_h = 0
            if not test:
                it.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), sz))
            x += w + sp
            line_h = max(line_h, h)
        return y + line_h - rect.y() + m.bottom()


class CollapsibleCard(QtWidgets.QFrame):
    """A titled rounded panel whose body folds away when the header is clicked."""
    def __init__(self, title, on_toggle=None, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self._title = title
        self._on_toggle = on_toggle
        self.collapsed = False
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 6)
        lay.setSpacing(4)
        self.header = QtWidgets.QToolButton()
        self.header.setText("  ▾  " + title)
        self.header.setAutoRaise(True)
        self.header.setCursor(Qt.CursorShape.PointingHandCursor)
        self.header.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                  QtWidgets.QSizePolicy.Policy.Fixed)
        self.header.setStyleSheet(
            f"QToolButton{{color:{PALETTE['accent']};font-weight:700;border:none;"
            f"text-align:left;padding:2px 2px;background:transparent;}}")
        self.header.clicked.connect(self.toggle)
        lay.addWidget(self.header)
        self.body = QtWidgets.QWidget()
        self.body_layout = QtWidgets.QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(4, 2, 4, 2)
        self.body_layout.setSpacing(4)
        lay.addWidget(self.body)

    def toggle(self):
        self.set_collapsed(not self.collapsed)

    def set_collapsed(self, val):
        val = bool(val)
        if val == self.collapsed:
            return
        self.collapsed = val
        self.body.setVisible(not val)
        self.header.setText(("  ▸  " if val else "  ▾  ") + self._title)
        if self._on_toggle:
            try:
                self._on_toggle()
            except Exception:
                pass


# ==============================================================================
# the application window
# ==============================================================================
class App(QtWidgets.QMainWindow):

    _SECTION_DEFS = [
        ("detect",      "Spike detection",
         "Stage 1 — wavelet spike detection on every recording. Uncheck to reuse "
         "spikes already on disk (or skip detection entirely)."),
        ("activity",    "Activity + bursts",
         "Per-recording firing-rate, electrode and burst / network-burst figures."),
        ("network",     "Network (STTC + graph)",
         "STTC functional connectivity, graph metrics and the network plots."),
        ("spectral",    "Spectral + propagation",
         "PSD, signal-propagation and the conduction-velocity field figures."),
        ("meanap",      "MEA-NAP original plots",
         "The MEA-NAP 'original output' plot tree (2A / 4A / 2B / 4B)."),
        ("comparisons", "Group comparisons",
         "Group / age comparison figures (Stage 2c). Needs >1 analysed group."),
        ("spkit",       "SpKit EGM (cond. velocity)",
         "spkit continuous-EGM conduction-velocity pipeline on the RAW voltage "
         "(independent of spike detection). Writes to the SpKit/ folder."),
        ("phase",       "Phase Lock-In (sweep / radial)",
         "Phase-lock pipeline — folds spikes onto the stimulus period to test "
         "whether the network locks to (and remembers) the laser sweep. Groups "
         "named radial* use the quadrant model for the centre-out ring. Writes to "
         "the 'Phase Lock' folder."),
        ("burstlat",    "Burst latency (propagation)",
         "Fits a latency plane to every fast network burst and asks whether the "
         "bursts travel along the trained direction (a ~50 ms corner-to-corner "
         "delay the phase lock-in cannot see). Writes to the 'Burst Latency' "
         "folder."),
        ("psdtrend",    "PSD trends (bursting rhythm)",
         "Population-rate power spectra and fast-burst rate / frequency per "
         "window, compared across training days (did stimulation change the "
         "spontaneous bursting rhythm?). Writes to the 'PSD Trends' folder."),
    ]

    def __init__(self):
        super().__init__()
        self.q = queue.Queue()
        self.rows = {}
        self.pvars = {}
        self.v = {}
        self.running = False
        self.group_detect = {}
        self.group_analyze = {}
        self._group_state = {}
        self._saved_by_group = None
        self._by_group_prev = False
        self._prerun_fold = (False, False)
        self._sash_restored = False
        self._env_result = None
        self._chip_widgets = []

        self.setWindowTitle("MEA Analyzer  —  Stage 1 → Stage 2")
        self.resize(1160, 1010)
        self.setMinimumSize(900, 620)

        io.QC_PARAMS["mode"] = "save"           # overnight-friendly default
        progress_gui.make = (lambda names, title="Progress", subtitle="":
                             EmbeddedProgress(self.q, names, title, subtitle))

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._build_banner(root)

        self.splitter = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(8)
        root.addWidget(self.splitter, 1)

        self.config_host = QtWidgets.QWidget()
        cfg = QtWidgets.QVBoxLayout(self.config_host)
        cfg.setContentsMargins(12, 10, 12, 4)
        cfg.setSpacing(6)
        self._build_header(cfg)
        self._build_outputs_panel(cfg)
        self._build_params_tabs(cfg)
        self.splitter.addWidget(self.config_host)

        self.views_stack = QtWidgets.QStackedWidget()
        self.splitter.addWidget(self.views_stack)
        self._build_setup_view()
        self._build_run_view()

        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([600, 380])

        self._build_controls(root)

        self._load_settings()
        self._refresh_groups()

        self._drain_timer = QTimer(self)
        self._drain_timer.timeout.connect(self._drain)
        self._drain_timer.start(POLL_MS)
        self._start_intro()

    # ---- small helpers ------------------------------------------------------
    def _err(self, title, msg):
        QtWidgets.QMessageBox.critical(self, title, msg)

    def _info(self, title, msg):
        QtWidgets.QMessageBox.information(self, title, msg)

    def _warn(self, title, msg):
        QtWidgets.QMessageBox.warning(self, title, msg)

    def _ask(self, title, msg):
        B = QtWidgets.QMessageBox.StandardButton
        return QtWidgets.QMessageBox.question(self, title, msg, B.Yes | B.No) == B.Yes

    def _help_icon(self, text):
        l = QtWidgets.QLabel("ⓘ")
        l.setStyleSheet(f"color:{PALETTE['accent']};")
        l.setToolTip(text)
        l.setCursor(Qt.CursorShape.WhatsThisCursor)
        return l

    @staticmethod
    def _clear_grid(grid):
        while grid.count():
            it = grid.takeAt(0)
            w = it.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    @staticmethod
    def _trunc(s, n):
        s = str(s)
        return s if len(s) <= n else s[:n - 1] + "…"

    def _fit_config(self):
        QTimer.singleShot(0, self._rebalance)

    def _rebalance(self):
        """Split the vertical space between the config pane (top) and the
        setup/run pane (bottom). The run view (progress rings + log) needs a
        generous fixed share; the setup view (group list) needs less, so the
        config pane keeps more of the parameters visible before a run."""
        try:
            total = self.splitter.height()
            if total < 200:
                return
            reserve = 470 if self.views_stack.currentIndex() == 1 else 300
            hint = self.config_host.sizeHint().height()
            top = max(200, min(hint, total - reserve))
            self.splitter.setSizes([top, total - top])
        except Exception:
            pass

    def showEvent(self, e):
        super().showEvent(e)
        if not getattr(self, "_shown_once", False):
            self._shown_once = True
            QTimer.singleShot(0, self._rebalance)

    # ---- banner -------------------------------------------------------------
    def _build_banner(self, root):
        bar = QtWidgets.QFrame()
        bar.setObjectName("banner")
        bar.setFixedHeight(56)
        bar.setStyleSheet(f"#banner{{background:{PALETTE['banner']};"
                          f"border-bottom:1px solid {PALETTE['line']};}}")
        h = QtWidgets.QHBoxLayout(bar)
        h.setContentsMargins(16, 0, 16, 0)
        title = QtWidgets.QLabel("MEA Analyzer")
        title.setStyleSheet("color:#ffffff;font-size:17pt;font-weight:700;")
        sub = QtWidgets.QLabel("MEA-NAP pipeline")
        sub.setStyleSheet("color:#c7d4f2;font-size:11pt;")
        h.addWidget(title)
        h.addSpacing(6)
        h.addWidget(sub)
        h.addStretch(1)
        self.banner_tag = QtWidgets.QLabel("Stage 1 → Stage 2")
        self.banner_tag.setStyleSheet("color:#9fb4de;font-style:italic;")
        h.addWidget(self.banner_tag)
        root.addWidget(bar)

    # ---- header: paths + environment ---------------------------------------
    def _build_header(self, parent_layout):
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        self.paths_card = CollapsibleCard("Paths", on_toggle=self._fit_config)
        self._build_paths(self.paths_card.body_layout)
        self.env_card = CollapsibleCard("Environment", on_toggle=self._fit_config)
        self._build_env_box(self.env_card.body_layout)
        row.addWidget(self.paths_card, 1)
        row.addWidget(self.env_card, 0)
        parent_layout.addLayout(row)

    def _build_paths(self, lay):
        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)

        def row(r, key, label):
            grid.addWidget(QtWidgets.QLabel(label), r, 0)
            e = QtWidgets.QLineEdit()
            self.v[key] = QtVar("text", e)
            grid.addWidget(e, r, 1)
            b = QtWidgets.QPushButton("Browse…")
            b.clicked.connect(lambda _=False, ed=e: self._browse(ed))
            grid.addWidget(b, r, 2)

        row(0, "DATA_DIR", "Data folder (.h5 + Batch.csv)")
        row(1, "SPIKE_DIR", "Spike output folder")
        row(2, "PLOT_DIR", "Plot output folder")
        grid.addWidget(QtWidgets.QLabel("Batch CSV name"), 3, 0)
        e = QtWidgets.QLineEdit("Batch.csv")
        self.v["BATCH_CSV"] = QtVar("text", e)
        grid.addWidget(e, 3, 1)
        grid.setColumnStretch(1, 1)
        lay.addLayout(grid)

    def _build_env_box(self, lay):
        self.env_card.setMinimumWidth(320)
        gpu = QtWidgets.QGridLayout()
        gpu.setHorizontalSpacing(6)
        gpu.setVerticalSpacing(2)
        self.env_gpu_dot = QtWidgets.QLabel("●")
        self.env_gpu_dot.setStyleSheet(f"color:{PALETTE['muted']};font-size:11pt;")
        self.env_gpu_txt = QtWidgets.QLabel("checking GPU…")
        self.env_gpu_txt.setStyleSheet("font-weight:700;")
        self.env_gpu_sub = QtWidgets.QLabel("")
        self.env_gpu_sub.setStyleSheet(f"color:{PALETTE['muted']};font-size:8pt;")
        gpu.addWidget(self.env_gpu_dot, 0, 0)
        gpu.addWidget(self.env_gpu_txt, 0, 1)
        gpu.addWidget(self.env_gpu_sub, 1, 1)
        gpu.setColumnStretch(1, 1)
        lay.addLayout(gpu)
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setStyleSheet(f"color:{PALETTE['line']};")
        lay.addWidget(line)
        cap = QtWidgets.QLabel("Packages")
        cap.setStyleSheet(f"color:{PALETTE['muted']};font-size:8pt;font-weight:700;")
        lay.addWidget(cap)
        self._pkg_holder = QtWidgets.QWidget()
        self._pkg_grid = QtWidgets.QGridLayout(self._pkg_holder)
        self._pkg_grid.setContentsMargins(0, 0, 0, 0)
        self._pkg_grid.setHorizontalSpacing(6)
        self._pkg_grid.setVerticalSpacing(1)
        self._pkg_placeholder = QtWidgets.QLabel("checking packages…")
        self._pkg_placeholder.setStyleSheet(f"color:{PALETTE['muted']};")
        self._pkg_grid.addWidget(self._pkg_placeholder, 0, 0)
        lay.addWidget(self._pkg_holder)
        rr = QtWidgets.QHBoxLayout()
        self.reload_status = LabelVar(QtWidgets.QLabel(""))
        self.reload_status.lbl.setStyleSheet(f"color:{PALETTE['muted']};font-size:8pt;")
        rr.addWidget(self.reload_status.lbl)
        rr.addStretch(1)
        self.reload_btn = QtWidgets.QPushButton("⟳ Reload scripts")
        self.reload_btn.clicked.connect(self._reload_code)
        self.reload_btn.setToolTip(
            "Re-import the pipeline .py files from disk so a freshly pushed version "
            "takes effect without relaunching. Parameter tabs rebuild and your "
            "current values are restored. GUI-layout changes still need a relaunch.")
        rr.addWidget(self.reload_btn)
        lay.addLayout(rr)

    def _browse(self, line_edit):
        start = line_edit.text() or os.getcwd()
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder", start)
        if d:
            line_edit.setText(os.path.normpath(d))

    # ---- hot-reload of the pipeline modules ---------------------------------
    def _rebuild_params_tabs(self):
        self.pvars = {}
        while self.nb.count():
            w = self.nb.widget(0)
            self.nb.removeTab(0)
            w.deleteLater()
        self._populate_params_tabs()

    def _reload_code(self):
        if getattr(self, "running", False):
            self._info("Busy", "A run is in progress — reload scripts after it finishes.")
            return
        import importlib
        snap = None
        try:
            snap = self._collect_config()
        except Exception:
            pass
        order = [progress_gui, det, digital, orig, spprop, io,
                 phase_lockin, burst_latency, psd_trend, spkit_egm, activity]
        ok, bad = [], []
        for m in order:
            try:
                importlib.reload(m)
                ok.append(m.__name__.replace("mea_nap_", ""))
            except Exception as e:
                bad.append(f"{m.__name__}: {e!r}")
        try:
            progress_gui.make = (lambda names, title="Progress", subtitle="":
                                 EmbeddedProgress(self.q, names, title, subtitle))
        except Exception:
            pass
        try:
            io.QC_PARAMS["mode"] = "save"
        except Exception:
            pass
        try:
            self._rebuild_params_tabs()
        except Exception as e:
            bad.append(f"params rebuild: {e!r}")
        if snap:
            try:
                self._apply_config(snap, apply_paths=True)
            except Exception as e:
                bad.append(f"restore values: {e!r}")
        import time as _t
        stamp = _t.strftime("%H:%M:%S")
        msg = f"reloaded {len(ok)} module(s) @ {stamp}"
        if bad:
            msg += f" · {len(bad)} issue(s)"
        try:
            self.reload_status.set(msg)
        except Exception:
            pass
        self.status_var.set(msg)
        try:
            self._append_log("[reload] " + msg
                             + (("\n  " + "\n  ".join(bad)) if bad else "") + "\n")
        except Exception:
            pass
        if bad:
            self._warn("Reload finished with issues",
                       "Some modules did not reload cleanly:\n\n" + "\n".join(bad)
                       + "\n\nIf the app misbehaves, relaunch it.")

    # ---- intro fade + async environment probe -------------------------------
    def _start_intro(self):
        try:
            self.setWindowOpacity(0.0)
            self._fade_anim = QtCore.QPropertyAnimation(self, b"windowOpacity")
            self._fade_anim.setDuration(260)
            self._fade_anim.setStartValue(0.0)
            self._fade_anim.setEndValue(1.0)
            self._fade_anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
            self._fade_anim.start()
        except Exception:
            self.setWindowOpacity(1.0)
        self._pulse_i = 0
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._pulse)
        self._pulse_timer.start(220)
        threading.Thread(target=self._probe_worker, daemon=True).start()
        self._env_timer = QTimer(self)
        self._env_timer.timeout.connect(self._env_poll)
        self._env_timer.start(120)

    def _pulse(self):
        shades = ["#c4ccda", "#9fb4de", "#6f8fd0", "#9fb4de"]
        self.env_gpu_dot.setStyleSheet(
            f"color:{shades[self._pulse_i % len(shades)]};font-size:11pt;")
        self._pulse_i += 1

    def _probe_worker(self):
        try:
            res = _probe_environment()
        except Exception as e:
            res = {"gpu": ("bad", "GPU: probe failed", type(e).__name__), "pkgs": []}
        self._env_result = res

    def _env_poll(self):
        if self._env_result is None:
            return
        self._env_timer.stop()
        self._pulse_timer.stop()
        self._render_env(self._env_result)

    def _render_env(self, res):
        col = {"ok": PALETTE["ok"], "warn": PALETTE["warn"], "bad": PALETTE["bad"]}
        state, label, detail = res["gpu"]
        self.env_gpu_dot.setStyleSheet(f"color:{col.get(state, PALETTE['muted'])};font-size:11pt;")
        self.env_gpu_txt.setText(label)
        self.env_gpu_sub.setText(detail)
        self._clear_grid(self._pkg_grid)
        pkgs = res.get("pkgs", [])
        self._chip_widgets = []
        for idx, (friendly, st, dt) in enumerate(pkgs):
            r = idx // 2
            c = (idx % 2) * 2
            dot = QtWidgets.QLabel("●")
            dot.setStyleSheet(f"color:{PALETTE['panel']};")
            txt = QtWidgets.QLabel(friendly)
            txt.setStyleSheet(f"color:{PALETTE['panel']};")
            tip = f"{friendly}: {dt}" if dt else friendly
            dot.setToolTip(tip)
            txt.setToolTip(tip)
            self._pkg_grid.addWidget(dot, r, c)
            self._pkg_grid.addWidget(txt, r, c + 1)
            self._chip_widgets.append((dot, txt, col.get(st, PALETTE["muted"])))
        self._pkg_grid.setColumnStretch(1, 1)
        self._pkg_grid.setColumnStretch(3, 1)
        good = sum(1 for _, st, _ in pkgs if st == "ok")
        if pkgs:
            allok = all(st == "ok" for _, st, _ in pkgs) and state == "ok"
            self.banner_tag.setText("✓ environment ready" if allok
                                    else f"packages {good}/{len(pkgs)} · see top-right")
            self.banner_tag.setStyleSheet(
                f"color:{PALETTE['ok'] if allok else PALETTE['warn']};font-style:italic;")
        self._reveal_i = 0
        self._reveal_timer = QTimer(self)
        self._reveal_timer.timeout.connect(self._reveal_chip)
        self._reveal_timer.start(45)

    def _reveal_chip(self):
        if self._reveal_i >= len(self._chip_widgets):
            self._reveal_timer.stop()
            return
        dot, txt, color = self._chip_widgets[self._reveal_i]
        dot.setStyleSheet(f"color:{color};")
        txt.setStyleSheet(f"color:{PALETTE['text']};")
        self._reveal_i += 1

    # ---- tabbed parameter panel --------------------------------------------
    def _scroll_tab(self, title):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(content)
        grid.setContentsMargins(10, 6, 10, 6)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        scroll.setWidget(content)
        self.nb.addTab(scroll, title)
        return content, grid

    def _build_outputs_panel(self, parent_layout):
        self.outputs_card = CollapsibleCard(
            "Analyses to run   (uncheck all → stop right after spike detection)",
            on_toggle=self._fit_config)
        lay = self.outputs_card.body_layout
        bar = QtWidgets.QHBoxLayout()
        b1 = QtWidgets.QPushButton("All on")
        b1.clicked.connect(lambda: self._set_all_sections(True))
        b2 = QtWidgets.QPushButton("Detection only")
        b2.clicked.connect(self._sections_detect_only)
        b3 = QtWidgets.QPushButton("All off")
        b3.clicked.connect(lambda: self._set_all_sections(False))
        for b in (b1, b2, b3):
            bar.addWidget(b)
        bar.addStretch(1)
        lay.addLayout(bar)
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(10)
        init = {
            "detect": True,
            "activity":    bool(activity.STAGE2_SECTIONS.get("activity", True)),
            "network":     bool(activity.STAGE2_SECTIONS.get("network", True)),
            "spectral":    bool(activity.STAGE2_SECTIONS.get("spectral", True)),
            "meanap":      bool(activity.STAGE2_SECTIONS.get("meanap", True)),
            "comparisons": bool(activity.STAGE2_SECTIONS.get("comparisons", True)),
            "spkit":       bool(spkit_egm.SPKIT_EGM.get("enable", False)),
            "phase":       bool(phase_lockin.PHASE_LOCKIN.get("enable", False)),
            "burstlat":    bool(burst_latency.BURST_LATENCY.get("enable", False)),
            "psdtrend":    bool(psd_trend.PSD_TREND.get("enable", False)),
        }
        self.section_vars = {}
        for i, (key, label, help_txt) in enumerate(self._SECTION_DEFS):
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(init.get(key, True))
            self.section_vars[key] = QtVar("bool", cb)
            cell = QtWidgets.QWidget()
            hl = QtWidgets.QHBoxLayout(cell)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(2)
            hl.addWidget(cb)
            hl.addWidget(self._help_icon(help_txt))
            hl.addStretch(1)
            grid.addWidget(cell, i // 4, i % 4)
        for c in range(4):
            grid.setColumnStretch(c, 1)
        lay.addLayout(grid)
        self.v["RUN1"] = self.section_vars["detect"]
        parent_layout.addWidget(self.outputs_card)

    def _set_all_sections(self, state):
        for var in self.section_vars.values():
            var.set(bool(state))

    def _sections_detect_only(self):
        for key, var in self.section_vars.items():
            var.set(key == "detect")

    def _build_params_tabs(self, parent_layout):
        self.params_card = CollapsibleCard(
            "Parameters  (value  ·  adjustment range)", on_toggle=self._fit_config)
        self.params_card.body_layout.setContentsMargins(4, 2, 4, 4)
        self.nb = QtWidgets.QTabWidget()
        self.nb.setDocumentMode(True)
        self.nb.setUsesScrollButtons(True)
        self.nb.setElideMode(Qt.TextElideMode.ElideNone)
        self.params_card.body_layout.addWidget(self.nb)
        self._populate_params_tabs()
        parent_layout.addWidget(self.params_card, 1)

    def _populate_params_tabs(self):
        muted = f"color:{PALETTE['muted']};"
        for title, specs in build_schema():
            _content, grid = self._scroll_tab(title)
            grid.setColumnMinimumWidth(1, 170)
            for c, t in enumerate(("Parameter", "Value", "Adjustment range")):
                lbl = QtWidgets.QLabel(t)
                lbl.setStyleSheet(muted)
                grid.addWidget(lbl, 0, c)
            for r, spec in enumerate(specs, start=1):
                self._param_row(grid, r, spec)
            grid.setColumnStretch(4, 1)
            grid.setRowStretch(len(specs) + 1, 1)
        self._build_plot_outputs_tab()
        self._build_preview_tab()

    def _param_row(self, grid, r, spec):
        grid.addWidget(QtWidgets.QLabel(spec["label"]), r, 0)
        t = spec["type"]
        if t == "bool":
            w = QtWidgets.QCheckBox()
            w.setChecked(bool(_cur(spec)))
            var = QtVar("bool", w)
        elif t in ("enum", "bool3"):
            w = QtWidgets.QComboBox()
            w.addItems([str(o) for o in (spec["options"] or [])])
            w.setMinimumWidth(150)
            var = QtVar("text", w)
            var.set(_fmt(spec))
        else:
            w = QtWidgets.QLineEdit()
            w.setText(_fmt(spec))
            w.setMinimumWidth(150)
            if t == "readonly":
                w.setReadOnly(True)
            var = QtVar("text", w)
        grid.addWidget(w, r, 1)
        rng = QtWidgets.QLabel(spec["rng"])
        rng.setStyleSheet(f"color:{PALETTE['muted']};")
        grid.addWidget(rng, r, 2)
        help_txt = PARAM_HELP.get(spec["skey"]) or spec.get("rng") or spec["label"]
        grid.addWidget(self._help_icon(help_txt), r, 3)
        self.pvars[spec["skey"]] = (var, spec)

    def _build_plot_outputs_tab(self):
        _content, grid = self._scroll_tab("Plot outputs")
        self._checkall_cb = QtWidgets.QCheckBox("Check all")
        self._checkall_cb.setChecked(True)
        self.checkall_var = QtVar("bool", self._checkall_cb)
        self._checkall_cb.clicked.connect(self._toggle_all_plots)
        grid.addWidget(self._checkall_cb, 0, 0, 1, 2)
        r = 0
        for key in orig.PLOT_ENABLE.keys():
            r += 1
            cb = QtWidgets.QCheckBox(PLOT_LABELS.get(key, key))
            cb.setChecked(bool(orig.PLOT_ENABLE.get(key, True)))
            grid.addWidget(cb, r, 0)
            grid.addWidget(self._help_icon(PLOT_HELP.get(key, key)), r, 1)
            self.pvars["PLOT_ENABLE." + key] = (QtVar("bool", cb), {
                "obj": orig.PLOT_ENABLE, "key": key, "type": "bool",
                "skey": "PLOT_ENABLE." + key, "rng": ""})
        r += 1
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setStyleSheet(f"color:{PALETTE['line']};")
        grid.addWidget(sep, r, 0, 1, 2)
        for key in spprop.SP_ENABLE.keys():
            r += 1
            cb = QtWidgets.QCheckBox(PLOT_LABELS.get(key, key))
            cb.setChecked(bool(spprop.SP_ENABLE.get(key, True)))
            grid.addWidget(cb, r, 0)
            grid.addWidget(self._help_icon(PLOT_HELP.get(key, key)), r, 1)
            self.pvars["SP_ENABLE." + key] = (QtVar("bool", cb), {
                "obj": spprop.SP_ENABLE, "key": key, "type": "bool",
                "skey": "SP_ENABLE." + key, "rng": ""})
        grid.setColumnStretch(2, 1)
        grid.setRowStretch(r + 1, 1)

    def _toggle_all_plots(self):
        state = self.checkall_var.get()
        for skey, (var, spec) in self.pvars.items():
            if skey.startswith("PLOT_ENABLE.") or skey.startswith("SP_ENABLE."):
                var.set(state)

    def _build_preview_tab(self):
        _content, grid = self._scroll_tab("Preview")
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Preview:"))
        self._preview_cb = QtWidgets.QComboBox()
        self._preview_cb.addItems(["Electrode heatmap", "Raster",
                                   "Firing-rate half-violin", "Network (participation)",
                                   "Node cartography", "Group half-violin"])
        self.preview_kind = QtVar("text", self._preview_cb)
        bar.addWidget(self._preview_cb)
        upd = QtWidgets.QPushButton("Update preview")
        upd.clicked.connect(self._render_preview)
        bar.addWidget(upd)
        bar.addStretch(1)
        holder = QtWidgets.QWidget()
        holder.setLayout(bar)
        grid.addWidget(holder, 0, 0)
        self.preview_label = QtWidgets.QLabel(
            "Pick a plot and click 'Update preview' to see how the\n"
            "current colour/size parameters render (synthetic data).")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.preview_label, 1, 0)
        grid.setRowStretch(2, 1)

    def _render_preview(self):
        import tempfile
        for skey, (var, spec) in self.pvars.items():
            try:
                val = _parse(spec, var)
                if spec.get("setter"):
                    spec["setter"](val)
                elif spec["type"] != "readonly":
                    spec["obj"][spec["key"]] = val
            except Exception:
                pass
        kind = self.preview_kind.get()
        tmp = tempfile.mkdtemp()
        old_dpi = orig.MEANAP_PLOT_PARAMS["dpi"]
        orig.MEANAP_PLOT_PARAMS["dpi"] = 92
        path = None
        try:
            unit, res = orig._synthetic_unit("PREVIEW", "stim", 14, seed=1)
            gmax = float(max(res["fr"])) if len(res["fr"]) else 1.0
            if kind == "Electrode heatmap":
                orig.fig_electrode_heatmap(unit, res, tmp, gmax)
                path = os.path.join(tmp, "2_Heatmap.png")
            elif kind == "Raster":
                orig.fig_raster(unit, res, tmp, gmax)
                path = os.path.join(tmp, "3_Raster.png")
            elif kind == "Firing-rate half-violin":
                orig.fig_firing_rate_by_electrode(unit, res, tmp)
                path = os.path.join(tmp, "1_FiringRateByElectrode.png")
            elif kind == "Network (participation)":
                nr = orig._synthetic_nr()
                orig.fig_network_plot(nr, tmp, 25, "PREVIEW", "stim", "6",
                                      metric=orig.NET_COLORMAP_METRICS[3])
                path = os.path.join(tmp, "6_MEA_NetworkPlot_ParticipationCoefficient.png")
            elif kind == "Node cartography":
                nr = orig._synthetic_nr()
                orig.fig_node_cartography(nr, tmp, 25, "PREVIEW", "stim")
                path = os.path.join(tmp, "9_adjM25msNodeCartography.png")
            elif kind == "Group half-violin":
                import pandas as pd
                import numpy as _np
                rng = _np.random.default_rng(3)
                rows = []
                for grp in ("Control", "Laser"):
                    for reg in ("pre", "stim", "post"):
                        b = 1.5 if reg == "stim" else 1.0
                        for _ in range(40):
                            rows.append({"group": grp, "region": reg, "age": 14,
                                         "degree": rng.random() * b})
                path = os.path.join(tmp, "preview_violin.png")
                orig.fig_region_by_x(pd.DataFrame(rows), "degree", "Node degree",
                                     "group", path, xlabel="Group")
            pix = QtGui.QPixmap(path)
            if pix.width() > 560:
                pix = pix.scaledToWidth(560, Qt.TransformationMode.SmoothTransformation)
            self.preview_label.setPixmap(pix)
            self.preview_label.setText("")
        except Exception as e:
            self.preview_label.setText(f"Preview failed: {e!r}")
        finally:
            orig.MEANAP_PLOT_PARAMS["dpi"] = old_dpi

    # ---- run controls -------------------------------------------------------
    def _build_controls(self, root):
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setStyleSheet(f"color:{PALETTE['line']};")
        root.addWidget(sep)
        bar = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(bar)
        h.setContentsMargins(12, 6, 12, 6)
        lbl = QtWidgets.QLabel("Sections chosen above ↑")
        lbl.setStyleSheet(f"color:{PALETTE['muted']};")
        h.addWidget(lbl)
        sc = QtWidgets.QPushButton("\U0001f4be Save config…")
        sc.clicked.connect(self._save_config_as)
        h.addWidget(sc)
        lc = QtWidgets.QPushButton("\U0001f4c2 Load config…")
        lc.clicked.connect(self._load_config_from)
        h.addWidget(lc)
        h.addStretch(1)
        self.status_var = LabelVar(QtWidgets.QLabel("Ready."))
        self.status_var.lbl.setStyleSheet(f"color:{PALETTE['muted']};")
        h.addWidget(self.status_var.lbl)
        self.groups_btn = QtWidgets.QPushButton("⚙  Groups")
        self.groups_btn.clicked.connect(self._show_setup_view)
        h.addWidget(self.groups_btn)
        self.replot_btn = QtWidgets.QPushButton("↻  Replot")
        self.replot_btn.setToolTip(
            "Redraw the Phase Lock-In figures from the cached results of the last "
            "run (phase_lockin_results.pkl in the Phase Lock output folder) "
            "without recomputing anything.\nEdit the plotting code, press "
            "“Reload scripts”, then Replot.")
        self.replot_btn.clicked.connect(self._on_replot)
        h.addWidget(self.replot_btn)
        self.run_btn = QtWidgets.QPushButton("▶  Run")
        self.run_btn.setProperty("primary", True)
        self.run_btn.clicked.connect(self._on_run)
        h.addWidget(self.run_btn)
        root.addWidget(bar)

    def _set_run_button(self, mode):
        try:
            self.run_btn.clicked.disconnect()
        except Exception:
            pass
        if mode == "abort":
            self.run_btn.setText("⏹  Abort")
            self.run_btn.setProperty("primary", False)
            self.run_btn.setStyleSheet(
                f"background:{PALETTE['bad']};color:#ffffff;font-weight:700;"
                f"border:none;border-radius:8px;padding:6px 16px;")
            self.run_btn.clicked.connect(self._on_abort)
        else:
            self.run_btn.setText("▶  Run")
            self.run_btn.setStyleSheet("")
            self.run_btn.setProperty("primary", True)
            self.run_btn.clicked.connect(self._on_run)
        self.run_btn.style().unpolish(self.run_btn)
        self.run_btn.style().polish(self.run_btn)
        self.run_btn.setEnabled(True)

    # ---- setup view (group panel) ------------------------------------------
    def _build_setup_view(self):
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(12, 4, 12, 10)
        box = QtWidgets.QGroupBox("Groups  (from Batch.csv)")
        bl = QtWidgets.QVBoxLayout(box)
        bar = QtWidgets.QHBoxLayout()
        ref = QtWidgets.QPushButton("⟳  Refresh from Batch.csv")
        ref.clicked.connect(self._refresh_groups)
        bar.addWidget(ref)
        self.group_info = LabelVar(QtWidgets.QLabel(""))
        self.group_info.lbl.setStyleSheet(f"color:{PALETTE['muted']};")
        bar.addWidget(self.group_info.lbl)
        bar.addStretch(1)
        for text, fn in (("Analyze: all", lambda: self._set_all_groups("analyze", True)),
                         ("Analyze: none", lambda: self._set_all_groups("analyze", False)),
                         ("Detect: all", lambda: self._set_all_groups("detect", True)),
                         ("Detect: none", lambda: self._set_all_groups("detect", False))):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            bar.addWidget(b)
        bl.addLayout(bar)
        self.group_scroll = QtWidgets.QScrollArea()
        self.group_scroll.setWidgetResizable(True)
        self.group_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.group_body = QtWidgets.QWidget()
        self.group_grid = QtWidgets.QGridLayout(self.group_body)
        self.group_grid.setContentsMargins(6, 4, 6, 4)
        self.group_grid.setHorizontalSpacing(16)
        self.group_scroll.setWidget(self.group_body)
        bl.addWidget(self.group_scroll, 1)
        bottom = QtWidgets.QHBoxLayout()
        self.by_group_chk = QtWidgets.QCheckBox(
            "Analysis by group  (produce group-comparison figures)")
        self.by_group_chk.setEnabled(False)
        self.by_group_var = QtVar("bool", self.by_group_chk)
        bottom.addWidget(self.by_group_chk)
        self.by_group_hint = LabelVar(
            QtWidgets.QLabel("Select ≥2 groups for analysis to enable."))
        self.by_group_hint.lbl.setStyleSheet(f"color:{PALETTE['muted']};")
        bottom.addWidget(self.by_group_hint.lbl)
        bottom.addStretch(1)
        bl.addLayout(bottom)
        v.addWidget(box)
        self.views_stack.addWidget(page)          # index 0

    def _build_run_view(self):
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(12, 4, 12, 10)
        v.setSpacing(6)
        self._build_progress(v)
        self._build_log(v)
        self.views_stack.addWidget(page)          # index 1

    def _show_run_view(self):
        try:
            self._prerun_fold = (self.paths_card.collapsed, self.env_card.collapsed)
            self.paths_card.set_collapsed(True)
            self.env_card.set_collapsed(True)
        except Exception:
            pass
        self.views_stack.setCurrentIndex(1)
        self._fit_config()

    def _show_setup_view(self):
        if self.running:
            return
        try:
            pf, ef = getattr(self, "_prerun_fold", (False, False))
            self.paths_card.set_collapsed(pf)
            self.env_card.set_collapsed(ef)
        except Exception:
            pass
        self.views_stack.setCurrentIndex(0)
        self._fit_config()

    # ---- group panel helpers ------------------------------------------------
    @staticmethod
    def _grp_sort_key(g):
        m = re.match(r"\s*(\d+(?:\.\d+)?)", str(g))
        return (1 if m else 0, float(m.group(1)) if m else 0.0, str(g).lower())

    def _refresh_groups(self):
        data_dir = self.v["DATA_DIR"].get().strip()
        csv_name = self.v["BATCH_CSV"].get().strip() or "Batch.csv"
        csv_path = os.path.join(data_dir, csv_name) if data_dir else ""
        if not csv_path or not os.path.exists(csv_path):
            self.group_info.set("Batch.csv not found — set the Data folder, then Refresh.")
            self._populate_group_rows([])
            return
        try:
            io.CONFIG["BATCH_CSV_NAME"] = csv_name
            rows = io.parse_batch(csv_path)
            summary = io.group_summary(rows)
        except Exception as e:
            self.group_info.set(f"Could not read Batch.csv: {e}")
            self._populate_group_rows([])
            return
        summary.sort(key=lambda t: self._grp_sort_key(t[0]))
        self._populate_group_rows(summary)
        nfiles = len({r["stem"] for r in rows})
        self.group_info.set(f"{len(summary)} group(s) · {nfiles} file(s) · {len(rows)} row(s)")

    def _populate_group_rows(self, summary):
        self._clear_grid(self.group_grid)
        self.group_detect = {}
        self.group_analyze = {}
        if not summary:
            lbl = QtWidgets.QLabel("(no groups)")
            lbl.setStyleSheet(f"color:{PALETTE['muted']};")
            self.group_grid.addWidget(lbl, 0, 0)
            self._update_by_group_state()
            return
        for c, t in enumerate(("Group", "Files", "Spike detection", "Analysis")):
            lbl = QtWidgets.QLabel(t)
            lbl.setStyleSheet(f"color:{PALETTE['muted']};font-weight:700;")
            self.group_grid.addWidget(lbl, 0, c)
        for i, (g, nrows, nfiles) in enumerate(summary, start=1):
            prev = self._group_state.get(g, {})
            dcb = QtWidgets.QCheckBox()
            dcb.setChecked(bool(prev.get("detect", True)))
            acb = QtWidgets.QCheckBox()
            acb.setChecked(bool(prev.get("analyze", True)))
            self.group_detect[g] = QtVar("bool", dcb)
            self.group_analyze[g] = QtVar("bool", acb)
            gl = QtWidgets.QLabel(g)
            gl.setStyleSheet("font-weight:700;")
            self.group_grid.addWidget(gl, i, 0)
            fl = QtWidgets.QLabel(f"{nfiles} file(s), {nrows} row(s)")
            fl.setStyleSheet(f"color:{PALETTE['muted']};")
            self.group_grid.addWidget(fl, i, 1)
            self.group_grid.addWidget(dcb, i, 2)
            self.group_grid.addWidget(acb, i, 3)
            acb.stateChanged.connect(self._update_by_group_state)
        self.group_grid.setColumnStretch(4, 1)
        self.group_grid.setRowStretch(len(summary) + 1, 1)
        self._update_by_group_state()

    def _set_all_groups(self, which, val):
        d = self.group_detect if which == "detect" else self.group_analyze
        for v in d.values():
            v.set(val)
        self._update_by_group_state()

    def _update_by_group_state(self, *_):
        n = sum(1 for v in self.group_analyze.values() if v.get())
        enabled = n >= 2
        if enabled and not self._by_group_prev:
            self.by_group_var.set(True if self._saved_by_group is None
                                  else bool(self._saved_by_group))
        if not enabled:
            self.by_group_var.set(False)
        try:
            self.by_group_chk.setEnabled(enabled)
        except Exception:
            pass
        self.by_group_hint.set("" if enabled
                               else "Select ≥2 groups for analysis to enable.")
        self._by_group_prev = enabled

    # ---- progress -----------------------------------------------------------
    def _build_progress(self, v):
        box = QtWidgets.QGroupBox("Progress")
        bl = QtWidgets.QVBoxLayout(box)
        top = QtWidgets.QHBoxLayout()
        self.big_pct = LabelVar(QtWidgets.QLabel("0%"))
        self.big_pct.lbl.setStyleSheet("font-size:15pt;font-weight:700;")
        top.addWidget(self.big_pct.lbl)
        self.bigbar = QtWidgets.QProgressBar()
        self.bigbar.setMaximum(100)
        self.bigbar.setValue(0)
        self.bigbar.setTextVisible(False)
        self.bigbar.setFixedHeight(24)
        top.addWidget(self.bigbar, 1)
        bl.addLayout(top)
        self.phase_var = LabelVar(QtWidgets.QLabel("—"))
        self.phase_var.lbl.setStyleSheet("font-weight:700;")
        bl.addWidget(self.phase_var.lbl)
        self.overall_var = LabelVar(QtWidgets.QLabel(""))
        self.overall_var.lbl.setStyleSheet(f"color:{PALETTE['muted']};")
        bl.addWidget(self.overall_var.lbl)
        self.ring_scroll = QtWidgets.QScrollArea()
        self.ring_scroll.setWidgetResizable(True)
        self.ring_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.ring_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.ring_scroll.setMinimumHeight(170)
        self.prog_body = QtWidgets.QWidget()
        self.ring_flow = FlowLayout(self.prog_body, margin=6, spacing=10)
        self.ring_scroll.setWidget(self.prog_body)
        bl.addWidget(self.ring_scroll, 1)
        box.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                          QtWidgets.QSizePolicy.Policy.Preferred)
        v.addWidget(box, 0)

    def _build_log(self, v):
        box = QtWidgets.QGroupBox("Log")
        bl = QtWidgets.QVBoxLayout(box)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(120)
        self.log.setStyleSheet(
            f"background:{PALETTE['log_bg']};color:{PALETTE['text']};"
            f"font-family:Consolas,monospace;font-size:9pt;border:none;")
        bl.addWidget(self.log)
        v.addWidget(box, 1)

    # ---- ring rows ----------------------------------------------------------
    def _build_rows(self, names, phase):
        self.ring_flow.clear()
        self.rows = {}
        self.phase_var.set(phase or "—")
        for nm in names:
            cell = QtWidgets.QWidget()
            cv = QtWidgets.QVBoxLayout(cell)
            cv.setContentsMargins(0, 0, 0, 0)
            cv.setSpacing(1)
            ring = RingWidget()
            name = QtWidgets.QLabel(self._trunc(nm, 20))
            name.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            name.setStyleSheet("font-size:8pt;font-weight:700;")
            note = QtWidgets.QLabel("pending")
            note.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            note.setStyleSheet(f"color:{PALETTE['muted']};font-size:7pt;")
            cv.addWidget(ring, 0, Qt.AlignmentFlag.AlignHCenter)
            cv.addWidget(name)
            cv.addWidget(note)
            cell.setFixedWidth(CELL_W - 16)
            self.ring_flow.addWidget(cell)
            self.rows[nm] = {"ring": ring, "note": note, "frac": 0.0, "state": "pending"}
        self.prog_body.updateGeometry()
        self._update_overall()

    def _update_row(self, name, frac, state, note):
        r = self.rows.get(name)
        if r is None:
            return
        if frac is not None:
            r["frac"] = max(0.0, min(float(frac), 1.0))
        if state:
            r["state"] = state
        if note is not None:
            r["note"].setText(self._trunc(note, 22))
        r["ring"].set_state(r["frac"], r["state"])
        self._update_overall()

    def _update_overall(self):
        if self.rows:
            done = sum(1 for x in self.rows.values()
                       if x["state"] in ("done", "skipped", "failed"))
            overall = 100.0 * sum(x["frac"] for x in self.rows.values()) / len(self.rows)
            self.overall_var.set(f"{done}/{len(self.rows)} samples complete")
        else:
            overall = 0.0
            self.overall_var.set("")
        self.bigbar.setValue(int(round(overall)))
        self.big_pct.set(f"{overall:.0f}%")

    def _append_log(self, text):
        self.log.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    # ---- queue drain --------------------------------------------------------
    def _drain(self):
        try:
            while True:
                self._apply(self.q.get_nowait())
        except queue.Empty:
            pass

    def _apply(self, cmd):
        kind = cmd[0]
        if kind == "newrun":
            self._build_rows(cmd[1], cmd[3] or cmd[2])
        elif kind == "update":
            self._update_row(cmd[1], cmd[2], cmd[3], cmd[4])
        elif kind == "log":
            self._append_log(cmd[1])
        elif kind == "status":
            self.status_var.set(cmd[1])
        elif kind == "error":
            self._append_log("\n" + "=" * 60 + "\n[ERROR]\n" + cmd[1] + "\n")
            self.status_var.set("Stopped with an error — see log.")
            self._err("Pipeline error",
                      "The run stopped with an error. See the Log panel.")
        elif kind == "finished":
            self.status_var.set(cmd[1])
        elif kind == "aborted":
            self._append_log("\n[abort] run stopped by user.\n")
            self.status_var.set("Aborted.")
        elif kind == "reenable":
            self.running = False
            self._set_run_button("run")
            self.groups_btn.setEnabled(True)
            self.replot_btn.setEnabled(True)

    # ---- run ----------------------------------------------------------------
    def _apply_settings_to_modules(self):
        io.CONFIG["DATA_DIR"] = self.v["DATA_DIR"].get().strip()
        io.CONFIG["SPIKE_DIR"] = self.v["SPIKE_DIR"].get().strip()
        io.CONFIG["PLOT_DIR"] = self.v["PLOT_DIR"].get().strip()
        io.CONFIG["BATCH_CSV_NAME"] = self.v["BATCH_CSV"].get().strip() or "Batch.csv"
        if self.group_detect:
            io.CONFIG["GROUPS_DETECT"] = [g for g, v in self.group_detect.items() if v.get()]
            io.CONFIG["GROUPS_ANALYZE"] = [g for g, v in self.group_analyze.items() if v.get()]
        else:
            io.CONFIG["GROUPS_DETECT"] = None
            io.CONFIG["GROUPS_ANALYZE"] = None
        io.CONFIG["BY_GROUP"] = bool(self.by_group_var.get())
        sv = self.section_vars
        activity.STAGE2_SECTIONS["activity"]    = bool(sv["activity"].get())
        activity.STAGE2_SECTIONS["network"]     = bool(sv["network"].get())
        activity.STAGE2_SECTIONS["spectral"]    = bool(sv["spectral"].get())
        activity.STAGE2_SECTIONS["meanap"]      = bool(sv["meanap"].get())
        activity.STAGE2_SECTIONS["comparisons"] = bool(sv["comparisons"].get())
        activity.MEANAP_ORIGINAL_PLOTS = bool(sv["meanap"].get())
        spkit_egm.SPKIT_EGM["enable"]  = bool(sv["spkit"].get())
        phase_lockin.PHASE_LOCKIN["enable"] = bool(sv["phase"].get())
        burst_latency.BURST_LATENCY["enable"] = bool(sv["burstlat"].get())
        psd_trend.PSD_TREND["enable"] = bool(sv["psdtrend"].get())
        for skey, (var, spec) in self.pvars.items():
            try:
                val = _parse(spec, var)
                if spec.get("setter"):
                    spec["setter"](val)
                elif spec["type"] != "readonly":
                    spec["obj"][spec["key"]] = val
            except Exception as e:
                print(f"[param warn] {skey}: {e!r} — kept current value")

    def _on_run(self):
        if self.running:
            return
        data_dir = self.v["DATA_DIR"].get().strip()
        if not data_dir or not os.path.isdir(data_dir):
            self._err("Missing data folder",
                      "Set a valid Data folder (with .h5 files and Batch.csv).")
            return
        csv = os.path.join(data_dir, self.v["BATCH_CSV"].get().strip() or "Batch.csv")
        if not os.path.exists(csv) and not self._ask(
                "Batch.csv not found", f"{csv}\n\ndoes not exist. Run anyway?"):
            return
        do1 = bool(self.v["RUN1"].get())
        try:
            self._apply_settings_to_modules()
        except Exception:
            self._err("Bad parameter",
                      "A parameter could not be parsed:\n\n" + traceback.format_exc())
            return
        do2 = any(activity.STAGE2_SECTIONS.values())
        do3 = bool(spkit_egm.SPKIT_EGM.get("enable"))
        do4 = bool(phase_lockin.PHASE_LOCKIN.get("enable"))
        do5 = bool(burst_latency.BURST_LATENCY.get("enable"))
        do6 = bool(psd_trend.PSD_TREND.get("enable"))
        if not (do1 or do2 or do3 or do4 or do5 or do6):
            self._info("Nothing to run",
                       "Check at least one analysis in the “Analyses to run” panel.")
            return
        if self.group_detect:
            det_on = [g for g, v in self.group_detect.items() if v.get()]
            ana_on = [g for g, v in self.group_analyze.items() if v.get()]
            if do1 and not det_on:
                self._err("No groups for detection",
                          "Stage 1 is enabled but no group is checked for spike "
                          "detection. Enable at least one group, or turn off Stage 1.")
                return
            if (do2 or do4 or do5 or do6) and not ana_on:
                self._err("No groups for analysis",
                          "An analysis stage is enabled but no group is checked for "
                          "analysis. Enable at least one group, or turn off the "
                          "analysis stages.")
                return
        self._save_settings()
        io.clear_abort()
        self._show_run_view()
        self._build_rows([], "starting…")
        self.log.clear()
        self.running = True
        self._set_run_button("abort")
        self.groups_btn.setEnabled(False)
        self.replot_btn.setEnabled(False)
        self.status_var.set("Running…")
        threading.Thread(target=self._worker, args=(do1, do2, do3, do4, do5, do6),
                         daemon=True).start()

    # ---- replot from cached results ------------------------------------------
    def _on_replot(self):
        if self.running:
            return
        try:
            self._apply_settings_to_modules()
        except Exception:
            self._err("Bad parameter",
                      "A parameter could not be parsed:\n\n" + traceback.format_exc())
            return
        plot_dir = io.CONFIG.get("PLOT_DIR", "")
        if not plot_dir:
            self._err("Missing output folder", "Set the Plot output folder first.")
            return
        root = phase_lockin.phase_lock_root(plot_dir)
        cache = phase_lockin.results_path(root)
        if not os.path.exists(cache):
            self._info("Nothing to replot",
                       "No cached Phase Lock-In results were found at\n\n"
                       f"{cache}\n\nRun the Phase Lock-In analysis once; every run "
                       "saves its results there, and Replot redraws the figures "
                       "from that file.")
            return
        self._save_settings()
        io.clear_abort()
        self._show_run_view()
        self._build_rows([], "replotting…")
        self.log.clear()
        self.running = True
        self.run_btn.setEnabled(False)
        self.groups_btn.setEnabled(False)
        self.replot_btn.setEnabled(False)
        self.status_var.set("Replotting Phase Lock-In…")
        threading.Thread(target=self._replot_worker, daemon=True).start()

    def _replot_worker(self):
        import time
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = _QueueWriter(self.q)
        sys.stderr = _Sink()
        try:
            print("\n===== Phase Lock-In: replot from cached results =====")
            _t = time.perf_counter()
            n = phase_lockin.replot_phase_lockin()
            print(f"  {n} figure(s) rewritten in {io._fmt_dur(time.perf_counter() - _t)}")
            self.q.put(("finished", f"Replot done - {n} figure(s)."))
        except Exception:
            self.q.put(("error", traceback.format_exc()))
        finally:
            io.clear_abort()
            sys.stdout, sys.stderr = old_out, old_err
            self.q.put(("reenable",))

    def _on_abort(self):
        if not self.running:
            return
        io.request_abort()
        self.status_var.set("Aborting — stopping after the current recording…")
        self.run_btn.setEnabled(False)
        self.run_btn.setText("⏹  Aborting…")
        try:
            self._append_log("\n[abort] requested — the run will stop after the "
                             "current recording finishes.\n")
        except Exception:
            pass

    def _worker(self, do1, do2, do3=False, do4=False, do5=False, do6=False):
        import time
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = _QueueWriter(self.q)
        sys.stderr = _Sink()
        timings = []
        try:
            if do1:
                self.q.put(("status", "Stage 1: spike detection…"))
                print("\n===== STAGE 1: spike detection =====")
                _t = time.perf_counter(); io.run_stage1()
                timings.append(("Stage 1 (spike detection)", time.perf_counter() - _t))
            if do2:
                self.q.put(("status", "Stage 2: activity + network…"))
                print("\n===== STAGE 2: activity + network =====")
                _t = time.perf_counter(); activity.run_stage2()
                timings.append(("Stage 2 (activity + network)", time.perf_counter() - _t))
            if do3:
                self.q.put(("status", "SpKit: continuous-EGM conduction velocity…"))
                print("\n===== SpKit EGM: continuous conduction velocity =====")
                _t = time.perf_counter(); spkit_egm.run_spkit_egm()
                timings.append(("SpKit EGM", time.perf_counter() - _t))
            if do4:
                self.q.put(("status", "Phase Lock-In: sweep phase propagation…"))
                print("\n===== Phase Lock-In: sweep phase propagation =====")
                _t = time.perf_counter(); phase_lockin.run_phase_lockin()
                timings.append(("Phase Lock-In", time.perf_counter() - _t))
            if do5:
                self.q.put(("status", "Burst latency: network-burst propagation…"))
                print("\n===== Burst latency: network-burst propagation =====")
                _t = time.perf_counter(); burst_latency.run_burst_latency()
                timings.append(("Burst latency", time.perf_counter() - _t))
            if do6:
                self.q.put(("status", "PSD trends: bursting rhythm by day…"))
                print("\n===== PSD trends: bursting rhythm by day =====")
                _t = time.perf_counter(); psd_trend.run_psd_trend()
                timings.append(("PSD trends", time.perf_counter() - _t))
            if timings:
                print("\n===== TIMING SUMMARY =====")
                for name, dt in timings:
                    print(f"  {name:<30s} {io._fmt_dur(dt)}")
                print(f"  {'TOTAL':<30s} {io._fmt_dur(sum(d for _, d in timings))}")
            self.q.put(("finished", "All done."))
            print("\n===== PIPELINE COMPLETE =====")
        except io.AbortRun:
            print("\n===== RUN ABORTED (by user) =====")
            if timings:
                print("Completed before abort:")
                for name, dt in timings:
                    print(f"  {name:<30s} {io._fmt_dur(dt)}")
            self.q.put(("aborted", "Run aborted."))
        except Exception:
            self.q.put(("error", traceback.format_exc()))
        finally:
            io.clear_abort()
            sys.stdout, sys.stderr = old_out, old_err
            self.q.put(("reenable",))

    # ---- config (de)serialization ------------------------------------------
    def _collect_config(self):
        data = {"_schema": "mea-nap-config/1"}
        for k in ("DATA_DIR", "SPIKE_DIR", "PLOT_DIR", "BATCH_CSV"):
            try:
                data[k] = self.v[k].get()
            except Exception:
                pass
        try:
            data["SECTIONS"] = {key: bool(var.get())
                                for key, var in self.section_vars.items()}
        except Exception:
            pass
        for skey, (var, spec) in self.pvars.items():
            try:
                data["P:" + skey] = var.get()
            except Exception:
                pass
        try:
            data["GROUP_STATE"] = {
                g: {"detect": bool(self.group_detect[g].get()),
                    "analyze": bool(self.group_analyze[g].get())}
                for g in self.group_detect}
            data["BY_GROUP"] = bool(self.by_group_var.get())
        except Exception:
            pass
        try:
            sizes = self.splitter.sizes()
            data["WINDOW"] = {
                "geometry": [self.x(), self.y(), self.width(), self.height()],
                "sash": sizes[0] if len(sizes) > 1 else None,
                "fold_paths": bool(self.paths_card.collapsed),
                "fold_env": bool(self.env_card.collapsed),
                "fold_analyses": bool(getattr(self, "outputs_card", None)
                                      and self.outputs_card.collapsed),
                "fold_params": bool(getattr(self, "params_card", None)
                                    and self.params_card.collapsed),
            }
        except Exception:
            pass
        return data

    def _apply_config(self, data, apply_paths=True):
        path_keys = {"DATA_DIR", "SPIKE_DIR", "PLOT_DIR", "BATCH_CSV"}
        for k, val in data.items():
            try:
                if k.startswith("P:"):
                    entry = self.pvars.get(k[2:])
                    if entry:
                        entry[0].set(val)
                elif k in self.v:
                    if k in path_keys and not apply_paths:
                        continue
                    self.v[k].set(val)
            except Exception:
                pass
        for key, val in (data.get("SECTIONS", {}) or {}).items():
            if key in self.section_vars:
                try:
                    self.section_vars[key].set(bool(val))
                except Exception:
                    pass
        self._group_state = data.get("GROUP_STATE", {}) or {}
        self._saved_by_group = data.get("BY_GROUP", None)
        w = data.get("WINDOW") or {}
        if w:
            g = w.get("geometry")
            try:
                if isinstance(g, (list, tuple)) and len(g) == 4:
                    self.setGeometry(int(g[0]), int(g[1]), int(g[2]), int(g[3]))
                elif isinstance(g, str):
                    m = re.match(r"(\d+)x(\d+)", g)
                    if m:
                        self.resize(int(m.group(1)), int(m.group(2)))
            except Exception:
                pass
            try:
                self.paths_card.set_collapsed(bool(w.get("fold_paths")))
                self.env_card.set_collapsed(bool(w.get("fold_env")))
                self._prerun_fold = (bool(w.get("fold_paths")), bool(w.get("fold_env")))
                if getattr(self, "outputs_card", None):
                    self.outputs_card.set_collapsed(bool(w.get("fold_analyses")))
                if getattr(self, "params_card", None):
                    self.params_card.set_collapsed(bool(w.get("fold_params")))
            except Exception:
                pass
            sash = w.get("sash")
            if sash:
                self._sash_restored = True
                QTimer.singleShot(180, lambda p=sash: self._apply_sash(p))

    def _apply_sash(self, s):
        try:
            total = self.splitter.height()
            s = int(s)
            if 60 < s < total - 60:
                self.splitter.setSizes([s, total - s])
        except Exception:
            pass

    def _save_settings(self):
        try:
            with open(SETTINGS_FILE, "w") as fh:
                json.dump(self._collect_config(), fh, indent=2)
        except Exception:
            pass

    def _load_settings(self):
        self.v["DATA_DIR"].set(io.CONFIG.get("DATA_DIR", ""))
        self.v["SPIKE_DIR"].set(io.CONFIG.get("SPIKE_DIR", ""))
        self.v["PLOT_DIR"].set(io.CONFIG.get("PLOT_DIR", ""))
        if not os.path.exists(SETTINGS_FILE):
            return
        try:
            with open(SETTINGS_FILE) as fh:
                data = json.load(fh)
        except Exception:
            return
        self._apply_config(data)

    def _save_config_as(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save MEA Analyzer config",
            os.path.join(self.v["PLOT_DIR"].get().strip() or os.getcwd(),
                         "mea_nap_config.json"),
            "MEA Analyzer config (*.json);;All files (*.*)")
        if not path:
            return
        try:
            with open(path, "w") as fh:
                json.dump(self._collect_config(), fh, indent=2)
            self.status_var.set(f"Config saved -> {os.path.basename(path)}")
        except Exception as e:
            self._err("Save config failed", str(e))

    def _load_config_from(self):
        if self.running:
            self._info("Run in progress",
                       "Wait for the current run to finish before loading a config.")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load MEA Analyzer config",
            self.v["PLOT_DIR"].get().strip() or os.getcwd(),
            "MEA Analyzer config (*.json);;All files (*.*)")
        if not path:
            return
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception as e:
            self._err("Load config failed", f"Could not read the file:\n{e}")
            return
        if not isinstance(data, dict):
            self._err("Load config failed", "That file is not a config file.")
            return
        keep_paths = self._ask(
            "Load config",
            "Also load the folder paths (Data / Spike / Plot / Batch CSV) from this "
            "config?\n\nYes = restore everything as saved.\n"
            "No  = keep your current paths, load only the parameters.")
        self._apply_config(data, apply_paths=keep_paths)
        self._refresh_groups()
        self.status_var.set(f"Config loaded <- {os.path.basename(path)}")

    def closeEvent(self, e):
        if self.running and not self._ask(
                "Quit while running?",
                "A run is in progress. Quitting will stop it. Quit anyway?"):
            e.ignore()
            return
        try:
            self._save_settings()
        except Exception:
            pass
        e.accept()


def main():
    QtWidgets.QApplication.setApplicationName("MEA Analyzer")
    app = QtWidgets.QApplication(sys.argv)
    try:
        mea_theme.apply_qt(app, "analyzer")
    except Exception:
        pass
    w = App()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
