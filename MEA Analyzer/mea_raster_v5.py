#!/usr/bin/env python3
"""
mea_raster_v2.py  –  MEA60 Network-Burst-Aware Spike Raster
═══════════════════════════════════════════════════════════════════════════════
Reads MCS HDF5 files (exported from Multi Channel DataManager) and generates
raster plots ordered along the diagonal from bottom-right → top-left to
reveal directional propagation patterns.

Electrode layout (MCS MEA60, label = col*10 + row, 1-indexed):
  col 1 : 12 13 14 [15=REF] 16 17          (rows 2-7, corners 11/18 absent)
  col 2-7: full columns 21-78              (rows 1-8)
  col 8 : 82 83 84 85 86 87               (rows 2-7, corners 81/88 absent)

Y-axis ordering on raster: bottom-right (y=0, plot bottom) → top-left (top).
  Primary sort: col+row DESCENDING   (large = bottom-right)
  Secondary: col ASCENDING           (breaks ties within same diagonal)

Burst detection:
  • Per-electrode: ISI threshold method (Chiappalone et al. 2005,
    J Neurosci Methods 142:89-101).
  • Network burst: smoothed population firing rate level (PFRL), adapted for
    high-activity cultures where the majority of spikes occur in network bursts.

Output: <stem>_raster.png saved next to each .h5 file, one per recording.

Usage:
    python mea_raster_v2.py ROOT_FOLDER
    python mea_raster_v2.py          # uses ROOT_DEFAULT below
"""

import sys
import re
from pathlib import Path
from typing import Optional

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# scipy is required only for raw-trace spike detection (USE_RAW_SPIKE_DETECTION).
# The pre-detected-timestamp path has no scipy dependency.
try:
    from scipy.signal import butter, sosfiltfilt
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


# ═══════════════════════════════ CONFIGURATION ══════════════════════════════

ROOT_DEFAULT = Path(r"C:\Users\Yikan\Nextcloud2\MEA Data\New MEA 06172026\H5")
SUBFOLDERS   = ["D1", "D2", "D3", "D4"]
#"D1", "D2", "D3", "D4"

# ─── MEA layout ──────────────────────────────────────────────────────────────
REFERENCE_ELECTRODE = 15            # col=1, row=5 — excluded from analysis
CORNER_ELECTRODES   = {11, 18, 81, 88}  # absent grid corners

# ─── Per-electrode ISI burst detection ───────────────────────────────────────
ISI_THRESHOLD_MS    = 100   # max intra-burst ISI (µs·1000 internally)
MIN_SPIKES_BURST    = 3     # discard bursts with fewer spikes
MERGE_GAP_MS        = 20    # merge two bursts separated by < this gap

# ─── Network burst detection (smoothed population firing rate) ───────────────
NB_BIN_MS           = 5     # spike-count bin width
NB_SIGMA_MS         = 50    # Gaussian smoothing σ
NB_MIN_FRACTION     = 0.15  # fraction of active electrodes needed per bin
                             # (lowered for cultures with >50% network bursts)
NB_MIN_DUR_MS       = 100   # minimum event duration to call a network burst
NB_MERGE_GAP_MS     = 100   # merge network bursts closer than this

# ─── Plot ─────────────────────────────────────────────────────────────────────
SPIKE_COLOR     = "#3191FF"   # deep blue — spikes outside network bursts
NB_SPIKE_COLOR  = "#D1AB83FF"   # deep orange (one shade lighter than before)
NB_FACECOLOR    = "#9A9A9A"   # blue-gray for network burst windows
NB_ALPHA        = 0.45
BURST_FACECOLOR = "#F2DEA2"   # amber per-electrode burst spans
BURST_ALPHA     = 0.22
FIG_HEIGHT_IN   = 2.0        # single-panel figure height
MIN_WIDTH_IN    = 18.0
MAX_WIDTH_IN    = 42.0
DPI_SHORT       = 300         # recordings < LONG_THRESHOLD_S
DPI_LONG        = 1200         # long recordings (keeps file size manageable)
LINEWIDTH       = 0.1
LINELENGTH      = 0.6
FIG_BG_COLOR    = "white"     # overall figure + axes background; e.g. "black".
                              # Axis text/ticks/spines auto-switch to a light
                              # color on dark backgrounds. For white spikes on a
                              # black background, also set SPIKE_COLOR = "white".

# ─── Multi-panel layout (long recordings) ────────────────────────────────────
# Recordings longer than this threshold are tiled into time-window panels so
# that individual network bursts remain visually resolvable.
LONG_THRESHOLD_S = 600        # > 10 min → switch to multi-panel layout
PANEL_DURATION_S = 600        # each panel covers this many seconds
PANEL_WIDTH_IN   = 36.0       # width of every panel (12" × 1200 DPI = 14400 px)
PANEL_HEIGHT_IN  = 1.2       # height per panel; keeps total figure portable
PANEL_GAP        = 0.30       # vertical gap between stacked panels (hspace)

# ─── Panel border (axes spines) ──────────────────────────────────────────────
PANEL_BORDER_WIDTH = 0.8      # spine line width; set 0 to hide the border
PANEL_BORDER_COLOR = "auto"   # "auto" follows the theme foreground; or any color

# ─── Duration cap ─────────────────────────────────────────────────────────────
MAX_DURATION_S   = 3600.0     # recordings longer than this are clipped to 60 min

# ─── Condition background windows ─────────────────────────────────────────────
# Each entry: (start_s, end_s, facecolor, alpha, legend_label)
# Training files: light purple covering the active recording window 2:30–57:30.
# D4 files:       light green covering a shorter window 2:30–4:30.
# Both windows are clipped per-panel for multi-panel layouts.
TRAINING_WIN = (150.0, 3450.0, "#BA99B8", 0.40, 'Sweep window (2:30–57:30)')
D4_WIN       = (150.0,  270.0, "#BA99B8", 0.45, 'Stim window (2:30–4:30)')

# Filename keywords that mark a sweep/training recording and so receive the
# TRAINING_WIN background. The original code only matched 'Slow'/'Fast', so the
# other speed conditions (Medium, Ridiculous, Ludicrous, …) got NO background.
# Matching is case-insensitive substring. Edit this to your exact file naming.
SWEEP_KEYWORDS = ('Slow', 'Fast', 'Faster',
                  'Fastest', 'Ludicrous')

# ─── Legend ───────────────────────────────────────────────────────────────────
SHOW_LEGEND               = True          # master on/off for the legend
LEGEND_SHOW_SPIKE_ISO     = True          # "Spike (isolated)" entry
LEGEND_SHOW_SPIKE_NB      = True          # "Spike (network burst)" entry
LEGEND_SHOW_ELEC_BURST    = True          # "Electrode burst" entry
LEGEND_SHOW_NETWORK_BURST = True          # "Network burst (n=…)" entry
LEGEND_SHOW_CONDITION     = True          # sweep / stim window entries
LEGEND_LOC                = 'upper right'  # any matplotlib legend loc string
LEGEND_FONTSIZE           = 7
LEGEND_FRAMEALPHA         = 0.9

# Whether network-burst spikes are drawn (and legended) in NB_SPIKE_COLOR. When
# False they render in SPIKE_COLOR like the rest (this was previously hard-off
# for single-panel and hard-on for multi-panel; now it is one control for both).
SHOW_NB_SPIKES            = True

# ─── Axis formatting ──────────────────────────────────────────────────────────
XTICK_LABELSIZE        = 7      # x (time) tick labels
YTICK_LABELSIZE        = 4.5    # electrode tick labels, full list (first/only panel)
YTICK_LABELSIZE_ABBR   = 5      # electrode tick labels, abbreviated (other panels)
XTICK_DIVISIONS        = 8      # ~how many major x ticks per panel
XMINOR_PER_MAJOR       = 5      # minor x ticks between majors
AXIS_LABEL_SIZE        = 80     # x/y axis title size, single-panel
AXIS_LABEL_SIZE_MULTI  = 7      # x/y axis title size, multi-panel
DIAG_LABEL_SIZE        = 60     # right-side "diagonal (col+row)" title, single-panel
DIAG_ANNOT_SIZE        = 4      # per-diagonal "c+r=" annotations, single-panel
TITLE_SIZE             = 100    # stats title size, single-panel
SUPTITLE_SIZE          = 8      # stats title size, multi-panel
PANEL_BADGE_SIZE       = 6      # per-panel time-range badge, multi-panel

# ─── Electrode (y-axis) diagonal separator lines ─────────────────────────────
SEP_SHOW       = True           # draw the dashed diagonal-band separators
SEP_LINEWIDTH  = 0.35
SEP_LINESTYLE  = '--'           # e.g. '-', '--', ':', '-.'
SEP_ALPHA      = 0.55

# ─── Per-electrode burst-span / DPI threshold ────────────────────────────────
# NOTE: this constant was referenced (lines for DPI selection and burst-span
# skipping) but never defined in the original file — leaving it undefined raised
# a NameError at runtime. Recordings at/above this length use DPI_LONG and have
# per-electrode burst spans suppressed (too dense to be legible / too slow).
SKIP_BURST_SPANS_ABOVE_S = 600.0   # 10 min; aligns with LONG_THRESHOLD_S

# ─── Raw-trace spike detection (band-pass + threshold) ───────────────────────
# When True, spikes are (re)DETECTED from the continuous AnalogStream voltage
# traces instead of reading MCS pre-detected timestamps. This is spike
# DETECTION, not spike SORTING — no waveform clustering into single units is
# performed (see notes accompanying this script).
#
# REQUIREMENT: the H5 export must contain continuous analog data (an
# AnalogStream with ChannelData). Timestamp-only exports cannot be re-detected;
# the pipeline falls back to read_spikes() and prints a notice in that case.
USE_RAW_SPIKE_DETECTION = False

DET_BAND_HZ       = (200.0, 3000.0)  # Butterworth band-pass corners (Hz)
DET_FILTER_ORDER  = 4                # Butterworth order (applied zero-phase, SOS)
DET_THRESH_K      = 5.0              # threshold = K × noise σ
DET_NOISE_EST     = 'mad'            # 'mad' → σ = median(|x|)/0.6745  (Quiroga 2004)
                                     # 'std' → σ = np.std(x)  (literal 5×SD as asked)
DET_WINDOW_S      = 300.0           # recompute σ / threshold every N s  (5 min)
DET_POLARITY      = 'neg'            # 'neg' | 'pos' | 'both'  threshold crossing
DET_REFRACTORY_MS = 1.0             # dead time after an accepted spike
DET_PEAK_WIN_MS   = 1.0             # window to locate the extremum after a crossing


# ═══════════════════════════════ THEME ═══════════════════════════════════════
# Foreground colours (axis text, ticks, spines, grid, separators, legend frame)
# are derived from FIG_BG_COLOR so that a dark background stays legible without
# editing every plotting call. Spike/burst/window colours remain user-set above.

def _is_dark_color(c) -> bool:
    """True if c has low relative luminance (Rec. 709)."""
    r, g, b = mcolors.to_rgb(c)
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.5


# Derived foreground colours. These are (re)computed by _apply_theme() from the
# CURRENT value of FIG_BG_COLOR on every render — so editing FIG_BG_COLOR at the
# top of the file (or setting it interactively before calling main()) always
# takes effect, with no stale import-time values.
_DARK_BG    = False
FG_COLOR    = "#37474F"   # primary text / labels
GRID_MAJOR  = "#E0E0E0"
GRID_MINOR  = "#F0F0F0"
SEP_COLOR   = "#90A4AE"   # diagonal-band separators
LEGEND_EDGE = "#B0BEC5"
MUTED_COLOR = "#546E7A"   # secondary annotations


def _apply_theme() -> None:
    """
    Recompute the derived palette from the current FIG_BG_COLOR and push the
    foreground colour into matplotlib rcParams (so spines, ticks, tick labels,
    axis labels and the figure title follow the theme). Called before each
    render, making FIG_BG_COLOR edits live even in an interactive session.
    """
    global _DARK_BG, FG_COLOR, GRID_MAJOR, GRID_MINOR, SEP_COLOR, LEGEND_EDGE, MUTED_COLOR
    _DARK_BG    = _is_dark_color(FIG_BG_COLOR)
    FG_COLOR    = "#ECEFF1" if _DARK_BG else "#37474F"
    GRID_MAJOR  = "#4A4A4A" if _DARK_BG else "#E0E0E0"
    GRID_MINOR  = "#333333" if _DARK_BG else "#F0F0F0"
    SEP_COLOR   = "#607D8B" if _DARK_BG else "#90A4AE"
    LEGEND_EDGE = "#607D8B" if _DARK_BG else "#B0BEC5"
    MUTED_COLOR = "#90A4AE" if _DARK_BG else "#546E7A"
    plt.rcParams.update({
        'text.color':      FG_COLOR,
        'axes.labelcolor': FG_COLOR,
        'axes.edgecolor':  FG_COLOR,
        'xtick.color':     FG_COLOR,
        'ytick.color':     FG_COLOR,
    })


_apply_theme()   # initialise from the default FIG_BG_COLOR


# ═══════════════════════════════ MEA LAYOUT ══════════════════════════════════

def parse_label(raw) -> Optional[int]:
    """Convert any HDF5 representation of an electrode label to int."""
    if isinstance(raw, (int, np.integer)):
        return int(raw)
    if isinstance(raw, (bytes, bytearray, np.bytes_)):
        raw = raw.decode('ascii', errors='replace')
    s = str(raw).strip().strip('\x00').strip()
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def label_to_col_row(label: int) -> tuple[int, int]:
    """MCS electrode label = col*10 + row (both 1-indexed)."""
    return label // 10, label % 10


def is_valid_electrode(label: int) -> bool:
    """
    True if label corresponds to a real signal electrode:
      - not a grid corner (11, 18, 81, 88)
      - not the reference electrode (15)
      - within the 8×8 grid
      - col 1 and col 8 only have rows 2-7 (corners absent)
    """
    if label == REFERENCE_ELECTRODE or label in CORNER_ELECTRODES:
        return False
    col, row = label_to_col_row(label)
    if not (1 <= col <= 8 and 1 <= row <= 8):
        return False
    if col in (1, 8) and row in (1, 8):   # additional corner guard
        return False
    return True


def build_electrode_order(labels: list[int]) -> list[int]:
    """
    Sort valid electrode labels for Y-axis placement:
      y=0 (plot bottom)   → most bottom-right electrode (largest col+row)
      y=N-1 (plot top)    → most top-left electrode   (smallest col+row)

    Primary key:  -(col+row)  — larger diagonal sum  → lower y (bottom)
    Secondary key: col        — leftmost on a diagonal → lower index
    """
    valid = [l for l in labels if is_valid_electrode(l)]

    def key(lbl: int):
        c, r = label_to_col_row(lbl)
        return (-(c + r), -c)   # ─(col+row) desc primary; col desc secondary

    return sorted(valid, key=key)


def diagonal_label(label: int) -> str:
    """Return 'col+row=N' annotation for a given electrode."""
    c, r = label_to_col_row(label)
    return f"d={c+r}"


# ═══════════════════════════════ HDF5 READING ════════════════════════════════

def find_neural_ts_stream(h5: h5py.File):
    """
    Locate the TimeStampStream group with DataSubType='NeuralSpike'
    and its parent Recording group.

    Returns (stream_grp, recording_grp).  Both may be None on failure.
    """
    # Accept /Data/Recording_x  or  /Recording_x  layouts
    rec_grp = None
    for root in (h5.get('Data'), h5):
        if root is None:
            continue
        for key in sorted(root.keys()):
            if re.match(r'Recording_\d+', key):
                rec_grp = root[key]
                break
        if rec_grp is not None:
            break

    if rec_grp is None or 'TimeStampStream' not in rec_grp:
        return None, rec_grp

    ts_root = rec_grp['TimeStampStream']

    # Prefer a stream explicitly labelled NeuralSpike
    for key in sorted(ts_root.keys()):
        grp = ts_root[key]
        sub = grp.attrs.get('DataSubType', b'')
        if isinstance(sub, bytes):
            sub = sub.decode('ascii', errors='replace')
        if 'NeuralSpike' in sub or ('Spike' in sub and 'Segment' not in sub):
            return grp, rec_grp

    # Fall back to first stream present
    keys = sorted(ts_root.keys())
    return (ts_root[keys[0]], rec_grp) if keys else (None, rec_grp)


def extract_labels_from_info(info_ds: h5py.Dataset) -> list[Optional[int]]:
    """
    Parse electrode labels from the InfoTimeStamp compound dataset.
    Handles both compound (named fields) and plain matrix layouts.
    """
    data = info_ds[:]

    # Compound type (most common in recent DataManager exports)
    if data.dtype.names:
        for field in ('Label', 'label', 'ChannelID', 'channelID'):
            if field in data.dtype.names:
                return [parse_label(v) for v in data[field]]

    # Plain 2D integer/string matrix – Label is column index 2
    if data.ndim == 2 and data.shape[1] >= 3:
        return [parse_label(v) for v in data[:, 2]]

    # Last resort: use sequential indices (caller will try to map to real labels)
    return list(range(len(data)))


def read_spikes(h5_path: Path) -> tuple[dict, int]:
    """
    Read spike timestamps from a DataManager-exported MCS HDF5 file.

    Returns
    -------
    spike_data : dict  {electrode_label (int): timestamps_µs (np.ndarray int64)}
                 Timestamps are relative to recording start (t=0).
    duration_us : int   Recording duration in µs.
    """
    spike_data: dict[int, np.ndarray] = {}

    with h5py.File(h5_path, 'r') as h5:
        stream, rec = find_neural_ts_stream(h5)
        if stream is None:
            return {}, 0

        # Attempt to read duration and start time from recording attributes
        dur_us   = int(rec.attrs.get('Duration',  0)) if rec is not None else 0
        start_us = int(rec.attrs.get('TimeStamp', 0)) if rec is not None else 0

        if 'InfoTimeStamp' not in stream:
            print('    WARNING: InfoTimeStamp dataset missing in stream.')
            return {}, dur_us

        labels = extract_labels_from_info(stream['InfoTimeStamp'])

        for i, label in enumerate(labels):
            if label is None:
                continue
            entity_key = f'TimeStampEntity_{i}'
            if entity_key not in stream:
                continue
            ts = np.asarray(stream[entity_key], dtype=np.int64).ravel()  # DataManager exports (n,1); flatten to (n,)
            if ts.size == 0:
                continue
            spike_data[label] = ts

    if not spike_data:
        return {}, 0

    # Make timestamps relative to recording start
    if start_us == 0:
        # Attribute absent or zero – infer start from earliest spike
        all_min = min(ts.min() for ts in spike_data.values())
        start_us = int(all_min)

    spike_data = {lbl: ts - start_us for lbl, ts in spike_data.items()}

    # Determine duration
    if dur_us == 0:
        all_max = max(ts.max() for ts in spike_data.values() if ts.size > 0)
        dur_us  = int(all_max) + 1_000_000   # +1 s buffer

    return spike_data, dur_us


# ═══════════════════ RAW-TRACE SPIKE DETECTION (band-pass + threshold) ════════
# This block adds optional spike DETECTION from the continuous voltage signal.
# Pipeline per electrode:
#   1. Band-pass (DET_BAND_HZ) with a zero-phase Butterworth filter (preserves
#      spike timing/shape; no phase distortion).
#   2. In each DET_WINDOW_S epoch, estimate the background noise σ and set
#      threshold = DET_THRESH_K · σ. Recomputing σ per epoch tracks slow drift
#      in noise floor across the recording.
#   3. Detect threshold crossings (negative-going by default for extracellular
#      APs), snap each to the local extremum, and enforce a refractory period.
# Reference for the robust (MAD) noise estimate and K·σ thresholding:
#   Quiroga, Nadasdy & Ben-Shaul (2004) Neural Computation 16(8):1661-1687,
#   σ = median(|x|)/0.6745. The 'std' option uses the literal sample SD instead.

def _butter_sos(fs: float):
    """Second-order-sections Butterworth band-pass for the given sample rate."""
    lo, hi = DET_BAND_HZ
    nyq = 0.5 * fs
    hi = min(hi, 0.45 * fs)            # keep below Nyquist with margin
    if lo >= hi:
        raise ValueError(f'Invalid band {DET_BAND_HZ} for fs={fs:.0f} Hz')
    return butter(DET_FILTER_ORDER, [lo / nyq, hi / nyq],
                  btype='band', output='sos')


def _noise_sigma(seg: np.ndarray) -> float:
    """Background-noise σ for a filtered segment."""
    if seg.size == 0:
        return 0.0
    if DET_NOISE_EST == 'std':
        return float(np.std(seg))
    # robust MAD-based estimate (Quiroga et al. 2004)
    return float(np.median(np.abs(seg)) / 0.6745)


def _detect_spikes_trace(v: np.ndarray, fs: float) -> np.ndarray:
    """
    Detect spike sample indices on one filtered-then-thresholded trace.

    Returns sorted sample indices (int64) of detected spike peaks.
    """
    sos = _butter_sos(fs)
    x   = sosfiltfilt(sos, v)                       # zero-phase band-pass

    n      = x.size
    win    = max(1, int(DET_WINDOW_S      * fs))
    refr   = max(1, int(DET_REFRACTORY_MS * 1e-3 * fs))
    pkw    = max(1, int(DET_PEAK_WIN_MS   * 1e-3 * fs))

    peaks: list[int] = []
    last  = -refr - 1                               # last accepted peak index

    for w0 in range(0, n, win):
        w1  = min(n, w0 + win)
        seg = x[w0:w1]
        sigma = _noise_sigma(seg)
        if sigma <= 0:
            continue
        thr = DET_THRESH_K * sigma

        if DET_POLARITY == 'neg':
            over = seg < -thr
        elif DET_POLARITY == 'pos':
            over = seg > thr
        else:                                       # 'both'
            over = np.abs(seg) > thr
        if not over.any():
            continue

        # rising edges of the over-threshold mask (start of each crossing)
        edges = np.flatnonzero(over[1:] & ~over[:-1]) + 1
        if over[0]:
            edges = np.r_[0, edges]

        for e in edges:
            g0 = w0 + int(e)
            g1 = min(n, g0 + pkw)
            local = x[g0:g1]
            if DET_POLARITY == 'pos':
                off = int(np.argmax(local))
            elif DET_POLARITY == 'both':
                off = int(np.argmax(np.abs(local)))
            else:
                off = int(np.argmin(local))         # most-negative sample
            pk = g0 + off
            if pk - last >= refr:
                peaks.append(pk)
                last = pk

    return np.asarray(sorted(peaks), dtype=np.int64)


def _find_analog_stream(h5: h5py.File):
    """
    Locate an AnalogStream sub-group containing ChannelData + InfoChannel.

    Returns (stream_grp, recording_grp). stream_grp is None if no continuous
    analog data is present (e.g. a timestamp-only export).
    """
    rec_grp = None
    for root in (h5.get('Data'), h5):
        if root is None:
            continue
        for key in sorted(root.keys()):
            if re.match(r'Recording_\d+', key):
                rec_grp = root[key]
                break
        if rec_grp is not None:
            break

    if rec_grp is None or 'AnalogStream' not in rec_grp:
        return None, rec_grp

    aroot = rec_grp['AnalogStream']
    fallback = None
    for key in sorted(aroot.keys()):
        g = aroot[key]
        if 'ChannelData' not in g or 'InfoChannel' not in g:
            continue
        fallback = fallback or g
        sub = g.attrs.get('DataSubType', b'')
        if isinstance(sub, bytes):
            sub = sub.decode('ascii', errors='replace')
        if any(tag in sub for tag in ('Electrode', 'Raw', 'Filter')):
            return g, rec_grp                       # prefer the electrode stream
    return fallback, rec_grp


def read_spikes_from_analog(h5_path: Path):
    """
    Detect spikes directly from the raw AnalogStream voltage traces.

    Drop-in alternative to read_spikes(); same return contract:
        (spike_data {label: ts_µs int64}, duration_µs int)
    Returns None if the file has no usable AnalogStream (caller should then fall
    back to read_spikes()).

    Channels are processed one at a time to bound memory; the full filter is
    applied per trace (zero-phase), then thresholds are recomputed per 5-min
    epoch. Expect this to be substantially slower than reading pre-detected
    timestamps (it touches every raw sample).
    """
    if not _SCIPY_OK:
        raise RuntimeError(
            'Raw spike detection requires scipy (scipy.signal). '
            'Install it (`pip install scipy`) or set '
            'USE_RAW_SPIKE_DETECTION = False.')

    spike_data: dict[int, np.ndarray] = {}

    with h5py.File(h5_path, 'r') as h5:
        stream, _rec = _find_analog_stream(h5)
        if stream is None:
            return None

        cd   = stream['ChannelData']                # (n_channels, n_samples)
        info = stream['InfoChannel'][:]
        names = info.dtype.names or ()

        def col(field, default=None):
            return info[field] if field in names else None

        rowidx_col = col('RowIndex')
        label_col  = col('Label')
        if label_col is None:
            label_col = col('ChannelID')
        tick_col   = col('Tick')                    # µs per sample
        conv_col   = col('ConversionFactor')
        adz_col    = col('ADZero')
        exp_col    = col('Exponent')

        if tick_col is None:
            raise RuntimeError('InfoChannel has no Tick field; cannot derive fs.')
        tick_us = float(np.asarray(tick_col)[0])
        fs      = 1e6 / tick_us

        n_samp  = cd.shape[1]
        n_cap   = int(MAX_DURATION_S * fs)
        n_use   = min(n_samp, n_cap)

        print(f'    ↳  Raw detection: fs={fs:.0f} Hz, band={DET_BAND_HZ} Hz, '
              f'{DET_THRESH_K:g}×σ ({DET_NOISE_EST}), σ/{DET_WINDOW_S:.0f}s, '
              f'{n_use:,} samples/ch')

        n_ch = info.shape[0]
        for i in range(n_ch):
            label = parse_label(label_col[i]) if label_col is not None else None
            if label is None or not is_valid_electrode(label):
                continue
            row = int(rowidx_col[i]) if rowidx_col is not None else i

            raw = np.asarray(cd[row, :n_use], dtype=np.float64)

            # ADC → physical units (cosmetic for thresholding, but keep correct)
            adz  = float(adz_col[i])  if adz_col  is not None else 0.0
            conv = float(conv_col[i]) if conv_col is not None else 1.0
            expo = float(exp_col[i])  if exp_col  is not None else 0.0
            v = (raw - adz) * conv * (10.0 ** expo)

            ts_samp = _detect_spikes_trace(v, fs)
            if ts_samp.size:
                spike_data[label] = np.round(ts_samp * tick_us).astype(np.int64)

        dur_us = int(round(n_use * tick_us))

    return (spike_data, dur_us) if spike_data else ({}, dur_us)


# ═══════════════════════════════ BURST DETECTION ═════════════════════════════

def detect_bursts(ts_us: np.ndarray) -> list[tuple[int, int]]:
    """
    ISI-threshold burst detection (Chiappalone et al. 2005).

    Parameters
    ----------
    ts_us : spike timestamps in µs (any order).

    Returns
    -------
    List of (burst_start_µs, burst_end_µs) tuples.
    """
    if ts_us.size < MIN_SPIKES_BURST:
        return []

    ts  = np.sort(ts_us)
    isi = np.diff(ts)
    thr = ISI_THRESHOLD_MS * 1_000   # → µs

    bursts: list[list[int]] = []
    i, n = 0, len(ts)

    while i < n - 1:
        if isi[i] <= thr:
            j = i + 1
            while j < n - 1 and isi[j] <= thr:
                j += 1
            if (j - i + 1) >= MIN_SPIKES_BURST:
                bursts.append([int(ts[i]), int(ts[j])])
            i = j
        else:
            i += 1

    if not bursts:
        return []

    # Merge close bursts
    gap = MERGE_GAP_MS * 1_000
    merged = [bursts[0]]
    for s, e in bursts[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    return [(s, e) for s, e in merged]


def _gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """1D Gaussian smoothing (no scipy dependency)."""
    radius = max(1, int(4.0 * sigma + 0.5))
    k = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (k / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(x.astype(float), kernel, mode='same')


def detect_network_bursts(
    spike_data: dict[int, np.ndarray],
    duration_us: int,
    active_labels: list[int],
) -> list[tuple[int, int]]:
    """
    Smoothed population firing rate level (PFRL) network burst detector.

    For each time bin, counts how many electrodes have at least one spike,
    smooths with a Gaussian, and finds contiguous regions above a threshold
    defined as a fraction of the total active electrode count.

    Optimised for cultures where the majority of activity is network bursts:
    NB_MIN_FRACTION is intentionally low (0.15) so that partial network bursts
    involving a subset of electrodes are still captured.

    Returns
    -------
    List of (nb_start_µs, nb_end_µs) tuples (already offset to t=0).
    """
    if not active_labels or duration_us <= 0:
        return []

    bin_us   = NB_BIN_MS * 1_000
    n_bins   = int(duration_us / bin_us) + 2
    n_active = len(active_labels)

    # Build electrode-presence histogram
    elec_active = np.zeros(n_bins, dtype=np.float32)
    for lbl in active_labels:
        ts = spike_data.get(lbl)
        if ts is None or ts.size == 0:
            continue
        bins = np.clip((ts // bin_us).astype(np.int64), 0, n_bins - 1)
        elec_active[np.unique(bins)] += 1.0

    # Smooth and threshold
    sigma_bins = NB_SIGMA_MS / NB_BIN_MS
    smoothed   = _gaussian_smooth(elec_active, sigma_bins)
    threshold  = max(1.5, n_active * NB_MIN_FRACTION)

    above = smoothed >= threshold

    # Extract contiguous "above-threshold" segments
    raw: list[list[int]] = []
    in_nb = False
    nb_s  = 0
    for idx in range(n_bins):
        if above[idx] and not in_nb:
            nb_s = idx
            in_nb = True
        elif not above[idx] and in_nb:
            raw.append([nb_s * bin_us, idx * bin_us])
            in_nb = False
    if in_nb:
        raw.append([nb_s * bin_us, n_bins * bin_us])

    # Filter by minimum duration
    min_dur = NB_MIN_DUR_MS * 1_000
    raw     = [r for r in raw if r[1] - r[0] >= min_dur]

    if not raw:
        return []

    # Merge close network bursts
    gap    = NB_MERGE_GAP_MS * 1_000
    merged = [raw[0]]
    for s, e in raw[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    return [(s, e) for s, e in merged]


# ═══════════════════════════════ PLOTTING ════════════════════════════════════

def _nice_tick_step(approx_s: float) -> float:
    """Return a 'round' time tick spacing close to approx_s seconds."""
    candidates = [0.5, 1, 2, 5, 10, 15, 20, 30, 60, 120, 300, 600, 1800, 3600]
    return min(candidates, key=lambda x: abs(x - approx_s))


def _in_nb_mask(ts_s: np.ndarray, network_bursts: list[tuple[int, int]]) -> np.ndarray:
    """
    Return a boolean mask marking spikes that fall inside any network burst.

    Uses a sort-then-binary-search approach: O(n_spikes · log n_bursts),
    no large intermediate matrix.  Assumes network bursts are non-overlapping
    and sorted (guaranteed by detect_network_bursts after merging).
    """
    if ts_s.size == 0 or not network_bursts:
        return np.zeros(ts_s.size, dtype=bool)

    starts = np.array([s / 1e6 for s, _ in network_bursts])
    ends   = np.array([e / 1e6 for _, e in network_bursts])

    # For each spike, find the index of the latest burst whose start ≤ spike time
    idx = np.searchsorted(starts, ts_s, side='right') - 1
    mask = np.zeros(ts_s.size, dtype=bool)
    valid = idx >= 0
    mask[valid] = ts_s[valid] < ends[idx[valid]]
    return mask


def _split_by_nb(
    ordered_labels: list[int],
    spike_data: dict[int, np.ndarray],
    t0_s: float,
    t1_s: float,
    network_bursts: list[tuple[int, int]],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Return (out_of_NB_spikes, in_NB_spikes) per electrode, clipped to [t0_s, t1_s).

    out_of_NB → blue SPIKE_COLOR
    in_NB     → orange NB_SPIKE_COLOR
    """
    cols_out, cols_in = [], []
    for lbl in ordered_labels:
        ts = spike_data.get(lbl, np.empty(0, np.int64))
        ts_s = ts / 1e6
        win  = (ts_s >= t0_s) & (ts_s < t1_s)
        ts_w = ts_s[win]
        if ts_w.size > 0:
            nb_mask = _in_nb_mask(ts_w, network_bursts)
            cols_out.append(ts_w[~nb_mask])
            cols_in.append(ts_w[nb_mask])
        else:
            cols_out.append(ts_w)
            cols_in.append(ts_w)
    return cols_out, cols_in


def _decorate_axes(
    ax,
    ordered_labels: list[int],
    n_elec: int,
    show_ytick_labels: bool,
    t0_s: float,
    t1_s: float,
) -> None:
    """Apply Y-axis electrode labels, diagonal separators, and X-axis styling."""
    y_pos = np.arange(n_elec)

    ax.set_ylim(-0.6, n_elec - 0.4)

    # Y-axis electrode labels (all labels on first panel, every-10th elsewhere)
    if show_ytick_labels:
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [str(lbl) for lbl in ordered_labels],
            fontsize=YTICK_LABELSIZE, fontfamily='monospace',
        )
    else:
        stride = max(1, n_elec // 8)
        ax.set_yticks(y_pos[::stride])
        ax.set_yticklabels(
            [str(ordered_labels[i]) for i in range(0, n_elec, stride)],
            fontsize=YTICK_LABELSIZE_ABBR, fontfamily='monospace',
        )

    # Diagonal-band separator lines
    if SEP_SHOW:
        prev_diag = None
        for y_idx, lbl in enumerate(ordered_labels):
            c, r = label_to_col_row(lbl)
            diag = c + r
            if prev_diag is not None and diag != prev_diag:
                ax.axhline(y_idx - 0.5, color=SEP_COLOR,
                           linewidth=SEP_LINEWIDTH, linestyle=SEP_LINESTYLE,
                           zorder=2, alpha=SEP_ALPHA)
            prev_diag = diag

    # X-axis
    dur_panel = t1_s - t0_s
    tick_step = _nice_tick_step(dur_panel / XTICK_DIVISIONS)
    ax.xaxis.set_major_locator(MultipleLocator(tick_step))
    ax.xaxis.set_minor_locator(AutoMinorLocator(XMINOR_PER_MAJOR))
    ax.tick_params(axis='x', which='major', labelsize=XTICK_LABELSIZE)
    ax.grid(axis='x', which='major', color=GRID_MAJOR,
            linewidth=0.45, zorder=2, alpha=0.8)
    ax.grid(axis='x', which='minor', color=GRID_MINOR,
            linewidth=0.2,  zorder=2, alpha=0.6)

    _style_spines(ax)


def _draw_nbs_and_bursts(
    ax,
    network_bursts: list[tuple[int, int]],
    electrode_bursts: dict[int, list[tuple[int, int]]],
    ordered_labels: list[int],
    t0_s: float,
    t1_s: float,
    show_elec_bursts: bool,
) -> None:
    """Draw network-burst spans and (optionally) per-electrode burst spans."""
    for nb_s, nb_e in network_bursts:
        s, e = nb_s / 1e6, nb_e / 1e6
        if e > t0_s and s < t1_s:
            ax.axvspan(max(s, t0_s), min(e, t1_s),
                       facecolor=NB_FACECOLOR, alpha=NB_ALPHA,
                       linewidth=0, zorder=1)    # above condition spans (zorder=0)

    if show_elec_bursts:
        for y_idx, lbl in enumerate(ordered_labels):
            for bs, be in electrode_bursts.get(lbl, []):
                bs_s, be_s = bs / 1e6, be / 1e6
                if be_s > t0_s and bs_s < t1_s:
                    ax.barh(
                        y_idx, min(be_s, t1_s) - max(bs_s, t0_s),
                        left=max(bs_s, t0_s),
                        height=LINELENGTH,
                        facecolor=BURST_FACECOLOR, alpha=BURST_ALPHA,
                        linewidth=0, zorder=2,   # above NB spans
                    )


def _draw_condition_spans(
    ax,
    spans: list[tuple],
    t0_s: float,
    t1_s: float,
) -> None:
    """
    Draw coloured background windows at zorder=0 (below NB spans and spikes).
    Each span is (start_s, end_s, facecolor, alpha, legend_label).
    Spans are clipped to the visible panel window [t0_s, t1_s].
    """
    for start_s, end_s, color, alpha, _label in spans:
        clipped_s = max(start_s, t0_s)
        clipped_e = min(end_s, t1_s)
        if clipped_e > clipped_s:
            ax.axvspan(clipped_s, clipped_e,
                       facecolor=color, alpha=alpha, linewidth=0, zorder=0)


def _build_stats(
    spike_data: dict[int, np.ndarray],
    ordered_labels: list[int],
    duration_us: int,
    network_bursts: list[tuple[int, int]],
    folder_tag: str,
    h5_path: Path,
) -> str:
    """Return a multi-line title string with recording statistics."""
    n_elec   = len(ordered_labels)
    n_spikes = sum(v.size for v in spike_data.values())
    n_active = sum(1 for lbl in ordered_labels
                   if spike_data.get(lbl, np.empty(0)).size > 0)
    n_nb      = len(network_bursts)
    dur_s     = duration_us / 1e6
    nb_frac   = (sum(e - s for s, e in network_bursts) / duration_us * 100
                 if network_bursts and duration_us > 0 else 0.0)
    return (
        f"[{folder_tag}]  {h5_path.stem}\n"
        f"Duration: {dur_s:.1f} s  ·  Active: {n_active}/{n_elec}"
        f"  ·  Spikes: {n_spikes:,}"
        f"  ·  Network bursts: {n_nb}  ({nb_frac:.1f}% of recording)"
    )


# ── Shared styling helpers ───────────────────────────────────────────────────

def _style_spines(ax) -> None:
    """Apply panel-border (spine) width and colour from config."""
    color = FG_COLOR if PANEL_BORDER_COLOR == "auto" else PANEL_BORDER_COLOR
    for spine in ax.spines.values():
        if PANEL_BORDER_WIDTH <= 0:
            spine.set_visible(False)
        else:
            spine.set_visible(True)
            spine.set_linewidth(PANEL_BORDER_WIDTH)
            spine.set_edgecolor(color)


def _build_legend_handles(condition_spans, n_nb, show_elec_bursts) -> list:
    """Assemble legend handles, honouring the LEGEND_SHOW_* toggles."""
    elems: list = []
    if LEGEND_SHOW_SPIKE_ISO:
        elems.append(Line2D([0], [0], color=SPIKE_COLOR, lw=1.3,
                            label='Spike (isolated)'))
    if LEGEND_SHOW_SPIKE_NB and SHOW_NB_SPIKES:
        elems.append(Line2D([0], [0], color=NB_SPIKE_COLOR, lw=1.3,
                            label='Spike (network burst)'))
    if LEGEND_SHOW_ELEC_BURST and show_elec_bursts:
        elems.append(Patch(facecolor=BURST_FACECOLOR, alpha=0.85,
                          label='Electrode burst'))
    if LEGEND_SHOW_NETWORK_BURST:
        elems.append(Patch(facecolor=NB_FACECOLOR, alpha=0.85,
                          label=f'Network burst (n={n_nb})'))
    if LEGEND_SHOW_CONDITION:
        for _s, _e, color, _a, label in condition_spans:
            elems.append(Patch(facecolor=color, alpha=0.85, label=label))
    return elems


def _draw_legend(ax, handles) -> None:
    """Draw the legend if enabled and there is anything to show."""
    if not SHOW_LEGEND or not handles:
        return
    ax.legend(handles=handles, loc=LEGEND_LOC, fontsize=LEGEND_FONTSIZE,
              framealpha=LEGEND_FRAMEALPHA, ncol=len(handles),
              edgecolor=LEGEND_EDGE, facecolor=FIG_BG_COLOR, labelcolor=FG_COLOR)


# ── Single-panel raster (short recordings ≤ LONG_THRESHOLD_S) ────────────────

def _plot_single(
    h5_path: Path,
    spike_data: dict[int, np.ndarray],
    duration_us: int,
    ordered_labels: list[int],
    electrode_bursts: dict[int, list[tuple[int, int]]],
    network_bursts: list[tuple[int, int]],
    folder_tag: str,
    show_elec_bursts: bool,
    condition_spans: list = (),
) -> Path:
    dur_s  = duration_us / 1e6
    n_elec = len(ordered_labels)
    y_pos  = np.arange(n_elec)
    dpi    = DPI_LONG if dur_s >= SKIP_BURST_SPANS_ABOVE_S else DPI_SHORT
    width  = float(np.clip(dur_s * 0.07, MIN_WIDTH_IN, MAX_WIDTH_IN))

    fig, ax = plt.subplots(figsize=(width, FIG_HEIGHT_IN), dpi=dpi)
    fig.patch.set_facecolor(FIG_BG_COLOR)
    ax.set_facecolor(FIG_BG_COLOR)

    _draw_condition_spans(ax, condition_spans, 0.0, dur_s)
    _draw_nbs_and_bursts(ax, network_bursts, electrode_bursts,
                          ordered_labels, 0.0, dur_s, show_elec_bursts)

    cols_out, cols_in = _split_by_nb(ordered_labels, spike_data,
                                      0.0, dur_s, network_bursts)
    _kw = dict(lineoffsets=y_pos, linelengths=LINELENGTH,
               linewidths=LINEWIDTH, zorder=3, rasterized=True)
    ax.eventplot(cols_out, colors=SPIKE_COLOR,    **_kw)
    if SHOW_NB_SPIKES:
        ax.eventplot(cols_in,  colors=NB_SPIKE_COLOR, **_kw)

    _decorate_axes(ax, ordered_labels, n_elec,
                   show_ytick_labels=True, t0_s=0.0, t1_s=dur_s)
    ax.set_ylabel('Electrode  ↑ top-left — bottom-right ↓', fontsize=AXIS_LABEL_SIZE)
    ax.set_xlim(0, dur_s)
    ax.set_xlabel('Time (s)', fontsize=AXIS_LABEL_SIZE)

    # Right-axis diagonal labels
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([])
    diag_groups: dict[int, list[int]] = {}
    for y_idx, lbl in enumerate(ordered_labels):
        c, r = label_to_col_row(lbl)
        diag_groups.setdefault(c + r, []).append(y_idx)
    for diag_val, y_idxs in diag_groups.items():
        mid = np.mean(y_idxs)
        ax2.annotate(f'c+r={diag_val}',
                     xy=(1.001, mid / max(n_elec - 1, 1)),
                     xycoords=('axes fraction', 'axes fraction'),
                     fontsize=DIAG_ANNOT_SIZE, color=MUTED_COLOR, va='center',
                     ha='left', annotation_clip=False)
    ax2.set_ylabel('← diagonal (col+row)', fontsize=DIAG_LABEL_SIZE, color=MUTED_COLOR, labelpad=22)

    n_nb = len(network_bursts)
    ax.set_title(_build_stats(spike_data, ordered_labels, duration_us,
                               network_bursts, folder_tag, h5_path),
                 fontsize=TITLE_SIZE, pad=6, loc='left')

    legend_elems = _build_legend_handles(condition_spans, n_nb, show_elec_bursts)
    _draw_legend(ax, legend_elems)

    plt.tight_layout(rect=[0, 0, 0.97, 1])
    out_path = h5_path.parent / (h5_path.stem + '_raster.png')
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight',
                    facecolor=FIG_BG_COLOR, edgecolor='none')
    finally:
        plt.close(fig)
    return out_path


# ── Multi-panel raster (long recordings > LONG_THRESHOLD_S) ──────────────────

def _plot_multi(
    h5_path: Path,
    spike_data: dict[int, np.ndarray],
    duration_us: int,
    ordered_labels: list[int],
    electrode_bursts: dict[int, list[tuple[int, int]]],
    network_bursts: list[tuple[int, int]],
    folder_tag: str,
    condition_spans: list = (),
) -> Path:
    """
    Multi-panel raster for recordings longer than LONG_THRESHOLD_S.

    Each panel covers PANEL_DURATION_S seconds at PANEL_WIDTH_IN width,
    giving ~5 px/s at DPI_LONG=100 — enough to resolve individual network
    bursts (typically 0.5–3 s) as distinct features rather than solid blocks.

    Layout (60-minute example, PANEL_DURATION_S=600):
        Panel 0:  0:00 – 10:00   full electrode labels on Y
        Panel 1: 10:00 – 20:00   abbreviated labels
        ...
        Panel 5: 50:00 – 60:00   X-axis label
    """
    dur_s     = duration_us / 1e6
    n_panels  = int(np.ceil(dur_s / PANEL_DURATION_S))
    n_elec    = len(ordered_labels)
    y_pos     = np.arange(n_elec)

    fig_h = n_panels * PANEL_HEIGHT_IN + 1.6  # +1.6 for suptitle + bottom margin
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(PANEL_WIDTH_IN, fig_h),
        dpi=DPI_LONG,
        gridspec_kw={'hspace': PANEL_GAP},
    )
    fig.patch.set_facecolor(FIG_BG_COLOR)

    if n_panels == 1:
        axes = [axes]

    n_nb = len(network_bursts)

    for p_idx, ax in enumerate(axes):
        ax.set_facecolor(FIG_BG_COLOR)
        t0 = p_idx * PANEL_DURATION_S
        t1 = min(t0 + PANEL_DURATION_S, dur_s)

        _draw_condition_spans(ax, condition_spans, t0, t1)
        _draw_nbs_and_bursts(
            ax, network_bursts, electrode_bursts,
            ordered_labels, t0, t1,
            show_elec_bursts=False,   # too dense at this scale
        )

        cols_out, cols_in = _split_by_nb(ordered_labels, spike_data,
                                          t0, t1, network_bursts)
        _kw = dict(lineoffsets=y_pos, linelengths=LINELENGTH,
                   linewidths=LINEWIDTH, zorder=3, rasterized=True)
        ax.eventplot(cols_out, colors=SPIKE_COLOR,    **_kw)
        if SHOW_NB_SPIKES:
            ax.eventplot(cols_in,  colors=NB_SPIKE_COLOR, **_kw)

        _decorate_axes(ax, ordered_labels, n_elec,
                       show_ytick_labels=(p_idx == 0),
                       t0_s=t0, t1_s=t1)

        ax.set_xlim(t0, t1)
        ax.set_ylim(-0.6, n_elec - 0.4)

        # Hide X tick labels on all but the last panel
        if p_idx < n_panels - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Time (s)', fontsize=AXIS_LABEL_SIZE_MULTI)

        # Y label only on first panel
        if p_idx == 0:
            ax.set_ylabel('Electrode\n↑TL — BR↓', fontsize=AXIS_LABEL_SIZE_MULTI, labelpad=4)

        # Panel time-range badge on the right
        m0, s0 = divmod(int(t0), 60)
        m1, s1 = divmod(int(t1), 60)
        ax.text(1.001, 0.5, f'{m0}:{s0:02d}–{m1}:{s1:02d}',
                transform=ax.transAxes,
                fontsize=PANEL_BADGE_SIZE, va='center', ha='left', color=MUTED_COLOR,
                rotation=90)

    # Shared title and legend on figure level
    fig.suptitle(
        _build_stats(spike_data, ordered_labels, duration_us,
                     network_bursts, folder_tag, h5_path),
        fontsize=SUPTITLE_SIZE, x=0.01, ha='left', y=1.0,
    )

    legend_elems = _build_legend_handles(condition_spans, n_nb,
                                          show_elec_bursts=False)
    _draw_legend(axes[0], legend_elems)

    plt.tight_layout(rect=[0, 0, 0.99, 0.985])
    fig.subplots_adjust(hspace=PANEL_GAP)   # enforce gap after tight_layout

    out_path = h5_path.parent / (h5_path.stem + '_raster.png')
    try:
        fig.savefig(out_path, dpi=DPI_LONG, bbox_inches='tight',
                    facecolor=FIG_BG_COLOR, edgecolor='none')
    finally:
        plt.close(fig)
    return out_path


# ── Public dispatcher ─────────────────────────────────────────────────────────

def plot_raster(
    h5_path: Path,
    spike_data: dict[int, np.ndarray],
    duration_us: int,
    ordered_labels: list[int],
    electrode_bursts: dict[int, list[tuple[int, int]]],
    network_bursts: list[tuple[int, int]],
    folder_tag: str,
    show_elec_bursts: bool = True,
) -> Path:
    """
    Dispatch to single-panel or multi-panel raster depending on recording length.

    Short (≤ LONG_THRESHOLD_S): one panel, width scales with duration.
    Long  (> LONG_THRESHOLD_S): tiled panels of PANEL_DURATION_S each,
        each rendered at PANEL_WIDTH_IN so network bursts stay resolvable.

    Condition background windows (zorder=0, behind NB spans):
      • any SWEEP_KEYWORDS token in filename → light purple between 2:30–57:30
      • folder_tag starts with 'D4'  → light green  between 2:30–4:30
      Windows are additive.
    """
    # Recompute the theme from the CURRENT FIG_BG_COLOR so edits always apply.
    _apply_theme()

    # Build list of coloured background windows for this file
    condition_spans: list = []
    if any(k.lower() in h5_path.stem.lower() for k in SWEEP_KEYWORDS):
        condition_spans.append(TRAINING_WIN)
    if folder_tag.upper().startswith('D4'):
        condition_spans.append(D4_WIN)

    if duration_us / 1e6 > LONG_THRESHOLD_S:
        return _plot_multi(h5_path, spike_data, duration_us, ordered_labels,
                           electrode_bursts, network_bursts, folder_tag,
                           condition_spans)
    return _plot_single(h5_path, spike_data, duration_us, ordered_labels,
                        electrode_bursts, network_bursts, folder_tag,
                        show_elec_bursts, condition_spans)


# ═══════════════════════════════ PIPELINE ════════════════════════════════════

def process_h5(h5_path: Path, folder_tag: str) -> None:
    """Full pipeline: read → detect → plot → save for one H5 file."""
    print(f'  {h5_path.name}')

    if USE_RAW_SPIKE_DETECTION:
        result = read_spikes_from_analog(h5_path)
        if result is None:
            print('    ↳  No AnalogStream in this file — falling back to '
                  'pre-detected timestamps.')
            spike_data, duration_us = read_spikes(h5_path)
        else:
            spike_data, duration_us = result
    else:
        spike_data, duration_us = read_spikes(h5_path)
    if not spike_data:
        print('    ✗  No spike data found — skipped.\n'
              '       (Check that "Spike time stamps" was selected during export.)')
        return

    # ── Clip to MAX_DURATION_S if the recording is longer ─────────────────
    cap_us = int(MAX_DURATION_S * 1_000_000)
    if duration_us > cap_us:
        print(f'    ↳  Recording {duration_us/1e6:.1f} s — clipping to '
              f'{MAX_DURATION_S:.0f} s ({MAX_DURATION_S/60:.0f} min).')
        duration_us = cap_us
        spike_data  = {lbl: ts[ts < cap_us]
                       for lbl, ts in spike_data.items()}

    dur_s = duration_us / 1_000_000

    # Build electrode list (valid labels only, diagonal order)
    valid_labels = [lbl for lbl in spike_data if is_valid_electrode(lbl)]
    if not valid_labels:
        print('    ✗  No recognised MEA60 electrode labels — skipped.\n'
              f'       Labels found: {sorted(spike_data.keys())[:10]}')
        return

    ordered = build_electrode_order(valid_labels)

    # Active electrodes (≥1 spike)
    active_labels = [lbl for lbl in valid_labels
                     if spike_data[lbl].size > 0]

    # ── Per-electrode burst detection ─────────────────────────────────────
    electrode_bursts: dict[int, list[tuple[int, int]]] = {}
    for lbl in active_labels:
        b = detect_bursts(spike_data[lbl])
        if b:
            electrode_bursts[lbl] = b

    # ── Network burst detection ───────────────────────────────────────────
    network_bursts = detect_network_bursts(spike_data, duration_us, active_labels)

    # Summary
    n_spikes    = sum(spike_data[lbl].size for lbl in valid_labels)
    n_eb        = sum(len(v) for v in electrode_bursts.values())
    nb_dur_frac = (sum(e - s for s, e in network_bursts) / duration_us * 100
                   if network_bursts and duration_us > 0 else 0.0)

    print(f'    Electrodes: {len(valid_labels)} valid / {len(active_labels)} active')
    print(f'    Spikes: {n_spikes:,}   Electrode bursts: {n_eb}   '
          f'Network bursts: {len(network_bursts)} ({nb_dur_frac:.1f}% of time)')
    print(f'    Duration: {dur_s:.1f} s')

    # Skip per-electrode burst spans for very long files (too slow / invisible)
    show_bursts = dur_s < SKIP_BURST_SPANS_ABOVE_S

    out = plot_raster(
        h5_path, spike_data, duration_us,
        ordered, electrode_bursts, network_bursts,
        folder_tag, show_elec_bursts=show_bursts,
    )
    print(f'    ✓  → {out.name}\n')


# ═══════════════════════════════ MAIN ════════════════════════════════════════

def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT_DEFAULT

    if not root.exists():
        print(f'Error: root folder not found:\n  {root}')
        sys.exit(1)

    # Collect all H5 files across subfolders
    h5_queue: list[tuple[Path, str]] = []
    for sub in SUBFOLDERS:
        folder = root / sub
        if not folder.is_dir():
            print(f'  Skipping (not found): {folder}')
            continue
        found = sorted(folder.glob('*.h5'))
        if not found:
            print(f'  No .h5 files in: {folder}')
        for f in found:
            h5_queue.append((f, sub))

    if not h5_queue:
        print(f'No .h5 files found under {root} in subfolders {SUBFOLDERS}.')
        sys.exit(1)

    print(f'MEA Raster Generator')
    print(f'Root  : {root}')
    print(f'Files : {len(h5_queue)} across {SUBFOLDERS}')
    print('═' * 65)

    for h5_path, folder_tag in h5_queue:
        print(f'\n[{folder_tag}]')
        try:
            process_h5(h5_path, folder_tag)
        except Exception as exc:
            print(f'    ✗  ERROR: {exc}')
            import traceback
            traceback.print_exc()

    print('═' * 65)
    print('Done.')


if __name__ == '__main__':
    main()
