#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 MEA-NAP pipeline -- STAGE 1: I/O, batch parsing, detection, windowing
================================================================================
Wraps the validated bior1.5 detector (mea_nap_wavelet_detect.py) with:
  * an MCS HDF5 reader (whole-file load into RAM),
  * a 6-column Batch.csv parser,
  * per-H5 whole-trace spike detection saved as MEA-NAP-style .mat + fast .npz,
  * per-row window slicing with clamp-and-warn and per-row electrode grounding.

Detection basis: spikes are detected ONCE per whole recording (thresholds/noise
estimated over the full trace); each Batch.csv row then just slices its window
from the cached spikes, so pre/stim/post share one detection basis.

Stage 2 (plots, network, node comparisons) imports `iter_batch_units()` and the
loaders here. Run `python mea_nap_io.py` to detect all files listed in Batch.csv.
================================================================================
"""

import os
import re
import glob
import json
import numpy as np
import h5py
from scipy.io import savemat
from tqdm import tqdm

import mea_nap_wavelet_detect as det
import mea_nap_progress as progress_gui


# ==============================================================================
# CONFIG  --  the three paths + run options
# ------------------------------------------------------------------------------
#   DATA_DIR  : flat folder containing the .h5 files AND Batch.csv
#   SPIKE_DIR : where per-H5 spike files (.mat/.npz) are written/read
#   PLOT_DIR  : where Stage 2 writes figures (created if missing)
#   BATCH_CSV_NAME : filename of the batch sheet inside DATA_DIR
#   REFERENCE_ELECTRODE : label always excluded everywhere (hardware reference)
#   RECOMPUTE_SPIKES : False -> reuse cached spike files when present
#   BATCH_HAS_HEADER : True / False / "auto" (auto = detect from row 1)
#   STREAM_INDEX : None -> auto-pick the analog stream with the most channels
# ==============================================================================
CONFIG = {
    "DATA_DIR":            r"C:\Users\Yikan\Nextcloud2\MEA Data\New MEA 06172026\H5",
    "SPIKE_DIR":           r"C:\Users\Yikan\Nextcloud2\MEA Data\New MEA 06172026\Output\SpikeData",
    "PLOT_DIR":            r"C:\Users\Yikan\Nextcloud2\MEA Data\New MEA 06172026\Output\Plots",
    "BATCH_CSV_NAME":      "Batch.csv",
    # Labels ALWAYS excluded everywhere (hardware reference). Real MCS 60-MEA
    # files label the grid-position-15 reference literally as "Ref"; older/other
    # exports may use "15". Both are dropped by default.
    "REFERENCE_ELECTRODE": ["Ref", "15"],  # list of labels always dropped
    "RECOMPUTE_SPIKES":    False,      # options: True | False (True forces re-detection)
    "BATCH_HAS_HEADER":    "auto",     # options: True | False | "auto"
    "STREAM_INDEX":        None,       # None = auto-pick largest stream, or int >= 0
    "VERBOSE":             True,       # options: True | False
    # A batch row whose usable window (after clamping to the file length) is
    # shorter than this is SKIPPED rather than analysed. Zero/near-zero windows
    # arise when a Batch.csv row starts at/after the end of a short recording
    # (e.g. a 420 s "post" row on a 300 s file) and previously produced empty
    # plots, divide-by-zero activity levels, and 0-bin STTC errors.
    "MIN_WINDOW_S":        1.0,        # min usable window (s). Range 0.5 - 5.0
    # GROUP GATING (set by the GUI group panel; None = all groups, the default).
    #   GROUPS_DETECT  : iterable of group names allowed into Stage-1 spike detection.
    #   GROUPS_ANALYZE : iterable of group names allowed into Stage-2 analysis.
    #   BY_GROUP       : master switch for the by-group comparison figures (Stage 2c
    #                    + MEA-NAP 2B/4B). Only meaningful with >=2 analysed groups;
    #                    the GUI auto-enables it only when >1 group is selected.
    "GROUPS_DETECT":       None,
    "GROUPS_ANALYZE":      None,
    "BY_GROUP":            True,
}

# ==============================================================================
# CHANNEL QUALITY  --  automatic grounding of dead / unusably noisy electrodes
# ------------------------------------------------------------------------------
# An electrode is auto-grounded (excluded from ALL downstream analysis) when it
# is not really recording, or when its noise floor is so high that thresholding
# yields little or nothing. Noise is the MAD of the BANDPASSED trace, estimated
# from a few short snippets (cheap, and drift/LFP-free).
#   flat_uv            : MAD below this -> electrode is not recording (flat/dead).
#   noisy_mad_ratio    : MAD above this x the median MAD across channels -> the
#                        electrode is unusually noisy...
#   noisy_max_rate_frac: ...and if its spike rate is also below this FRACTION of
#                        the median channel rate, its noise floor is swamping
#                        detection -> ground it. Relative (not absolute) so it
#                        works whether the culture fires at 0.5 Hz or 50 Hz. A
#                        noisy electrode that still yields a normal rate is KEPT.
#   silent_max_rate_hz : rate at/below this with a NORMAL noise floor -> nothing
#                        to analyse (a network node with no spikes is dead weight).
#   Set auto_ground=False to disable and rely only on Batch.csv column 6.
# ==============================================================================
QUALITY_PARAMS = {
    "auto_ground":         True,   # options: True | False (False = use Batch.csv col 6 only)
    "flat_uv":             0.5,    # dead-electrode MAD floor (uV). Range 0.2 - 1.0
    "noisy_mad_ratio":     3.0,    # x median MAD to call "noisy". Range 2.0 - 5.0
    "noisy_max_rate_frac": 0.2,    # ...and rate below this frac of median. Range 0.1 - 0.5
    "silent_max_rate_hz":  0.0,    # rate at/below this = dead (Hz). Range 0.0 - 0.1
    "noise_snippets":      3,      # snippets used to estimate noise. Range 2 - 6
    "noise_snippet_s":     20.0,   # snippet length (s). Range 5 - 60
}


def _log(*a):
    if CONFIG["VERBOSE"]:
        print(*a, flush=True)


# ==============================================================================
# cooperative abort — the GUI's Abort button sets this; the pipeline stages check
# it at each recording boundary and raise AbortRun to unwind cleanly.
# ==============================================================================
import threading as _threading

ABORT_EVENT = _threading.Event()


class AbortRun(Exception):
    """Raised by check_abort() when the user has requested an abort."""


def request_abort():
    ABORT_EVENT.set()


def clear_abort():
    ABORT_EVENT.clear()


def abort_requested():
    return ABORT_EVENT.is_set()


def check_abort():
    """Raise AbortRun if an abort was requested. Call at loop boundaries."""
    if ABORT_EVENT.is_set():
        raise AbortRun("aborted by user")


def _fmt_dur(seconds):
    """Human-readable duration: '4.2s', '3m 07s', '1h 12m'."""
    s = float(seconds)
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(round(s)), 60)
    if m < 60:
        return f"{m}m {sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def get_device():
    """Return 'cuda' if a CUDA torch build sees the GPU, else 'cpu'. (The
    detector currently runs on CPU/scipy; Stage 2 swaps a torch-conv1d path in.)"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ==============================================================================
# BLOCK A  --  MCS HDF5 reader (whole-file load)
# ------------------------------------------------------------------------------
# MCS layout: /Data/Recording_0/AnalogStream/Stream_k/{ChannelData,InfoChannel}
# Voltage[uV] = (raw - ADZero) * ConversionFactor * 10^Exponent * 1e6
# fs[Hz]      = 1e6 / Tick[us]
# The whole ChannelData is read in one sequential pass (fast on a Gen5 SSD,
# avoids per-channel chunk read-amplification) and kept in RAM.
# ==============================================================================
class MCSFile:
    def __init__(self, filepath, stream_index=None):
        self.filepath = filepath
        with h5py.File(filepath, "r") as f:
            rec_keys = sorted(k for k in f["Data"] if k.startswith("Recording_"))
            astream = f["Data"][rec_keys[0]]["AnalogStream"]
            streams = [(k, astream[k]["InfoChannel"][:]) for k in sorted(astream)
                       if k.startswith("Stream_")]
            if stream_index is not None:
                key = f"Stream_{stream_index}"
                sk = next(k for k, _ in streams if k == key)
            else:
                sk = max(streams, key=lambda s: len(s[1]))[0]
            info = astream[sk]["InfoChannel"][:]
            ds = astream[sk]["ChannelData"]
            self.n_samples = int(ds.shape[1])
            self._raw = ds[:]                                   # whole file -> RAM

        labels = [l.decode() if isinstance(l, (bytes, bytearray)) else str(l)
                  for l in info["Label"]]
        self.rows = np.asarray(info["RowIndex"]).astype(int)
        tick = float(np.asarray(info["Tick"])[0])
        self.fs = 1.0e6 / tick
        self.adzero = np.asarray(info["ADZero"]).astype(np.float64)
        self.scale = (np.asarray(info["ConversionFactor"]).astype(np.float64)
                      * 10.0 ** np.asarray(info["Exponent"]).astype(np.float64) * 1e6)
        self.labels = labels
        self.n_channels = len(labels)
        self.duration = self.n_samples / self.fs

    def get_channel(self, i):
        """Calibrated full trace (uV) for channel index i."""
        row = int(self.rows[i])
        return (self._raw[row].astype(np.float64) - self.adzero[i]) * self.scale[i]

    def get_channel_window(self, i, s0, s1):
        """Calibrated trace (uV) for channel i over sample range [s0, s1)."""
        row = int(self.rows[i])
        s0 = max(0, int(s0)); s1 = min(self.n_samples, int(s1))
        return (self._raw[row, s0:s1].astype(np.float64) - self.adzero[i]) * self.scale[i]

    def get_matrix(self):
        """Calibrated (n_channels, n_samples) float32 matrix, label-ordered.
        ~n_ch*n_samples*4 bytes in RAM (e.g. 60ch x 60min x 20kHz ~= 17 GB)."""
        raw = self._raw[self.rows].astype(np.float32)
        return (raw - self.adzero[:, None].astype(np.float32)) \
            * self.scale[:, None].astype(np.float32)

    def free(self):
        self._raw = None


# ==============================================================================
# BLOCK B  --  Batch.csv parser (6 columns + optional 7th)
# ------------------------------------------------------------------------------
# Columns (in order):
#   1 filename stem   -> file = stem + ".h5"  (day token in name is ignored)
#   2 age (days)      -> integer/float
#   3 group           -> string
#   4 window "s-s"    -> "start-end" in SECONDS, e.g. 0-30 or 3450-3600
#   5 region          -> pre / stim / post (free label)
#   6 grounded elecs  -> SPACE-separated labels, e.g. "26 63 67" (may be empty)
#   7 frames          -> OPTIONAL: Polygon frames in one cycle of THIS
#                        recording's pattern (e.g. 160 for the BR->TL sweep,
#                        133 for the radial ring). Overrides the global
#                        frames_per_sweep parameter for the period seed in the
#                        Phase Lock-In and SpKit sweep-latency stages. Blank or
#                        absent -> the global parameter is used.
# Header row optional (auto-detected: if column 2 of row 1 isn't numeric).
# ==============================================================================
def _parse_ground(cell):
    """Parse the grounded-electrode cell -> list of label strings. Tolerates
    brackets/commas/semicolons in case the sheet wasn't space-separated."""
    if cell is None:
        return []
    s = str(cell).strip().strip("[]").strip()
    if not s or s.lower() in ("nan", "none"):
        return []
    return [tok for tok in re.split(r"[\s,;]+", s) if tok]


def _parse_window(cell):
    m = re.match(r"^\s*([-+]?\d*\.?\d+)\s*-\s*([-+]?\d*\.?\d+)\s*$", str(cell))
    if not m:
        raise ValueError(f"bad window '{cell}' (expected 'start-end' seconds)")
    return float(m.group(1)), float(m.group(2))


def _parse_frames(cell):
    """Optional per-row frame count -> float, or None when blank / not a number."""
    if cell is None:
        return None
    s = str(cell).strip()
    if not s or s.lower() in ("nan", "none", "auto", "-"):
        return None
    try:
        v = float(s)
    except ValueError:
        _log(f"  [warn] frames column value {cell!r} is not a number -> ignored")
        return None
    return v if v > 0 else None


def parse_batch(csv_path):
    """Return list of row dicts: file, stem, age, group, win, region, ground,
    frames (None unless the optional 7th column is filled)."""
    with open(csv_path, "r", encoding="utf-8-sig") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    if not lines:
        return []

    def split(line):
        return [c.strip() for c in line.split(",")]

    # header auto-detect: is field 2 of line 0 numeric?
    has_header = CONFIG["BATCH_HAS_HEADER"]
    if has_header == "auto":
        f0 = split(lines[0])
        try:
            float(f0[1]); has_header = False
        except (ValueError, IndexError):
            has_header = True
    body = lines[1:] if has_header else lines

    rows = []
    for ln in body:
        c = split(ln)
        if len(c) < 5:
            _log(f"  [skip] malformed batch row (<5 fields): {ln}")
            continue
        stem = c[0]
        ground = _parse_ground(c[5]) if len(c) >= 6 else []
        frames = _parse_frames(c[6]) if len(c) >= 7 else None
        try:
            win = _parse_window(c[3])
        except ValueError as e:
            _log(f"  [skip] {e}")
            continue
        rows.append({
            "file": stem + ".h5", "stem": stem,
            "age": float(c[1]), "group": c[2],
            "win": win, "region": c[4],
            "ground": ground, "frames": frames,
        })
    # one pattern per recording: warn if a file's rows disagree about frames
    seen = {}
    for r in rows:
        if r["frames"] is not None:
            seen.setdefault(r["stem"], set()).add(r["frames"])
    for stem, vals in seen.items():
        if len(vals) > 1:
            _log(f"  [warn] {stem}: Batch.csv rows give different frame counts "
                 f"{sorted(vals)} -> using {max(vals):g}")
    return rows


def frames_for_rows(rows):
    """The per-recording frame count from a list of that recording's rows, or
    None when no row sets one."""
    vals = [r.get("frames") for r in rows if r.get("frames")]
    return max(vals) if vals else None


def group_summary(rows):
    """Ordered [(group, n_rows, n_files)] for the batch, one entry per distinct
    group. File order of first appearance is preserved so the GUI can list groups
    the way they appear in Batch.csv (the GUI applies its own speed ordering)."""
    seen = []
    agg = {}
    for r in rows:
        g = r["group"]
        if g not in agg:
            agg[g] = {"rows": 0, "files": set()}
            seen.append(g)
        agg[g]["rows"] += 1
        agg[g]["files"].add(r["stem"])
    return [(g, agg[g]["rows"], len(agg[g]["files"])) for g in seen]


def filter_rows_by_group(rows, which):
    """Keep only rows whose group is enabled for `which` ('detect' | 'analyze').

    Reads CONFIG['GROUPS_DETECT'] / CONFIG['GROUPS_ANALYZE']; when that value is
    None (the default) every group passes, so existing headless runs are
    unaffected. An empty iterable means 'no groups selected' -> no rows."""
    key = "GROUPS_DETECT" if which == "detect" else "GROUPS_ANALYZE"
    allowed = CONFIG.get(key)
    if allowed is None:
        return rows
    allowed = set(allowed)
    kept = [r for r in rows if r["group"] in allowed]
    dropped = len(rows) - len(kept)
    if dropped:
        _log(f"  [group filter] {which}: kept {len(kept)}/{len(rows)} row(s) "
             f"from groups {sorted(allowed)}; skipped {dropped} row(s) in "
             f"disabled group(s).")
    return kept


# ==============================================================================
# BLOCK C  --  spike-file save / load (per H5, whole trace)
# ------------------------------------------------------------------------------
# .npz  : fast Python reuse  (object array of per-channel spike-time arrays)
# .mat  : MEA-NAP-style      (spikeTimes cell of structs w/ method field;
#                             spikeDetectionResult; channels; fs)
# ==============================================================================
_METHOD_TAG = "bior1p5"


def spike_paths(stem):
    base = os.path.join(CONFIG["SPIKE_DIR"], stem)
    return base + "_spikes.npz", base + "_spikes.mat"


def estimate_channel_noise(mcs, qp=QUALITY_PARAMS):
    """Per-channel noise floor (uV) = MAD/0.6745 of the BANDPASSED trace,
    estimated from a few evenly spaced snippets. Cheap: filters seconds, not
    hours. Returns an array (n_channels,)."""
    fs = mcs.fs
    n = qp["noise_snippets"]
    win = int(min(qp["noise_snippet_s"], mcs.duration) * fs)
    win = max(win, int(0.5 * fs))
    if mcs.n_samples <= win:
        starts = [0]
    else:
        starts = np.linspace(0, mcs.n_samples - win, n).astype(int)
    noise = np.zeros(mcs.n_channels)
    for i in range(mcs.n_channels):
        vals = []
        for s0 in starts:
            x = det.bandpass(mcs.get_channel_window(i, s0, s0 + win), fs)
            if x.size:
                vals.append(np.median(np.abs(x - np.median(x))) / 0.6745)
        noise[i] = float(np.median(vals)) if vals else 0.0
    return noise


def classify_bad_channels(labels, noise_uv, rates_hz, qp=QUALITY_PARAMS):
    """Decide which electrodes to auto-ground. Returns (bad_labels, reasons)."""
    bad, reasons = [], {}
    if not qp["auto_ground"]:
        return bad, reasons
    finite = noise_uv[np.isfinite(noise_uv) & (noise_uv > 0)]
    med = float(np.median(finite)) if finite.size else 0.0
    live = rates_hz[rates_hz > 0]
    med_rate = float(np.median(live)) if live.size else 0.0
    for i, lab in enumerate(labels):
        nz, rt = float(noise_uv[i]), float(rates_hz[i])
        if nz < qp["flat_uv"]:
            bad.append(lab); reasons[lab] = f"flat/not recording (noise {nz:.2f} uV)"
        elif (med > 0 and nz > qp["noisy_mad_ratio"] * med and
              med_rate > 0 and rt < qp["noisy_max_rate_frac"] * med_rate):
            bad.append(lab)
            reasons[lab] = (f"noise {nz:.1f} uV = {nz/med:.1f}x median, and only "
                            f"{rt:.2f} Hz = {100*rt/med_rate:.0f}% of median rate")
        elif rt <= qp["silent_max_rate_hz"]:
            bad.append(lab); reasons[lab] = f"no spikes detected ({rt:.3f} Hz)"
    return bad, reasons


def _quality_hash():
    """Hash of the auto-grounding settings. Kept SEPARATE from the detector hash:
    changing QUALITY_PARAMS must not force a (multi-hour) re-detection, it only
    needs the cheap quality backfill below."""
    return hashlib.md5(
        json.dumps(QUALITY_PARAMS, sort_keys=True, default=str).encode()
    ).hexdigest()[:10]


def _noise_snippets_lazy(stem, qp=QUALITY_PARAMS):
    """Per-channel noise floor WITHOUT loading the whole recording into RAM.
    Opens the H5 and slices only the few short snippets we actually need, so this
    is seconds per file instead of a multi-GB read. Returns (labels, noise_uv)."""
    fp = os.path.join(CONFIG["DATA_DIR"], stem + ".h5")
    with h5py.File(fp, "r") as f:
        rec = sorted(k for k in f["Data"] if k.startswith("Recording_"))[0]
        astream = f["Data"][rec]["AnalogStream"]
        streams = [(k, astream[k]["InfoChannel"][:]) for k in sorted(astream)
                   if k.startswith("Stream_")]
        if CONFIG["STREAM_INDEX"] is not None:
            sk = f"Stream_{CONFIG['STREAM_INDEX']}"
        else:
            sk = max(streams, key=lambda s: len(s[1]))[0]
        info = astream[sk]["InfoChannel"][:]
        ds = astream[sk]["ChannelData"]
        n_samples = int(ds.shape[1])
        labels = [l.decode() if isinstance(l, (bytes, bytearray)) else str(l)
                  for l in info["Label"]]
        rows = np.asarray(info["RowIndex"]).astype(int)
        fs = 1.0e6 / float(np.asarray(info["Tick"])[0])
        adzero = np.asarray(info["ADZero"]).astype(np.float64)
        scale = (np.asarray(info["ConversionFactor"]).astype(np.float64)
                 * 10.0 ** np.asarray(info["Exponent"]).astype(np.float64) * 1e6)

        win = int(max(min(qp["noise_snippet_s"], n_samples / fs) * fs, 0.5 * fs))
        win = min(win, n_samples)
        starts = ([0] if n_samples <= win else
                  np.linspace(0, n_samples - win, qp["noise_snippets"]).astype(int))
        noise = np.zeros(len(labels))
        for i in range(len(labels)):
            vals = []
            for s0 in starts:
                raw = ds[int(rows[i]), int(s0):int(s0) + win]
                x = det.bandpass((raw.astype(np.float64) - adzero[i]) * scale[i], fs)
                if x.size:
                    vals.append(np.median(np.abs(x - np.median(x))) / 0.6745)
            noise[i] = float(np.median(vals)) if vals else 0.0
    return labels, noise


def ensure_quality(stem, prog, qhash):
    """Backfill auto-grounding onto an ALREADY-DETECTED file, without redoing
    detection. Spike files made before auto-grounding existed (or made with
    different QUALITY_PARAMS) carry no/stale auto_ground, and because the
    detector-params hash is unchanged they are otherwise 'cached' forever.
    Returns True if the spike file was updated."""
    if not QUALITY_PARAMS["auto_ground"]:
        return False
    sp = load_spike_file(stem)
    if sp is None:
        return False
    st = prog.get(stem, {})
    if st.get("quality_hash") == qhash and sp.get("noise_uv") is not None:
        return False                                  # already current
    try:
        labels, noise = _noise_snippets_lazy(stem)
        dur = _file_duration(stem)
        rates = np.array([len(s) / dur if dur > 0 else 0.0
                          for s in sp["spike_times"]])
        bad, reasons = classify_bad_channels(sp["labels"], noise, rates)
        params = dict(det.DETECT_PARAMS); params["fs"] = sp["fs"]
        save_spike_file(stem, sp["spike_times"], sp["labels"], sp["fs"], params,
                        auto_ground=bad, noise_uv=noise)
        st["quality_hash"] = qhash
        prog[stem] = st
        save_progress(prog)
        if bad:
            _log(f"  [quality] {stem}: auto-grounded {len(bad)}/{len(sp['labels'])}")
            for lab in bad:
                _log(f"      {lab}: {reasons[lab]}")
        else:
            _log(f"  [quality] {stem}: no electrodes auto-grounded")
        return True
    except Exception as e:
        _log(f"  [quality warning] {stem}: {e!r}")
        return False


def save_spike_file(stem, spike_times, labels, fs, params, auto_ground=None,
                    noise_uv=None):
    """Write .npz + MEA-NAP .mat ATOMICALLY (temp file + os.replace) so a crash
    mid-write never leaves a truncated file masquerading as complete."""
    os.makedirs(CONFIG["SPIKE_DIR"], exist_ok=True)
    npz_path, mat_path = spike_paths(stem)

    npz_tmp = npz_path[:-4] + ".tmp.npz"      # ends in .npz so savez won't append
    np.savez_compressed(
        npz_tmp,
        spike_times=np.array([np.asarray(s, np.float32) for s in spike_times],
                             dtype=object),
        labels=np.array(labels), fs=float(fs),
        params=json.dumps(params), complete=True, n_channels=len(labels),
        auto_ground=np.array(list(auto_ground or [])),
        noise_uv=np.asarray(noise_uv if noise_uv is not None
                            else np.full(len(labels), np.nan), float),
    )
    cells = np.empty((len(spike_times),), dtype=object)
    for i, s in enumerate(spike_times):
        cells[i] = {_METHOD_TAG: np.asarray(s, np.float64).reshape(-1, 1)}
    mat_tmp = mat_path + ".tmp"
    savemat(mat_tmp, {
        "spikeTimes": cells,
        "spikeDetectionResult": {"method": _METHOD_TAG,
                                 "params_json": json.dumps(params),
                                 "fs": float(fs)},
        "channels": np.array([int(l) if str(l).isdigit() else 0 for l in labels]),
        "fs": float(fs),
    }, do_compression=True, long_field_names=True, appendmat=False)

    os.replace(npz_tmp, npz_path)      # atomic on NTFS/ext4
    os.replace(mat_tmp, mat_path)
    return npz_path, mat_path


def spike_file_ok(stem, expected_n=None):
    """True only if the spike .npz exists, loads, is flagged complete, and has
    the expected channel count. A truncated/partial file fails this."""
    npz_path, _ = spike_paths(stem)
    if not os.path.exists(npz_path):
        return False
    try:
        z = np.load(npz_path, allow_pickle=True)
        if not bool(z["complete"]):
            return False
        if expected_n is not None and int(z["n_channels"]) != expected_n:
            return False
        return len(z["spike_times"]) > 0
    except Exception:
        return False


def load_spike_file(stem):
    npz_path, _ = spike_paths(stem)
    if not os.path.exists(npz_path):
        return None
    z = np.load(npz_path, allow_pickle=True)
    return {"spike_times": list(z["spike_times"]),
            "labels": [str(l) for l in z["labels"]],
            "fs": float(z["fs"]),
            "auto_ground": ([str(l) for l in z["auto_ground"]]
                            if "auto_ground" in z.files else []),
            "noise_uv": (np.asarray(z["noise_uv"], float)
                         if "noise_uv" in z.files else None)}


# ==============================================================================
# BLOCK D  --  resumable detection driver (once per unique H5)
# ------------------------------------------------------------------------------
# Crash/power-loss safety:
#   * Each H5's spike file is written atomically (temp + os.replace).
#   * A live manifest `_progress.json` in SPIKE_DIR records per-file status
#     (pending / in_progress / done / failed) with timestamps and spike counts,
#     rewritten atomically after every state change -> it always reflects where
#     the run is, even if the machine dies.
#   * On restart: files marked `done` with a valid spike file and matching
#     detector params are skipped; a file left `in_progress` (i.e. the process
#     died mid-detection) is redone from scratch WITHOUT counting as a failure;
#     a file that raised an error is retried ONCE, then skipped and reported
#     (MAX_ATTEMPTS = 2).
#   * Changing detector params invalidates old spike files (they are recomputed).
# ==============================================================================
import time
import hashlib

MAX_ATTEMPTS = 2                       # retries for erroring files. Range 1 - 5 (2 = retry once)
PROGRESS_NAME = "_progress.json"


def _progress_path():
    return os.path.join(CONFIG["SPIKE_DIR"], PROGRESS_NAME)


def _params_hash():
    return hashlib.md5(
        json.dumps(det.DETECT_PARAMS, sort_keys=True, default=str).encode()
    ).hexdigest()[:10]


def load_progress():
    p = _progress_path()
    if os.path.exists(p):
        try:
            with open(p) as fh:
                return json.load(fh)
        except Exception:
            return {}
    return {}


_ATOMIC_SWAP_OK = True                 # cleared if os.replace proves unusable


def save_progress(prog, _retries=6):
    """Persist the resume manifest, tolerating a locked destination.

    On Windows os.replace() raises PermissionError (WinError 5) whenever
    another process holds a handle on the destination -- a Nextcloud or
    OneDrive sync client, an antivirus scanner, or an open editor.  That is
    transient, and it is far more likely here than elsewhere because a run
    with many missing files rewrites this manifest once per file in quick
    succession, which is exactly when a sync client is mid-upload.

    Strategy: retry the atomic swap with backoff, then fall back to a direct
    (non-atomic) write, and finally warn and carry on.  The manifest is a
    resume convenience, not analysis output, so losing it must never abort a
    detection run that is otherwise fine.
    """
    global _ATOMIC_SWAP_OK
    os.makedirs(CONFIG["SPIKE_DIR"], exist_ok=True)
    dst = _progress_path()
    tmp = dst + ".tmp"

    # Once the atomic swap has provably lost the race on this machine, stop
    # paying the retry backoff on every subsequent file.
    if not _ATOMIC_SWAP_OK:
        try:
            with open(dst, "w") as fh:
                json.dump(prog, fh, indent=2)
            return True
        except OSError:
            return False

    try:
        with open(tmp, "w") as fh:
            json.dump(prog, fh, indent=2)
    except OSError as exc:
        _log(f"  [warn] could not write progress manifest: {exc}")
        return False

    delay = 0.1
    for attempt in range(_retries):
        try:
            os.replace(tmp, dst)
            return True
        except PermissionError:
            if attempt == _retries - 1:
                break
            time.sleep(delay)
            delay = min(delay * 2.0, 1.5)
        except OSError as exc:
            _log(f"  [warn] progress manifest swap failed: {exc}")
            break

    # The atomic swap kept losing the race -- write in place from now on.
    _ATOMIC_SWAP_OK = False
    try:
        with open(dst, "w") as fh:
            json.dump(prog, fh, indent=2)
        _discard(tmp)
        return True
    except OSError as exc:
        _discard(tmp)
        _log(f"  [warn] progress manifest is locked ({exc}); this run will "
             f"still complete, but resume info may be incomplete. Pausing "
             f"Nextcloud sync on the Output folder usually clears it.")
        return False


def _discard(path):
    try:
        os.remove(path)
    except OSError:
        pass


def detect_file(stem, prog, phash, device, gui=None):
    """Detect one H5 with resume logic. Mutates & persists `prog`. Returns one of
    'detected' | 'cached' | 'skipped' | 'failed' | 'missing'."""
    st = prog.get(stem, {})
    fails = st.get("fail_count", 0)

    # already complete, valid, same params -> skip
    if (not CONFIG["RECOMPUTE_SPIKES"] and st.get("status") == "done"
            and st.get("params_hash") == phash and spike_file_ok(stem)):
        return "cached"
    # erroring file that already used up its retries -> skip & report
    if (not CONFIG["RECOMPUTE_SPIKES"] and st.get("status") == "failed"
            and fails >= MAX_ATTEMPTS):
        _log(f"  [skip] {stem}: failed {fails}x previously ({st.get('error','')})")
        return "skipped"
    if st.get("status") == "done" and st.get("params_hash") != phash:
        _log(f"  [params changed] re-detecting {stem}")

    fp = os.path.join(CONFIG["DATA_DIR"], stem + ".h5")
    if not os.path.exists(fp):
        prog[stem] = {"status": "missing", "params_hash": phash}
        save_progress(prog)
        _log(f"  [skip] not on disk yet: {os.path.basename(fp)}")
        return "missing"

    # mark in_progress (a crash here leaves this marker -> redone next run)
    prog[stem] = {"status": "in_progress", "fail_count": fails,
                  "params_hash": phash, "started": time.time()}
    save_progress(prog)

    try:
        mcs = MCSFile(fp, CONFIG["STREAM_INDEX"])
        bar = tqdm(total=mcs.n_channels, desc=f"    {stem}", leave=False, unit="ch")
        nch = mcs.n_channels

        def _prog(done):
            bar.update(done - bar.n)
            if gui is not None:
                gui.set(stem, done / max(nch, 1),
                        note=f"detecting: channel {done}/{nch}")

        spikes = det.detect_spikes_matrix(
            mcs.get_matrix(), mcs.fs, device=device, progress=_prog)
        bar.close()
        params = dict(det.DETECT_PARAMS); params["fs"] = mcs.fs
        auto_ground, reasons, noise = [], {}, None
        if QUALITY_PARAMS["auto_ground"]:
            try:
                noise = estimate_channel_noise(mcs)
                rates = np.array([len(s) / mcs.duration if mcs.duration > 0 else 0.0
                                  for s in spikes])
                auto_ground, reasons = classify_bad_channels(mcs.labels, noise, rates)
                if auto_ground:
                    _log(f"  auto-grounded {len(auto_ground)}/{mcs.n_channels} "
                         f"electrode(s) on {stem}:")
                    for lab in auto_ground:
                        _log(f"      {lab}: {reasons[lab]}")
            except Exception as e:
                _log(f"  [auto-ground warning] {stem}: {e!r}")
        save_spike_file(stem, spikes, mcs.labels, mcs.fs, params,
                        auto_ground=auto_ground, noise_uv=noise)
        if QC_PARAMS["mode"] != "off":              # QC popup/PNG; never fatal
            try:
                qc_plot_file(mcs, spikes, mcs.labels, mcs.fs, stem)
            except Exception as e:
                _log(f"  [qc warning] {stem}: {e!r}")
        mcs.free()
        prog[stem] = {"status": "done", "fail_count": fails, "params_hash": phash,
                      "n_channels": len(mcs.labels),
                      "spikes": int(sum(len(s) for s in spikes)),
                      "started": prog[stem]["started"], "finished": time.time()}
        save_progress(prog)
        return "detected"
    except Exception as e:                 # caught error -> count a failure
        prog[stem] = {"status": "failed", "fail_count": fails + 1,
                      "params_hash": phash, "error": repr(e),
                      "finished": time.time()}
        save_progress(prog)
        _log(f"  [error] {stem}: {e!r}")
        return "failed"


def detect_all(rows):
    """Detect every unique H5 in the batch, resuming from the manifest."""
    stems = sorted({r["stem"] for r in rows})
    prog = load_progress()
    phash = _params_hash()
    device = det.auto_device()

    # startup report: how many already done vs remaining
    done0 = sum(1 for s in stems
                if prog.get(s, {}).get("status") == "done"
                and prog.get(s, {}).get("params_hash") == phash and spike_file_ok(s))
    _log(f"Spike detection: {len(stems)} unique file(s), {done0} already complete, "
         f"resuming {len(stems)-done0} [device={device}, params={phash}]")

    absent = [s for s in stems
              if not os.path.exists(os.path.join(CONFIG["DATA_DIR"], s + ".h5"))]
    if absent:
        _log(f"  {len(absent)} of {len(stems)} file(s) in the batch are not on "
             f"disk yet and will be skipped this run: {', '.join(absent)}")

    gui = progress_gui.make(stems, "MEA Analyzer — spike detection",
                            f"Spike detection ({device})")
    qhash = _quality_hash()
    stats = {"detected": 0, "cached": 0, "skipped": 0, "failed": 0, "missing": 0}
    try:
        for stem in tqdm(stems, desc="Spike detection", unit="file"):
            check_abort()
            gui.start(stem, "loading H5...")
            outcome = detect_file(stem, prog, phash, device, gui=gui)
            stats[outcome] += 1
            info = prog.get(stem, {})
            if outcome == "detected":
                info["quality_hash"] = qhash          # computed during detection
                prog[stem] = info; save_progress(prog)
                gui.done(stem, f"{info.get('n_channels','?')} ch, "
                               f"{info.get('spikes','?'):,} spikes")
            elif outcome == "cached":
                # Detection is current, but the auto-grounding may predate this
                # file (or QUALITY_PARAMS changed) -> cheap backfill, no re-detect.
                gui.set(stem, 0.5, note="checking channel quality...")
                updated = ensure_quality(stem, prog, qhash)
                gui.done(stem, "cached + quality backfilled" if updated else "cached")
            elif outcome == "missing":
                gui.skipped(stem, "H5 not found")
            elif outcome == "skipped":
                gui.skipped(stem, "skipped")
            else:
                gui.failed(stem, str(info.get("error", "failed"))[:60])
    finally:
        gui.close()
    _log(f"  detection summary: {stats}")
    failed = [s for s in stems if prog.get(s, {}).get("status") == "failed"]
    if failed:
        _log(f"  files needing attention: {failed}")
    return stats


# ==============================================================================
# BLOCK E  --  per-row window slicing + grounding  (feeds Stage 2)
# ==============================================================================
def _file_duration(stem):
    fp = os.path.join(CONFIG["DATA_DIR"], stem + ".h5")
    with h5py.File(fp, "r") as f:
        rec = sorted(k for k in f["Data"] if k.startswith("Recording_"))[0]
        astream = f["Data"][rec]["AnalogStream"]
        sk = (f"Stream_{CONFIG['STREAM_INDEX']}" if CONFIG["STREAM_INDEX"] is not None
              else max((k for k in astream if k.startswith("Stream_")),
                       key=lambda k: len(astream[k]["InfoChannel"])))
        info = astream[sk]["InfoChannel"][:]
        n = int(astream[sk]["ChannelData"].shape[1])
        fs = 1.0e6 / float(np.asarray(info["Tick"])[0])
    return n / fs


def window_unit(row):
    """Slice one Batch.csv row into an analysis unit.

    Returns dict: stem, age, group, region, duration, labels (kept), spikes
    (list of per-kept-channel arrays, times relative to window start), and any
    warning. Electrode 15 + the row's grounded electrodes are dropped.
    Returns None if the spike file is missing.
    """
    sp = load_spike_file(row["stem"])
    if sp is None:
        _log(f"  [warn] no spike file for {row['stem']} (run detection first)")
        return None

    dur_file = _file_duration(row["stem"])
    t0, t1 = row["win"]
    warn = None
    if t1 > dur_file + 1e-6:
        warn = (f"{row['stem']} window end {t1:.1f}s exceeds file length "
                f"{dur_file:.2f}s -> clamped")
        _log(f"  [warn] {warn}")
        t1 = dur_file
    t0 = max(0.0, t0)
    dur = max(0.0, t1 - t0)
    if dur < CONFIG["MIN_WINDOW_S"]:
        _log(f"  [skip] {row['stem']} [{row['region']}] window {row['win'][0]:.1f}-"
             f"{row['win'][1]:.1f}s leaves {dur:.2f}s inside the {dur_file:.2f}s "
             f"file (< MIN_WINDOW_S={CONFIG['MIN_WINDOW_S']}s) -> row skipped")
        return None

    ref = CONFIG["REFERENCE_ELECTRODE"]
    ref = [ref] if isinstance(ref, str) else list(ref)
    drop = set(row["ground"]) | set(ref) | set(sp.get("auto_ground") or [])
    kept_labels, kept_spikes = [], []
    for lab, st in zip(sp["labels"], sp["spike_times"]):
        if lab in drop:
            continue
        st = np.asarray(st)
        w = st[(st >= t0) & (st < t1)] - t0                    # window-relative
        kept_labels.append(lab)
        kept_spikes.append(w.astype(np.float32))

    return {"stem": row["stem"], "age": row["age"], "group": row["group"],
            "region": row["region"], "duration": dur, "win": (t0, t1),
            "labels": kept_labels, "spikes": kept_spikes,
            "n_grounded": len(drop), "warning": warn,
            "frames": row.get("frames")}


def iter_batch_units(rows):
    """Yield one windowed analysis unit per batch row (Stage 2 consumes these)."""
    for r in tqdm(rows, desc="Windowing units", unit="row"):
        u = window_unit(r)
        if u is not None:
            yield u


# ==============================================================================
# ==============================================================================
# BLOCK F  --  spike-detection QC snippets (popup per file when detection ends)
# ------------------------------------------------------------------------------
# After a file finishes detection, plot random 1-min snippets of the FILTERED
# trace with detected spikes marked and the normalized wavelet response overlaid
# (=1 at the detection threshold), so you can eyeball whether detection is sane.
#
#   mode          : "popup" (show a window + save PNG) | "save" (PNG only) | "off"
#   block         : popup mode -- if True, the window blocks and the NEXT file's
#                   detection waits until you close it (good for interactive
#                   checking). Set False, or use mode="save", for unattended
#                   overnight batches so a blocking window never stalls the run.
#   snippet_len_s : snippet length (1 min).
#   short_max_min : recordings shorter than this -> n_short snippets (middle+end).
#   long_min_min  : recordings longer than this  -> n_long snippets (edges+middle).
#   edge_min      : width (min) of the first/last band used for long recordings.
#   n_short/n_long/n_mid : snippet counts for the three duration regimes. The
#                   10-50 min regime (n_mid) was NOT specified by MEA-NAP/you;
#                   default = 4 snippets spread across the recording.
#   prefer_active : pick channels that actually spike in the window (else random).
#   seed          : RNG seed for window/channel choice (None = new each run).
#   display_bins  : min/max decimation target so 1-min traces plot fast while
#                   still preserving spike peaks.
# ==============================================================================
QC_PARAMS = {
    "mode":          "popup",   # options: "popup" | "save" | "off"
    "popup_backend": "viewer",  # options: "viewer" | "matplotlib". "viewer" = open the
                                # saved PNG in the OS image viewer (its own process ->
                                # stays responsive, never freezes or blocks detection;
                                # RECOMMENDED, esp. for long recordings). "matplotlib" =
                                # interactive figure window (zoomable) but can show
                                # "(Not Responding)" while the next big file detects.
    "block":         False,     # options: True | False (only used for backend="matplotlib")
    "snippet_len_s": 60.0,      # QC snippet length (s). Range 30 - 120
    "short_max_min": 10.0,      # recordings shorter than this use n_short. Range 5 - 20 (min)
    "long_min_min":  50.0,      # recordings longer than this use n_long. Range 30 - 90 (min)
    "edge_min":      3.0,       # first/last band width for long recordings. Range 1 - 5 (min)
    "n_short":       2,         # snippets for short recordings. Range 1 - 4
    "n_long":        8,         # snippets for long recordings. Range 4 - 12
                                #   (laid out as 2 first-edge / 4 middle / 2 last-edge)
    "n_mid":         4,         # snippets for mid-length recordings. Range 2 - 8
    "prefer_active": True,      # options: True | False (pick channels that actually spike)
    "seed":          None,      # None = new each run, or an int for reproducible picks
    "display_bins":  8000,      # min/max decimation target. Range 2000 - 20000
    "save_dir_name": "QC",      # subfolder name under PLOT_DIR (any string)
}


def _open_in_viewer(path):
    """Open an image in the OS default viewer -- a separate process that stays
    responsive and never blocks/freezes the detection loop. The viewer window is
    titled by the file, e.g. '<stem>_qc.png'."""
    import sys, subprocess
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)                              # noqa: returns at once
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        _log(f"  [qc] could not open viewer for {path}: {e!r}")


def _minmax_decimate(y, n_bins):
    """Downsample to ~2*n_bins points preserving per-bin min & max (keeps spikes
    visible). Returns indices into the original array."""
    n = len(y)
    if n <= 2 * n_bins:
        return np.arange(n)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    idx = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            seg = y[a:b]
            idx.append(a + int(np.argmin(seg)))
            idx.append(a + int(np.argmax(seg)))
    return np.array(sorted(set(idx)))


def _pick_qc_windows(dur, rng):
    """List of (band_label, t0, t1) snippet windows per the duration rules."""
    L = min(QC_PARAMS["snippet_len_s"], dur)
    mins = dur / 60.0

    def rand_win(a, b, label):
        hi = max(a, b - L)
        t0 = rng.uniform(a, hi) if hi > a else a
        return (label, t0, min(t0 + L, dur))

    if mins < QC_PARAMS["short_max_min"]:                    # <10 min: middle + end
        c = max(0.0, dur / 2 - L / 2)
        return [("middle", c, min(c + L, dur)),
                ("end", max(0.0, dur - L), dur)]
    if mins > QC_PARAMS["long_min_min"]:                     # >50 min: 2 / 4 / 2
        e = QC_PARAMS["edge_min"] * 60.0
        return ([rand_win(0, e, "first-3min") for _ in range(2)]
                + [rand_win(e, dur - e, "middle") for _ in range(4)]
                + [rand_win(dur - e, dur, "last-3min") for _ in range(2)])
    edges = np.linspace(0, dur, QC_PARAMS["n_mid"] + 1)       # 10-50 min: spread
    return [rand_win(edges[i], edges[i + 1], "spread")
            for i in range(QC_PARAMS["n_mid"])]


def _wavelet_response(x, fs):
    """Normalized wavelet response R(t) (=1 at threshold), max over scales,
    mirroring the detector's per-scale MAD threshold. For QC display only."""
    p = det.DETECT_PARAMS
    coeffs = det._cwt_coeffs(x, fs, p)          # already sign-negated by the port
    scales = det.determine_scales(fs, p)
    n = x.size
    R = np.zeros(n)
    for i, c in enumerate(coeffs):
        DTh = det.wavelet_threshold_np(c, scales[i], n, p)
        if DTh is None or DTh <= 0:
            continue
        # only POSITIVE (negated-CWT) coefficients count, matching detection:
        # a spike is where R crosses 1.0
        R = np.maximum(R, np.maximum(c, 0.0) / float(DTh))
    return R


def qc_plot_file(mcs, spikes, labels, fs, stem):
    """Build the per-file QC figure (filtered trace + spikes + wavelet/threshold
    for each random snippet); popup and/or save per QC_PARAMS."""
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(QC_PARAMS["seed"])
    wins = _pick_qc_windows(mcs.duration, rng)
    ref = CONFIG["REFERENCE_ELECTRODE"]
    ref = {ref} if isinstance(ref, str) else set(ref)
    eligible = [i for i, l in enumerate(labels) if l not in ref]

    fig, axes = plt.subplots(len(wins), 1, figsize=(13, 2.1 * len(wins)),
                             squeeze=False)
    axes = axes[:, 0]
    for ax, (band, t0, t1) in zip(axes, wins):
        s0, s1 = int(t0 * fs), int(t1 * fs)
        cand = eligible
        if QC_PARAMS["prefer_active"]:
            active = [i for i in eligible
                      if np.any((spikes[i] >= t0) & (spikes[i] < t1))]
            cand = active or eligible
        ch = int(rng.choice(cand))
        x = det.bandpass(mcs.get_channel_window(ch, s0, s1), fs)
        tt = t0 + np.arange(x.size) / fs
        R = _wavelet_response(x, fs)
        sp = spikes[ch][(spikes[ch] >= t0) & (spikes[ch] < t1)]

        di = _minmax_decimate(x, QC_PARAMS["display_bins"])
        ax.plot(tt[di], x[di], lw=0.4, color="0.25")
        ax.set_ylabel("uV", fontsize=8)
        for s in sp:
            ax.axvline(s, color="tab:red", lw=0.4, alpha=0.6)
        ax2 = ax.twinx()
        ri = _minmax_decimate(R, QC_PARAMS["display_bins"])
        ax2.plot(tt[ri], R[ri], lw=0.5, color="tab:blue", alpha=0.5)
        ax2.axhline(1.0, color="tab:blue", ls="--", lw=0.8, alpha=0.8)
        ax2.set_ylabel("wavelet / thr", fontsize=8, color="tab:blue")
        top = float(np.nanpercentile(R, 99.9)) if R.size else 3.0
        ax2.set_ylim(0, max(3.0, top))
        ax.set_title(f"ch {labels[ch]} | {t0:.0f}-{t1:.0f}s [{band}] | "
                     f"{len(sp)} spikes", fontsize=9, loc="left")
        ax.set_xlim(t0, t1)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Spike-detection QC — {stem}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    qc_dir = os.path.join(CONFIG["PLOT_DIR"], QC_PARAMS["save_dir_name"])
    os.makedirs(qc_dir, exist_ok=True)
    png = os.path.join(qc_dir, f"{stem}_qc.png")
    fig.savefig(png, dpi=110)

    if QC_PARAMS["mode"] == "popup":
        if QC_PARAMS["popup_backend"] == "viewer":
            plt.close(fig)                 # hand off to the OS viewer (responsive)
            _open_in_viewer(png)
        else:                              # interactive matplotlib window
            try:
                fig.canvas.manager.set_window_title(stem)
            except Exception:
                pass
            plt.show(block=QC_PARAMS["block"])
    else:
        plt.close(fig)


def run_stage1():
    csv_path = os.path.join(CONFIG["DATA_DIR"], CONFIG["BATCH_CSV_NAME"])
    if not os.path.exists(csv_path):
        _log(f"Batch.csv not found at {csv_path}")
        return
    rows = parse_batch(csv_path)
    _log(f"Parsed {len(rows)} batch row(s) "
         f"({len({r['stem'] for r in rows})} unique file(s)).")
    rows = filter_rows_by_group(rows, "detect")
    if not rows:
        _log("Stage 1: no rows to detect (all groups disabled for spike detection).")
        return []
    os.makedirs(CONFIG["PLOT_DIR"], exist_ok=True)
    detect_all(rows)
    _log("Stage 1 complete: spike files written to SPIKE_DIR. "
         "Stage 2 will read them for plotting/network analysis.")
    return rows


if __name__ == "__main__":
    run_stage1()
