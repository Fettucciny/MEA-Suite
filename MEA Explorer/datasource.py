"""Data adapter over the tested MEA-NAP loaders.

This is the ONLY module that touches mea_nap_io / mea_nap_digital_events /
the meanap pipeline. It exposes plain numpy to the rest of the viewer, so the
UI never couples to the loader internals and the loaders can evolve
independently.

Reused, not reimplemented:
  io.load_spike_file(stem)      -> the raster data (per-channel spike seconds)
  io._file_duration(stem)       -> true recording length (cheap; shape only)
  io.spike_paths(stem)          -> locate detected spike files
  io.MCSFile(path)              -> raw voltage (lazy; whole file into RAM)
  io._minmax_decimate(y, n)     -> waveform envelope decimation
  de.read_digital_events(path)  -> laser / digital trigger times (seconds)
  det.bandpass(x, fs, p)        -> the detector's own Butterworth bandpass
  meanap.pipeline.burst_detection -> Bakkum ISI_N single-channel bursts
"""
import os
import glob
import json
import threading
import numpy as np

import config

config.ensure_loaders_on_path()
import mea_nap_io as io                     # noqa: E402
import mea_nap_digital_events as de         # noqa: E402
import mea_nap_original_plots as orig       # noqa: E402  (mcs_coord geometry)
import mea_nap_wavelet_detect as det        # noqa: E402  (bandpass + DETECT_PARAMS)


def electrode_xy(label):
    """(x, y) grid position for an MCS label, or None for non-grid labels.
    Reuses orig.mcs_coord -> (col, 9-row), so row 1 is at the top."""
    return orig.mcs_coord(str(label))


def grid_labels():
    """All 60 MCS 8x8 electrode labels (corners removed)."""
    return orig._mcs60_labels()


def _apply_dir_overrides():
    if config.DATA_DIR:
        io.CONFIG["DATA_DIR"] = config.DATA_DIR
    if config.SPIKE_DIR:
        io.CONFIG["SPIKE_DIR"] = config.SPIKE_DIR


def set_spike_dir(path):
    """Point the loaders at a spike-data folder chosen at runtime (the GUI
    folder picker). Persists across later list_recordings/Recording calls."""
    config.SPIKE_DIR = path                       # keep override consistent
    io.CONFIG["SPIKE_DIR"] = path


def get_spike_dir():
    """Current spike-data folder the viewer will scan."""
    _apply_dir_overrides()
    return io.CONFIG.get("SPIKE_DIR", "")


def set_data_dir(path):
    """Point the loaders at the folder with the .h5 files (triggers/waveforms)."""
    config.DATA_DIR = path
    io.CONFIG["DATA_DIR"] = path


def list_recordings():
    """Stems that have a detected spike file in SPIKE_DIR (sorted)."""
    _apply_dir_overrides()
    spike_dir = io.CONFIG["SPIKE_DIR"]
    stems = []
    for p in sorted(glob.glob(os.path.join(spike_dir, "*_spikes.npz"))):
        name = os.path.basename(p)
        stems.append(name[:-len("_spikes.npz")])
    return stems


def get_data_dir():
    return io.CONFIG.get("DATA_DIR", "")


def _h5_candidates(stem):
    """Where the raw <stem>.h5 may live, most specific first.

    io.CONFIG["DATA_DIR"] is a hard-coded default that goes stale as soon as a
    new dataset is opened, so after it we try the pipeline's own layout,
    <root>/H5 beside <root>/Output/SpikeData, derived from the spike folder
    the user actually picked.
    """
    name = stem + ".h5"
    dirs = [config.DATA_DIR, io.CONFIG.get("DATA_DIR", "")]
    sd = io.CONFIG.get("SPIKE_DIR", "")
    if sd:
        out_dir = os.path.dirname(os.path.normpath(sd))       # <root>/Output
        root = os.path.dirname(out_dir)                         # <root>
        dirs += [os.path.join(root, "H5"), os.path.join(out_dir, "H5"),
                 sd, out_dir, root]
    seen, out = set(), []
    for d in dirs:
        if not d:
            continue
        key = os.path.normcase(os.path.abspath(d))
        if key not in seen:
            seen.add(key)
            out.append(os.path.join(d, name))
    return out


def resolve_h5(stem):
    """Path of the raw recording for `stem`, or None if it can't be found."""
    for p in _h5_candidates(stem):
        if os.path.isfile(p):
            return p
    return None


def detection_params(stem):
    """The detector settings this recording's spikes were found with.

    save_spike_file() stores them as JSON in the .npz; falls back to the
    detector's current defaults for older files.
    """
    p = dict(getattr(det, "DETECT_PARAMS", {}) or {})
    try:
        npz_path, _ = io.spike_paths(stem)
        with np.load(npz_path, allow_pickle=True) as z:
            if "params" in z.files:
                saved = json.loads(str(z["params"]))
                if isinstance(saved, dict):
                    p.update(saved)
    except Exception:
        pass
    return p


def detect_bursts(spike_times, fs, min_spikes=5, isi_threshold="automatic",
                  duration_s=0.0):
    """Single-electrode burst detection with the pipeline's own code.

    Bakkum's ISI_N method from meanap.pipeline.burst_detection. The threshold
    choice mirrors single_channel_burst_detection() exactly (automatic only
    when there are >10 distinct ISIs, else 0.1 s), and the summary metrics
    come straight from single_channel_burst_detection() so their definitions
    can't drift from what Stage 2 reports.
    """
    from meanap.pipeline import burst_detection as bd

    t = np.sort(np.asarray(spike_times, dtype=np.float64))
    n = max(2, int(min_spikes))
    auto = str(isi_threshold).lower() == "automatic"

    if t.size >= n:
        if auto:
            th = (bd.get_isin_threshold(t, n=n)
                  if len(np.unique(np.diff(t))) > 10 else 0.1)
        else:
            th = float(isi_threshold)
        info, sbn = bd.burst_detect_isin(t, n, th)
    else:
        th = (float("nan") if auto else float(isi_threshold))
        info, sbn = {"T_start": [], "T_end": [], "S": []}, np.full(t.size, -1)

    ts = np.asarray(info["T_start"], dtype=np.float64)
    te = np.asarray(info["T_end"], dtype=np.float64)
    size = np.asarray(info["S"], dtype=np.int64)
    dur = te - ts
    with np.errstate(divide="ignore", invalid="ignore"):
        fr = np.where(dur > 0, size / dur, np.nan)

    res = {
        "n_spikes": int(t.size), "min_spikes": n, "auto": auto,
        "isin_th": float(th), "spike_times": t,
        "spike_burst": np.asarray(sbn, dtype=np.int64),
        "t_start": ts, "t_end": te, "size": size,
        "duration_ms": dur * 1000.0, "in_burst_fr": fr,
        "burst_rate_per_min": 0.0, "mean_in_burst_fr_hz": float("nan"),
        "mean_burst_dur_ms": float("nan"), "mean_isi_within_ms": float("nan"),
        "mean_isi_outside_ms": float("nan"), "frac_in_bursts": 0.0,
        "mean_spikes_per_burst": float(size.mean()) if size.size else float("nan"),
    }
    if ts.size:
        m = bd.single_channel_burst_detection(
            {0: t}, 1, fs, min_spikes=n,
            isi_threshold=("automatic" if auto else th),
            recording_duration_s=float(duration_s or 0.0))
        if len(m["bursting_units"]):
            res.update({
                "burst_rate_per_min": float(m["all_burstRates"][0]),
                "mean_in_burst_fr_hz": float(m["all_inBurstFRs"][0]),
                "mean_burst_dur_ms": float(m["all_burstDurs"][0]),
                "mean_isi_within_ms": float(m["all_ISIs_within"][0]),
                "mean_isi_outside_ms": float(m["all_ISIs_outside"][0]),
                "frac_in_bursts": float(m["all_fracsInBursts"][0]),
            })
    return res


class Recording:
    """One recording's raster state: sorted per-channel spike times + metadata.

    All spike arrays are sorted once at construction so every window query is an
    O(log n) `np.searchsorted` slice rather than an O(n) mask -> redraw cost
    scales with what's visible, not the whole recording.
    """

    def __init__(self, stem, apply_qc=True, drop_reference=True):
        _apply_dir_overrides()
        sp = io.load_spike_file(stem)
        if sp is None:
            raise FileNotFoundError(
                "No detected spikes for '{}'.\n"
                "Run Stage-1 detection for this recording first "
                "(load_spike_file returned None).".format(stem)
            )

        self.stem = stem
        self.fs = float(sp["fs"])
        self.noise_uv = sp.get("noise_uv")

        drop = set()
        if drop_reference:
            drop |= set(map(str, io.CONFIG.get("REFERENCE_ELECTRODE", []) or []))
        self.auto_ground = list(sp.get("auto_ground") or [])
        if apply_qc:
            drop |= set(self.auto_ground)

        labels, times = [], []
        for lab, s in zip(sp["labels"], sp["spike_times"]):
            if str(lab) in drop:
                continue
            arr = np.asarray(s, dtype=np.float64)
            arr.sort()
            labels.append(str(lab))
            times.append(arr)

        self.labels = labels
        self.spikes = times                      # list[np.ndarray], sorted
        self.n_channels = len(labels)
        self._label_index = {lab: i for i, lab in enumerate(labels)}
        self._det_params = None                  # lazy: detection_params(stem)

        # Find the raw file first: io._file_duration (and the trigger reader)
        # look in io.CONFIG["DATA_DIR"], which is stale for any dataset other
        # than the hard-coded one.
        self.h5_path = resolve_h5(stem)
        if self.h5_path:
            io.CONFIG["DATA_DIR"] = os.path.dirname(self.h5_path)

        # True duration from the h5 (cheap). Fall back to last spike if the raw
        # file is not reachable (viewer can run off spike files alone).
        self.duration = self._read_duration()

        self._events = None                      # cached trigger dict
        self._mcs = None                         # lazy MCSFile
        self._mcs_lock = threading.Lock()        # explorer loads it off-thread

    def _read_duration(self):
        try:
            return float(io._file_duration(self.stem))
        except Exception:
            return float(max((a[-1] for a in self.spikes if a.size), default=1.0))

    def set_raw_path(self, path):
        """Use an .h5 the user located by hand; refreshes duration/triggers."""
        set_data_dir(os.path.dirname(path))
        self.h5_path = path
        self._events = None
        self.free_raw()
        self.duration = self._read_duration()

    def raw_file_size(self):
        try:
            return os.path.getsize(self.h5_path) if self.h5_path else 0
        except OSError:
            return 0

    def spike_times_of(self, label):
        j = self._label_index.get(str(label))
        return self.spikes[j] if j is not None else np.empty(0)

    def filter_band(self):
        """(low, high) Hz of the detector's bandpass for this recording."""
        if self._det_params is None:
            self._det_params = detection_params(self.stem)
        p = self._det_params
        lo = float(p.get("bp_low_hz", 600.0))
        hi = float(p.get("bp_high_hz", 8000.0))
        nyq = self.fs / 2.0
        hi = min(hi, 0.95 * nyq)                 # butter() needs hi < Nyquist
        lo = min(lo, 0.5 * hi)
        return lo, hi

    # -- raster queries -----------------------------------------------------
    def _channel_iter(self, idx):
        return range(self.n_channels) if idx is None else idx

    def visible_count(self, t0, t1, idx=None):
        """Total spikes within [t0, t1) across the selected channels
        (idx = list of channel indices, or None for all)."""
        c = 0
        for j in self._channel_iter(idx):
            s = self.spikes[j]
            lo = np.searchsorted(s, t0, "left")
            hi = np.searchsorted(s, t1, "left")
            c += hi - lo
        return int(c)

    def window_counts(self, t0, t1, n_bins, idx=None):
        """(n_selected, n_bins) float32 spike counts over [t0, t1]. Overview.
        Rows follow the order of `idx` (or channel order if idx is None)."""
        n_bins = max(1, int(n_bins))
        sel = list(self._channel_iter(idx))
        edges = np.linspace(t0, t1, n_bins + 1)
        M = np.zeros((len(sel), n_bins), dtype=np.float32)
        for r, j in enumerate(sel):
            s = self.spikes[j]
            lo = np.searchsorted(s, t0, "left")
            hi = np.searchsorted(s, t1, "right")
            if hi > lo:
                M[r] = np.histogram(s[lo:hi], bins=edges)[0]
        return M

    def window_spikes(self, t0, t1, idx=None):
        """list of spike-time arrays within [t0, t1), one per selected channel
        (in `idx` order, or channel order if idx is None). Detail/export."""
        out = []
        for j in self._channel_iter(idx):
            s = self.spikes[j]
            lo = np.searchsorted(s, t0, "left")
            hi = np.searchsorted(s, t1, "left")
            out.append(s[lo:hi])
        return out

    # -- triggers -----------------------------------------------------------
    def trigger_on_times(self):
        """ON times (seconds) of the laser/digital trigger, or empty array."""
        if self._events is None:
            self._events = {}
            path = self.h5_path or os.path.join(io.CONFIG["DATA_DIR"],
                                                self.stem + ".h5")
            try:
                ev = de.read_digital_events(path)
                if ev:
                    self._events = ev
            except Exception:
                self._events = {}
        on = self._events.get("on_s")
        return np.asarray(on, dtype=np.float64) if on is not None else np.empty(0)

    # -- raw voltage (lazy; heavy) -----------------------------------------
    def has_label(self, label):
        return str(label) in self._label_index

    def _ensure_mcs(self):
        with self._mcs_lock:                     # never load the file twice
            return self._ensure_mcs_locked()

    def _ensure_mcs_locked(self):
        if self._mcs is None:
            path = self.h5_path or resolve_h5(self.stem)
            if not path or not os.path.exists(path):
                raise FileNotFoundError(
                    "Raw file '{}.h5' not found. Looked in:\n  {}".format(
                        self.stem, "\n  ".join(
                            os.path.dirname(p) for p in _h5_candidates(self.stem))))
            self.h5_path = path
            self._mcs = io.MCSFile(path)         # whole file -> RAM
        return self._mcs

    def channel_waveform(self, label, t0, t1, max_points=4000):
        """(t, uv) decimated raw trace for `label` over [t0, t1].

        Constructs (and caches) the MCSFile on first use -- this loads the whole
        recording into RAM, so it is gated behind an explicit user action.
        """
        mcs = self._ensure_mcs()
        try:
            i = mcs.labels.index(str(label))
        except ValueError:
            raise KeyError("Label {} not in raw file".format(label))
        s0 = int(round(t0 * mcs.fs))
        s1 = int(round(t1 * mcs.fs))
        uv = mcs.get_channel_window(i, s0, s1)
        if uv.size == 0:
            return np.empty(0), np.empty(0)
        idx = io._minmax_decimate(uv, max_points)
        t = t0 + idx.astype(np.float64) / mcs.fs
        return t, uv[idx]

    TRACE_PAD_S = 0.05      # filter edge padding (s), trimmed off afterwards

    def filtered_window(self, label, t0, t1, max_points=4000):
        """Bandpassed trace of `label` over [t0, t1], ready to plot.

        Uses the detector's own filter (det.bandpass) with the corner
        frequencies saved alongside this recording's spikes, so the trace is
        what the spike detector actually looked at. The window is padded by
        TRACE_PAD_S on each side before filtfilt and trimmed afterwards, so
        there are no edge transients at the borders of the view.

        Returns a dict:
          t, y      decimated (min/max envelope) arrays for plotting
          full      the full-resolution filtered samples for [t0, t1]
          s0, fs    sample index of full[0] and the sampling rate
          band      (low, high) Hz actually used
        """
        mcs = self._ensure_mcs()
        try:
            i = mcs.labels.index(str(label))
        except ValueError:
            raise KeyError("Label {} not in raw file".format(label))
        fs = float(mcs.fs)
        s0 = max(0, int(np.floor(t0 * fs)))
        s1 = min(int(mcs.n_samples), int(np.ceil(t1 * fs)))
        band = self.filter_band()
        empty = {"t": np.empty(0), "y": np.empty(0), "full": np.empty(0),
                 "s0": s0, "fs": fs, "band": band}
        if s1 - s0 < 2:
            return empty
        pad = int(round(self.TRACE_PAD_S * fs))
        a, b = max(0, s0 - pad), min(int(mcs.n_samples), s1 + pad)
        raw = np.asarray(mcs.get_channel_window(i, a, b), dtype=np.float64)
        if raw.size < 64:                       # too short for filtfilt's padding
            return empty
        p = dict(self._det_params or detection_params(self.stem))
        p["bp_low_hz"], p["bp_high_hz"] = band
        filt = np.asarray(det.bandpass(raw, fs, p), dtype=np.float64)
        full = filt[s0 - a: s0 - a + (s1 - s0)]
        idx = io._minmax_decimate(full, max_points)
        return {"t": (s0 + idx.astype(np.float64)) / fs, "y": full[idx],
                "full": full, "s0": s0, "fs": fs, "band": band}

    def raw_loaded(self):
        return self._mcs is not None

    def free_raw(self):
        with self._mcs_lock:
            self._free_raw_locked()

    def _free_raw_locked(self):
        if self._mcs is not None:
            try:
                self._mcs.free()
            except Exception:
                pass
            self._mcs = None
