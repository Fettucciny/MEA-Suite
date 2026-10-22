"""Data adapter over the tested MEA-NAP loaders.

This is the ONLY module that imports mea_nap_io / mea_nap_digital_events /
mea_nap_original_plots / mea_nap_wavelet_detect. It exposes plain numpy and small
dataclasses to the rest of the app, so the UI and the analysis code never couple
to the loader internals.

Reused, not reimplemented:
  io.MCSFile(path)                 -> raw calibrated voltage (uV), lazy whole-file
  io._file_duration(stem)          -> true recording length (cheap; shape only)
  io.parse_batch / io.CONFIG       -> Batch.csv rows + reference-electrode config
  io.load_spike_file(stem)         -> validated biological spikes (per channel)
  de.read_digital_events(path)     -> laser flood ON/OFF times (seconds)
  det.bandpass / det.detect_spikes -> the SAND/MEA-NAP bior1.5 detector
  orig.mcs_coord / _mcs60_labels   -> MCS 8x8 electrode geometry

The mea_nap_* modules are imported lazily (first use) so this file can be
imported for unit-testing the pure-numpy analysis code in an environment that
does not have the loaders / torch / h5py installed.
"""
import os
import glob
import numpy as np

import config

_MODS = {}


def _mods():
    """Import and cache the mea_nap loaders on first use."""
    if not _MODS:
        config.ensure_loaders_on_path()
        import mea_nap_io as io                      # noqa: E402
        import mea_nap_digital_events as de          # noqa: E402
        import mea_nap_original_plots as orig        # noqa: E402
        import mea_nap_wavelet_detect as det         # noqa: E402
        _MODS.update(io=io, de=de, orig=orig, det=det)
    return _MODS


# ---------------------------------------------------------------------------
# electrode geometry (delegates to the MEA-NAP layout so heatmaps line up)
# ---------------------------------------------------------------------------
def electrode_xy(label):
    """(x, y) grid position for an MCS label ('47' -> col 4, row 7), y flipped so
    row 1 is at the top; None for non-grid labels ('Ref')."""
    return _mods()["orig"].mcs_coord(str(label))


def grid_labels():
    """All 60 MCS 8x8 electrode labels (four corners removed)."""
    return _mods()["orig"]._mcs60_labels()


def reference_labels():
    """Labels always excluded (hardware reference)."""
    io = _mods()["io"]
    ref = io.CONFIG.get("REFERENCE_ELECTRODE", [])
    return [ref] if isinstance(ref, str) else list(ref)


# ---------------------------------------------------------------------------
# batch sheet + folder scanning
# ---------------------------------------------------------------------------
def parse_batch(csv_path):
    """List of Batch.csv row dicts (file, stem, age, group, win, region, ground)."""
    return _mods()["io"].parse_batch(csv_path)


def list_h5_stems(data_dir):
    """Sorted stems of every *.h5 in a folder."""
    stems = []
    for p in sorted(glob.glob(os.path.join(data_dir, "*.h5"))):
        stems.append(os.path.splitext(os.path.basename(p))[0])
    return stems


def file_duration(data_dir, stem):
    io = _mods()["io"]
    old = io.CONFIG.get("DATA_DIR")
    io.CONFIG["DATA_DIR"] = data_dir
    try:
        return float(io._file_duration(stem))
    finally:
        io.CONFIG["DATA_DIR"] = old


# ---------------------------------------------------------------------------
# laser flood events
# ---------------------------------------------------------------------------
def laser_events(h5_path):
    """{"on_s","off_s","intervals","source","fs","detail"} for the laser floods.
    ON times are the flood onsets (one exposure every ~5 s in these recordings)."""
    de = _mods()["de"]
    try:
        return de.read_digital_events(h5_path)
    except Exception as exc:
        return {"on_s": np.empty(0), "off_s": np.empty(0), "intervals": [],
                "source": None, "fs": None, "detail": "read failed: %r" % (exc,)}


# ---------------------------------------------------------------------------
# raw voltage (heavy; whole file into RAM)
# ---------------------------------------------------------------------------
class RawRecording:
    """Thin wrapper over io.MCSFile giving label-indexed calibrated traces (uV)."""

    def __init__(self, h5_path, stream_index=None):
        io = _mods()["io"]
        self.path = h5_path
        self.mcs = io.MCSFile(h5_path, stream_index)
        self.fs = float(self.mcs.fs)
        self.labels = [str(l) for l in self.mcs.labels]
        self.n_samples = int(self.mcs.n_samples)
        self.duration = float(self.mcs.duration)
        self._index = {lab: i for i, lab in enumerate(self.labels)}

    def has_label(self, label):
        return str(label) in self._index

    def channel(self, label):
        """Full calibrated trace (uV) for one electrode label."""
        return self.mcs.get_channel(self._index[str(label)])

    def channel_window(self, label, t0, t1):
        """Calibrated trace (uV) for a label over [t0, t1) seconds, and its t0."""
        i = self._index[str(label)]
        s0 = int(round(t0 * self.fs))
        s1 = int(round(t1 * self.fs))
        uv = self.mcs.get_channel_window(i, s0, s1)
        return uv, max(0, s0) / self.fs

    def matrix(self):
        """(n_channels, n_samples) float32 calibrated matrix, label-ordered."""
        return self.mcs.get_matrix()

    def free(self):
        try:
            self.mcs.free()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# validated detection primitives (reused for spike detection on residuals)
# ---------------------------------------------------------------------------
def bandpass(signal, fs, low_hz=None, high_hz=None):
    """3rd-order Butterworth bandpass -- the lab's own filter (MEA-NAP default
    600-8000 Hz). low/high override the detector defaults when given."""
    det = _mods()["det"]
    p = dict(det.DETECT_PARAMS)
    if low_hz is not None:
        p["bp_low_hz"] = low_hz
    if high_hz is not None:
        p["bp_high_hz"] = high_hz
    return det.bandpass(signal, fs, p)


def detect_spikes(trace, fs, prefiltered=False):
    """Validated bior1.5 wavelet spike detection on ONE trace. Returns spike times
    (s). Used to re-detect biological spikes on an artifact-subtracted residual."""
    det = _mods()["det"]
    return np.asarray(det.detect_spikes(trace, fs, prefiltered=prefiltered), float)


def detect_spikes_matrix(mat, fs, prefiltered=False, progress=None):
    """Validated detection on (n_ch, n_samples). Returns list of spike-time arrays."""
    det = _mods()["det"]
    return det.detect_spikes_matrix(mat, fs, prefiltered=prefiltered,
                                    progress=progress)


def detection_wavelet_info(fs=20000.0):
    """The validated detector's calibration for a given sampling rate:
    {params, scales (CWT scales for the wid_ms range), psi_d + x (the bior1.5
    analysis wavelet), fs}. Fields default to None if the loaders can't be
    imported so the report can still fall back to a PyWavelets reconstruction."""
    info = {"fs": float(fs), "params": None, "scales": None,
            "psi_d": None, "x": None}
    try:
        det = _mods()["det"]
        info["params"] = dict(det.DETECT_PARAMS)
        try:
            info["scales"] = np.asarray(det.determine_scales(fs), float)
        except Exception:
            pass
        try:                                   # the lab port's own wavelet fn
            psi_d, x = det._sand._get_bior15_wavefn()
            info["psi_d"] = np.asarray(psi_d, float)
            info["x"] = np.asarray(x, float)
        except Exception:
            pass
    except Exception:
        pass
    return info


def load_spike_file(spike_dir, stem):
    """Validated per-recording biological spikes if Stage-1 detection has been run
    (a <stem>_spikes.npz in spike_dir). Returns dict or None."""
    io = _mods()["io"]
    old = io.CONFIG.get("SPIKE_DIR")
    io.CONFIG["SPIKE_DIR"] = spike_dir
    try:
        return io.load_spike_file(stem)
    finally:
        io.CONFIG["SPIKE_DIR"] = old


def save_meanap_spike_file(spike_dir, stem, spikes_by_label, fs,
                           noise_by_label=None, extra_params=None):
    """Write the cleaned biological spikes in the EXACT MEA-NAP format
    (`<stem>_spikes.npz` + `.mat`) so MEA Analyzer (MEA-NAP) Stage 2 and the raster
    viewer read/plot them directly. Reuses the lab's own io.save_spike_file, so the
    layout (spike_times object array in seconds, labels, fs, params, complete,
    n_channels, auto_ground, noise_uv; MEA-NAP .mat cell) is guaranteed compatible.

    spikes_by_label : {electrode label -> spike times (s)} (whole recording).
    Returns the written .npz path.
    """
    m = _mods()
    io, det = m["io"], m["det"]
    labels = list(spikes_by_label.keys())
    spike_times = [np.asarray(spikes_by_label[l], float).ravel() for l in labels]
    params = dict(getattr(det, "DETECT_PARAMS", {}))
    params["fs"] = float(fs)
    params["source"] = "MEA Solar Cell Analyzer (biological, flood artifacts removed)"
    if extra_params:
        params.update(extra_params)
    noise_uv = None
    if noise_by_label is not None:
        noise_uv = np.array([float(noise_by_label.get(l, np.nan)) for l in labels],
                            float)
    old = io.CONFIG.get("SPIKE_DIR")
    io.CONFIG["SPIKE_DIR"] = spike_dir
    try:
        npz_path, _mat = io.save_spike_file(stem, spike_times, labels, float(fs),
                                            params, auto_ground=[], noise_uv=noise_uv)
    finally:
        io.CONFIG["SPIKE_DIR"] = old
    return npz_path
