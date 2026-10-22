"""Test stub of mea_nap_io: the API surface MEA Explorer uses, backed by the
synthetic fixture in tests/fixtures (no HDF5, no real recording needed).

The real module's CONFIG["DATA_DIR"] is a hard-coded default that goes stale;
the stub deliberately points it at a folder that does not exist so the tests
prove that datasource.resolve_h5 finds <root>/H5 next to <root>/Output/SpikeData.
"""
import os, json, numpy as np
import mea_nap_wavelet_detect as det          # the REAL module (via sys.path), like the real io

_FIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")
CONFIG = {"DATA_DIR": os.path.join(_FIX, "wrong", "H5"),          # deliberately stale
          "SPIKE_DIR": os.path.join(_FIX, "MEA 071826", "Output", "SpikeData"),
          "REFERENCE_ELECTRODE": ["Ref", "15"]}


def spike_paths(stem):
    b = os.path.join(CONFIG["SPIKE_DIR"], stem)
    return b + "_spikes.npz", b + "_spikes.mat"


def load_spike_file(stem):
    p, _ = spike_paths(stem)
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return {"spike_times": list(z["spike_times"]), "labels": [str(l) for l in z["labels"]],
            "fs": float(z["fs"]), "auto_ground": [str(l) for l in z["auto_ground"]],
            "noise_uv": np.asarray(z["noise_uv"], float) if "noise_uv" in z.files else None}


def _meta(path):
    with open(path) as fh:
        return json.load(fh)          # the fixture .h5 is a JSON header stand-in


def _file_duration(stem):             # the real one reads CONFIG["DATA_DIR"] too
    m = _meta(os.path.join(CONFIG["DATA_DIR"], stem + ".h5"))
    return m["n"] / m["fs"]


def _minmax_decimate(y, n_bins):
    n = len(y)
    if n <= 2 * n_bins:
        return np.arange(n)
    e = np.linspace(0, n, n_bins + 1).astype(int)
    idx = []
    for a, b in zip(e[:-1], e[1:]):
        if b > a:
            s = y[a:b]
            idx += [a + int(np.argmin(s)), a + int(np.argmax(s))]
    return np.array(sorted(set(idx)))


class MCSFile:
    """Synthetic raw recording: drift + 50 Hz mains + fixed noise + a spike
    waveform (trough at the detected time) on every channel."""
    LOADS = 0

    def __init__(self, filepath, stream_index=None):
        MCSFile.LOADS += 1
        m = _meta(filepath)
        self.fs = float(m["fs"]); self.n_samples = int(m["n"])
        self.labels = list(m["labels"]); self.n_channels = len(self.labels)
        self.duration = self.n_samples / self.fs
        stem = os.path.basename(filepath)[:-3]
        sp = load_spike_file(stem)
        self._sp = dict(zip(sp["labels"], sp["spike_times"]))
        self._noise = {}

    def get_channel_window(self, i, s0, s1):
        s0 = max(0, int(s0)); s1 = min(self.n_samples, int(s1))
        k = np.arange(s0, s1); t = k / self.fs
        x = 40 * np.sin(2 * np.pi * 0.7 * t) + 15 * np.sin(2 * np.pi * 50 * t)   # drift + mains
        if i not in self._noise:            # one fixed noise realisation per channel
            self._noise[i] = np.random.default_rng(i * 1000003).normal(0, 6, self.n_samples)
        x = x + self._noise[i][s0:s1]
        st = np.asarray(self._sp.get(self.labels[i], []), float)
        idx = np.round(st * self.fs).astype(int)
        idx = idx[(idx > s0 - 60) & (idx < s1 + 60)]
        w = np.arange(-25, 41)
        wave = -70 * np.exp(-(w / 5.0) ** 2) + 25 * np.exp(-((w - 12) / 9.0) ** 2)
        for c in idx:
            a, b = max(c - 25, s0), min(c + 41, s1)
            if b > a:
                x[a - s0:b - s0] += wave[a - c + 25:b - c + 25]
        return x

    def free(self):
        pass
