"""Broadband, multi-polarity peak detection + snippet features (pure numpy/scipy).

The MEA-NAP bior1.5 detector is tuned for negative-going biological action
potentials (600-8000 Hz). Solar-cell voltage transients and the piezoelectric
laser artifact are larger and lower-frequency and can be EITHER polarity, so this
module detects ALL candidate events on a lightly high-passed trace (default 100
Hz, to kill drift but keep artifact shape) and hands each one to the classifier.

A detected peak carries: sample index, time (s), signed amplitude (µV), polarity,
half-width (ms), and an amplitude-aligned waveform snippet for template matching.
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def highpass(trace, fs, hp_hz=100.0, order=3):
    """Zero-phase Butterworth high-pass. Preserves the (possibly slow) artifact
    waveform while removing baseline drift/LFP."""
    trace = np.asarray(trace, float)
    if hp_hz <= 0 or trace.size < 3 * order:
        return trace - np.median(trace)
    ny = 0.5 * fs
    b, a = butter(order, min(hp_hz / ny, 0.99), btype="highpass")
    return filtfilt(b, a, trace)


def robust_noise(x):
    """Noise floor = MAD/0.6745 (robust std, spike-insensitive)."""
    x = np.asarray(x, float)
    if x.size == 0:
        return 0.0
    return float(np.median(np.abs(x - np.median(x))) / 0.6745)


class Peaks:
    """Detected peaks for one channel over one window.

    Attributes (all length n, except snippets which is (n, L)):
      idx        sample indices into the (windowed) trace
      t          peak times (s), absolute if t_offset given
      amp        signed amplitude (µV) at the extremum on the filtered trace
      polarity   +1 / -1
      width_ms   half-maximum width (ms)
      snippets   (n, L) filtered-trace waveforms aligned on the extremum
      snip_t_ms  (L,) snippet time axis (ms, 0 at the extremum)
    """

    def __init__(self, idx, t, amp, polarity, width_ms, snippets, snip_t_ms,
                 fs, noise_uv, label=None):
        self.idx = np.asarray(idx, int)
        self.t = np.asarray(t, float)
        self.amp = np.asarray(amp, float)
        self.polarity = np.asarray(polarity, int)
        self.width_ms = np.asarray(width_ms, float)
        self.snippets = np.asarray(snippets, float)
        self.snip_t_ms = np.asarray(snip_t_ms, float)
        self.fs = float(fs)
        self.noise_uv = float(noise_uv)
        self.label = label

    def __len__(self):
        return int(self.idx.size)

    def normalized(self):
        """Zero-mean, unit-norm snippets (for correlation). (n, L)."""
        s = self.snippets - self.snippets.mean(axis=1, keepdims=True)
        nrm = np.linalg.norm(s, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        return s / nrm


def _half_width_ms(snip, peak_i, fs):
    """Full width at half maximum (ms) around the extremum of a snippet."""
    v = snip - np.median(snip)
    a = np.abs(v)
    half = a[peak_i] / 2.0
    if half <= 0:
        return 0.0
    lo = peak_i
    while lo > 0 and a[lo] > half:
        lo -= 1
    hi = peak_i
    while hi < a.size - 1 and a[hi] > half:
        hi += 1
    return (hi - lo) / fs * 1000.0


def detect_peaks(trace, fs, hp_hz=100.0, k=5.0, refractory_ms=1.0,
                 pre_ms=1.5, post_ms=1.5, t_offset=0.0, label=None,
                 prefiltered=False):
    """Detect all threshold-crossing peaks (either polarity) on a trace.

    trace      : raw calibrated voltage (µV), or already high-passed if prefiltered.
    k          : threshold in robust-MAD units.
    Returns a Peaks object (possibly empty).
    """
    trace = np.asarray(trace, float)
    filt = trace if prefiltered else highpass(trace, fs, hp_hz)
    noise = robust_noise(filt)
    pre = max(1, int(round(pre_ms * 1e-3 * fs)))
    post = max(1, int(round(post_ms * 1e-3 * fs)))
    dist = max(1, int(round(refractory_ms * 1e-3 * fs)))
    snip_t_ms = (np.arange(-pre, post + 1) / fs) * 1000.0

    empty = Peaks([], [], [], [], [], np.empty((0, pre + post + 1)),
                  snip_t_ms, fs, noise, label)
    if noise <= 0 or filt.size < pre + post + 3:
        return empty

    height = k * noise
    idx, _ = find_peaks(np.abs(filt), height=height, distance=dist)
    # keep only peaks whose full snippet window fits inside the trace
    idx = idx[(idx >= pre) & (idx < filt.size - post)]
    if idx.size == 0:
        return empty

    snippets = np.stack([filt[i - pre:i + post + 1] for i in idx])
    amp = filt[idx]
    polarity = np.sign(amp).astype(int)
    polarity[polarity == 0] = 1
    width = np.array([_half_width_ms(snippets[j], pre, fs)
                      for j in range(idx.size)])
    t = t_offset + idx / fs
    return Peaks(idx, t, amp, polarity, width, snippets, snip_t_ms, fs, noise,
                 label)


def flood_latency(peak_times, flood_on, lock_pre_s, lock_post_s):
    """For each peak, the latency (s) to the NEAREST flood onset within the lock
    window [-lock_pre, +lock_post]; np.nan if none. Positive = peak AFTER a flood.

    Because the neurons are spontaneous, a peak that reliably falls in this window
    is stimulus-locked and therefore an artifact (solar or piezo), not biological.
    """
    peak_times = np.asarray(peak_times, float)
    flood_on = np.sort(np.asarray(flood_on, float))
    out = np.full(peak_times.size, np.nan)
    if flood_on.size == 0 or peak_times.size == 0:
        return out
    # nearest preceding flood for each peak
    pos = np.searchsorted(flood_on, peak_times, side="right") - 1
    for j, t in enumerate(peak_times):
        cand = []
        if 0 <= pos[j] < flood_on.size:
            cand.append(flood_on[pos[j]])
        if pos[j] + 1 < flood_on.size:
            cand.append(flood_on[pos[j] + 1])
        best = np.nan
        for f in cand:
            lat = t - f
            if -lock_pre_s <= lat <= lock_post_s:
                if np.isnan(best) or abs(lat) < abs(best):
                    best = lat
        out[j] = best
    return out


def event_windows(on, off, pre_s, tail_s, fallback_post_s, min_post_s=0.0):
    """Per-flood artifact windows (start, end) in seconds, spanning the WHOLE
    exposure so both the ON and OFF solar transients fall inside:
        [onset - pre_s, offset + tail_s]
    When the offset is unknown (no OFF times), falls back to
        [onset - pre_s, onset + fallback_post_s].
    `min_post_s` guarantees each window reaches at least onset + min_post_s, so a
    short/misread exposure never makes the window narrower than the classic fixed
    window (prevents artifacts from leaking past a too-narrow window).
    """
    on = np.sort(np.asarray(on, float))
    off = np.asarray(off, float) if off is not None else np.array([])
    wins = []
    if off.size == on.size and off.size:
        for a, b in zip(on, off):
            end = (b + tail_s) if (np.isfinite(b) and b > a) else a + fallback_post_s
            wins.append((a - pre_s, max(end, a + min_post_s)))
    else:
        off_sorted = np.sort(off)
        for a in on:
            end = a + fallback_post_s
            if off_sorted.size:
                nxt = off_sorted[np.searchsorted(off_sorted, a, "right"):]
                if nxt.size and nxt[0] - a < fallback_post_s * 20:
                    end = nxt[0] + tail_s
            wins.append((a - pre_s, max(end, a + min_post_s)))
    return np.asarray(wins, float).reshape(-1, 2)


def train_starts(on, stim_train_s):
    """Collapse the onset times of a pulsed flood into ONE start per stimulation
    train. The flood.seq fires a burst of pulses (e.g. 10 pulses over 250 ms); the
    digital-event recorder may log one onset per pulse OR one per train. Any onset
    within `stim_train_s` of the current train's start is treated as part of that
    train, so this returns just the train-start times either way."""
    on = np.sort(np.asarray(on, float))
    if on.size == 0:
        return on
    starts = [on[0]]
    for t in on[1:]:
        if t - starts[-1] > stim_train_s:
            starts.append(t)
    return np.asarray(starts, float)


def train_windows(starts, pre_s, span_s, tail_s):
    """Artifact window per train: [start - pre, start + span + tail]. `span_s` is
    the whole stimulation-train duration, so every pulse's transient falls inside."""
    starts = np.asarray(starts, float)
    if starts.size == 0:
        return np.empty((0, 2))
    return np.column_stack([starts - pre_s, starts + span_s + tail_s])


def locked_to_events(peak_times, windows):
    """(mask, latency_to_onset) for peaks that fall inside any event window.
    latency is measured from the window's onset (= start + pre); NaN if unlocked."""
    peak_times = np.asarray(peak_times, float)
    mask = np.zeros(peak_times.size, bool)
    lat = np.full(peak_times.size, np.nan)
    if windows.size == 0 or peak_times.size == 0:
        return mask, lat
    starts = windows[:, 0]
    ends = windows[:, 1]
    order = np.argsort(starts)
    starts, ends = starts[order], ends[order]
    idx = np.searchsorted(starts, peak_times, "right") - 1
    for j, t in enumerate(peak_times):
        i = idx[j]
        if 0 <= i < starts.size and starts[i] <= t <= ends[i]:
            mask[j] = True
            lat[j] = t - starts[i]
    return mask, lat


def event_average(trace, fs, on, dur_s, pre_ms, tail_ms, t_offset=0.0):
    """Flood-ONSET-aligned average over the whole exposure window
    [-pre_ms, dur_s + tail_ms], so the returned waveform shows BOTH the ON and OFF
    transients (the two curves per solar event). Returns (avg, t_ms, n, p2p)."""
    post_ms = dur_s * 1000.0 + tail_ms
    return flood_triggered_average(trace, fs, on, pre_ms=pre_ms, post_ms=post_ms,
                                   t_offset=t_offset)


def flood_triggered_average(trace, fs, flood_on, pre_ms=5.0, post_ms=40.0,
                            t_offset=0.0):
    """Average the trace over all flood onsets -> the stimulus-triggered artifact
    waveform for one electrode (STA). Returns (sta, t_ms, n_used, peak_to_peak_uv).

    The STA is the model of the flood-evoked artifact on this electrode: its
    peak-to-peak amplitude is the electrode's response strength (drives the field
    map), and subtracting it removes the artifact (drives the filter).
    """
    trace = np.asarray(trace, float)
    flood_on = np.asarray(flood_on, float)
    pre = max(1, int(round(pre_ms * 1e-3 * fs)))
    post = max(1, int(round(post_ms * 1e-3 * fs)))
    t_ms = (np.arange(-pre, post + 1) / fs) * 1000.0
    segs = []
    for f in flood_on:
        i = int(round((f - t_offset) * fs))
        if i - pre >= 0 and i + post < trace.size:
            segs.append(trace[i - pre:i + post + 1])
    if not segs:
        return np.zeros(pre + post + 1), t_ms, 0, 0.0
    sta = np.mean(segs, axis=0)
    sta = sta - np.median(sta[:pre])       # baseline to pre-flood level
    p2p = float(np.max(sta) - np.min(sta))
    return sta, t_ms, len(segs), p2p


def flood_plateau(trace, fs, pulse_on, dur_s, pre_ms=5.0, edge_ms=1.5):
    """Sustained (held) photovoltage during illumination — the DC level that an
    AC/high-pass view differentiates away, as opposed to the fast edge-spike p2p.

    Averages the trace over every INDIVIDUAL light-ON pulse (baseline-subtracted to
    the pre-pulse median), then returns the MEDIAN level over the middle of the
    illumination window [edge_ms, dur - edge_ms] — i.e. skipping a guard band at the
    ON and OFF edges so the capacitive charging transients don't contaminate the
    held level. The coupling test showed this plateau persists in the DC-coupled raw
    trace, so it is the right quantity for "sustained photovoltage on this cell".

    trace    : one electrode's full calibrated trace (µV).
    pulse_on : INDIVIDUAL light-ON pulse onset times (s) — NOT train starts, so each
               averaged segment is a single illumination pulse.
    dur_s    : single-pulse ON duration (s). <= 0 (unknown) -> plateau undefined.
    edge_ms  : guard band skipped at each edge of the illumination window.

    Returns (plateau_uv, sta, t_ms, n_used). plateau_uv is SIGNED (held level
    relative to the pre-pulse baseline); NaN if the illumination window is too short
    to expose a plateau clear of both edges, or no pulse fit in the trace.
    """
    dur_ms = float(dur_s) * 1000.0
    post_ms = max(dur_ms + edge_ms, pre_ms + 1.0)
    sta, t_ms, n, _p2p = flood_triggered_average(
        trace, fs, pulse_on, pre_ms=pre_ms, post_ms=post_ms)
    if n == 0 or dur_ms <= 2.0 * edge_ms + 0.5:
        return float("nan"), sta, t_ms, n
    mid = (t_ms >= edge_ms) & (t_ms <= dur_ms - edge_ms)
    if not np.any(mid):
        return float("nan"), sta, t_ms, n
    return float(np.median(sta[mid])), sta, t_ms, n
