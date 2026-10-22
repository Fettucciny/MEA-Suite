"""Remove solar-cell and piezoelectric artifacts, leaving pure biological spikes.

Two complementary paths:

1. PEAK-LEVEL (cheap): keep only peaks the classifier labelled biological -> a
   cleaned spike train per electrode. Good when artifacts and spikes don't
   overlap in time.

2. WAVEFORM-LEVEL (recovers spikes buried under an artifact): every laser flood
   evokes nearly the SAME artifact waveform on a given electrode, so its
   flood-triggered average (STA) is that artifact. Subtract the STA (least-squares
   scaled per flood) at each flood onset, then re-run the VALIDATED bior1.5
   detector on the residual -> biological spikes even during the stim window.

The STA subtraction is "filtering the sample with the learned peak shape": the
learned shape is the per-electrode flood-evoked artifact.
"""
import numpy as np

import peaks as pk


def subtract_flood_artifact(trace, fs, flood_on, sta, sta_t_ms, t_offset=0.0,
                            fit_scale=True):
    """Subtract the STA artifact from `trace` at every flood onset.

    fit_scale: least-squares scale the STA to each flood's actual segment (handles
    flood-to-flood amplitude jitter) before subtracting. Returns the residual
    trace (same length)."""
    trace = np.asarray(trace, float).copy()
    sta = np.asarray(sta, float)
    if sta.size == 0 or np.allclose(sta, 0):
        return trace
    pre = int(round(-sta_t_ms[0] * 1e-3 * fs))
    L = sta.size
    sta0 = sta - sta.mean()
    denom = float(sta0 @ sta0)
    for f in np.asarray(flood_on, float):
        i = int(round((f - t_offset) * fs))
        a, b = i - pre, i - pre + L
        if a < 0 or b > trace.size:
            continue
        seg = trace[a:b]
        if fit_scale and denom > 0:
            g = float((seg - seg.mean()) @ sta0) / denom
            g = float(np.clip(g, 0.0, 3.0))       # only remove same-sign artifact
        else:
            g = 1.0
        trace[a:b] = seg - g * sta
    return trace


def train_starts(flood_on, analysis):
    """Train-start onsets for the pulsed flood (one per stimulation train)."""
    stim_train = analysis.get("stim_train_ms", 250.0) * 1e-3
    return pk.train_starts(flood_on, stim_train)


def _butter_bandpass(trace, fs, low_hz, high_hz, order=2):
    from scipy.signal import butter, filtfilt
    trace = np.asarray(trace, float)
    ny = 0.5 * fs
    lo = max(1e-6, float(low_hz) / ny)
    hi = min(0.999, float(high_hz) / ny)
    if hi <= lo:
        return trace
    b, a = butter(int(order), [lo, hi], btype="band")
    return filtfilt(b, a, trace)


def threshold_gate(trace, fs, spike_times, low_hz=200.0, high_hz=3000.0,
                   order=2, k=5.0):
    """MEA-NAP-style amplitude gate ANDed with the bior1.5 detections: keep only
    spikes whose amplitude on a `low`-`high` Hz Butterworth band-pass exceeds
    k × robust-noise, so low-amplitude non-biological wavelet detections are dropped.

    The check is over a ±0.5 ms window around each spike (robust to sub-sample
    alignment between the wavelet detector and the band-pass extremum)."""
    st = np.asarray(spike_times, float)
    if st.size == 0:
        return st
    bp = _butter_bandpass(trace, fs, low_hz, high_hz, order)
    thr = float(k) * max(pk.robust_noise(bp), 1e-9)
    w = max(1, int(round(0.0005 * fs)))
    idx = np.clip(np.round(st * fs).astype(int), 0, bp.size - 1)
    keep = np.zeros(st.size, bool)
    for j, i in enumerate(idx):
        a, b = max(0, i - w), min(bp.size, i + w + 1)
        seg = bp[a:b]
        keep[j] = bool(seg.size) and float(np.max(np.abs(seg))) >= thr
    return st[keep]


def _maybe_gate(trace, fs, spikes, analysis):
    """Apply the amplitude gate if enabled in the analysis config."""
    if not analysis.get("prefilter_enabled"):
        return spikes
    return threshold_gate(trace, fs, spikes,
                          analysis.get("prefilter_low_hz", 200.0),
                          analysis.get("prefilter_high_hz", 3000.0),
                          analysis.get("prefilter_order", 2),
                          analysis.get("prefilter_k", 5.0))


def reject_flood_locked(bio_spikes, artifact_labels, starts, analysis):
    """On artifact-contaminated electrodes, DROP any re-detected spike inside a
    flood-train window. These are residual solar/piezo transients the STA
    subtraction could not fully remove — on a 4 mV artifact electrode even a small
    residual dwarfs the biological threshold, so bior1.5 re-detects it as a "spike".
    A spontaneous neuron is, by definition, not time-locked to the laser, so this
    removes artifacts without discarding anything biological we could trust.

    Mutates bio_spikes in place. Controlled by analysis['reject_flood_locked'].
    """
    if not analysis.get("reject_flood_locked", True):
        return
    starts = np.asarray(starts, float) if starts is not None else np.empty(0)
    if starts.size == 0:
        return
    lock_pre = analysis.get("lock_pre_ms", 2.0) * 1e-3
    span = analysis.get("stim_train_ms", 250.0) * 1e-3
    tail = analysis.get("event_tail_ms", 15.0) * 1e-3
    windows = pk.train_windows(starts, lock_pre, span, tail)
    for lab in artifact_labels:
        t = bio_spikes.get(lab)
        if t is not None and np.size(t):
            locked, _ = pk.locked_to_events(np.asarray(t, float), windows)
            bio_spikes[lab] = np.asarray(t, float)[~locked]


def build_sta_bank(raw, flood_on, analysis, dur_s=0.0, labels=None, progress=None):
    """Per-electrode TRAIN-onset-triggered average + peak-to-peak + noise, over a
    window spanning the whole stimulation TRAIN ([-pre, stim_train + tail]) so the
    STA (and thus the subtraction) covers ALL the pulsed solar transients — not
    just the first pulse.

    Each entry also carries "plateau": the SUSTAINED (held) photovoltage measured on
    a SINGLE-pulse-triggered average (median over the middle of the illumination
    window, edges excluded) — the DC level a high-pass view differentiates away,
    reported alongside the edge-spike p2p. NaN when the single-pulse duration is
    unknown (dur_s <= 0).

    Returns dict label -> {"sta","t_ms","n","p2p","noise","plateau"}."""
    labels = labels or raw.labels
    pre = analysis["lock_pre_ms"] + 3.0
    tail = analysis.get("event_tail_ms", 15.0)
    stim_train_ms = analysis.get("stim_train_ms", 250.0)
    post = stim_train_ms + tail + 10.0
    starts = train_starts(flood_on, analysis)
    edge_ms = analysis.get("plateau_edge_ms", 1.5)
    bank = {}
    for n, lab in enumerate(labels):
        trace = raw.channel(lab)
        sta, t_ms, nused, p2p = pk.flood_triggered_average(
            trace, raw.fs, starts, pre_ms=pre, post_ms=post)
        noise = pk.robust_noise(pk.highpass(trace, raw.fs, analysis["hp_hz"]))
        # sustained photovoltage on a SINGLE pulse (not the whole train)
        plateau = float("nan")
        if dur_s and dur_s > 0:
            plateau, _sp, _tp, _np = pk.flood_plateau(
                trace, raw.fs, flood_on, dur_s, edge_ms=edge_ms)
        bank[lab] = {"sta": sta, "t_ms": t_ms, "n": nused, "p2p": p2p,
                     "noise": noise, "plateau": plateau}
        if progress:
            progress(n + 1, len(labels))
    return bank


class FilterResult:
    def __init__(self, bio_spikes, artifact_labels, sta_bank, fs, grounded=None):
        self.bio_spikes = bio_spikes          # label -> spike times (s)
        self.artifact_labels = set(artifact_labels)
        self.sta_bank = sta_bank
        self.fs = float(fs)
        self.grounded = set(grounded or [])   # electrodes turned OFF (Batch.csv col 6)

    def total_spikes(self):
        return int(sum(len(v) for v in self.bio_spikes.values()))

    def rate_map(self, duration):
        """label -> biological spike rate (Hz)."""
        return {lab: (len(t) / duration if duration > 0 else 0.0)
                for lab, t in self.bio_spikes.items()}


def filter_recording(raw, flood_on, analysis, detector, artifact_k=5.0,
                     dur_s=0.0, labels=None, progress=None, sta_bank=None,
                     exclude=None):
    """Full artifact removal on a RawRecording.

    detector : callable(trace, fs) -> spike times (s). Pass loaders.detect_spikes
               (the validated bior1.5 detector) so recovered spikes match the rest
               of the pipeline.
    Returns a FilterResult. Electrodes whose STA p2p exceeds artifact_k x noise are
    artifact-subtracted before detection; others are detected as-is.
    """
    labels = labels or raw.labels
    exclude = set(exclude or [])
    if sta_bank is None:
        sta_bank = build_sta_bank(raw, flood_on, analysis, dur_s, labels)
    starts = train_starts(flood_on, analysis)     # subtract the train STA per train
    bio_spikes, artifact_labels = {}, []
    for n, lab in enumerate(labels):
        if lab in exclude:                        # grounded -> OFF, no detection
            bio_spikes[lab] = np.empty(0)
            if progress:
                progress(n + 1, len(labels))
            continue
        trace = raw.channel(lab)
        info = sta_bank[lab]
        if info["p2p"] >= artifact_k * max(info["noise"], 1e-9) and info["n"] > 0:
            artifact_labels.append(lab)
            trace = subtract_flood_artifact(trace, raw.fs, starts,
                                            info["sta"], info["t_ms"])
        sp = np.asarray(detector(trace, raw.fs), float)
        bio_spikes[lab] = _maybe_gate(trace, raw.fs, sp, analysis)
        if progress:
            progress(n + 1, len(labels))
    reject_flood_locked(bio_spikes, artifact_labels, starts, analysis)
    return FilterResult(bio_spikes, artifact_labels, sta_bank, raw.fs,
                        grounded=exclude & set(labels))


def filter_recording_matrix(raw, flood_on, analysis, detector_matrix,
                            artifact_k=5.0, labels=None, progress=None,
                            sta_bank=None, exclude=None):
    """Same artifact removal as filter_recording, but subtract into the whole
    (n_ch, n_samples) matrix and detect every channel in ONE batched call
    (detector_matrix(residual, fs, progress=...) -> list of spike arrays). Much
    faster over many channels/recordings (the CWT batches on the GPU), so it is the
    path used by batch filtering. Returns a FilterResult identical in shape to the
    per-channel path.

    detector_matrix : callable(mat, fs, progress=None) -> list of spike-time arrays
                      (s), one per row. Pass loaders.detect_spikes_matrix.
    """
    labels = labels or raw.labels
    if sta_bank is None:
        sta_bank = build_sta_bank(raw, flood_on, analysis, labels=labels)
    starts = train_starts(flood_on, analysis)
    mat = np.asarray(raw.matrix(), dtype=np.float32)
    index = {lab: i for i, lab in enumerate(raw.labels)}
    residual = mat.copy()
    artifact_labels = []
    for lab in labels:
        info = sta_bank[lab]
        if info["p2p"] >= artifact_k * max(info["noise"], 1e-9) and info["n"] > 0:
            artifact_labels.append(lab)
            i = index[lab]
            residual[i] = subtract_flood_artifact(residual[i], raw.fs, starts,
                                                  info["sta"], info["t_ms"])
    sel = [index[lab] for lab in labels]

    prog = None
    if progress is not None:
        total = len(sel)

        def prog(n_done):                     # detector contract: progress(n_done)
            progress(int(n_done), total)
    exclude = set(exclude or [])
    spike_lists = detector_matrix(residual[sel], raw.fs, progress=prog)
    bio_spikes = {}
    for k, lab in enumerate(labels):
        if lab in exclude:                        # grounded -> OFF
            bio_spikes[lab] = np.empty(0)
            continue
        sp = np.asarray(spike_lists[k], float)
        bio_spikes[lab] = _maybe_gate(residual[index[lab]], raw.fs, sp, analysis)
    artifact_labels = [l for l in artifact_labels if l not in exclude]
    reject_flood_locked(bio_spikes, artifact_labels, starts, analysis)
    return FilterResult(bio_spikes, artifact_labels, sta_bank, raw.fs,
                        grounded=exclude & set(labels))


def bio_spikes_from_classification(results_by_label):
    """Peak-level cleaning: {label -> ClassifyResult} -> {label -> bio spike times}.
    Keeps only peaks classified biological."""
    from classify import BIO
    out = {}
    for lab, res in results_by_label.items():
        out[lab] = res.peaks.t[res.mask(BIO)] if len(res.peaks) else np.empty(0)
    return out
