"""Classify detected peaks into solar / piezo / biological.

Master discriminator (given the neurons fire SPONTANEOUSLY): a peak that is
time-locked to a laser flood cannot be a spontaneous action potential, so it is
an artifact -- either the solar-cell voltage transient or the piezoelectric laser
transient. Which one is decided by template shape correlation, broken ties by
spatial prior (solar peaks concentrate near the cell; piezo is array-wide).

A peak that is NOT flood-locked and matches the biological template with a
physiological width is called biological. Everything else is 'unknown'.
"""
import numpy as np

import peaks as pk
import templates as tpl

SOLAR, PIEZO, BIO, UNKNOWN = tpl.SOLAR, tpl.PIEZO, tpl.BIO, "unknown"


class ClassifyResult:
    def __init__(self, labels, corr, latency, locked, peaks):
        self.labels = np.asarray(labels, object)   # per-peak class string
        self.corr = corr                            # dict class -> (n,) corr
        self.latency = np.asarray(latency, float)   # s to flood (nan = unlocked)
        self.locked = np.asarray(locked, bool)
        self.peaks = peaks

    def mask(self, cls):
        return self.labels == cls

    def counts(self):
        u, c = np.unique(self.labels, return_counts=True)
        return dict(zip((str(x) for x in u), (int(n) for n in c)))


def classify_peaks(peaks, on, off, dur_s, templateset, analysis,
                   near_solar=None):
    """Classify a Peaks object. Returns a ClassifyResult.

    peaks       : peaks.Peaks for one electrode/window.
    on, off     : laser flood ON / OFF times (s), same clock as peaks.t. A solar
                  cell fires an ON transient at onset and an OFF transient at
                  offset, so a peak anywhere in [onset-pre, offset+tail] is
                  flood-locked (an artifact) — this catches BOTH curves.
    dur_s       : median flood duration (s), used only when `off` is unavailable.
    templateset : templates.TemplateSet (bio template used for the biological test).
    near_solar  : optional per-peak bool (peak's electrode is inside/near a solar
                  cell). Because both artifacts are flood-locked, this spatial prior
                  is the solar-vs-piezo discriminator: solar voltage is localised at
                  the cell, the piezo transient is array-wide.
    """
    n = len(peaks)
    corr = {c: np.zeros(n) for c in tpl.CLASSES}
    if n == 0:
        return ClassifyResult(np.array([], object), corr,
                              np.array([]), np.array([], bool), peaks)

    # Per-peak SHAPE correlation against the extremum-aligned PEAK templates for
    # ALL three classes (solar/piezo templates now carry a peak template for
    # classification, plus a separate onset event waveform used only for display).
    sn = peaks.normalized()
    for c in tpl.CLASSES:
        t = templateset.get(c)
        if t is not None and t.n > 0 and getattr(t, "aligned", "peak") == "peak" \
                and t.mean_norm.size:
            corr[c] = np.atleast_1d(t.correlate(sn))
    bio_t = templateset.get(BIO)

    lock_pre = analysis["lock_pre_ms"] * 1e-3
    tail = analysis.get("event_tail_ms", 15.0) * 1e-3
    # The flood is a pulsed TRAIN (e.g. 10 pulses over 250 ms), so the artifact
    # window spans the whole train from each train start — this catches all of the
    # 10 solar transients (both the down and up lobes), not just the first pulse.
    stim_train = analysis.get("stim_train_ms", 250.0) * 1e-3
    starts = pk.train_starts(on, stim_train)
    windows = pk.train_windows(starts, lock_pre, stim_train, tail)
    locked, latency = pk.locked_to_events(peaks.t, windows)

    corr_min = analysis.get("corr_min", 0.6)
    artifact_corr = analysis.get("artifact_corr_min", 0.7)
    bio_width_max = (2.0 if bio_t is None or bio_t.width_med <= 0
                     else max(1.5, 2.5 * bio_t.width_med))

    if near_solar is None:
        near_solar = np.zeros(n, bool)
    near_solar = np.asarray(near_solar, bool)

    labels = np.empty(n, object)
    for j in range(n):
        cs, cp, cb = corr[SOLAR][j], corr[PIEZO][j], corr[BIO][j]
        art = max(cs, cp)
        # A peak is an artifact if it is flood-locked OR clearly matches a solar/
        # piezo SHAPE better than the biological shape — the shape backstop catches
        # sharp artifacts that fall just outside the flood window.
        is_artifact = locked[j] or (art >= artifact_corr and art >= cb)
        if is_artifact:
            # Solar voltage is LOCALISED at the cell; the piezo laser transient is
            # array-wide. Solar is a 250 ms train (no per-peak solar shape), so the
            # spatial prior is the master solar/piezo discriminator: a flood-locked
            # peak inside/near the cell is solar. Per-peak shape (cs/cp) only breaks
            # ties away from the cell (e.g. a piezo peak template still present).
            if near_solar[j]:
                labels[j] = SOLAR
            elif cs >= cp + 0.1:
                labels[j] = SOLAR
            else:
                labels[j] = PIEZO
        elif cb >= corr_min and peaks.width_ms[j] <= bio_width_max:
            labels[j] = BIO
        else:
            labels[j] = UNKNOWN
    return ClassifyResult(labels, corr, latency, locked, peaks)


def electrode_flood_response(sta_p2p, noise_uv, k=5.0):
    """Is an electrode flood-driven? True when its flood-triggered-average
    peak-to-peak exceeds k x its noise floor. Used to decide which electrodes are
    artifact-contaminated (for filtering and the field map)."""
    return bool(sta_p2p >= k * max(noise_uv, 1e-9))
