"""Learned peak-shape templates for the three peak classes.

A template is the average waveform of one class of peak, learned from the
recording DEDICATED to that class:

  PIEZO  <- laser_only  (bare MEA): flood-locked peaks are the piezoelectric
            mechanical transient of the laser hitting the array.
  SOLAR  <- solar_only  (no culture): flood-locked peaks are the solar cell's
            photovoltaic voltage transient.
  BIO    <- culture_only OR the non-flood-locked peaks of any recording: the
            physiological action-potential shape.

Templates are amplitude-aligned (extremum at t=0), stored unit-norm for
correlation plus the raw-µV mean for display, and carry amplitude/width
distributions. Correlation of a candidate peak's normalized snippet with each
template is the shape feature the classifier uses.
"""
import numpy as np

import peaks as pk

PIEZO = "piezo"
SOLAR = "solar"
BIO = "bio"
CLASSES = [SOLAR, PIEZO, BIO]


class Template:
    def __init__(self, name, t_ms, mean_norm, mean_uv, std_uv, n,
                 amp_med, amp_mad, width_med, fs, aligned="peak",
                 event_t_ms=None, event_uv=None, event_std=None, event_n=0):
        self.name = name
        self.t_ms = np.asarray(t_ms, float)
        self.mean_norm = np.asarray(mean_norm, float)   # zero-mean, unit-norm
        self.mean_uv = np.asarray(mean_uv, float)       # average raw µV waveform
        self.std_uv = np.asarray(std_uv, float)
        self.n = int(n)
        self.amp_med = float(amp_med)
        self.amp_mad = float(amp_mad)
        self.width_med = float(width_med)
        self.fs = float(fs)
        # `mean_*` is ALWAYS the extremum-aligned PEAK template used for per-peak
        # classification (aligned == "peak"). For solar/piezo we ALSO keep the
        # flood-onset-aligned EVENT average (both ON+OFF transients) purely for the
        # calibration/Templates display — it is NOT used for classification.
        self.aligned = aligned
        self.event_t_ms = None if event_t_ms is None else np.asarray(event_t_ms, float)
        self.event_uv = None if event_uv is None else np.asarray(event_uv, float)
        self.event_std = None if event_std is None else np.asarray(event_std, float)
        self.event_n = int(event_n)

    @property
    def has_event(self):
        return self.event_uv is not None and self.event_uv.size > 0

    @property
    def length(self):
        return int(self.mean_norm.size)

    def correlate(self, snippets_norm):
        """Correlation of each normalized snippet (n, L) with this template.
        Both sides are zero-mean unit-norm, so this is a cosine similarity in
        [-1, 1]. Returns (n,) or a scalar for a single 1-D snippet."""
        s = np.atleast_2d(np.asarray(snippets_norm, float))
        L = self.mean_norm.size
        if s.shape[1] != L:                        # length mismatch -> resample
            s = _resample_rows(s, L)
        c = s @ self.mean_norm
        return c if c.size > 1 else float(c[0])


def _resample_rows(arr, L):
    arr = np.atleast_2d(arr)
    if arr.shape[1] == L:
        return arr
    xo = np.linspace(0, 1, arr.shape[1])
    xn = np.linspace(0, 1, L)
    return np.stack([np.interp(xn, xo, row) for row in arr])


def _sign_align_rows(rows, iters=4):
    """Flip each row (±1) to a common polarity by correlation, so waveforms that
    are mere sign-inversions of each other (opposite edges of a dipolar solar
    cell) reinforce instead of cancelling. Preserves biphasic (up+down) shape."""
    rows = np.atleast_2d(np.asarray(rows, float))
    if rows.shape[0] < 2:
        return rows
    m = rows - rows.mean(axis=1, keepdims=True)
    ref = m[np.argmax(np.linalg.norm(m, axis=1))].copy()   # strongest as seed
    out = rows
    for _ in range(iters):
        c = m @ ref
        signs = np.sign(c)
        signs[signs == 0] = 1.0
        out = rows * signs[:, None]
        mo = out - out.mean(axis=1, keepdims=True)
        ref = mo.mean(axis=0)
    return out


def _robust_inliers(rows, min_corr=0.55, keep_frac=0.6, iters=3):
    """Boolean mask of rows consistent with the class's mean SHAPE.

    Iteratively builds the zero-mean unit-norm class mean, correlates every row to
    it (cosine similarity), and drops rows below `min_corr` — but never keeps fewer
    than `keep_frac` of the rows (falls back to the top-correlated ones). This is
    what stops a handful of contaminating peaks (a sharp artifact that slipped past
    the flood window, a positive noise transient) from injecting the big spurious
    lobes that otherwise show up in the ±SD error band."""
    rows = np.atleast_2d(np.asarray(rows, float))
    n = rows.shape[0]
    keep = np.ones(n, bool)
    if n < 4:
        return keep
    rc = rows - rows.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(rc, axis=1)
    norms[norms == 0] = 1.0
    floor = max(3, int(round(keep_frac * n)))
    for _ in range(iters):
        ref = rc[keep].mean(axis=0)
        rn = np.linalg.norm(ref)
        if rn == 0:
            break
        corr = (rc @ (ref / rn)) / norms
        newkeep = corr >= min_corr
        if int(newkeep.sum()) < floor:          # keep the best `floor` instead
            thresh = np.sort(corr)[::-1][floor - 1]
            newkeep = corr >= thresh
        if np.array_equal(newkeep, keep):
            break
        keep = newkeep
    return keep


def build_template(name, snippets, snip_t_ms, amps, widths, fs,
                   align_polarity=True, aligned="peak", sign_align=False,
                   robust=False, min_corr=0.55, keep_frac=0.6, amp_mad_k=6.0,
                   amp_normalize=False, orient_down=False):
    """Average a stack of aligned snippets (n, L) into a Template.

    amp_normalize=True (the solar EVENT template): equalise each electrode's
    amplitude (peak-to-peak) before averaging, so an electrode sitting right on the
    cell contact — measured at ~12 mV, ~19× the next electrode — can't dominate the
    average and hide the shape every other electrode sees. orient_down=True then
    flips the mean so its LARGEST transient is negative, matching the physical solar
    unit (a deep DOWN peak at each pulse onset, then the relaxation / OFF up-peak).

    align_polarity=True (biological peaks): flip only the rare opposite-polarity
    outlier to the class's dominant sign so the average doesn't cancel, while
    keeping the class's natural polarity. align_polarity=False (onset-aligned event
    waveforms, e.g. solar/piezo with two transients): average the waveforms AS-IS,
    since flipping by a single extremum would corrupt a biphasic event.

    robust=True (the per-peak class wavelets): before averaging, reject (1) AMPLITUDE
    outliers — a rare electrode artifact can be orders of magnitude larger than a real
    spike and, though it barely moves the mean, it dominates the ±SD band (verified on
    real data: one 21 mV event took the trough SD from 9 to 331 µV) — and (2) shape
    outliers whose waveform is inconsistent with the class mean. So the mean waveform
    and its ±SD band reflect the true class, not a handful of artifacts. NOT used for
    the onset event waveforms."""
    snippets = np.atleast_2d(np.asarray(snippets, float))
    if snippets.shape[0] == 0:
        L = np.asarray(snip_t_ms).size
        z = np.zeros(L)
        return Template(name, snip_t_ms, z, z, z, 0, 0.0, 0.0, 0.0, fs, aligned)
    amps = np.asarray(amps, float)
    if sign_align:
        # correlation-based polarity alignment across electrodes (dipolar solar
        # cell): keeps the biphasic up+down event from cancelling.
        rows = _sign_align_rows(snippets)
    elif align_polarity:
        signs = np.sign(amps)
        signs[signs == 0] = 1.0
        class_sign = np.sign(np.median(amps))
        class_sign = 1.0 if class_sign == 0 else class_sign
        rows = snippets * (signs * class_sign)[:, None]
    else:
        rows = snippets
    keep = np.ones(rows.shape[0], bool)
    if robust:
        # (1) amplitude-outlier rejection (the band-blowing artifacts) — robust MAD
        #     ceiling on |amplitude at the extremum|.
        aa = np.abs(amps)
        if aa.size == keep.size and np.any(aa > 0):
            med = np.median(aa)
            mad = np.median(np.abs(aa - med)) / 0.6745
            # floor the scale so a near-constant bulk (MAD≈0) can't disable the cut;
            # an order-of-magnitude artifact is still caught.
            scale = max(mad, 0.1 * med)
            if scale > 0:
                keep &= aa <= med + amp_mad_k * scale
        # (2) shape-outlier rejection on the amplitude-kept subset.
        idx = np.where(keep)[0]
        if idx.size >= 4:
            sub = _robust_inliers(rows[idx], min_corr, keep_frac)
            keep[idx[~sub]] = False
    krows = rows[keep]
    kamps = amps[keep] if amps.size == keep.size else amps
    widths = np.asarray(widths, float)
    kw = widths[keep] if widths.size == keep.size else widths
    if amp_normalize and krows.shape[0]:
        # equalise per-electrode amplitude so one contact electrode can't dominate;
        # rescale to the representative (median) amplitude for a µV-meaningful mean.
        p2p = np.ptp(krows, axis=1)
        p2p[p2p == 0] = 1.0
        rep = float(np.median(p2p))
        krows = krows / p2p[:, None] * rep
    mean_uv = krows.mean(axis=0)
    std_uv = krows.std(axis=0)
    if orient_down and mean_uv.size and \
            abs(float(mean_uv.max())) > abs(float(mean_uv.min())):
        mean_uv = -mean_uv                    # largest transient -> negative (down)
    m = mean_uv - mean_uv.mean()
    nrm = np.linalg.norm(m)
    mean_norm = m / nrm if nrm > 0 else m
    return Template(name, snip_t_ms, mean_norm, mean_uv, std_uv,
                    int(krows.shape[0]),
                    float(np.median(np.abs(kamps))) if kamps.size else 0.0,
                    float(np.median(np.abs(kamps - np.median(kamps))) / 0.6745)
                    if kamps.size else 0.0,
                    float(np.median(kw)) if np.size(kw) else 0.0, fs,
                    aligned)


class TemplateSet:
    def __init__(self, templates=None, meta=None):
        self.templates = dict(templates or {})   # name -> Template
        self.meta = dict(meta or {})

    def __contains__(self, name):
        return name in self.templates

    def get(self, name):
        return self.templates.get(name)

    def names(self):
        return [n for n in CLASSES if n in self.templates] + \
               [n for n in self.templates if n not in CLASSES]

    def save(self, path):
        d = {"_names": np.array(list(self.templates.keys()))}
        for name, t in self.templates.items():
            d[f"{name}__t_ms"] = t.t_ms
            d[f"{name}__mean_norm"] = t.mean_norm
            d[f"{name}__mean_uv"] = t.mean_uv
            d[f"{name}__std_uv"] = t.std_uv
            d[f"{name}__scalars"] = np.array(
                [t.n, t.amp_med, t.amp_mad, t.width_med, t.fs], float)
            d[f"{name}__aligned"] = np.array(t.aligned)
            if t.has_event:
                d[f"{name}__event_t_ms"] = t.event_t_ms
                d[f"{name}__event_uv"] = t.event_uv
                d[f"{name}__event_std"] = (t.event_std if t.event_std is not None
                                           else np.zeros_like(t.event_uv))
                d[f"{name}__event_n"] = np.array(t.event_n)
        np.savez_compressed(path, **d)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        names = [str(n) for n in z["_names"]]
        templates = {}
        for name in names:
            sc = z[f"{name}__scalars"]
            aligned = (str(z[f"{name}__aligned"])
                       if f"{name}__aligned" in z.files else "peak")
            ev_kw = {}
            if f"{name}__event_uv" in z.files:
                ev_kw = dict(event_t_ms=z[f"{name}__event_t_ms"],
                             event_uv=z[f"{name}__event_uv"],
                             event_std=z[f"{name}__event_std"],
                             event_n=int(z[f"{name}__event_n"]))
            templates[name] = Template(
                name, z[f"{name}__t_ms"], z[f"{name}__mean_norm"],
                z[f"{name}__mean_uv"], z[f"{name}__std_uv"],
                int(sc[0]), sc[1], sc[2], sc[3], sc[4], aligned, **ev_kw)
        return cls(templates)


# ---------------------------------------------------------------------------
# high-level learning from tagged recordings (uses the loaders; heavy)
# ---------------------------------------------------------------------------
def _match_mask(times, ref_times, tol_s):
    """Boolean mask of `times` that fall within tol_s of ANY `ref_times`."""
    times = np.asarray(times, float)
    ref = np.sort(np.asarray(ref_times, float))
    if times.size == 0 or ref.size == 0:
        return np.zeros(times.size, bool)
    pos = np.searchsorted(ref, times)
    out = np.zeros(times.size, bool)
    for j, t in enumerate(times):
        for q in (pos[j] - 1, pos[j]):
            if 0 <= q < ref.size and abs(t - ref[q]) <= tol_s:
                out[j] = True
                break
    return out


def learn_class_from_recording(raw, flood_on, analysis, locked=True,
                               max_channels=None, progress=None, keep_polarity=0,
                               detector=None, match_tol_ms=1.0):
    """Detect peaks on every electrode of a RawRecording and collect the snippets
    of the wanted kind: flood-LOCKED (artifact classes) or NON-locked (biological).

    keep_polarity : +1/-1 keeps only positive/negative-extremum peaks; 0 keeps both.
    For the BIOLOGICAL template this is set to -1 to MIRROR the validated bior1.5
    detector, which itself hard-gates to negative-going APs (mea_nap_wavelet_detect
    `_detect_from_cwt`: "keep negative-going APs") — so the learned bio wavelet
    represents the same spikes the detector actually returns, not an added assumption.

    detector : optional callable(trace, fs) -> spike times (s). When given (the
    BIOLOGICAL template), only broadband peaks CONFIRMED by this validated detector
    (within match_tol_ms) enter the template — so noise threshold-crossings that the
    broadband detector picks up but bior1.5 rejects never contaminate the bio wavelet
    or its error band. The snippets are still cut from the same broadband trace as
    the candidate peaks, so template↔peak correlation stays consistent.

    Returns (snippets, snip_t_ms, amps, widths) ready for build_template.
    `raw` is a loaders.RawRecording; kept dependency-free here so the caller wires
    the loader. This is the only heavy path and is skipped by the unit tests.
    """
    lock_pre = analysis["lock_pre_ms"] * 1e-3
    lock_post = analysis["lock_post_ms"] * 1e-3
    tol_s = match_tol_ms * 1e-3
    labels = raw.labels if max_channels is None else raw.labels[:max_channels]
    all_snips, all_amps, all_w = [], [], []
    snip_t = None
    for n, lab in enumerate(labels):
        trace = raw.channel(lab)
        p = pk.detect_peaks(trace, raw.fs, hp_hz=analysis["hp_hz"],
                            k=analysis["peak_k"],
                            refractory_ms=analysis["refractory_ms"],
                            pre_ms=analysis["snippet_pre_ms"],
                            post_ms=analysis["snippet_post_ms"], label=lab)
        if len(p) == 0:
            if progress:
                progress(n + 1, len(labels))
            continue
        snip_t = p.snip_t_ms
        lat = pk.flood_latency(p.t, flood_on, lock_pre, lock_post)
        mask = ~np.isnan(lat) if locked else np.isnan(lat)
        if keep_polarity:
            mask = mask & (np.sign(p.polarity) == np.sign(keep_polarity))
        if detector is not None:
            mask = mask & _match_mask(p.t, detector(trace, raw.fs), tol_s)
        if mask.any():
            all_snips.append(p.snippets[mask])
            all_amps.append(p.amp[mask])
            all_w.append(p.width_ms[mask])
        if progress:
            progress(n + 1, len(labels))
    if not all_snips:
        L = snip_t.size if snip_t is not None else 1
        return np.empty((0, L)), (snip_t if snip_t is not None else np.zeros(1)), \
            np.empty(0), np.empty(0)
    return (np.concatenate(all_snips), snip_t,
            np.concatenate(all_amps), np.concatenate(all_w))


def learn_event_class_from_recording(raw, flood_on, dur_s, analysis,
                                     max_channels=None, k_response=5.0,
                                     progress=None):
    """Learn a flood-locked ARTIFACT class (solar / piezo) as an ONSET-aligned
    event average. For each electrode, average the trace over all flood onsets on a
    window that spans the whole exposure ([-pre, dur+tail]) so BOTH the ON and OFF
    transients are captured — this is the fix for the two-curves-per-event problem
    (simple extremum-aligned peak averaging would smear the two transients).

    Keeps only electrodes whose event average is a real response (peak-to-peak >=
    k_response × noise). Returns (event_waves, t_ms, p2p_amps, widths) — one row
    per responsive electrode — ready for build_template(align_polarity=False)."""
    pre_ms = 3.0
    tail_ms = analysis.get("event_tail_ms", 15.0)
    # Show the WHOLE stimulation train (e.g. 250 ms, all pulses) aligned on the
    # train start, so the event waveform reveals every solar transient.
    stim_train_ms = analysis.get("stim_train_ms", 250.0)
    post_ms = stim_train_ms + tail_ms
    starts = pk.train_starts(flood_on, stim_train_ms * 1e-3)
    labels = raw.labels if max_channels is None else raw.labels[:max_channels]
    waves, amps = [], []
    t_ms = None
    for n, lab in enumerate(labels):
        trace = raw.channel(lab)
        avg, tt, nused, p2p = pk.flood_triggered_average(
            trace, raw.fs, starts, pre_ms=pre_ms, post_ms=post_ms)
        noise = pk.robust_noise(pk.highpass(trace, raw.fs, analysis["hp_hz"]))
        if nused > 0 and p2p >= k_response * max(noise, 1e-9):
            t_ms = tt
            waves.append(avg)
            amps.append(p2p)
        if progress:
            progress(n + 1, len(labels))
    if not waves:
        L = t_ms.size if t_ms is not None else 1
        return np.empty((0, L)), (t_ms if t_ms is not None else np.zeros(1)), \
            np.empty(0), np.empty(0)
    waves = np.stack(waves)
    return waves, t_ms, np.asarray(amps, float), np.full(len(amps), np.nan)
