#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 mea_nap_phase_lockin.py  --  stimulus-locked ("lock-in") sweep propagation
================================================================================
An OPTIONAL analysis stage that detects whether the network LOCKS to the periodic
laser sweep, and recovers the sweep's direction / apparent speed as a spatial
PHASE gradient. Ported from the user's standalone `sweep_lockin.py` and wired into
the MEA-NAP pipeline (spike units, MCS geometry, Batch.csv regions, group gating).
Output goes to a subfolder of the GUI output path:  "Phase Lock"/.

OUTPUTS (per group; every figure uses the same grid = row per sample/training day
sorted youngest-first, column per region pre/stim/post, with a left day strip.
Every panel is normalised to its OWN maximum so weak Pre/Post windows stay
readable, and one 0-1 colour legend is drawn per sample row):
    phase_lockin_<group>.png    electrode map, colour = relative lock strength,
                                black arrow = recovered heading, red dashed =
                                nominal sweep axis
    phase_flow_<group>.png      streamline flow map of the interpolated phase
                                gradient — the path the sweep phase travels
    phase_compass_<group>.png   polar compass of per-electrode preferred phase
                                (angle) vs relative lock strength (radius), with
                                a phase-histogram rose and the mean resultant
    phase_direction_<group>.png lock-on convergence vs time: the heading is
                                re-fitted in consecutive direction_window_s
                                bins and scored as % convergence on the training
                                angle, 50*(1+alignment): 135 deg -> 100 %,
                                -45 deg -> 0 %, with the pre / stim / post
                                epochs shaded. One row per sample; this is the
                                only figure whose x axis is session time
    RADIAL groups (Batch.csv group name starting "radial", e.g. radial1) use
    the quadrant alignment model in the SAME files: the array is split into
    four quadrants about the ring centre (between 44/45/54/55), a plane wave
    is fitted per quadrant and scored against that quadrant's outward
    direction. The lock map and flow map show the four quadrant headings in
    place of the single sweep heading; the direction-over-time curve plots
    the mean of the four quadrants' % convergence on outward. See
    quadrant_alignment().
    phase_direction_mean_<condition>.png
                                the same convergence curve AVERAGED across
                                replicate samples, one row per training day,
                                with a +-1 SEM ribbon. Replicates = groups that
                                differ only by a trailing number (sweep1/2/3 ->
                                "sweep") or several files per group.
    phase_lockin_summary.csv    absolute numbers (direction, speed, coherence,
                                period, peak_ratio, alignment, phase span,
                                propagation flag, p-value)

WHY (vs the spike-LAT / spkit continuous-EGM propagation)
--------------------------------------------------------
The LAT / -dV/dt methods pick ONE activation time per event and fit its spatial
gradient -> they measure the FAST intra-burst neural wave (10^4-10^5 um/s), NOT
the slow imposed sweep (200-3200 um/s). This module instead folds EVERY spike
onto the sweep phase phi = 2*pi*frac((t-t0)/T) and, per electrode, forms the
phase-locking resultant vector
        z_e = (1/n_e) * sum_k exp(i*phi_k)
    |z_e|   = vector strength  -> how strongly electrode e locks to the sweep
    arg z_e = preferred phase  -> WHEN in the cycle e fires (its sweep position)
Non-locked background spikes scatter across phases over many cycles and cancel;
sweep-driven spikes accumulate coherently (SNR ~ sqrt(n_cycles)). The propagation
is the SPATIAL gradient of preferred phase, recovered with a wrap-immune plane-
wave matched filter maximised over spatial wavevector k:
    direction = atan2(k_y, k_x)          (heading of the travelling phase)
    speed     = 2*pi / (|k|*T)           (= wavelength / period)
    coherence = P(k*) in [0,1]           (1 = perfect plane wave)
A circular-shift / phase-rotation permutation null gives a p-value.

Because Pre/Post windows have no external drive, a low peak_ratio / coherence
there is the EXPECTED result; a HIGH one that persists post-stim is the memory /
entrainment signal. Each panel is flagged so real locks are not confused with
spurious fits to spontaneous rhythm.

PERIOD for this rig: the laser trigger advances one Polygon frame; one full sweep
= `frames_per_sweep` frames (160 for Sept2021_SweepBRtoTL.seq), so the sweep
repeats every T ~= frames_per_sweep / trigger_hz seconds (e.g. 160/80Hz = 2.0 s).
That is used only as a SEED; refine_period searches around it for the period that
best explains the data (a seed off by even 5% smears the fit).

Import-safe: pure numpy/scipy/matplotlib. Run directly for a synthetic self-test.
================================================================================
"""

import os
import re as _re
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mea_nap_io as io
import mea_nap_original_plots as orig
import mea_nap_digital_events as de
import mea_nap_progress as progress_gui

# ---- parula colormap (MATLAB default; shared by SpKit + Phase-Lock heatmaps) ----
from matplotlib.colors import ListedColormap as _LCM
_PARULA_DATA = [
    (0.2081,0.1663,0.5292),(0.2116,0.1898,0.5777),(0.2123,0.2138,0.6270),(0.2081,0.2386,0.6771),
    (0.1959,0.2645,0.7279),(0.1707,0.2919,0.7792),(0.1253,0.3242,0.8303),(0.0591,0.3598,0.8683),
    (0.0117,0.3875,0.8819),(0.0060,0.4086,0.8828),(0.0165,0.4266,0.8786),(0.0329,0.4430,0.8720),
    (0.0498,0.4586,0.8641),(0.0629,0.4737,0.8554),(0.0723,0.4887,0.8467),(0.0779,0.5040,0.8384),
    (0.0793,0.5200,0.8312),(0.0749,0.5375,0.8263),(0.0641,0.5570,0.8240),(0.0488,0.5772,0.8228),
    (0.0343,0.5966,0.8199),(0.0265,0.6137,0.8135),(0.0239,0.6287,0.8038),(0.0231,0.6418,0.7913),
    (0.0228,0.6535,0.7768),(0.0267,0.6642,0.7607),(0.0384,0.6743,0.7436),(0.0590,0.6838,0.7254),
    (0.0843,0.6928,0.7062),(0.1133,0.7015,0.6859),(0.1453,0.7098,0.6646),(0.1801,0.7177,0.6424),
    (0.2178,0.7250,0.6193),(0.2586,0.7317,0.5954),(0.3022,0.7376,0.5712),(0.3482,0.7424,0.5473),
    (0.3953,0.7459,0.5244),(0.4420,0.7481,0.5033),(0.4871,0.7491,0.4840),(0.5300,0.7491,0.4661),
    (0.5709,0.7485,0.4494),(0.6099,0.7473,0.4337),(0.6473,0.7456,0.4188),(0.6834,0.7435,0.4044),
    (0.7184,0.7411,0.3905),(0.7525,0.7384,0.3768),(0.7858,0.7356,0.3633),(0.8185,0.7327,0.3498),
    (0.8507,0.7299,0.3360),(0.8824,0.7274,0.3217),(0.9139,0.7258,0.3063),(0.9450,0.7261,0.2886),
    (0.9739,0.7314,0.2666),(0.9938,0.7455,0.2403),(0.9990,0.7653,0.2164),(0.9955,0.7861,0.1967),
    (0.9880,0.8066,0.1794),(0.9789,0.8271,0.1633),(0.9697,0.8481,0.1475),(0.9626,0.8705,0.1309),
    (0.9589,0.8949,0.1132),(0.9598,0.9218,0.0948),(0.9661,0.9514,0.0755),(0.9763,0.9831,0.0538)]
PARULA = _LCM(_PARULA_DATA, name="parula")


# ==============================================================================
# PARAMETERS  (adjustment ranges noted; exposed in the GUI "Phase Lock-In" tab)
# ==============================================================================
PHASE_LOCKIN = {
    # enable : master switch. When True the GUI runs this pipeline after Stage 2
    #   (and after SpKit, if that is also on) and writes to the "Phase Lock" folder.
    "enable":               False,
    # use_digital_events : seed the sweep period from the laser trigger rate
    #   (frames_per_sweep / trigger_hz) when the recording has digital events.
    #   True | False.
    "use_digital_events":   True,
    # frames_per_sweep : Polygon frames in one full sweep of the pattern; the
    #   trigger advances one frame, so period seed = frames_per_sweep / trigger_hz.
    #   Sept2021_SweepBRtoTL.seq = 160. Range 1 - 100000.
    "frames_per_sweep":     160.0,
    # period_s : explicit sweep-period override (s). None -> derive from triggers
    #   (or population-rate autocorrelation for windows without triggers).
    "period_s":             None,
    # refine : search around the seed for the true cycle period. True | False.
    "refine":               True,
    # refine_search_frac : fractional half-width of the period search around the
    #   seed (0.30 = +/-30%). Range 0.05 - 0.50.
    "refine_search_frac":   0.30,
    # refine_objective : "pooled" (fast, non-spatial locking strength) or
    #   "spatial" (weights each candidate period by plane-wave coherence; ~15x
    #   slower but rejects non-spatial period aliases).
    "refine_objective":     "pooled",
    # min_spikes : an electrode needs at least this many spikes in the window to
    #   contribute a lock (fewer -> its resultant vector is too noisy). Range 5 - 60.
    "min_spikes":           15,
    # min_cycles : the window must contain at least this many complete sweep
    #   periods for the fold to be meaningful. A period near/longer than the window
    #   (e.g. stray-trigger Control recordings whose trigger rate implies a ~500 s
    #   "sweep") collapses every electrode's phase to one value -> spurious |z|~1
    #   everywhere (lock strength clipped at max). Such windows are rejected.
    #   Range 1.5 - 10.
    "min_cycles":           2.0,
    # max_trigger_period_s : reject a trigger-derived period longer than this and
    #   fall back to population-rate autocorrelation. Guards against the flood /
    #   simultaneous control, where frames_per_sweep*ISI overshoots to ~800 s.
    #   Range 60 - 600.
    "max_trigger_period_s": 120.0,
    # min_phase_span_rad : a real travelling wave must vary the preferred phase by
    #   at least this much ACROSS the array (|k|*array_extent). A flood/simultaneous
    #   response is flat (phase span ~0) -> flagged "no propagation" (k~0, non-
    #   physical speed) so the negative control reads correctly. Range 0.2 - 1.5.
    "min_phase_span_rad":   0.6,
    # peak_ratio_ok : period-search peak/median ratio below which the panel is
    #   flagged low-confidence (no sharply tuned period found). Range 1.2 - 4.0.
    "peak_ratio_ok":        1.8,
    # coherence_floor : plane-wave coherence below which direction/speed are flagged
    #   as near the noise floor (more cycles / less background needed). Range 0.0-0.5.
    "coherence_floor":      0.10,
    # do_pvalue : compute a phase-rotation permutation p-value per window. ON by
    #   default so the p_value column of the summary CSV is populated; uncheck in
    #   the GUI to speed up runs. True | False.
    "do_pvalue":            True,
    # n_perm : permutations for the p-value null. Range 100 - 1000.
    "n_perm":               200,
    # pitch_um : inter-electrode spacing (um); sets the spatial Nyquist and the
    #   speed scale (MCS 60-MEA = 200). Range 100 - 700.
    "pitch_um":             200.0,
    # sweep_angle_deg : nominal sweep axis (BR->TL = 135) for the alignment score.
    #   Range 0 - 360.
    "sweep_angle_deg":      135.0,
    # direction_window_s : bin width (s) for the heading-over-time curve. The
    #   session is cut into consecutive bins of this length and the plane-wave
    #   fit is re-run inside each one, giving a lock-on direction trace. Short
    #   bins follow fast changes but carry fewer spikes per electrode (noisier,
    #   and bins with <6 locked electrodes are dropped); long bins are smooth
    #   but blur the pre/stim transition. Read ONLY by phase_direction_<group>
    #   .png -- no other output or parameter depends on it. Range 10 - 600.
    "direction_window_s":   60.0,
    # ---- keys added by later builds. Present here ONLY so the current GUI
    #      schema resolves when this legacy module is rolled in; they are not
    #      read by this build. ----
    "n_workers":            1,
    "post_map_window_s":    0.0,
    "direction_smooth_bins": 1.0,
}

DPI = 200


# ==============================================================================
# output location + small helpers
# ==============================================================================
def phase_lock_root(plot_dir):
    """The 'Phase Lock' folder, a subfolder UNDER the selected output path."""
    return os.path.join(os.path.normpath(plot_dir), "Phase Lock")


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _sweep_suffix(stem):
    """Suptitle fragment noting the nominal laser-sweep speed parsed from the
    filename prefix (e.g. '4000um...' -> 4000 um/s), or '' for Control names."""
    m = _re.match(r"\s*(\d+(?:\.\d+)?)\s*um", str(stem), _re.IGNORECASE)
    if not m:
        return ""
    v = float(m.group(1))
    lab = (f"{v / 1000.0:.3g} mm/s ({v:.0f} um/s)" if v >= 1000 else f"{v:.0f} um/s")
    return f"  -  nominal laser sweep {lab}"


def _label_xy(labels, pitch):
    """{label: (x_um, y_um)} in the MEA display frame (x=col, y=9-row), microns."""
    out = {}
    for l in labels:
        c = orig.mcs_coord(l)
        if c is not None:
            out[l] = (c[0] * pitch, c[1] * pitch)
    return out


# ==============================================================================
# period estimation
# ==============================================================================
def _population_rate(spike_s, t0_s, t1_s, bin_s=0.05):
    edges = np.arange(t0_s, t1_s + bin_s, bin_s)
    rate = np.zeros(max(1, len(edges) - 1))
    for ts in spike_s.values():
        s = ts[(ts >= t0_s) & (ts < t1_s)]
        if s.size:
            idx = np.clip(((s - t0_s) / bin_s).astype(int), 0, len(rate) - 1)
            np.add.at(rate, idx, 1)
    return rate, bin_s


def estimate_period(spike_s, t0_s, t1_s, p_lo=0.5, p_hi=60.0, bin_s=0.05):
    """Coarse population-rate autocorrelation period (s) — a rough SEED only."""
    rate, bs = _population_rate(spike_s, t0_s, t1_s, bin_s)
    rate = rate - rate.mean()
    if not np.any(rate):
        return None
    ac = np.correlate(rate, rate, mode="full")[len(rate) - 1:]
    lags = np.arange(len(ac)) * bs
    band = (lags >= p_lo) & (lags <= p_hi)
    if not np.any(band):
        return None
    bi = np.where(band)[0]
    return float(lags[bi[np.argmax(ac[bi])]])


# ==============================================================================
# per-electrode locking
# ==============================================================================
@dataclass
class ElecLock:
    label: object
    x_um: float
    y_um: float
    z: complex
    n: int

    @property
    def strength(self):
        return abs(self.z)

    @property
    def phase(self):
        return np.angle(self.z)


def electrode_locking(spike_s, t0_s, t1_s, period_s, label_xy, min_spikes=15):
    """Per-electrode phase-locking resultant vector at a trial period."""
    out = []
    if not period_s or period_s <= 0:
        return out
    for lbl, (x, y) in label_xy.items():
        ts = spike_s.get(lbl)
        if ts is None:
            continue
        s = ts[(ts >= t0_s) & (ts < t1_s)]
        if s.size < min_spikes:
            continue
        phi = 2.0 * np.pi * (((s - t0_s) / period_s) % 1.0)
        z = np.exp(1j * phi).mean()
        out.append(ElecLock(lbl, float(x), float(y), z, int(s.size)))
    return out


# ==============================================================================
# plane-wave matched filter
# ==============================================================================
def _matched_power(locks, kx, ky):
    w = np.array([np.sqrt(l.n) for l in locks])
    z = np.array([l.z for l in locks])
    x = np.array([l.x_um for l in locks])
    y = np.array([l.y_um for l in locks])
    acc = (w * z * np.exp(-1j * (kx * x + ky * y))).sum()
    return abs(acc) / w.sum()


def fit_plane_wave(locks, period_s, pitch_um, n_grid=121, refine=True):
    """Grid-search the wavevector k maximising plane-wave coherence, then refine."""
    if len(locks) < 6:
        return None
    k_max = np.pi / pitch_um                    # spatial Nyquist (min lambda = 2*pitch)
    ks = np.linspace(-k_max, k_max, n_grid)
    P = np.empty((n_grid, n_grid))
    for i, kx in enumerate(ks):
        for j, ky in enumerate(ks):
            P[i, j] = _matched_power(locks, kx, ky)
    i0, j0 = np.unravel_index(np.argmax(P), P.shape)
    kx0, ky0 = ks[i0], ks[j0]
    coh = float(P[i0, j0])
    if refine and 0 < i0 < n_grid - 1 and 0 < j0 < n_grid - 1:
        def parab(pm, p0, pp, step):
            denom = (pm - 2 * p0 + pp)
            return 0.5 * (pm - pp) / denom * step if denom != 0 else 0.0
        dk = ks[1] - ks[0]
        kx0 += parab(P[i0 - 1, j0], P[i0, j0], P[i0 + 1, j0], dk)
        ky0 += parab(P[i0, j0 - 1], P[i0, j0], P[i0, j0 + 1], dk)
        coh = _matched_power(locks, kx0, ky0)
    kmag = np.hypot(kx0, ky0)
    speed = (2 * np.pi / (period_s * kmag)) if kmag > 1e-12 else np.inf
    return {"direction_deg": float(np.degrees(np.arctan2(ky0, kx0))),
            "speed_um_per_s": float(speed),
            "wavelength_um": float(2 * np.pi / kmag) if kmag > 1e-12 else np.inf,
            "coherence": coh, "k": (float(kx0), float(ky0)),
            "n_electrodes": len(locks)}


# ==============================================================================
# period refinement
# ==============================================================================
def _total_locking_strength(spike_s, t0_s, t1_s, period_s, label_xy, min_spikes):
    locks = electrode_locking(spike_s, t0_s, t1_s, period_s, label_xy, min_spikes)
    return sum(l.strength * np.sqrt(l.n) for l in locks) if locks else 0.0


def _spatial_coherence_at_period(spike_s, t0_s, t1_s, period_s, label_xy,
                                 min_spikes, pitch_um):
    locks = electrode_locking(spike_s, t0_s, t1_s, period_s, label_xy, min_spikes)
    if len(locks) < 6:
        return 0.0
    fit = fit_plane_wave(locks, period_s, pitch_um, n_grid=41, refine=False)
    if fit is None:
        return 0.0
    pooled = sum(l.strength * np.sqrt(l.n) for l in locks)
    return fit["coherence"] * pooled


def refine_period(spike_s, t0_s, t1_s, label_xy, seed_period, pitch_um,
                  search_frac=0.30, coarse_n=61, fine_n=41, fine_frac=0.03,
                  min_period=0.3, max_period=120.0, min_spikes=15,
                  objective="pooled"):
    """Find the period maximising the search objective around the seed. Returns
    (best_period_s, search_curve, peak_ratio)."""
    if objective == "spatial":
        score_fn = lambda p: _spatial_coherence_at_period(
            spike_s, t0_s, t1_s, p, label_xy, min_spikes, pitch_um)
    else:
        score_fn = lambda p: _total_locking_strength(
            spike_s, t0_s, t1_s, p, label_xy, min_spikes)

    if seed_period and seed_period > 0:
        lo = max(min_period, seed_period * (1 - search_frac))
        hi = min(max_period, seed_period * (1 + search_frac))
    else:
        lo, hi = min_period, max_period
    if hi <= lo:
        hi = lo + max(0.1, 0.1 * lo)
    coarse = np.linspace(lo, hi, coarse_n)
    cs = [(p, score_fn(p)) for p in coarse]
    best_p = max(cs, key=lambda t: t[1])[0]

    flo = max(min_period, best_p * (1 - fine_frac))
    fhi = min(max_period, best_p * (1 + fine_frac))
    fine = np.linspace(flo, fhi, fine_n)
    fs = [(p, score_fn(p)) for p in fine]
    all_pts = sorted(cs + fs)
    best_p2, best_s2 = max(fs, key=lambda t: t[1])

    scores = np.array([s for _, s in all_pts])
    med = float(np.median(scores)) or 1e-9
    return best_p2, all_pts, float(best_s2 / med)


# ==============================================================================
# permutation null
# ==============================================================================
def permutation_pvalue(locks, pitch_um, fit, n_perm=200, seed=0):
    """Null: independently rotate each electrode's preferred phase, recompute peak
    plane-wave coherence. p = fraction of nulls whose best coherence >= observed.

    IMPORTANT: the observed statistic is recomputed here on the SAME grid the null
    uses (no parabolic refinement) so observed and permuted data go through an
    identical test statistic. Comparing the refined fit["coherence"] against a
    coarse-grid null biased every window to the p floor (1/(n_perm+1))."""
    rng = np.random.default_rng(seed)
    k_max = np.pi / pitch_um
    ks = np.linspace(-k_max, k_max, 61)
    w = np.array([np.sqrt(l.n) for l in locks]); ws = w.sum()
    xx = np.array([l.x_um for l in locks]); yy = np.array([l.y_um for l in locks])
    z_obs = np.array([l.z for l in locks])
    amp = np.abs(z_obs)
    # precompute the plane-wave basis once; the k-grid search is then two matmuls
    # per permutation instead of a 61x61 Python loop (same result, ~1000x faster).
    Ex = np.exp(-1j * np.outer(ks, xx))          # (nk, nelec)
    Ey = np.exp(-1j * np.outer(ks, yy))          # (nk, nelec)

    def _best(z):
        M = Ex @ ((w * z)[:, None] * Ey.T)       # (nk, nk): coherence at every k
        return float(np.abs(M).max()) / ws

    obs = _best(z_obs)                           # same statistic as the null
    ge = 0
    for _ in range(n_perm):
        zz = amp * np.exp(1j * rng.uniform(0, 2 * np.pi, len(locks)))
        ge += (_best(zz) >= obs)
    return (ge + 1) / (n_perm + 1)


# ==============================================================================
# orchestration for one window (region)
# ==============================================================================
def lockin_window(spike_s, dur_s, label_xy, p, seed_period=None):
    """Run the full lock-in on one region window (spikes in seconds, 0..dur_s).
    Returns a fit dict (ok=True) or a failure dict (ok=False, reason)."""
    pitch = p["pitch_um"]
    min_spikes = int(p["min_spikes"])
    seed = seed_period if (seed_period and seed_period > 0) else p.get("period_s")
    if not seed or seed <= 0:
        seed = estimate_period(spike_s, 0.0, dur_s)
    if not seed or seed <= 0:
        return {"ok": False, "reason": "no_period_seed"}

    search_curve = peak_ratio = None
    if p.get("refine", True):
        period, search_curve, peak_ratio = refine_period(
            spike_s, 0.0, dur_s, label_xy, seed_period=seed, pitch_um=pitch,
            search_frac=p.get("refine_search_frac", 0.30), min_spikes=min_spikes,
            objective=p.get("refine_objective", "pooled"))
    else:
        period = seed

    # a period near/longer than the window folds every spike into <min_cycles
    # sweeps, collapsing all phases together -> spurious |z|~1 everywhere. Reject.
    min_cycles = float(p.get("min_cycles", 2.0))
    n_cycles = (dur_s / period) if period and period > 0 else 0.0
    if n_cycles < min_cycles:
        return {"ok": False,
                "reason": f"{n_cycles:.1f} sweeps in window (<{min_cycles:g})",
                "period_s": float(period), "seed_period_s": float(seed),
                "n_cycles": round(float(n_cycles), 2)}

    locks = electrode_locking(spike_s, 0.0, dur_s, period, label_xy, min_spikes)
    if len(locks) < 6:
        return {"ok": False, "reason": "too_few_locked", "period_s": period,
                "seed_period_s": seed, "peak_ratio": peak_ratio, "n_locked": len(locks)}
    fit = fit_plane_wave(locks, period, pitch)
    if fit is None:
        return {"ok": False, "reason": "fit_failed", "period_s": period,
                "seed_period_s": seed, "peak_ratio": peak_ratio}
    fit.update(ok=True, period_s=float(period), seed_period_s=float(seed),
               search_curve=search_curve, peak_ratio=peak_ratio, locks=locks,
               mean_strength=float(np.mean([l.strength for l in locks])))
    # phase span the fit predicts ACROSS the array. A flood / simultaneous drive
    # gives a flat phase map (k~0) -> tiny span -> "no propagation" (still a high
    # coherence, since flat IS trivially a k=0 plane wave; the tell is span/|k|).
    kx, ky = fit["k"]
    proj = np.array([kx * l.x_um + ky * l.y_um for l in locks])
    fit["phase_span_rad"] = float(proj.max() - proj.min()) if proj.size else 0.0
    fit["no_propagation"] = fit["phase_span_rad"] < float(
        p.get("min_phase_span_rad", 0.6))
    low = []
    if fit["no_propagation"]:
        low.append(f"phase flat across array (span={fit['phase_span_rad']:.2f} rad, "
                   f"k≈0) — simultaneous, no propagation")
    if peak_ratio is not None and peak_ratio < p.get("peak_ratio_ok", 1.8):
        low.append(f"peak_ratio={peak_ratio:.1f} (period not sharply tuned)")
    if fit["coherence"] < p.get("coherence_floor", 0.10):
        low.append(f"coherence={fit['coherence']:.2f} (near noise floor)")
    if low:
        fit["low_confidence"] = True
        fit["note"] = "; ".join(low)
    if p.get("do_pvalue"):
        fit["p_value"] = permutation_pvalue(locks, pitch, fit,
                                            n_perm=int(p.get("n_perm", 200)))
    return fit


# ==============================================================================
# figures  --  one figure per GROUP: a row per sample (training day), a column
#              per region, plus a left "day strip" marking the day progression
#
# Three products share the SAME grid:
#   phase_lockin_<group>.png     electrode lock-strength map + heading arrows
#   phase_flow_<group>.png       streamline flow map of the interpolated phase
#                                gradient (the path the sweep phase travels)
#   phase_compass_<group>.png    circular (compass) plot of preferred phases
#
# COLOUR SCALING: every panel is normalised to its OWN maximum, so the colour
# axis is a dimensionless 0-1 "relative lock strength". That makes the spatial
# PATTERN comparable between panels whose absolute locking differs by an order
# of magnitude (Pre/Post are usually far weaker than Stim and would otherwise
# render as a uniform dark tile). One colour legend is drawn per SAMPLE ROW and
# is shared by that sample's region panels. Absolute lock strength is not lost -
# it stays in phase_lockin_summary.csv (mean_strength / coherence), and each
# panel is annotated with its own max in the corner.
# ==============================================================================
from matplotlib.colors import Normalize as _Normalize
from matplotlib.cm import ScalarMappable as _ScalarMappable

_NORM01 = _Normalize(vmin=0.0, vmax=1.0)


def _norm_strengths(locks):
    """(normalised 0-1 strengths, panel max). Empty -> (empty, 0)."""
    st = np.array([l.strength for l in locks], float)
    if st.size == 0:
        return st, 0.0
    smax = float(st.max())
    return (st / smax if smax > 1e-12 else np.zeros_like(st)), smax


def _no_lock(ax, fit, polar=False):
    """Render the 'no lock' placeholder. Returns True if the tile is empty."""
    if fit is not None and fit.get("ok"):
        return False
    reason = (fit or {}).get("reason", "no data")
    if polar:
        ax.set_yticklabels([]); ax.set_xticklabels([]); ax.grid(False)
    ax.text(0.5, 0.5, f"no lock\n({reason})", ha="center", va="center",
            fontsize=8, color="#888", style="italic", transform=ax.transAxes)
    return True


def _sweep_axis(ax, sweep_deg):
    sdx, sdy = np.cos(np.radians(sweep_deg)), np.sin(np.radians(sweep_deg))
    ax.annotate("", xy=(4.5 + 2.4 * sdx, 4.5 + 2.4 * sdy),
                xytext=(4.5 - 2.4 * sdx, 4.5 - 2.4 * sdy),
                arrowprops=dict(arrowstyle="-", lw=1.4, color="crimson",
                                linestyle="--"), zorder=4)


def _heading_arrow(ax, fit, lw=2.4, reach=2.4):
    """Recovered heading. Greyed when the fit is low-confidence; suppressed (and
    labelled) when the phase map is flat, i.e. simultaneous / no propagation."""
    if fit.get("no_propagation"):
        ax.text(4.5, 4.5, "no propagation\n(phase flat, k≈0)", ha="center",
                va="center", fontsize=7.5, color="#B03A2E", style="italic",
                zorder=7, bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                    ec="#B03A2E", lw=0.6, alpha=0.85))
        return
    kx, ky = fit["k"]; g = np.hypot(kx, ky)
    if g <= 0:
        return
    low = fit.get("low_confidence", False)
    ux, uy = kx / g, ky / g
    ax.annotate("", xy=(4.5 + reach * ux, 4.5 + reach * uy),
                xytext=(4.5 - reach * ux, 4.5 - reach * uy),
                arrowprops=dict(arrowstyle="-|>", lw=lw,
                                color="#999" if low else "black",
                                mutation_scale=16), zorder=6)


def _panel_tag(ax, text, polar=False):
    """Small corner annotation carrying the absolute scale that normalisation
    removes, so no information is lost by going to 0-1."""
    x, y = (1.02, 1.02) if polar else (0.985, 0.012)
    ax.text(x, y, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5, color="#555", zorder=8,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                      alpha=0.72))


# ==============================================================================
# radial (centre-out) stimulation  --  quadrant alignment model
# ------------------------------------------------------------------------------
# A ring expanding from the array centre is not a plane wave, so the single
# whole-array plane-wave fit returns a confident-looking but meaningless heading
# on radial data. Instead the array is split into four quadrants about the ring
# centre and the EXISTING fit_plane_wave is run inside each one. Locally a ring
# is close to a plane wave travelling away from the centre, so each quadrant's
# heading is scored against its OWN expected outward direction: the bearing from
# the ring centre to that quadrant's locked-electrode centroid.
#
# Using the centroid rather than a fixed 45/135/225/315 diagonal is deliberate.
# A plane-wave fit over a patch of a ring recovers roughly the radial direction
# at the patch centroid, and grounded/unlocked electrodes move that centroid, so
# referencing the centroid cancels that geometric bias instead of reporting it
# as misalignment.
#
# Per-electrode phase (electrode_locking), the period, and every PHASE_LOCKIN
# parameter come from the normal region pipeline unchanged; only the spatial
# readout differs. Radial groups are recognised by their Batch.csv group name.
# ==============================================================================
_RADIAL_GROUP_PREFIX = "radial"          # Batch.csv groups radial1, radial2, ...
_RADIAL_CENTRE_CELLS = (4.5, 4.5)        # ring centre: between 44 / 45 / 54 / 55
_RADIAL_MIN_QUADRANTS = 2                # quadrants needed to form an average
_QUADRANT_ORDER = ("TL", "TR", "BL", "BR")


def _is_radial_group(group):
    """True for Batch.csv groups named radial1, Radial2, ... (case-insensitive)."""
    return str(group or "").strip().lower().startswith(_RADIAL_GROUP_PREFIX)


def _quadrant_name(x_cells, y_cells, cx, cy):
    """Display frame: x = column (left -> right), y = 9 - row (row 1 at top)."""
    return ("T" if y_cells > cy else "B") + ("R" if x_cells > cx else "L")


def quadrant_alignment(locks, period_s, pitch_um, min_phase_span_rad=0.6):
    """Per-quadrant plane-wave fit, scored against the outward direction.

    Returns a dict with
      quadrants : {TL/TR/BL/BR: {n_locked, ok, centroid_cells, expected_deg,
                   direction_deg, deviation_deg, alignment, coherence,
                   speed_um_per_s, k, phase_span_rad, flat}}
      n_quadrants           quadrants that reached a fit (fit_plane_wave
                            needs >= 6 locked electrodes)
      mean_deviation_deg    CIRCULAR mean of the quadrant deviations from
                            outward: 0 = outward, +-180 = inward
      deviation_consistency |mean resultant| of those deviations (1 = all
                            quadrants deviate identically; low = they
                            disagree, so the mean alone can hide error)
      radial_alignment      mean cos(deviation): +1 outward, -1 inward, 0
                            tangential/random -- the radial analogue of the
                            linear 'alignment' column
      mean_coherence, median_speed_um_per_s
      ok                    n_quadrants >= _RADIAL_MIN_QUADRANTS
    """
    cx, cy = _RADIAL_CENTRE_CELLS
    members = {q: [] for q in _QUADRANT_ORDER}
    for l in locks:
        members[_quadrant_name(l.x_um / pitch_um, l.y_um / pitch_um,
                               cx, cy)].append(l)

    quads = {}
    for q in _QUADRANT_ORDER:
        ql = members[q]
        entry = {"n_locked": len(ql), "ok": False}
        quads[q] = entry
        if not ql:
            continue
        mx = float(np.mean([l.x_um for l in ql])) / pitch_um
        my = float(np.mean([l.y_um for l in ql])) / pitch_um
        expected = float(np.degrees(np.arctan2(my - cy, mx - cx)))
        entry.update(centroid_cells=(mx, my), expected_deg=expected)
        pw = fit_plane_wave(ql, period_s, pitch_um)
        if pw is None:
            continue
        kx, ky = pw["k"]
        proj = np.array([kx * l.x_um + ky * l.y_um for l in ql])
        span = float(proj.max() - proj.min()) if proj.size else 0.0
        dev = float(_wrap180(pw["direction_deg"] - expected))
        entry.update(ok=True, direction_deg=float(pw["direction_deg"]),
                     deviation_deg=dev,
                     convergence_pct=float(_convergence_pct(dev)),
                     alignment=float(np.cos(np.radians(dev))),
                     coherence=float(pw["coherence"]),
                     speed_um_per_s=float(pw["speed_um_per_s"]),
                     k=pw["k"], phase_span_rad=span,
                     flat=bool(span < float(min_phase_span_rad)))

    fitted = [quads[q] for q in _QUADRANT_ORDER if quads[q]["ok"]]
    out = {"quadrants": quads, "n_quadrants": len(fitted),
           "centre_cells": (cx, cy),
           "ok": len(fitted) >= _RADIAL_MIN_QUADRANTS}
    if fitted:
        R = np.exp(1j * np.radians([e["deviation_deg"] for e in fitted])).mean()
        speeds = [e["speed_um_per_s"] for e in fitted
                  if np.isfinite(e["speed_um_per_s"])]
        out.update(
            mean_deviation_deg=float(np.degrees(np.angle(R))),
            deviation_consistency=float(abs(R)),
            radial_alignment=float(np.mean([e["alignment"] for e in fitted])),
            mean_convergence_pct=float(np.mean([e["convergence_pct"]
                                                for e in fitted])),
            mean_coherence=float(np.mean([e["coherence"] for e in fitted])),
            median_speed_um_per_s=(float(np.median(speeds)) if speeds
                                   else float("inf")))
    return out


def _radial_summary_fields(rad, coherence_floor):
    """Summary-CSV overrides for a radial row. The whole-array plane-wave
    numbers (direction, phase span, p-value) describe the wrong model on a ring
    and are blanked rather than left in the table to be misread."""
    row = {"pattern": "radial", "direction_deg": None, "p_value": None,
           "phase_span_rad": None, "propagation": None,
           "n_quadrants": rad.get("n_quadrants", 0)}
    if rad.get("n_quadrants"):
        row.update(
            alignment=round(rad["radial_alignment"], 3),
            mean_deviation_deg=round(rad["mean_deviation_deg"], 1),
            deviation_consistency=round(rad["deviation_consistency"], 3),
            coherence=round(rad["mean_coherence"], 3),
            speed_um_per_s=(round(rad["median_speed_um_per_s"], 1)
                            if np.isfinite(rad["median_speed_um_per_s"])
                            else None))
    flat_any = False
    for q in _QUADRANT_ORDER:
        e = rad["quadrants"].get(q, {})
        row[f"q{q}_heading_deg"] = (round(e["direction_deg"], 1)
                                    if e.get("ok") else None)
        row[f"q{q}_deviation_deg"] = (round(e["deviation_deg"], 1)
                                      if e.get("ok") else None)
        row[f"q{q}_coherence"] = (round(e["coherence"], 3)
                                  if e.get("ok") else None)
        flat_any = flat_any or bool(e.get("flat"))
    row["low_confidence"] = bool(
        rad.get("n_quadrants", 0) < len(_QUADRANT_ORDER) or flat_any
        or rad.get("mean_coherence", 0.0) < float(coherence_floor))
    return row


def _quadrant_decor(ax, fit, lw=2.2, reach=1.25):
    """Radial replacement for _heading_arrow + _sweep_axis: quadrant dividers,
    the ring centre, each quadrant's expected outward direction (crimson dashed,
    the same visual role the sweep axis plays for linear runs) and its fitted
    heading (black; grey when that quadrant's phase is flat)."""
    rad = fit.get("radial") or {}
    cx, cy = rad.get("centre_cells", _RADIAL_CENTRE_CELLS)
    ax.axvline(cx, color="#BBBBBB", lw=0.8, ls="--", zorder=1)
    ax.axhline(cy, color="#BBBBBB", lw=0.8, ls="--", zorder=1)
    ax.plot(cx, cy, "o", color="crimson", ms=4.5, zorder=7)
    for q in _QUADRANT_ORDER:
        e = (rad.get("quadrants") or {}).get(q, {})
        if "centroid_cells" not in e:
            continue
        mx, my = e["centroid_cells"]
        er = np.radians(e["expected_deg"])
        ax.annotate("", xy=(mx + reach * np.cos(er), my + reach * np.sin(er)),
                    xytext=(mx - reach * np.cos(er), my - reach * np.sin(er)),
                    arrowprops=dict(arrowstyle="-", lw=1.3, color="crimson",
                                    linestyle="--"), zorder=4)
        if not e.get("ok"):
            ax.text(mx, my, "no fit", ha="center", va="center", fontsize=6.5,
                    color="#888", style="italic", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                              alpha=0.8))
            continue
        hr = np.radians(e["direction_deg"])
        ax.annotate("", xy=(mx + reach * np.cos(hr), my + reach * np.sin(hr)),
                    xytext=(mx - reach * np.cos(hr), my - reach * np.sin(hr)),
                    arrowprops=dict(arrowstyle="-|>", lw=lw,
                                    color="#999" if e.get("flat") else "black",
                                    mutation_scale=13), zorder=6)


# ------------------------------------------------------------------ panel 1/3
def _draw_panel(ax, fit, sweep_deg, pitch):
    """Electrode map for one sample/region: dots coloured by RELATIVE lock
    strength (parula, each panel normalised to its own max), black arrow =
    recovered heading, red dashed = sweep axis."""
    ax.set_facecolor("#FAFAFA")
    ax.set_xlim(0.3, 8.7); ax.set_ylim(0.3, 8.7); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if _no_lock(ax, fit):
        return
    locks = fit["locks"]
    xs = np.array([l.x_um for l in locks]) / pitch
    ys = np.array([l.y_um for l in locks]) / pitch
    st, smax = _norm_strengths(locks)
    ax.scatter(xs, ys, c=st, cmap=PARULA, norm=_NORM01,
               s=90, edgecolors="k", linewidths=0.4, zorder=3)
    if fit.get("radial") is not None:          # radial run: quadrant model
        _quadrant_decor(ax, fit)
        return
    _heading_arrow(ax, fit)
    _sweep_axis(ax, sweep_deg)


# ------------------------------------------------------------------ panel 2/3
def _phase_field(locks, pitch, n=30, sigma_cells=1.15):
    """Gaussian-weighted interpolation of the COMPLEX resultant vectors onto a
    regular grid (wrap-safe: interpolating z, never the raw angle). Returns
    (gx, gy, Z) with Z complex; |Z| is a local locking/consistency amplitude and
    arg Z is the local preferred phase."""
    xs = np.array([l.x_um for l in locks]) / pitch
    ys = np.array([l.y_um for l in locks]) / pitch
    zz = np.array([l.z for l in locks])
    w = np.array([np.sqrt(l.n) for l in locks], float)
    gx = np.linspace(0.5, 8.5, n)
    gy = np.linspace(0.5, 8.5, n)
    X, Y = np.meshgrid(gx, gy)
    d2 = (X[..., None] - xs) ** 2 + (Y[..., None] - ys) ** 2
    K = np.exp(-d2 / (2.0 * sigma_cells ** 2)) * w
    den = np.maximum(K.sum(axis=-1), 1e-12)
    Z = (K * zz).sum(axis=-1) / den
    return gx, gy, Z


def _draw_flow_panel(ax, fit, sweep_deg, pitch):
    """Streamlines of the local phase GRADIENT — i.e. the direction in which the
    sweep phase advances across the array. Streamline colour = relative locking
    amplitude of the interpolated field (0-1, per-panel normalised)."""
    ax.set_facecolor("#FAFAFA")
    ax.set_xlim(0.3, 8.7); ax.set_ylim(0.3, 8.7); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if _no_lock(ax, fit):
        return
    locks = fit["locks"]
    gx, gy, Z = _phase_field(locks, pitch)
    # phase gradient without unwrapping:  grad(arg Z) = Im(conj(Z) * grad Z)/|Z|^2
    dZy, dZx = np.gradient(Z, gy, gx)
    den = np.abs(Z) ** 2 + 1e-12
    U = np.imag(np.conj(Z) * dZx) / den
    V = np.imag(np.conj(Z) * dZy) / den
    mag = np.abs(Z)
    mmax = float(mag.max()) if mag.size else 0.0
    magn = mag / mmax if mmax > 1e-12 else np.zeros_like(mag)
    # de-noise: where the field is essentially unlocked the gradient is garbage
    weak = magn < 0.12
    U = np.where(weak, np.nan, U); V = np.where(weak, np.nan, V)
    try:
        ax.streamplot(gx, gy, U, V, color=magn, cmap=PARULA, norm=_NORM01,
                      density=1.15, linewidth=1.1, arrowsize=0.9, zorder=3)
    except Exception:
        # streamplot is fussy about degenerate / all-NaN fields; fall back to a
        # plain quiver so the tile is still informative.
        s = 3
        ax.quiver(gx[::s], gy[::s], U[::s, ::s], V[::s, ::s], magn[::s, ::s],
                  cmap=PARULA, norm=_NORM01, zorder=3)
    ax.scatter(np.array([l.x_um for l in locks]) / pitch,
               np.array([l.y_um for l in locks]) / pitch,
               s=8, c="#333", alpha=0.35, zorder=2)
    if fit.get("radial") is not None:          # radial run: quadrant model
        _quadrant_decor(ax, fit, lw=2.0)
        return
    _heading_arrow(ax, fit, lw=2.0, reach=2.2)
    _sweep_axis(ax, sweep_deg)


# ------------------------------------------------------------------ panel 3/3
def _draw_compass_panel(ax, fit, sweep_deg, pitch):
    """Compass (polar) view of WHEN each electrode fires in the sweep cycle.
    angle = preferred phase (0 deg = cycle start, CCW), radius = relative lock
    strength (0-1, per-panel normalised), grey rose = phase histogram, thick
    arrow = population mean resultant vector. A tight angular cluster with a
    long resultant = the population locks but fires together (no propagation);
    phase spread around the circle = a travelling sweep."""
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_facecolor("#FAFAFA")
    ax.set_ylim(0.0, 1.06)
    ax.set_yticks([0.5, 1.0]); ax.set_yticklabels([])
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["0°", "90°", "180°", "270°"], fontsize=6.0, color="#777")
    # pull the degree labels inside the circle so they cannot collide with the
    # column titles above / the next row below
    ax.tick_params(axis="x", pad=-4.0)
    ax.grid(True, color="#CCC", lw=0.5)
    if _no_lock(ax, fit, polar=True):
        return
    locks = fit["locks"]
    th = np.array([l.phase for l in locks], float)
    st, smax = _norm_strengths(locks)
    # phase histogram (rose), scaled to the radial axis
    cnt, edges = np.histogram(th, bins=18, range=(-np.pi, np.pi))
    if cnt.max() > 0:
        ctr = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(ctr, cnt / cnt.max(), width=(2 * np.pi / 18) * 0.95, bottom=0.0,
               color="#B9C6D2", alpha=0.35, edgecolor="none", zorder=1)
    ax.scatter(th, st, c=st, cmap=PARULA, norm=_NORM01, s=42,
               edgecolors="k", linewidths=0.3, zorder=3)
    # population mean resultant (n-weighted), radius on the same 0-1 scale
    w = np.array([np.sqrt(l.n) for l in locks], float)
    zz = np.array([l.z for l in locks])
    R = (w * zz).sum() / max(w.sum(), 1e-12)
    rr = float(abs(R) / smax) if smax > 1e-12 else 0.0
    ax.annotate("", xy=(float(np.angle(R)), min(rr, 1.0)), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color="black",
                                mutation_scale=14), zorder=6)
    _panel_tag(ax, f"|R| = {abs(R):.2f}", polar=True)


# ------------------------------------------------------------------ grid driver
def _regions_of(samples):
    regions = [r for r in ("pre", "stim", "post")
               if any(r in res for _s, res, _a in samples)]
    extra = []
    for _s, res, _a in samples:
        for r in res:
            if r not in regions and r not in extra:
                extra.append(r)
    return regions + extra


def _day_strip(ax, i, nr, stem, age):
    """Left-hand sequential shade marking the training-day progression."""
    rgb = plt.get_cmap("Blues")(0.25 + 0.60 * (i / max(1, nr - 1)))
    ax.set_facecolor(rgb)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    m = _re.search(r"_D(\d+)", str(stem))
    day = m.group(1) if m else str(i + 1)
    agestr = f"age {age:g}" if age is not None else ""
    ax.text(0.5, 0.5, f"Day {day}\n{agestr}\n{stem}", rotation=90,
            ha="center", va="center", color=("white" if lum < 0.6 else "#111111"),
            fontsize=9, fontweight="bold", transform=ax.transAxes)


def _grid_figure(group, samples, p, out_dir, panel_fn, fname, suptitle,
                 cbar_label, polar=False):
    """Build the shared day-strip + sample x region grid and let `panel_fn` fill
    each tile. Rows are sorted YOUNGEST age -> oldest (top -> bottom = increasing
    days of sweep training). One 0-1 colour legend is attached per SAMPLE ROW and
    shared by that sample's region panels, because every panel is normalised to
    its own maximum."""
    sweep = p["sweep_angle_deg"]; pitch = p["pitch_um"]
    # youngest first (top); ties broken by stem name
    samples = sorted(samples, key=lambda t: (t[2] if t[2] is not None else 1e9, t[0]))
    regions = _regions_of(samples)
    if not regions or not samples:
        return
    nr, nc = len(samples), len(regions)
    fig = plt.figure(figsize=(3.0 * nc + 2.4, (3.3 if polar else 3.0) * nr),
                     constrained_layout=True)
    gs = fig.add_gridspec(nr, nc + 1, width_ratios=[0.20] + [1] * nc)
    for i, (stem, res, age) in enumerate(samples):
        _day_strip(fig.add_subplot(gs[i, 0]), i, nr, stem, age)
        row_axes = []
        for j, region in enumerate(regions):
            ax = fig.add_subplot(gs[i, j + 1], polar=polar)
            row_axes.append(ax)
            try:
                panel_fn(ax, res.get(region), sweep, pitch)
            except Exception as e:                       # never lose the figure
                ax.text(0.5, 0.5, f"panel failed\n{e!r}"[:60], fontsize=7,
                        ha="center", va="center", color="crimson",
                        transform=ax.transAxes)
            if i == 0:
                ax.set_title(region, fontsize=12, fontweight="bold",
                             pad=(12 if polar else 6))
        sm = _ScalarMappable(norm=_NORM01, cmap=PARULA); sm.set_array([])
        cb = fig.colorbar(sm, ax=row_axes, location="right",
                          fraction=0.04, pad=0.01)
        cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        cb.ax.tick_params(labelsize=6)
        cb.set_label(cbar_label, fontsize=7)
    fig.suptitle(suptitle, fontsize=11)
    _save(fig, os.path.join(out_dir, fname))


def _fig_group(group, samples, p, out_dir):
    """Electrode lock-strength map. samples : list of (stem, results, age).
    Radial groups get the quadrant alignment overlay in the same file."""
    sweep = p["sweep_angle_deg"]
    if _is_radial_group(group):
        title = (f"Radial phase lock-in — group {group}   (rows = training days, "
                 f"youngest at top; colour = lock strength normalised PER PANEL; "
                 f"black arrows = per-quadrant heading, red dashed = outward "
                 f"from ring centre)")
    else:
        title = (f"Sweep phase lock-in — group {group}   (rows = training days, youngest "
                 f"at top; colour = lock strength normalised PER PANEL; black arrow = "
                 f"heading, red dashed = sweep {sweep:.0f}° BR→TL)")
    _grid_figure(
        group, samples, p, out_dir, _draw_panel,
        f"phase_lockin_{group}.png", title,
        "relative lock strength (per-panel 0–1)")


def _fig_group_flow(group, samples, p, out_dir):
    """Streamline flow map of the phase gradient, same grid as _fig_group."""
    sweep = p["sweep_angle_deg"]
    if _is_radial_group(group):
        title = (f"Radial phase flow — group {group}   (rows = training days, "
                 f"youngest at top; streamlines follow the local phase gradient; "
                 f"black arrows = per-quadrant heading, red dashed = outward "
                 f"from ring centre)")
    else:
        title = (f"Sweep phase flow — group {group}   (rows = training days, youngest at "
                 f"top; streamlines follow the local phase gradient, colour = relative "
                 f"locking amplitude; black arrow = heading, red dashed = sweep "
                 f"{sweep:.0f}° BR→TL)")
    _grid_figure(
        group, samples, p, out_dir, _draw_flow_panel,
        f"phase_flow_{group}.png", title,
        "relative locking amplitude (per-panel 0–1)")


def _fig_group_compass(group, samples, p, out_dir):
    """Per-electrode preferred-phase compass, same grid as _fig_group."""
    _grid_figure(
        group, samples, p, out_dir, _draw_compass_panel,
        f"phase_compass_{group}.png",
        f"Sweep phase-lock compass — group {group}   (rows = training days, "
        f"youngest at top; angle = preferred phase in the sweep cycle, radius & "
        f"colour = lock strength normalised per panel, grey rose = phase "
        f"histogram, black arrow = mean resultant)",
        "relative lock strength (per-panel 0–1)",
        polar=True)


# ==============================================================================
# lock-on direction over time   ->  phase_direction_<group>.png
# ------------------------------------------------------------------------------
# The three region figures answer "where was the wave heading during pre / stim /
# post". This one answers "WHEN did the heading change", by re-running the SAME
# plane-wave fit inside consecutive direction_window_s bins and laying the result
# on a session clock. It is a read-only second pass over the same spike units:
# nothing here feeds back into the region fits, the summary CSV, or any existing
# parameter.
# ==============================================================================

# Epoch bands reuse the module's existing palette rather than introducing one:
# the same "Blues" ramp the day strip is drawn from, stepped light -> dark in
# pre -> stim -> post order (which is also their time order). PARULA stays
# reserved for the data itself, exactly as in the map / flow / compass panels,
# and the bands are additionally labelled in text so the epochs never rest on
# colour alone.
_BAND_FILL_STEP = {"pre": 0.06, "stim": 0.26, "post": 0.15}
_BAND_INK_STEP = {"pre": 0.74, "stim": 0.97, "post": 0.86}


def _region_ink(region, step_table=None, default=0.70):
    """A shade from the day strip's Blues ramp, keyed by epoch."""
    table = step_table if step_table is not None else _BAND_INK_STEP
    return plt.get_cmap("Blues")(table.get(str(region).strip().lower(), default))


def _region_fill(region):
    return _region_ink(region, _BAND_FILL_STEP, default=0.12)


def _wrap180(deg):
    """Map an angle (difference) into (-180, +180]."""
    return (np.asarray(deg, dtype=float) + 180.0) % 360.0 - 180.0


def _convergence_pct(deviation_deg):
    """% convergence on the target direction = 50 * (1 + cos(deviation)),
    i.e. the summary CSV's `alignment` column (cos of the deviation) rescaled
    from [-1, +1] to [0, 100] %. On target = 100 %, perpendicular = 50 %,
    opposite = 0 %; forgiving near the target, steep near opposite. Which SIDE
    of the target a heading falls on is folded away (+90 and -90 are both
    50 %)."""
    return 50.0 * (1.0 + np.cos(np.radians(np.asarray(deviation_deg, float))))


def _direction_timecourse(by_region, results, p, seed_T, radial=False):
    """Re-fit the plane wave in consecutive bins across the whole session.

    Returns {"bins": [...], "bands": [(t0_s, t1_s, region), ...], "window_s": n}
    in ABSOLUTE file time, taken from each unit's own ``win`` so the epochs land
    where they really sit in the recording rather than end to end.

    Each bin dict carries t_mid_s, direction_deg, coherence, n_locked, region.
    A bin that cannot be fitted is simply absent -- fit_plane_wave needs six
    locked electrodes, and a sparse minute legitimately has fewer. Their absence
    is what breaks the drawn line instead of bridging a gap that has no data.

    The period is taken from that region's own completed fit, so the binning
    inherits the refined period rather than re-deriving one per bin (a 60 s bin
    rarely holds enough cycles to refine a period honestly).

    radial=True (radial groups): each bin runs the quadrant alignment model
    instead. Each fitted quadrant is scored as % convergence on ITS outward
    direction and the bin value is the mean over quadrants -- averaging the
    convergences (not the signed deviations) so quadrants that err in opposite
    directions cannot cancel into a false 100 %. A bin needs at least
    _RADIAL_MIN_QUADRANTS fitted quadrants to count.
    """
    win_s = float(p.get("direction_window_s", 60.0) or 0.0)
    if win_s <= 0:
        return None
    pitch = p["pitch_um"]
    min_spikes = p["min_spikes"]

    bins, bands = [], []
    for region, u in by_region.items():
        t_off = float((u.get("win") or (0.0, 0.0))[0])
        dur = float(u.get("duration") or 0.0)
        if dur <= 0:
            continue
        bands.append((t_off, t_off + dur, region))

        fit = results.get(region) or {}
        period = fit.get("period_s") or seed_T
        if not period or float(period) <= 0:
            continue
        period = float(period)

        spike_s = {u["labels"][i]: np.asarray(u["spikes"][i], float)
                   for i in range(len(u["labels"]))}
        label_xy = _label_xy(u["labels"], pitch)

        for b in range(int(np.floor(dur / win_s))):
            b0, b1 = b * win_s, (b + 1) * win_s
            locks = electrode_locking(spike_s, b0, b1, period, label_xy,
                                      min_spikes)
            if radial:
                qa = quadrant_alignment(locks, period, pitch,
                                        p.get("min_phase_span_rad", 0.6))
                if not qa["ok"]:
                    continue
                bins.append({"t_mid_s": t_off + 0.5 * (b0 + b1),
                             "direction_deg": qa["mean_deviation_deg"],
                             "convergence_pct": qa["mean_convergence_pct"],
                             "coherence": qa["mean_coherence"],
                             "n_locked": len(locks),
                             "n_quadrants": qa["n_quadrants"],
                             "region": region})
                continue
            pw = fit_plane_wave(locks, period, pitch)
            if pw is None:
                continue
            bins.append({"t_mid_s": t_off + 0.5 * (b0 + b1),
                         "direction_deg": float(pw["direction_deg"]),
                         "convergence_pct": float(_convergence_pct(
                             pw["direction_deg"] - float(p["sweep_angle_deg"]))),
                         "coherence": float(pw["coherence"]),
                         "n_locked": int(pw["n_electrodes"]),
                         "region": region})

    bands.sort(key=lambda t: t[0])
    bins.sort(key=lambda d: d["t_mid_s"])
    return {"bins": bins, "bands": bands, "window_s": win_s,
            "mode": "radial" if radial else "linear"}


def _draw_direction_axes(ax, series, p, label_bands=False):
    """One sample's convergence trace. Returns the row's max coherence.
    y = % convergence on the training direction: the sweep angle for linear
    runs (135 deg -> 100 %, -45 deg -> 0 %), each quadrant's outward direction
    for radial runs (mean over quadrants)."""
    radial = (series or {}).get("mode") == "radial"
    sweep = float(p["sweep_angle_deg"])
    ax.set_facecolor("#FAFAFA")
    ax.tick_params(labelsize=8, colors="#555")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#CCCCCC")

    bands = (series or {}).get("bands") or []
    bins = (series or {}).get("bins") or []
    if not bands:
        ax.text(0.5, 0.5, "no windows", ha="center", va="center", fontsize=9,
                color="#888", style="italic", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        return 0.0

    t_lo = min(b[0] for b in bands) / 60.0
    t_hi = max(b[1] for b in bands) / 60.0
    ax.set_xlim(t_lo, t_hi)
    ax.set_ylim(-4.0, 114.0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("convergence on training angle", fontsize=8.5, color="#555")
    ax.grid(axis="y", color="#E2E2E2", lw=0.7, zorder=0)
    ax.set_axisbelow(True)

    # epoch shading + direct labels
    for (bt0, bt1, region) in bands:
        ax.axvspan(bt0 / 60.0, bt1 / 60.0, color=_region_fill(region), lw=0,
                   zorder=1)
        ax.axvline(bt0 / 60.0, color="#BFCBD8", lw=0.8, zorder=2)
        if label_bands:
            ax.text(0.5 * (bt0 + bt1) / 60.0, 112.0, str(region),
                    ha="center", va="top", fontsize=8.5, fontweight="bold",
                    color=_region_ink(region), zorder=5)

    # 100 % = on the training direction, same crimson dashed language as the
    # sweep axis in the map panels
    ax.axhline(100.0, color="crimson", lw=1.2, ls="--", alpha=0.85, zorder=3)
    ax.text(t_hi, 100.0, (" outward" if radial else f" sweep {sweep:.0f}°"),
            fontsize=7, color="crimson", va="bottom", ha="right", zorder=5)

    if not bins:
        ax.text(0.5, 0.5, "no bin reached a plane-wave fit",
                ha="center", va="center", fontsize=9, color="#888",
                style="italic", transform=ax.transAxes, zorder=6)
        return 0.0

    t = np.array([b["t_mid_s"] for b in bins], float) / 60.0
    # % convergence is continuous in heading (no +-180 seam), so no unwrapping
    y = np.array([b["convergence_pct"] for b in bins], float)
    coh = np.array([b["coherence"] for b in bins], float)
    reg = [b["region"] for b in bins]

    cmax = float(coh.max()) if coh.size else 0.0
    cn = coh / cmax if cmax > 1e-12 else np.zeros_like(coh)

    # break the line across epoch changes and across dropped bins
    win_min = float(series.get("window_s", 60.0)) / 60.0
    yline = y.astype(float).copy()
    tline = t.astype(float).copy()
    seg_t, seg_y = [tline[0]], [yline[0]]
    for i in range(1, len(tline)):
        gap = (tline[i] - tline[i - 1]) > 1.6 * win_min
        if gap or reg[i] != reg[i - 1]:
            seg_t.append(np.nan); seg_y.append(np.nan)
        seg_t.append(tline[i]); seg_y.append(yline[i])
    if all("sem" in b for b in bins):
        # sample-averaged rows: +-1 SEM ribbon, broken at the same gaps/epochs
        sem = np.array([b["sem"] for b in bins], float)
        rt, rlo, rhi = [t[0]], [y[0] - sem[0]], [y[0] + sem[0]]
        for i in range(1, len(t)):
            if (t[i] - t[i - 1]) > 1.6 * win_min or reg[i] != reg[i - 1]:
                rt.append(np.nan); rlo.append(np.nan); rhi.append(np.nan)
            rt.append(t[i]); rlo.append(y[i] - sem[i]); rhi.append(y[i] + sem[i])
        ax.fill_between(rt, np.clip(rlo, 0, 100), np.clip(rhi, 0, 100),
                        color="#7A8694", alpha=0.22, lw=0, zorder=3.5)
    ax.plot(seg_t, seg_y, color="#7A8694", lw=1.6, zorder=4, solid_capstyle="round")
    ax.scatter(t, y, c=cn, cmap=PARULA, norm=_NORM01, s=34, zorder=5,
               edgecolors="white", linewidths=0.6)
    return cmax


def _fig_group_direction(group, samples, p, out_dir):
    """Heading-vs-time rows. samples : list of (stem, series, age)."""
    samples = [t for t in samples if t[1] and (t[1].get("bands"))]
    samples = sorted(samples,
                     key=lambda t: (t[2] if t[2] is not None else 1e9, t[0]))
    if not samples:
        return
    nr = len(samples)
    win_s = float(p.get("direction_window_s", 60.0) or 60.0)
    fig = plt.figure(figsize=(13.0, 2.7 * nr + 0.9), constrained_layout=True)
    gs = fig.add_gridspec(nr, 2, width_ratios=[0.18, 1])
    for i, (stem, series, age) in enumerate(samples):
        _day_strip(fig.add_subplot(gs[i, 0]), i, nr, stem, age)
        ax = fig.add_subplot(gs[i, 1])
        try:
            _draw_direction_axes(ax, series, p, label_bands=(i == 0))
        except Exception as e:                       # never lose the figure
            ax.text(0.5, 0.5, f"row failed\n{e!r}"[:70], fontsize=7,
                    ha="center", va="center", color="crimson",
                    transform=ax.transAxes)
        if i == nr - 1:
            ax.set_xlabel("session time (min)", fontsize=9, color="#555")
        sm = _ScalarMappable(norm=_NORM01, cmap=PARULA); sm.set_array([])
        cb = fig.colorbar(sm, ax=[ax], location="right", fraction=0.025,
                          pad=0.01)
        cb.set_ticks([0.0, 0.5, 1.0])
        cb.ax.tick_params(labelsize=6)
        cb.set_label("coherence (per-row 0–1)", fontsize=7)
    if _is_radial_group(group):
        title = (f"Radial lock-on convergence over time — group {group}   (rows = "
                 f"training days, youngest at top; mean % convergence of the four "
                 f"quadrants on outward, re-fitted every {win_s:g} s; 100% = "
                 f"outward, 0% = inward)")
    else:
        title = (f"Lock-on convergence over time — group {group}   (rows = training "
                 f"days, youngest at top; heading re-fitted every {win_s:g} s; 100% "
                 f"= sweep {float(p['sweep_angle_deg']):.0f}° BR→TL, 0% = opposite; "
                 f"gaps = bins with too few locked electrodes)")
    fig.suptitle(title, fontsize=11)
    _save(fig, os.path.join(out_dir, f"phase_direction_{group}.png"))


# ==============================================================================
# sample-averaged lock-on convergence  ->  phase_direction_mean_<condition>.png
# ------------------------------------------------------------------------------
# One row per training day; each row is the MEAN convergence curve across the
# replicate samples recorded that day, with a +-1 SEM ribbon. Replicates are
# found whichever way Batch.csv encodes them:
#   * one group per replicate  (sweep1 / sweep2 / sweep3)  -> condition "sweep"
#   * one group per condition, replicate in the file name  (4000um, with
#     4000um_1_D1 / 4000um_2_D1 ...)                       -> condition "4000um"
# Bins are aligned on (epoch, bin index within the epoch) rather than raw clock
# time, so replicates whose pre / stim / post windows start a few seconds apart
# still average stim-onset with stim-onset.
# ==============================================================================
def _condition_of(group):
    """sweep1 -> sweep, radial_2 -> radial; 4000um / Control stay as they are."""
    g = str(group or "").strip()
    base = _re.sub(r"[\s_-]*\d+$", "", g)
    return base or g


def _training_day(stem, age):
    """Day index from the '_D<n>' part of the file name, else the age."""
    m = _re.search(r"_D(\d+)", str(stem))
    if m:
        return int(m.group(1))
    return float(age) if age is not None else None


def _mean_timecourse(members, win_s):
    """Average several samples' convergence series bin-by-bin."""
    acc, t0s, durs, mode = {}, {}, {}, "linear"
    for sr in members:
        mode = sr.get("mode", mode)
        starts = {}
        for (bt0, bt1, region) in sr["bands"]:
            starts[region] = bt0
            t0s.setdefault(region, []).append(bt0)
            durs.setdefault(region, []).append(bt1 - bt0)
        for b in sr["bins"]:
            r = b["region"]
            if r not in starts:
                continue
            k = int(round((b["t_mid_s"] - starts[r]) / win_s - 0.5))
            acc.setdefault((r, k), []).append((b["convergence_pct"],
                                                b["coherence"]))
    regions = sorted(t0s, key=lambda r: float(np.median(t0s[r])))
    bands = [(float(np.median(t0s[r])),
              float(np.median(t0s[r]) + np.median(durs[r])), r) for r in regions]
    span = {r: (a, b) for (a, b, r) in bands}
    bins = []
    for (r, k), vals in acc.items():
        a, b = span[r]
        tm = a + (k + 0.5) * win_s
        if not (a <= tm <= b):
            continue
        conv = np.array([v[0] for v in vals], float)
        coh = np.array([v[1] for v in vals], float)
        n = conv.size
        bins.append({"t_mid_s": tm, "region": r, "n": n,
                     "convergence_pct": float(conv.mean()),
                     "sem": (float(conv.std(ddof=1) / np.sqrt(n)) if n > 1
                             else 0.0),
                     "coherence": float(coh.mean())})
    bins.sort(key=lambda d: d["t_mid_s"])
    return {"bins": bins, "bands": bands, "window_s": win_s, "mode": mode,
            "n_samples": len(members)}


def _mean_day_strip(ax, i, nr, day, n, ages):
    """Same look as _day_strip, labelled for an averaged row."""
    rgb = plt.get_cmap("Blues")(0.25 + 0.60 * (i / max(1, nr - 1)))
    ax.set_facecolor(rgb)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    ages = sorted({a for a in ages if a is not None})
    agestr = ("" if not ages else f"age {ages[0]:g}" if len(ages) == 1
              else f"age {ages[0]:g}\u2013{ages[-1]:g}")
    label = f"Day {day}" if isinstance(day, int) else f"age {day:g}"
    ax.text(0.5, 0.5, f"{label}\n{agestr}\nmean of n={n}", rotation=90,
            ha="center", va="center", color=("white" if lum < 0.6 else "#111111"),
            fontsize=9, fontweight="bold", transform=ax.transAxes)


def _fig_direction_means(dir_samples, p, out_dir):
    """Write one averaged figure per condition that has >= 2 replicates on at
    least one day. Returns the number of figures written."""
    win_s = float(p.get("direction_window_s", 60.0) or 60.0)
    by_cond = {}
    for grp, items in dir_samples.items():
        cond = _condition_of(grp)
        for stem, series, age in items:
            if not series or not series.get("bands"):
                continue
            day = _training_day(stem, age)
            by_cond.setdefault(cond, {}).setdefault(day, []).append(
                (series, age))
    written = 0
    for cond, days in by_cond.items():
        if max(len(v) for v in days.values()) < 2:
            continue                                  # nothing to average
        order = sorted(days, key=lambda d: (isinstance(d, float), d))
        nr = len(order)
        fig = plt.figure(figsize=(13.0, 2.7 * nr + 0.9), constrained_layout=True)
        gs = fig.add_gridspec(nr, 2, width_ratios=[0.18, 1])
        for i, day in enumerate(order):
            members = days[day]
            mean = _mean_timecourse([m[0] for m in members], win_s)
            _mean_day_strip(fig.add_subplot(gs[i, 0]), i, nr, day,
                            len(members), [m[1] for m in members])
            ax = fig.add_subplot(gs[i, 1])
            try:
                _draw_direction_axes(ax, mean, p, label_bands=(i == 0))
            except Exception as e:                   # never lose the figure
                ax.text(0.5, 0.5, f"row failed\n{e!r}"[:70], fontsize=7,
                        ha="center", va="center", color="crimson",
                        transform=ax.transAxes)
            if i == nr - 1:
                ax.set_xlabel("session time (min)", fontsize=9, color="#555")
            sm = _ScalarMappable(norm=_NORM01, cmap=PARULA); sm.set_array([])
            cb = fig.colorbar(sm, ax=[ax], location="right", fraction=0.025,
                              pad=0.01)
            cb.set_ticks([0.0, 0.5, 1.0])
            cb.ax.tick_params(labelsize=6)
            cb.set_label("mean coherence (per-row 0\u20131)", fontsize=7)
        target = ("outward" if _is_radial_group(cond) else
                  f"sweep {float(p['sweep_angle_deg']):.0f}\u00b0 BR\u2192TL")
        fig.suptitle(
            f"Lock-on convergence over time \u2014 {cond}, mean of samples   "
            f"(rows = training days; line = mean across samples, shaded = "
            f"\u00b11 SEM; re-fitted every {win_s:g} s; 100% = {target})",
            fontsize=11)
        _save(fig, os.path.join(out_dir, f"phase_direction_mean_{cond}.png"))
        written += 1
    return written


# ==============================================================================
# driver  --  called by the GUI (do4) after Stage 2 / SpKit
# ==============================================================================
def _period_seed_for_stem(stem, p):
    """Sweep-period seed (s) for one recording: explicit override, else derived
    from the laser trigger rate (frames_per_sweep / trigger_hz); else None."""
    if p.get("period_s"):
        return float(p["period_s"])
    if not p.get("use_digital_events"):
        return None
    fpath = os.path.join(io.CONFIG["DATA_DIR"], stem + ".h5")
    if not os.path.exists(fpath):
        return None
    try:
        ev = de.read_digital_events(fpath, io.CONFIG.get("STREAM_INDEX"))
        m = de.sweep_metrics(ev)
        thz = m.get("trigger_hz")
        fps = p.get("frames_per_sweep")
        if thz and np.isfinite(thz) and thz > 0 and fps and fps > 0:
            seed = float(fps) / float(thz)
            # frames_per_sweep is the 160-frame SWEEP model; a flood/simultaneous
            # control fires ONE trigger per event, so fps*ISI overshoots to a period
            # longer than the recording (e.g. 160 * 5 s = 800 s). Reject anything
            # implausibly long and let lockin_window fall back to population-rate
            # autocorrelation, which recovers the true repeat period (e.g. 5 s flood).
            if seed > float(p.get("max_trigger_period_s", 120.0)):
                io._log(f"  [PhaseLock] {stem}: trigger period {seed:.0f}s implausible "
                        f"(flood/sparse triggers?) — using autocorrelation instead")
                return None
            return seed
    except Exception:
        pass
    return None


def run_phase_lockin():
    """Run the lock-in sweep-propagation analysis for every Batch.csv row that is
    enabled for analysis, writing to the 'Phase Lock' folder under PLOT_DIR."""
    from collections import OrderedDict
    p = PHASE_LOCKIN
    csv = os.path.join(io.CONFIG["DATA_DIR"], io.CONFIG.get("BATCH_CSV_NAME", "Batch.csv"))
    if not os.path.exists(csv):
        io._log(f"[PhaseLock] Batch.csv not found at {csv}; nothing to do.")
        return
    rows = io.filter_rows_by_group(io.parse_batch(csv), "analyze")
    if not rows:
        io._log("[PhaseLock] no rows to analyse (all groups disabled for analysis).")
        return
    root = phase_lock_root(io.CONFIG["PLOT_DIR"])
    io._log(f"[PhaseLock] sweep phase lock-in -> {root}")

    by_stem = OrderedDict()
    for unit in io.iter_batch_units(rows):
        by_stem.setdefault(unit["stem"], []).append(unit)
    if not by_stem:
        io._log("[PhaseLock] no spike units (run Stage 1 detection first).")
        return

    gui = progress_gui.make(list(by_stem.keys()), "MEA Analyzer — Phase Lock-In",
                            "Sweep phase lock-in")
    summaries = []
    group_samples = OrderedDict()            # group -> [(stem, results-dict), ...]
    dir_samples = OrderedDict()              # group -> [(stem, series, age), ...]
    try:
        for stem, units in by_stem.items():
            io.check_abort()
            gui.start(stem, "reading laser triggers...")
            seed_T = _period_seed_for_stem(stem, p)
            is_radial = _is_radial_group(units[0].get("group", "") if units else "")
            by_region = {}
            for u in units:                       # last window per region wins
                by_region[u["region"]] = u
            # stim first so its refined period seeds pre/post (carry-over prior)
            order = [r for r in ("stim",) if r in by_region]
            order += [r for r in by_region if r not in order]
            results = {}
            carry_T = None
            for k, region in enumerate(order):
                u = by_region[region]
                gui.set(stem, (k + 0.5) / max(1, len(order)),
                        note=f"lock-in {region}...")
                spike_s = {u["labels"][i]: np.asarray(u["spikes"][i], float)
                           for i in range(len(u["labels"]))}
                label_xy = _label_xy(u["labels"], p["pitch_um"])
                seed = carry_T or seed_T
                try:
                    fit = lockin_window(spike_s, float(u["duration"]), label_xy, p,
                                        seed_period=seed)
                except Exception as e:
                    fit = {"ok": False, "reason": f"error:{e!r}"[:40]}
                if is_radial and fit.get("ok"):
                    try:
                        fit["radial"] = quadrant_alignment(
                            fit["locks"], fit["period_s"], p["pitch_um"],
                            p.get("min_phase_span_rad", 0.6))
                    except Exception as e:
                        io._log(f"  [PhaseLock] {stem} [{region}]: quadrant "
                                f"model failed ({e!r})")
                results[region] = fit
                if region == "stim" and fit.get("ok"):
                    carry_T = fit["period_s"]
                if fit.get("ok"):
                    al = float(np.cos(np.radians(fit["direction_deg"]
                                                 - p["sweep_angle_deg"])))
                    summaries.append({
                        "stem": stem, "region": region, "group": u.get("group", ""),
                        "direction_deg": round(fit["direction_deg"], 1),
                        "speed_um_per_s": round(fit["speed_um_per_s"], 1),
                        "coherence": round(fit["coherence"], 3),
                        "period_s": round(fit["period_s"], 3),
                        "peak_ratio": (round(fit["peak_ratio"], 2)
                                       if fit.get("peak_ratio") is not None else None),
                        "alignment": round(al, 3),
                        "n_locked": fit["n_electrodes"],
                        "phase_span_rad": round(float(fit.get("phase_span_rad", 0.0)), 3),
                        "propagation": (not bool(fit.get("no_propagation", False))),
                        "p_value": fit.get("p_value"),
                        "low_confidence": bool(fit.get("low_confidence", False)),
                    })
                    if fit.get("radial") is not None:
                        summaries[-1].update(_radial_summary_fields(
                            fit["radial"], p.get("coherence_floor", 0.10)))
            try:
                gui.set(stem, 0.97, note="heading over time...")
                dseries = _direction_timecourse(by_region, results, p, seed_T,
                                                radial=is_radial)
            except Exception as e:
                dseries = None
                io._log(f"  [PhaseLock] {stem}: direction time course failed ({e!r})")

            grp = units[0].get("group", "") if units else ""
            age = units[0].get("age") if units else None
            group_samples.setdefault(grp, []).append((stem, results, age))
            dir_samples.setdefault(grp, []).append((stem, dseries, age))
            okregs = [r for r, f in results.items() if f.get("ok")]
            gui.done(stem, f"{len(okregs)}/{len(results)} region(s) locked"
                     if okregs else "no lock")
    finally:
        gui.close()

    # three combined figures per group (rows = samples/training days,
    # cols = pre/stim/post): lock-strength map, phase-gradient streamline flow
    # map, and preferred-phase compass
    n_figs = 0
    for grp, samples in group_samples.items():
        for name, fn in (("lock map", _fig_group),
                         ("flow map", _fig_group_flow),
                         ("compass", _fig_group_compass)):
            try:
                fn(grp, samples, p, root)
                n_figs += 1
            except Exception as e:
                io._log(f"  [PhaseLock] group {grp}: {name} figure failed ({e!r})")
        try:
            _fig_group_direction(grp, dir_samples.get(grp, []), p, root)
            n_figs += 1
        except Exception as e:
            io._log(f"  [PhaseLock] group {grp}: direction figure failed ({e!r})")
    try:
        n_figs += _fig_direction_means(dir_samples, p, root)
    except Exception as e:
        io._log(f"  [PhaseLock] sample-averaged direction figure failed ({e!r})")
    if any(r.get("pattern") == "radial" for r in summaries):
        for r in summaries:
            r.setdefault("pattern", "linear")
    _write_summary(summaries, root)
    io._log(f"[PhaseLock] done - {len(summaries)} region window(s) analysed, "
            f"{n_figs} figure(s) across {len(group_samples)} group(s).")


def _write_summary(summaries, root):
    if not summaries:
        return
    try:
        import pandas as pd
        os.makedirs(root, exist_ok=True)
        pd.DataFrame(summaries).to_csv(
            os.path.join(root, "phase_lockin_summary.csv"), index=False)
    except Exception as e:
        io._log(f"  [PhaseLock] summary CSV failed ({e!r})")


# ==============================================================================
# SELF-TEST  --  synthetic periodic sweep travelling BR->TL (135 deg)
# ==============================================================================
def _synthetic_region(prop_angle=135.0, period_s=2.0, dur_s=120.0, cv_um_s=None,
                      seed=0, locked=True):
    """Spikes: each electrode fires once per sweep cycle at a phase set by its
    position projected on prop_angle, plus Poisson background. locked=False gives
    background only (Pre/Post-like)."""
    rng = np.random.default_rng(seed)
    labels = orig._mcs60_labels()
    pitch = PHASE_LOCKIN["pitch_um"]
    pos = {l: np.array(orig.mcs_coord(l)) * pitch for l in labels}
    dvec = np.array([np.cos(np.radians(prop_angle)), np.sin(np.radians(prop_angle))])
    proj0 = min(float(pos[l] @ dvec) for l in labels)
    span = max(float(pos[l] @ dvec) for l in labels) - proj0
    spikes = {}
    n_cycles = int(dur_s / period_s)
    for l in labels:
        s = []
        if locked:
            frac = (float(pos[l] @ dvec) - proj0) / (span or 1.0)   # 0..1 along sweep
            t_in_cycle = frac * period_s * 0.8 + 0.05 * period_s
            for c in range(n_cycles):
                s.append(c * period_s + t_in_cycle + rng.normal(0, 0.02 * period_s))
        # Poisson background at ~2 Hz
        s.extend(rng.uniform(0, dur_s, int(2.0 * dur_s)))
        spikes[l] = np.sort(np.array(s, float))
    return spikes, dur_s


def _synthetic_radial_region(period_s=2.0, dur_s=120.0, seed=0, locked=True,
                             outward=True):
    """Radial analogue of _synthetic_region: each electrode fires once per cycle
    at a phase set by its distance from the ring centre (between 44/45/54/55),
    plus the same ~2 Hz Poisson background. outward=False reverses it."""
    rng = np.random.default_rng(seed)
    labels = orig._mcs60_labels()
    pitch = PHASE_LOCKIN["pitch_um"]
    c = np.array(_RADIAL_CENTRE_CELLS) * pitch
    pos = {l: np.array(orig.mcs_coord(l)) * pitch for l in labels}
    r = {l: float(np.hypot(*(pos[l] - c))) for l in labels}
    r0, r1 = min(r.values()), max(r.values())
    spikes = {}
    n_cycles = int(dur_s / period_s)
    for l in labels:
        s = []
        if locked:
            frac = (r[l] - r0) / ((r1 - r0) or 1.0)          # 0 centre .. 1 rim
            if not outward:
                frac = 1.0 - frac
            t_in_cycle = frac * period_s * 0.8 + 0.05 * period_s
            for cyc in range(n_cycles):
                s.append(cyc * period_s + t_in_cycle
                         + rng.normal(0, 0.02 * period_s))
        s.extend(rng.uniform(0, dur_s, int(2.0 * dur_s)))
        spikes[l] = np.sort(np.array(s, float))
    return spikes, dur_s


if __name__ == "__main__":
    out = os.path.join(os.getcwd(), "_phaselock_selftest", "4000um_2_D3")
    p = PHASE_LOCKIN
    print("phase lock-in self-test (synthetic BR->TL sweep, T=2.0s):")
    results = {}
    by_region = {}
    t_cursor = 0.0
    for region, ang, locked, sd in [("pre", 40, False, 1),
                                    ("stim", 135, True, 2),
                                    ("post", 135, True, 3)]:
        spk, dur = _synthetic_region(prop_angle=ang, period_s=2.0, seed=sd,
                                     locked=locked)
        lxy = _label_xy(list(spk.keys()), p["pitch_um"])
        fit = lockin_window(spk, dur, lxy, p, seed_period=2.0)
        results[region] = fit
        labels = list(spk.keys())
        by_region[region] = {"labels": labels,
                             "spikes": [spk[l] for l in labels],
                             "duration": dur, "win": (t_cursor, t_cursor + dur)}
        t_cursor += dur
        if fit.get("ok"):
            al = np.cos(np.radians(fit["direction_deg"] - p["sweep_angle_deg"]))
            print(f"  {region:4s}: dir={fit['direction_deg']:+.0f} deg "
                  f"v={fit['speed_um_per_s']:,.0f} um/s coh={fit['coherence']:.2f} "
                  f"PR={fit['peak_ratio']:.1f} T={fit['period_s']:.2f}s align={al:+.2f}")
        else:
            print(f"  {region:4s}: no lock ({fit.get('reason')})")
    samples = [("4000um_2_D3", results, 15),
               ("4000um_2_D1", results, 13),
               ("4000um_2_D2", results, 14)]
    _fig_group("4000um", samples, p, out)
    _fig_group_flow("4000um", samples, p, out)
    _fig_group_compass("4000um", samples, p, out)

    import time as _time
    _t = _time.time()
    series = _direction_timecourse(by_region, results, p, 2.0)
    nb = len(series["bins"]) if series else 0
    print(f"  direction time course: {nb} bin(s) fitted of "
          f"{int(t_cursor // float(p['direction_window_s']))} attempted "
          f"({_time.time() - _t:.1f}s)")
    _fig_group_direction("4000um", [("4000um_2_D3", series, 15),
                                    ("4000um_2_D1", series, 13),
                                    ("4000um_2_D2", series, 14)], p, out)
    print("SELFTEST done ->", out)

    # ---------------- radial (centre-out) self-test -------------------------
    rout = os.path.join(os.getcwd(), "_phaselock_selftest", "radial1_D1")
    print("radial self-test (synthetic centre-out ring, T=2.0s):")
    rres, rby = {}, {}
    t_cursor = 0.0
    for region, locked, sd in [("pre", False, 21), ("stim", True, 22),
                               ("post", True, 23)]:
        spk, dur = _synthetic_radial_region(period_s=2.0, seed=sd, locked=locked)
        lxy = _label_xy(list(spk.keys()), p["pitch_um"])
        fit = lockin_window(spk, dur, lxy, p, seed_period=2.0)
        if fit.get("ok"):
            fit["radial"] = quadrant_alignment(fit["locks"], fit["period_s"],
                                               p["pitch_um"],
                                               p["min_phase_span_rad"])
            ra = fit["radial"]
            qs = "  ".join(
                f"{q}:{ra['quadrants'][q]['deviation_deg']:+4.0f}"
                if ra["quadrants"][q].get("ok") else f"{q}: n/a"
                for q in _QUADRANT_ORDER)
            print(f"  {region:4s}: quadrants={ra['n_quadrants']}  "
                  f"mean dev={ra.get('mean_deviation_deg', float('nan')):+5.1f} deg  "
                  f"radial align={ra.get('radial_alignment', float('nan')):+.2f}  "
                  f"[{qs}]")
        else:
            print(f"  {region:4s}: no lock ({fit.get('reason')})")
        rres[region] = fit
        labels = list(spk.keys())
        rby[region] = {"labels": labels, "spikes": [spk[l] for l in labels],
                       "duration": dur, "win": (t_cursor, t_cursor + dur)}
        t_cursor += dur
    rsamples = [("radial1_D1", rres, 13), ("radial1_D2", rres, 14)]
    _fig_group("radial1", rsamples, p, rout)
    _fig_group_flow("radial1", rsamples, p, rout)
    _fig_group_compass("radial1", rsamples, p, rout)
    rseries = _direction_timecourse(rby, rres, p, 2.0, radial=True)
    print(f"  radial time course: {len(rseries['bins'])} bin(s) fitted")
    _fig_group_direction("radial1", [("radial1_D1", rseries, 13),
                                     ("radial1_D2", rseries, 14)], p, rout)
    print("RADIAL SELFTEST done ->", rout)
