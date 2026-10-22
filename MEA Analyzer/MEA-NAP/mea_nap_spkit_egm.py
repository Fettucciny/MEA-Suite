#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 mea_nap_spkit_egm.py  --  spkit continuous-EGM conduction-velocity pipeline
================================================================================
An OPTIONAL third stage that runs the *actual* spkit `bsp_mea` continuous-signal
pipeline on the raw MCS voltage traces (not on our detected spikes), as an
independent, method-diverse cross-check of the spike-based propagation analysis.
Output goes to a NEW folder beside Plots/ called  "SpKit"  (SpKit/<stem>/<region>/).

Per Batch.csv row (one region window of one recording) it:
  1. reads the raw calibrated voltage matrix for the window (via mea_nap_io.MCSFile),
  2. detects network-event onsets from the population signal,
  3. averages the per-channel voltage segments aligned on those onsets -> one clean
     representative electrogram (EGM) per electrode  (spkit's align-cycles idea,
     driven by spontaneous/evoked network events instead of a periodic stimulus),
  4. runs spkit.mea.activation_time_loc  (activation = max negative dV/dt),
  5. arranges activation times on the 8x8 MEA grid, interpolates gaps
     (spkit.fill_nans_2d) and runs spkit.mea.compute_cv  -> conduction-velocity
     field + mean direction, exactly the CinC-2025 / spkit method,
  6. writes an activation-time map, a CV vector field, and CV_df.csv per region,
     plus a per-recording pre/stim/post summary.

spkit is imported LAZILY (only when the pipeline actually runs) so importing this
module — e.g. from the GUI — never fails if spkit is not installed. Enable it from
the GUI "SpKit EGM" tab (SPKIT_EGM["enable"]) for an overnight run.
================================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import re as _re

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


def _sweep_suffix(stem):
    """Suptitle fragment noting the nominal laser-sweep speed parsed from the
    filename prefix (e.g. '4000um...' -> 4000 µm/s), or '' for Control names.
    The prefix is the ground-truth speed the laser bar swept across the sample,
    set by the Polygon trigger rate — independent of electrode pitch / neural CV."""
    m = _re.match(r"\s*(\d+(?:\.\d+)?)\s*um", str(stem), _re.IGNORECASE)
    if not m:
        return ""
    v = float(m.group(1))
    lab = (f"{v / 1000.0:.3g} mm/s ({v:.0f} µm/s)" if v >= 1000
           else f"{v:.0f} µm/s")
    return f"  ·  nominal laser sweep {lab}"

# ==============================================================================
# PARAMETERS  (adjustment ranges noted; exposed in the GUI "SpKit EGM" tab)
# ==============================================================================
SPKIT_EGM = {
    # enable : master switch. When True the GUI runs this pipeline after Stage 2.
    "enable":               False,
    # use_digital_events : if True, use the laser ON times from the MCS Digital
    #   Event Recorder as the event onsets (stimulus-locked) whenever the window
    #   contains them; fall back to population detection when it doesn't (e.g. the
    #   pre/post spontaneous regions). True | False.
    "use_digital_events":   True,
    # digital_lag_ms : shift applied to each laser ON time before cutting the EGM
    #   segment (ms). 0 = align on the trigger; positive = start later. Range -20..50.
    "digital_lag_ms":       0.0,
    # --- event detection on the continuous population signal (fallback) ---
    # evt_bp_low_hz / evt_bp_high_hz : band used to build the population activity
    #   envelope for event detection (Hz). Range 100-500 / 1000-3000.
    "evt_bp_low_hz":        200.0,
    "evt_bp_high_hz":       3000.0,
    # evt_k_mad : event threshold = median + k*MAD of the population envelope.
    #   Range 3 - 8; higher = fewer, stronger events.
    "evt_k_mad":            4.0,
    "evt_min_isi_ms":       300.0,   # min spacing between events (ms). Range 100-1000.
    "evt_max_events":       300,     # cap events averaged. Range 20 - 1000.
    "evt_min_events":       5,       # need at least this many to analyse. Range 2-20.
    # --- EGM segment around each onset ---
    "egm_pre_ms":           15.0,    # ms before onset. Range 5 - 40.
    "egm_post_ms":          45.0,    # ms after onset. Range 20 - 120.
    # --- spkit activation-time detection ---
    # at_method : spkit activation method. min_dvdt | max_dvdt | max_abs_dvdt.
    "at_method":            "min_dvdt",
    # at_gradient : spkit gradient method. fdiff | fgrad | sgdiff | gauss_diff ...
    #   NOTE: the smoothing methods (gauss_diff, sg*_diff) need a NON-zero window
    #   below, or spkit builds an empty kernel and raises "v cannot be empty".
    "at_gradient":          "fdiff",
    # gauss_window / sg_window : smoothing-kernel lengths (samples, odd) used by the
    #   smoothing gradient methods above. Ignored by fdiff/fgrad. Must be >= 3 for
    #   gauss_diff and >= 5 for the sg* methods. Range 3 - 51.
    "gauss_window":         11,
    "sg_window":            11,
    # --- spkit conduction-velocity ---
    # eD_um : inter-electrode distance passed to compute_cv (MCS 60-MEA = 200 um).
    #   Sets the velocity scale. Range 100 - 700.
    "eD_um":                200.0,
    # cv_thr_cm_s : spkit CV ceiling (cm/s); local CV above this is dropped as noise.
    #   Range 50 - 500.  (neuronal CV ~ 10-50 cm/s = 100-500 mm/s)
    "cv_thr_cm_s":          100.0,
    # sweep_angle_deg : nominal stimulation-sweep axis (BR->TL = 135), for the
    #   alignment annotation. Range 0 - 360.
    "sweep_angle_deg":      135.0,

    # ======================================================================
    # SLOW-SWEEP LATENCY MODE  (physically matched to the laser raster sweep)
    # ----------------------------------------------------------------------
    # The event-triggered EGM/CV method above is built for FAST propagation
    # (a burst that crosses the array in ms). The optical sweep is ~1000x
    # slower and hits each electrode at a different phase of the sweep cycle,
    # so averaging on triggers washes it out -> random CV directions. This
    # mode instead folds each electrode's multiunit ENVELOPE onto the sweep
    # PERIOD (derived from the laser trigger rate) and reads out its preferred
    # sweep phase; the spatial phase gradient across the array recovers the
    # sweep direction & speed, and is compared to the geometric prediction of
    # a laser edge crossing a square field centred on the electrode pad.
    # run_egm_cv : also run the classic fast-EGM CV pipeline (above). Turn OFF
    #   to run ONLY the slow-sweep latency analysis. True | False.
    "run_egm_cv":           True,
    # run_sweep_latency : run the slow-sweep latency/direction analysis. True | False.
    "run_sweep_latency":    False,
    # frames_per_sweep : Polygon frames in one full BR->TL crossing (the trigger
    #   advances one frame). Sweep period = frames_per_sweep / trigger_hz.
    #   Sept2021_SweepBRtoTL.seq = 160. Overridden per recording by the optional
    #   7th 'frames' column of Batch.csv. Range 1 - 100000.
    "frames_per_sweep":     160.0,
    # sweep_box_mm : side length (mm) of the square the laser bar sweeps across,
    #   centred on the electrode pad; the bar edge starts at the box corner and
    #   travels to the opposite corner (the diagonal). Range 1 - 10.
    "sweep_box_mm":         4.0,
    # lat_bp_low_hz / lat_bp_high_hz : band for the multiunit envelope used in the
    #   latency folding (Hz). Range 100 - 1000 / 1000 - 6000.
    "lat_bp_low_hz":        300.0,
    "lat_bp_high_hz":       3000.0,
    # lat_min_coh : electrodes whose folding coherence is below this are dropped
    #   from the plane-wave fit as too noisy. Range 0 - 0.5.
    "lat_min_coh":          0.02,
    # lat_min_electrodes : need at least this many coherent electrodes to fit. 6-40.
    "lat_min_electrodes":   6,
}

DPI = 200


# ==============================================================================
# output location
# ==============================================================================
def spkit_root(plot_dir):
    """The 'SpKit' folder, a subfolder UNDER the selected output path (PLOT_DIR).
    Previously this was written to the PARENT of PLOT_DIR, which put it outside the
    folder chosen in the GUI and made the SpKit output look 'missing'."""
    return os.path.join(os.path.normpath(plot_dir), "SpKit")


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ==============================================================================
# signal helpers
# ==============================================================================
def _bandpass(X, fs, lo, hi):
    """Zero-phase band-pass each row of X (n_ch, n). Falls back to raw on error."""
    from scipy.signal import butter, sosfiltfilt
    try:
        ny = 0.5 * fs
        lo = max(1e-3, lo) / ny
        hi = min(hi, ny * 0.99) / ny
        sos = butter(4, [lo, hi], btype="band", output="sos")
        return sosfiltfilt(sos, X, axis=1)
    except Exception:
        return X


def _detect_events(X, fs, p):
    """Onset sample indices of network events, from the population envelope."""
    from scipy.signal import find_peaks
    Xb = _bandpass(X, fs, p["evt_bp_low_hz"], p["evt_bp_high_hz"])
    env = np.mean(np.abs(Xb), axis=0)                      # population activity
    # light smoothing (~2 ms boxcar)
    w = max(1, int(0.002 * fs))
    if w > 1:
        env = np.convolve(env, np.ones(w) / w, mode="same")
    med = np.median(env)
    mad = np.median(np.abs(env - med)) + 1e-12
    thr = med + p["evt_k_mad"] * 1.4826 * mad
    dist = max(1, int(p["evt_min_isi_ms"] / 1000.0 * fs))
    peaks, _ = find_peaks(env, height=thr, distance=dist)
    if peaks.size > p["evt_max_events"]:                   # keep the strongest
        order = np.argsort(env[peaks])[::-1][: p["evt_max_events"]]
        peaks = np.sort(peaks[order])
    return peaks


def _representative_egm(X, fs, onsets, pre_ms, post_ms):
    """Average per-channel voltage segments aligned on onsets -> (X1B, onset_idx)."""
    pre = int(pre_ms / 1000.0 * fs)
    post = int(post_ms / 1000.0 * fs)
    n = X.shape[1]
    segs = [X[:, o - pre:o + post] for o in onsets if o - pre >= 0 and o + post <= n]
    if not segs:
        return None, pre
    return np.mean(np.stack(segs, axis=0), axis=0), pre     # (n_ch, pre+post)


# ==============================================================================
# per-region spkit run
# ==============================================================================
def _int_labels(labels):
    """(int_labels, ok_mask) — numeric MCS labels; non-numeric channels excluded."""
    ilab, ok = [], []
    for l in labels:
        s = str(l).strip()
        if s.isdigit():
            ilab.append(int(s)); ok.append(True)
        else:
            ilab.append(-1); ok.append(False)
    return np.array(ilab), np.array(ok, bool)


# ==============================================================================
# SLOW-SWEEP LATENCY MODE
# Folds each electrode's multiunit envelope onto the laser-sweep period and reads
# out its preferred sweep phase. The spatial phase gradient across the array is
# the sweep direction & speed (plane-wave fit, reusing the Phase-Lock-In matched
# filter), compared to the geometric prediction of a laser edge crossing a square
# field centred on the pad. This is the physically-matched analogue of the fast
# EGM/CV method for the SLOW optical sweep.
# ==============================================================================
def _sweep_period(onsets_s, fps):
    """Sweep period (s) and trigger ISI (s) from laser trigger times."""
    on = np.sort(np.asarray(onsets_s, float)) if onsets_s is not None else np.array([])
    if on.size < 3:
        return None, None
    isi = float(np.median(np.diff(on)))
    if not np.isfinite(isi) or isi <= 0:
        return None, None
    return float(fps) * isi, isi


def _multiunit_envelope(X, fs, lo, hi):
    """|band-passed voltage| per channel, lightly smoothed (~2 ms)."""
    Xb = _bandpass(X, fs, lo, hi)
    env = np.abs(Xb)
    w = max(1, int(0.002 * fs))
    if w > 1:
        k = np.ones(w) / w
        env = np.vstack([np.convolve(r, k, mode="same") for r in env])
    return env


def _fold_resultants(env, fs, period_s):
    """Amplitude-weighted circular mean of sweep phase per channel (complex)."""
    n = env.shape[1]
    t = np.arange(n) / fs
    e = np.exp(1j * 2.0 * np.pi * ((t / period_s) % 1.0))
    num = env @ e
    den = env.sum(axis=1) + 1e-12
    return num / den


def _sweep_geom_phase(x_um, y_um, sweep_deg, diag_um):
    """Predicted contact phase: a laser edge entering the square field at one
    corner and crossing to the opposite corner over one period. Speed cancels, so
    this is purely geometric."""
    ang = np.radians(sweep_deg)
    s = x_um * np.cos(ang) + y_um * np.sin(ang)
    s0 = -0.5 * diag_um                              # box corner (edge start)
    return (2.0 * np.pi * ((s - s0) / diag_um)) % (2.0 * np.pi)


def _circ_corr(a, b):
    """Circular correlation between two angle arrays (Fisher–Lee)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a - np.angle(np.mean(np.exp(1j * a)))
    b = b - np.angle(np.mean(np.exp(1j * b)))
    num = np.sum(np.sin(a) * np.sin(b))
    den = np.sqrt(np.sum(np.sin(a) ** 2) * np.sum(np.sin(b) ** 2)) + 1e-12
    return float(num / den)


def run_region_sweep_latency(X, labels, fs, bad_labels, stem, region, p, onsets_s):
    """Slow-sweep latency/direction analysis for one region window. ALWAYS returns
    a plot-data dict (ok=True with the fit, or ok=False with a 'reason') so the
    grouped figure can show a tile for every sample instead of silently dropping
    windows. Thresholds are relaxed; a plane-wave fit still needs >=6 electrodes."""
    import mea_nap_phase_lockin as plk
    sweep_deg = float(p.get("sweep_angle_deg", 135.0))
    base = {"stem": stem, "region": region, "ok": False, "reason": "",
            "sweep_deg": sweep_deg}

    T, isi = _sweep_period(onsets_s, p.get("frames_per_sweep", 160.0))
    if T is None:
        base["reason"] = "no laser triggers"
        return base
    pitch = float(p.get("eD_um", 200.0))
    diag = float(p.get("sweep_box_mm", 4.0)) * np.sqrt(2.0) * 1000.0   # µm
    V_meas = diag / T                                # ground-truth sweep speed (µm/s)
    base.update({"T": T, "V_meas": V_meas, "n_trig": int(np.size(onsets_s))})

    ilab, ok = _int_labels(labels)
    badset = {int(str(b)) for b in (bad_labels or []) if str(b).isdigit()}
    xy = {}
    for i, l in enumerate(labels):
        c = orig.mcs_coord(l)
        if c is None or not ok[i] or ilab[i] in badset:
            continue
        xy[i] = (c[0] * pitch, c[1] * pitch)
    if len(xy) < 6:
        base["reason"] = f"only {len(xy)} electrodes"
        return base
    idx = sorted(xy)
    xs = np.array([xy[i][0] for i in idx], float)
    ys = np.array([xy[i][1] for i in idx], float)
    xs -= xs.mean(); ys -= ys.mean()                 # centre on the pad

    env = _multiunit_envelope(X[idx], fs, p.get("lat_bp_low_hz", 300.0),
                              p.get("lat_bp_high_hz", 3000.0))
    z = _fold_resultants(env, fs, T)
    coh = np.abs(z)
    obs_phi = np.angle(z) % (2.0 * np.pi)
    pred_phi = _sweep_geom_phase(xs, ys, sweep_deg, diag)
    keep = coh >= float(p.get("lat_min_coh", 0.02))
    # the phase map can always be drawn from every electrode's preferred phase
    base.update({"xs": xs, "ys": ys, "obs_phi": obs_phi, "pred_phi": pred_phi,
                 "coh": coh, "keep": keep})
    need = int(p.get("lat_min_electrodes", 6))
    if int(keep.sum()) < max(6, need):
        base["reason"] = f"{int(keep.sum())} coherent electrodes"
        return base

    locks = [plk.ElecLock(idx[i], float(xs[i]), float(ys[i]), complex(z[i]),
                          int(env.shape[1]))
             for i in range(len(idx)) if keep[i]]
    fit = plk.fit_plane_wave(locks, T, pitch)
    if not fit:
        base["reason"] = "plane-wave fit failed"
        return base

    obs_dir = fit["direction_deg"] % 360.0
    align = float(np.cos(np.radians(obs_dir - sweep_deg)))
    cc = _circ_corr(obs_phi[keep], pred_phi[keep])
    low = float(np.median(coh[keep])) < 0.05        # flag but still draw
    base.update({"ok": True, "fit": fit, "obs_dir": obs_dir, "align": align,
                 "cc": cc, "coh_med": float(np.median(coh[keep])),
                 "n_locked": int(keep.sum()), "low_conf": low})
    io._log(f"  [SpKit/latency] {stem}/{region}: dir {obs_dir:.0f}° "
            f"align {align:+.2f}, speed {fit['speed_um_per_s']:.0f} µm/s "
            f"(measured {V_meas:.0f}), coh {base['coh_med']:.2f}, "
            f"phase-corr {cc:+.2f}, T {T:.2f}s{'  [low-conf]' if low else ''}")
    return base


def _lat_summary_row(d):
    """CSV row from a latency plot-data dict (ok fits only)."""
    if not d.get("ok"):
        return None
    return {"stem": d["stem"], "region": d["region"], "n_triggers": d.get("n_trig", 0),
            "period_s": round(d["T"], 3), "sweep_speed_um_s": round(d["V_meas"], 1),
            "obs_dir_deg": round(d["obs_dir"], 1),
            "obs_speed_um_s": round(float(d["fit"]["speed_um_per_s"]), 1),
            "alignment": round(d["align"], 3), "coherence": round(d["coh_med"], 3),
            "phase_corr": round(d["cc"], 3), "n_locked": d["n_locked"],
            "low_confidence": bool(d.get("low_conf", False))}


def _draw_lat_row(fig, nr, r, d):
    """Draw one sample's 3 latency panels (obs phase map | obs-vs-pred | compass)
    into row r of an (nr x 3) grid. Empty/failed windows get a 'no lock' tile."""
    sweep_deg = d.get("sweep_deg", 135.0)
    stem = d["stem"]
    # column 1 : observed sweep-phase map (cyclic -> twilight)
    a1 = fig.add_subplot(nr, 3, r * 3 + 1)
    a1.set_ylabel(f"{stem}", fontsize=8, fontweight="bold", rotation=90, labelpad=8)
    if "xs" not in d:
        a1.text(0.5, 0.5, f"no lock\n({d.get('reason','no data')})", ha="center",
                va="center", transform=a1.transAxes, fontsize=8, color="#888",
                style="italic")
        a1.set_xticks([]); a1.set_yticks([])
        for c in (2, 3):
            ax = fig.add_subplot(nr, 3, r * 3 + c); ax.axis("off")
        return a1, None
    xs, ys = d["xs"], d["ys"]; keep = d["keep"]
    ext = max(np.ptp(xs), np.ptp(ys)) * 0.6 + 200.0
    sc = a1.scatter(xs[keep], ys[keep], c=d["obs_phi"][keep], cmap="twilight",
                    vmin=0, vmax=2 * np.pi, s=70, edgecolors="k", linewidths=0.3)
    if (~keep).any():
        a1.scatter(xs[~keep], ys[~keep], facecolors="none", edgecolors="0.75",
                   s=70, linewidths=0.5)
    a1.set_aspect("equal"); a1.set_xlim(-ext, ext); a1.set_ylim(-ext, ext)
    a1.set_xticks([]); a1.set_yticks([])
    if r == 0:
        a1.set_title("Observed sweep phase", fontsize=9)
    # column 2 : observed vs predicted phase (colour = coherence, parula)
    a2 = fig.add_subplot(nr, 3, r * 3 + 2)
    if d.get("ok"):
        sc2 = a2.scatter(d["pred_phi"][keep], d["obs_phi"][keep], c=d["coh"][keep],
                         cmap=PARULA, s=32)
        a2.plot([0, 2 * np.pi], [0, 2 * np.pi], "--", color="0.5", lw=1)
        a2.text(0.03, 0.95, f"corr {d['cc']:+.2f}", transform=a2.transAxes,
                fontsize=8, va="top")
        a2.set_xlim(0, 2 * np.pi); a2.set_ylim(0, 2 * np.pi); a2.set_aspect("equal")
        a2.set_xticks([0, np.pi, 2 * np.pi]); a2.set_yticks([0, np.pi, 2 * np.pi])
        a2.set_xticklabels(["0", "π", "2π"], fontsize=7)
        a2.set_yticklabels(["0", "π", "2π"], fontsize=7)
    else:
        a2.text(0.5, 0.5, f"no fit\n({d.get('reason','')})", ha="center", va="center",
                transform=a2.transAxes, fontsize=8, color="#888", style="italic")
        a2.axis("off")
    if r == 0:
        a2.set_title("Observed vs predicted", fontsize=9)
    # column 3 : direction compass
    a3 = fig.add_subplot(nr, 3, r * 3 + 3, projection="polar")
    a3.plot([np.radians(sweep_deg)] * 2, [0, 1.0], "--", color="crimson", lw=1.6)
    if d.get("ok"):
        obs_dir = d["obs_dir"]
        a3.bar(np.radians(obs_dir), 1.0, width=0.16,
               color="#999" if d.get("low_conf") else "#2f6df6", alpha=0.85)
        a3.annotate("", xy=(np.radians(obs_dir), 1.0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", lw=2.2,
                                    color="#999" if d.get("low_conf") else "black"))
        ttl = (f"obs {obs_dir:.0f}° · align {d['align']:+.2f}"
               f"\nv={d['fit']['speed_um_per_s']:,.0f} µm/s")
    else:
        ttl = "no fit"
    a3.set_theta_zero_location("E"); a3.set_theta_direction(1); a3.set_rticks([])
    a3.set_title(("Direction\n" if r == 0 else "") + ttl, fontsize=8, pad=10)
    return a1, sc


def _fig_latency_group(group, plots, p, out_dir):
    """One slow-sweep-latency figure per GROUP: a row per sample, 3 detailed
    panels each (observed phase map | obs-vs-predicted | direction compass)."""
    plots = sorted(plots, key=lambda d: d["stem"])
    nr = max(1, len(plots))
    fig = plt.figure(figsize=(12.0, 3.2 * nr), constrained_layout=True)
    col0, last = [], None
    for r, d in enumerate(plots):
        a1, sc = _draw_lat_row(fig, nr, r, d)
        col0.append(a1)
        if sc is not None:
            last = sc
    if last is not None and col0:
        cb = fig.colorbar(last, ax=col0, location="left", fraction=0.04, pad=0.02,
                          ticks=[0, np.pi, 2 * np.pi])
        cb.ax.set_yticklabels(["0", "π", "2π"])
        cb.set_label("observed sweep phase (rad)", fontsize=8)
    sweep = float(p.get("sweep_angle_deg", 135.0))
    fig.suptitle(f"Slow-sweep latency — group {group}   "
                 f"(black arrow = recovered heading, red dashed = sweep axis "
                 f"{sweep:.0f}° BR→TL)", fontsize=11)
    _save(fig, os.path.join(out_dir, f"spkit_sweep_latency_{group}.png"))


def run_region(X, labels, fs, bad_labels, stem, region, out_dir, p,
               onsets_s=None):
    """Full spkit continuous-EGM CV pipeline for one region window.
    onsets_s : optional window-relative event onset times (s) — e.g. laser ON
    times from the Digital Event Recorder. If given (and enough), they are used
    stimulus-locked; otherwise events are detected from the population signal.
    Returns a summary dict (or None if too few events)."""
    import spkit as sp
    import spkit.mea as mea

    src = "population"
    if (p.get("use_digital_events") and onsets_s is not None
            and np.size(onsets_s) >= p["evt_min_events"]):
        lag = p.get("digital_lag_ms", 0.0) / 1000.0
        onsets = np.round((np.asarray(onsets_s, float) + lag) * fs).astype(int)
        onsets = onsets[(onsets >= 0) & (onsets < X.shape[1])]
        cap = int(p.get("evt_max_events", 300))          # don't average 100k+ segments
        if cap and onsets.size > cap:
            onsets = onsets[np.linspace(0, onsets.size - 1, cap).astype(int)]
        src = "laser/digital"
    else:
        onsets = _detect_events(X, fs, p)
    if onsets.size < p["evt_min_events"]:
        io._log(f"  [SpKit] {stem}/{region}: only {onsets.size} events "
                f"({src}, < {p['evt_min_events']}) — skipped")
        return None
    io._log(f"  [SpKit] {stem}/{region}: {onsets.size} events ({src})")
    X1B, onset_idx = _representative_egm(X, fs, onsets, p["egm_pre_ms"], p["egm_post_ms"])
    if X1B is None:
        io._log(f"  [SpKit] {stem}/{region}: no usable segments — skipped")
        return None

    # spkit activation times (samples) per channel, over the whole segment.
    # Smoothing gradient methods (gauss_diff / sg*_diff) build a kernel from these
    # windows; spkit's default gauss_window=0 makes an EMPTY kernel and raises
    # "v cannot be empty", so force valid, odd, non-trivial windows here.
    gwin = max(3, int(p.get("gauss_window", 11) or 11))
    swin = max(5, int(p.get("sg_window", 11) or 11))
    if gwin % 2 == 0:
        gwin += 1
    if swin % 2 == 0:
        swin += 1
    at_loc = mea.activation_time_loc(
        X1B, fs=fs, t_range=[None, None], method=p["at_method"],
        gradient_method=p["at_gradient"], gauss_window=gwin, sg_window=swin,
        plot=False)
    at_ms = 1000.0 * np.asarray(at_loc, float) / fs
    at_ms = at_ms - np.nanmin(at_ms)                       # relative to earliest

    ilab, ok = _int_labels(labels)
    badset = {int(str(b)) for b in (bad_labels or []) if str(b).isdigit()}
    good = np.array([ok[i] and ilab[i] not in badset for i in range(len(labels))])

    # arrange on the 8x8 MEA grid; bad/absent channels -> NaN
    at_grid = mea.arrange_mea_grid(np.where(good, at_ms, np.nan),
                                   ch_labels=ilab, grid=(8, 8), default=np.nan)
    good_grid = mea.arrange_mea_grid(np.where(good, 1.0, np.nan),
                                     ch_labels=ilab, grid=(8, 8), default=np.nan)
    AxI, _AxIx = sp.fill_nans_2d(at_grid * good_grid, clip_range=[None, None])
    CV_df, CV_thetas, CV0, CV = mea.compute_cv(
        AxI, good_grid, eD=p["eD_um"], cv_thr=p["cv_thr_cm_s"],
        plots=0, verbose=False)

    # ---- per-electrode EGM features: peak-to-peak, duration, fractionation ----
    p2p = np.full(len(labels), np.nan)
    egmdur = np.full(len(labels), np.nan)
    findex = np.full(len(labels), np.nan)
    try:
        XE, ATloc = mea.extract_egm(X1B, act_loc=np.asarray(at_loc, int), fs=fs,
                                    verbose=False, plot=False)
        for i in range(len(labels)):
            if not good[i]:
                continue
            try:
                egmf, names = mea.egm_features(XE[i], act_loc=int(ATloc[i]), fs=fs,
                                               verbose=0, plot=0)
                d = dict(zip(names, np.asarray(egmf, float)))
                p2p[i] = d.get("peak_to_peak", np.nan)
                egmdur[i] = d.get("egm_duration", np.nan) * 1000.0 / fs   # samples -> ms
                findex[i] = d.get("f_index", np.nan)
            except Exception:
                continue
    except Exception as e:
        io._log(f"  [SpKit] {stem}/{region}: EGM features failed ({e!r})")
    feat_grids = {
        "p2p": mea.arrange_mea_grid(np.where(good, p2p, np.nan), ch_labels=ilab,
                                    grid=(8, 8), default=np.nan),
        "dur": mea.arrange_mea_grid(np.where(good, egmdur, np.nan), ch_labels=ilab,
                                    grid=(8, 8), default=np.nan),
        "findex": mea.arrange_mea_grid(np.where(good, findex, np.nan), ch_labels=ilab,
                                       grid=(8, 8), default=np.nan),
    }

    # ---- data ----------------------------------------------------------------
    try:
        CV_df.to_csv(os.path.join(out_dir, "CV_df.csv"))
    except Exception:
        pass

    cvmean_cm = float(CV_df.loc["CV_mean", "Original"]) if "Original" in CV_df else float("nan")
    cvmed_cm = float(CV_df.loc["CV_median", "Original"]) if "Original" in CV_df else float("nan")
    ang = float(CV_df.loc["avg_angle", "Original"]) if "Original" in CV_df else float("nan")
    ang_deg = np.degrees(ang) % 360.0 if np.isfinite(ang) else float("nan")
    al = (np.cos(np.radians(ang_deg - p["sweep_angle_deg"]))
          if np.isfinite(ang_deg) else float("nan"))
    summary = {"stem": stem, "region": region, "n_events": int(onsets.size),
               "event_source": src,
               "cv_mean_mm_s": cvmean_cm * 10.0, "cv_median_mm_s": cvmed_cm * 10.0,
               "mean_dir_deg": ang_deg, "alignment": al}
    # everything the combined per-sample figures need for this region row
    plotdata = {"region": region, "AxI": AxI, "CV": CV, "CV_thetas": CV_thetas,
                "feat_grids": {"AxI": AxI, **feat_grids}, "n_events": int(onsets.size)}
    return summary, plotdata


def _ordered_region_plots(plots):
    """Region plot-data dicts in pre/stim/post order (unknowns appended)."""
    order = {r: i for i, r in enumerate(orig.REGION_ORDER)}
    return sorted(plots, key=lambda d: order.get(d["region"], 99))


def _draw_cv_mag(ax, CVmm):
    im = ax.imshow(CVmm, cmap=PARULA, origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="CV (mm/s)")
    _grid_ticks(ax)


def _draw_cv_vectors(ax, CVmm, TH, sweep):
    ys, xs = np.where(np.isfinite(TH) & np.isfinite(CVmm))
    mdir = float("nan")
    if xs.size:
        u = np.cos(TH[ys, xs]); v = np.sin(TH[ys, xs]); cvv = CVmm[ys, xs]
        norm = plt.Normalize(0, max(np.percentile(cvv, 95), 1e-6))
        q = ax.quiver(xs, ys, u, -v, cvv, cmap=PARULA, norm=norm, angles="xy",
                      scale_units="xy", scale=1.6, width=0.012, pivot="middle")
        plt.colorbar(q, ax=ax, fraction=0.046, pad=0.04, label="CV (mm/s)")
        mdir = np.degrees(np.arctan2(np.mean(np.sin(TH[ys, xs])),
                                     np.mean(np.cos(TH[ys, xs])))) % 360.0
        ax.annotate("", xy=(0.6 + 1.1 * np.cos(np.radians(sweep)),
                            0.6 - 1.1 * np.sin(np.radians(sweep))), xytext=(0.6, 0.6),
                    arrowprops=dict(arrowstyle="-|>", color="crimson", lw=1.8))
    ax.set_xlim(-0.7, 7.7); ax.set_ylim(7.7, -0.7); ax.set_aspect("equal")
    _grid_ticks(ax)
    return mdir


def _draw_cv_streamline(ax, AxI, CVmm, TH):
    try:
        U = np.cos(TH); V = -np.sin(TH)                  # image y is down
        U = np.where(np.isfinite(U), U, 0.0); V = np.where(np.isfinite(V), V, 0.0)
        spd = np.where(np.isfinite(CVmm), CVmm, 0.0)
        ax.streamplot(np.arange(8), np.arange(8), U, V, color=spd, cmap=PARULA,
                      density=1.1, linewidth=1.3, arrowsize=1.2)
        ax.imshow(AxI, cmap="Greys", origin="upper", alpha=0.20)
    except Exception:
        ax.text(0.5, 0.5, "streamline n/a", ha="center", transform=ax.transAxes)
    ax.set_xlim(-0.5, 7.5); ax.set_ylim(7.5, -0.5); ax.set_aspect("equal")
    _grid_ticks(ax)


def _draw_cv_compass(ax, TH, sweep):
    ys, xs = np.where(np.isfinite(TH))
    mdir = float("nan")
    if xs.size:
        th = TH[ys, xs]
        ax.hist(th % (2 * np.pi), bins=np.linspace(0, 2 * np.pi, 25),
                color="#2f6df6", alpha=0.6)
        C, S = np.mean(np.cos(th)), np.mean(np.sin(th))
        Rr = np.hypot(C, S)
        mdir = np.degrees(np.arctan2(S, C)) % 360.0
        ax.annotate("", xy=(np.arctan2(S, C), ax.get_ylim()[1] * Rr), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=2.4))
    ax.plot([np.radians(sweep)] * 2, [0, ax.get_ylim()[1]], "--", color="crimson", lw=1.6)
    ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
    return mdir


def _fig_cv_combined(stem, plots, p, out_dir):
    """One CV figure per SAMPLE: a row per region (pre/stim/post), 4 tiles each —
    CV magnitude map | CV vector field | streamline flow | velocity compass."""
    sweep = p["sweep_angle_deg"]
    plots = _ordered_region_plots(plots)
    nr = len(plots)
    fig = plt.figure(figsize=(16.5, 3.9 * nr))
    for r, d in enumerate(plots):
        AxI = np.asarray(d["AxI"], float)
        CVmm = np.asarray(d["CV"], float) * 10.0
        TH = np.asarray(d["CV_thetas"], float)
        a1 = fig.add_subplot(nr, 4, r * 4 + 1)
        _draw_cv_mag(a1, CVmm)
        a1.set_ylabel(f"{d['region']}\n({d['n_events']} ev)", fontsize=10,
                      fontweight="bold", rotation=90, labelpad=8)
        if r == 0:
            a1.set_title("CV magnitude (mm/s)", fontsize=9)
        a2 = fig.add_subplot(nr, 4, r * 4 + 2)
        mdir = _draw_cv_vectors(a2, CVmm, TH, sweep)
        if r == 0:
            a2.set_title("CV vector field", fontsize=9)
        a3 = fig.add_subplot(nr, 4, r * 4 + 3)
        _draw_cv_streamline(a3, AxI, CVmm, TH)
        if r == 0:
            a3.set_title("Streamline flow map", fontsize=9)
        a4 = fig.add_subplot(nr, 4, r * 4 + 4, projection="polar")
        _draw_cv_compass(a4, TH, sweep)
        al = np.cos(np.radians(mdir - sweep)) if np.isfinite(mdir) else float("nan")
        a4.set_title(("Velocity compass\n" if r == 0 else "")
                     + f"mean {mdir:.0f}° · align {al:+.2f}", fontsize=9, pad=12)
    fig.suptitle(f"spkit continuous-EGM conduction velocity — {stem}"
                 f"{_sweep_suffix(stem)}; sweep axis {sweep:.0f}° BR→TL", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, os.path.join(out_dir, "spkit_cv.png"))


def _fig_features_combined(stem, plots, out_dir):
    """One features figure per SAMPLE: a row per region (pre/stim/post), 4 tiles
    each — activation time | peak-to-peak | EGM duration | fractionation index."""
    plots = _ordered_region_plots(plots)
    nr = len(plots)
    cols = [("Activation time (ms)", "AxI", PARULA, "AT (ms)"),
            ("Peak-to-peak (µV)", "p2p", PARULA, "P2P"),
            ("EGM duration (ms)", "dur", PARULA, "ms"),
            ("Fractionation index", "findex", PARULA, "index")]
    fig, axes = plt.subplots(nr, 4, figsize=(16.5, 3.7 * nr), squeeze=False)
    for r, d in enumerate(plots):
        fg = d["feat_grids"]
        for c, (title, key, cmap, lbl) in enumerate(cols):
            ax = axes[r][c]
            grid = fg.get(key)
            if grid is None:
                ax.text(0.5, 0.5, "n/a", ha="center", transform=ax.transAxes)
                ax.axis("off")
                continue
            g = np.asarray(grid, float)
            finite = g[np.isfinite(g)]
            vmin = np.percentile(finite, 2) if finite.size else None
            vmax = np.percentile(finite, 98) if finite.size else None
            im = ax.imshow(g, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=lbl)
            _grid_ticks(ax)
            if r == 0:
                ax.set_title(title, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{d['region']}\n({d['n_events']} ev)", fontsize=10,
                              fontweight="bold", rotation=90, labelpad=8)
    fig.suptitle(f"spkit continuous-EGM features — {stem}{_sweep_suffix(stem)}",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, os.path.join(out_dir, "spkit_features.png"))


def _grid_ticks(ax):
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


# ==============================================================================
# driver — iterate Batch.csv, one region window at a time
# ==============================================================================
def run_spkit_egm():
    """Run the spkit continuous-EGM CV pipeline for every Batch.csv row."""
    try:
        import spkit  # noqa: F401  (fail loud & early if missing)
    except Exception as e:
        io._log(f"[SpKit] spkit not installed ({e!r}); install with "
                f"'python -m pip install spkit'. Skipping SpKit stage.")
        return

    p = SPKIT_EGM
    data_dir = io.CONFIG["DATA_DIR"]
    csv = os.path.join(data_dir, io.CONFIG.get("BATCH_CSV_NAME", "Batch.csv"))
    rows = io.parse_batch(csv)
    if not rows:
        io._log("[SpKit] no Batch.csv rows found; nothing to do.")
        return
    root = spkit_root(io.CONFIG["PLOT_DIR"])
    ref = set(str(r) for r in io.CONFIG.get("REFERENCE_ELECTRODE", []))
    io._log(f"[SpKit] continuous-EGM CV -> {root}  ({len(rows)} region windows)")

    do_cv = bool(p.get("run_egm_cv", True))
    do_lat = bool(p.get("run_sweep_latency", False))
    if not (do_cv or do_lat):
        io._log("[SpKit] both EGM-CV and slow-sweep latency are off; nothing to do.")
        return
    summaries = []
    lat_summaries = []
    lat_by_group = {}                         # group -> [latency plot-data per sample]
    by_stem = {}
    stem_group = {}
    for r in rows:
        by_stem.setdefault(r["stem"], []).append(r)
        stem_group[r["stem"]] = r.get("group", "")

    gui = progress_gui.make(list(by_stem.keys()), "MEA Analyzer — SpKit EGM",
                            "Continuous-EGM conduction velocity")
    try:
        for stem, srows in by_stem.items():
            io.check_abort()
            gui.start(stem, "reading raw voltage…")
            fpath = os.path.join(data_dir, stem + ".h5")
            if not os.path.exists(fpath):
                io._log(f"[SpKit] {stem}: file not found — skipped")
                gui.done(stem, "file not found")
                continue
            try:
                mcs = io.MCSFile(fpath, io.CONFIG.get("STREAM_INDEX"))
            except Exception as e:
                io._log(f"[SpKit] {stem}: read failed ({e!r}) — skipped")
                gui.done(stem, "read failed")
                continue
            fs = mcs.fs
            labels = list(mcs.labels)
            keep_idx = [i for i, l in enumerate(labels) if str(l) not in ref]
            klabels = [labels[i] for i in keep_idx]
            # laser ON times from the MCS Digital Event Recorder (once per file)
            laser = None
            if p.get("use_digital_events"):
                try:
                    laser = de.read_digital_events(fpath, io.CONFIG.get("STREAM_INDEX"))
                    if laser.get("source"):
                        io._log(f"  [SpKit] {stem}: laser events from {laser['source']} "
                                f"({laser['on_s'].size} ON)")
                    else:
                        io._log(f"  [SpKit] {stem}: no digital/laser stream found; "
                                f"using population detection")
                except Exception as e:
                    io._log(f"  [SpKit] {stem}: digital-event read failed ({e!r}); "
                            f"using population detection")
                    laser = None
            n_reg = max(1, len(srows))
            n_ok = 0
            cv_plots = []                    # per-region CV plot-data for this sample
            for k, row in enumerate(srows):
                io.check_abort()
                region = row["region"]
                gui.set(stem, (k + 0.5) / n_reg, note=f"CV: {region}…")
                s0 = int(max(0, row["win"][0]) * fs)
                s1 = int(min(mcs.duration, row["win"][1]) * fs)
                if s1 - s0 < int(0.5 * fs):
                    io._log(f"[SpKit] {stem}/{region}: window too short — skipped")
                    continue
                X = np.stack([mcs.get_channel_window(i, s0, s1) for i in keep_idx])
                bad = list(row.get("ground", []))
                onsets_s = (de.events_in_window(laser, row["win"][0], row["win"][1])
                            if laser else None)
                out_dir = os.path.join(root, stem, region)
                if do_cv:
                    try:
                        res = run_region(X, klabels, fs, bad, stem, region, out_dir, p,
                                         onsets_s=onsets_s)
                        if res:
                            s, pdata = res
                            summaries.append(s)
                            cv_plots.append(pdata)
                            n_ok += 1
                            io._log(f"  [SpKit] {stem}/{region}: {s['n_events']} events, "
                                    f"CV {s['cv_median_mm_s']:.0f} mm/s, dir "
                                    f"{s['mean_dir_deg']:.0f}°, align {s['alignment']:+.2f}")
                    except Exception as e:
                        io._log(f"  [SpKit] {stem}/{region}: EGM-CV failed ({e!r})")
                # slow-sweep latency runs on the STIM window only (that is when the
                # laser sweeps); one tile per sample in the per-group figure.
                if do_lat and region == "stim":
                    try:
                        p_stem = dict(p)              # per-recording frame count
                        fr = io.frames_for_rows(srows)
                        if fr:
                            p_stem["frames_per_sweep"] = fr
                        sl = run_region_sweep_latency(X, klabels, fs, bad, stem,
                                                      region, p_stem, onsets_s)
                        grp = stem_group.get(stem, "")
                        lat_by_group.setdefault(grp, []).append(sl)
                        srow = _lat_summary_row(sl)
                        if srow:
                            lat_summaries.append(srow)
                    except Exception as e:
                        io._log(f"  [SpKit/latency] {stem}/{region}: failed ({e!r})")
            mcs.free()
            # one combined CV + features figure per sample (all regions stacked)
            if do_cv and cv_plots:
                stem_dir = os.path.join(root, stem)
                try:
                    _fig_cv_combined(stem, cv_plots, p, stem_dir)
                    _fig_features_combined(stem, cv_plots, stem_dir)
                except Exception as e:
                    io._log(f"  [SpKit] {stem}: combined figure failed ({e!r})")
            gui.done(stem, f"{n_ok}/{len(srows)} region(s)")
    finally:
        try:
            gui.close()
        except Exception:
            pass

    if do_cv:
        _write_summary(summaries, root)
    if do_lat:
        # one combined slow-sweep-latency figure per group (rows = samples)
        for grp, plots in lat_by_group.items():
            if not plots:
                continue
            try:
                _fig_latency_group(grp, plots, p, root)
            except Exception as e:
                io._log(f"  [SpKit/latency] group {grp}: figure failed ({e!r})")
        _write_latency_summary(lat_summaries, root)
    io._log(f"[SpKit] done — {len(summaries)} EGM-CV, "
            f"{len(lat_summaries)} slow-sweep-latency stim windows locked.")


def _write_summary(summaries, root):
    if not summaries:
        return
    try:
        import pandas as pd
        df = pd.DataFrame(summaries)
        os.makedirs(root, exist_ok=True)
        df.to_csv(os.path.join(root, "spkit_cv_summary.csv"), index=False)
        # per-stem pre/stim/post comparison figure of median CV + alignment
        for stem, sub in df.groupby("stem"):
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
            order = [r for r in orig.REGION_ORDER if r in list(sub["region"])]
            order += [r for r in sub["region"] if r not in order]
            sub = sub.set_index("region").loc[order].reset_index()
            axes[0].bar(sub["region"], sub["cv_median_mm_s"],
                        color=[orig._region_color(r, i) for i, r in enumerate(sub["region"])])
            axes[0].set_ylabel("median CV (mm/s)"); axes[0].set_title("Conduction velocity")
            axes[1].bar(sub["region"], sub["alignment"],
                        color=[orig._region_color(r, i) for i, r in enumerate(sub["region"])])
            axes[1].axhline(0, color="k", lw=0.8)
            axes[1].set_ylim(-1, 1); axes[1].set_ylabel("sweep alignment")
            axes[1].set_title("Alignment with sweep axis")
            for a in axes:
                orig._aesthetics(a)
            fig.suptitle(f"spkit continuous-EGM CV — {stem}{_sweep_suffix(stem)}",
                         fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.92])
            _save(fig, os.path.join(root, stem, "spkit_cv_summary.png"))
    except Exception as e:
        io._log(f"[SpKit] summary failed ({e!r})")


def _write_latency_summary(summaries, root):
    """CSV + per-stem pre/stim/post figure for the slow-sweep latency analysis:
    sweep alignment and folding coherence per region."""
    if not summaries:
        return
    try:
        import pandas as pd
        df = pd.DataFrame(summaries)
        os.makedirs(root, exist_ok=True)
        df.to_csv(os.path.join(root, "spkit_sweep_latency_summary.csv"), index=False)
        for stem, sub in df.groupby("stem"):
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
            order = [r for r in orig.REGION_ORDER if r in list(sub["region"])]
            order += [r for r in sub["region"] if r not in order]
            sub = sub.set_index("region").loc[order].reset_index()
            cols = [orig._region_color(r, i) for i, r in enumerate(sub["region"])]
            axes[0].bar(sub["region"], sub["alignment"], color=cols)
            axes[0].axhline(0, color="k", lw=0.8); axes[0].set_ylim(-1, 1)
            axes[0].set_ylabel("sweep alignment"); axes[0].set_title("Alignment with sweep axis")
            axes[1].bar(sub["region"], sub["coherence"], color=cols)
            axes[1].set_ylabel("folding coherence"); axes[1].set_title("Sweep-locking strength")
            for a in axes:
                orig._aesthetics(a)
            fig.suptitle(f"slow-sweep latency — {stem}{_sweep_suffix(stem)}", fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.92])
            _save(fig, os.path.join(root, stem, "spkit_sweep_latency_summary.png"))
    except Exception as e:
        io._log(f"[SpKit] latency summary failed ({e!r})")


# ==============================================================================
# SELF-TEST  --  synthetic propagating EGM wave (bypasses file IO)
# ==============================================================================
def _synthetic_region(region, prop_angle=135.0, cv_mm_s=200.0, seed=0):
    """Build a raw voltage matrix whose EGM activation propagates along prop_angle."""
    rng = np.random.default_rng(seed)
    labels = orig._mcs60_labels()
    pitch = SPKIT_EGM["eD_um"]
    pos = {l: np.array(orig.mcs_coord(l)) * pitch for l in labels}
    fs = 25000.0
    dur = 30.0
    n = int(dur * fs)
    X = 2.0 * rng.standard_normal((len(labels), n))        # baseline noise
    dvec = np.array([np.cos(np.radians(prop_angle)), np.sin(np.radians(prop_angle))])
    proj0 = min(pos[l] @ dvec for l in labels)
    onsets = np.sort(rng.uniform(1.0, dur - 1.0, 40))       # 40 network events
    wsig = int(0.4e-3 * fs)                                  # deflection width
    for on in onsets:
        for i, l in enumerate(labels):
            lat = (pos[l] @ dvec - proj0) / cv_mm_s / 1000.0  # s
            loc = int((on + lat) * fs)
            t = np.arange(max(0, loc - 5 * wsig), min(n, loc + 5 * wsig))
            X[i, t] += -60.0 * (t - loc) * np.exp(-0.5 * ((t - loc) / wsig) ** 2) / wsig
    return X, labels, fs


if __name__ == "__main__":
    out = os.path.join(os.getcwd(), "_spkit_selftest", "MEA_TEST")
    print("spkit continuous-EGM self-test (synthetic 135° wave, CV 200 mm/s):")
    cv_plots = []
    for region, ang, cv, sd in [("pre", 40, 150, 1), ("stim", 135, 180, 2),
                                ("post", 135, 260, 3)]:
        X, labels, fs = _synthetic_region(region, prop_angle=ang, cv_mm_s=cv, seed=sd)
        res = run_region(X, labels, fs, [], "MEA_TEST", region,
                         os.path.join(out, region), SPKIT_EGM)
        if res:
            s, pdata = res
            cv_plots.append(pdata)
            print(f"  {region}: n={s['n_events']} CV_med={s['cv_median_mm_s']:.0f} mm/s "
                  f"dir={s['mean_dir_deg']:.0f}° align={s['alignment']:+.2f}")
        else:
            print(f"  {region}: no result")
    if cv_plots:
        _fig_cv_combined("MEA_TEST", cv_plots, SPKIT_EGM, out)
        _fig_features_combined("MEA_TEST", cv_plots, out)
    print("SELFTEST done ->", out)
