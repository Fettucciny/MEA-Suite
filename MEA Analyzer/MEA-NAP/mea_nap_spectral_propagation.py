#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 mea_nap_spectral_propagation.py  --  Stage-2 add-ons written into Plots/<stem>/
================================================================================
Two extra analyses, per recording, comparing the pre / stim / post regions:

1. PSD (power spectral density) of the POPULATION firing and of the network
   bursting, per region -> did the light stimulation change the spontaneous
   firing / bursting frequency of the sample?
     * firing signal  : array-summed spikes binned at psd_bin_s -> Welch PSD.
     * bursting signal : network-burst envelope (1 during a network burst) -> PSD;
                         its peak is the burst-recurrence frequency.

2. Signal-propagation direction & speed (Local Activation Time / conduction-
   velocity method, after the spkit bsp_mea / CinC-2025 toolkit, adapted for
   spike trains). For each propagation event we take every electrode's FIRST-SPIKE
   time (its LAT), fit a spatial plane LAT = a*x + b*y + c across the MEA grid, and
   read off:
       propagation direction  theta = atan2(b, a)   (direction of increasing LAT)
       conduction velocity     CV   = 1 / sqrt(a^2 + b^2)   [mm/s]
   Events are BOTH detected network bursts and fixed sliding windows. Per region
   we draw an example LAT activation map, a direction rose, a CV distribution, and
   an alignment score against the nominal bottom-right -> top-left sweep axis, so
   pre-vs-post reveals any directional / speed "memory" (STDP) effect.

Import-safe (numpy + scipy + matplotlib). Run directly for a synthetic self-test
(a burst that propagates along the bottom-right -> top-left axis).
================================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mea_nap_original_plots as orig     # mcs_coord, REGION_ORDER, region colours

# ==============================================================================
# PARAMETERS  (adjustment ranges noted; exposed in the GUI)
# ==============================================================================
SPECTRAL_PARAMS = {
    # psd_bin_s : firing/burst rate bin (s); sampling fs = 1/bin, Nyquist = fs/2.
    #   Range 0.02 - 0.20 (0.05 -> 20 Hz sampling, sees rhythms up to 10 Hz).
    "psd_bin_s":        0.05,
    # psd_fmax_hz : highest frequency shown. Range 1 - 25 (burst rhythms are slow).
    "psd_fmax_hz":      10.0,
    # psd_welch_nperseg : Welch segment length (samples). Range 256 - 4096; longer
    #   = finer frequency resolution but needs a longer window.
    "psd_welch_nperseg": 1024,
    # psd_logy : log y-axis for the PSD. True | False.
    "psd_logy":         True,
}

PROP_PARAMS = {
    # electrode_pitch_um : inter-electrode spacing (µm). MCS 60-MEA = 200 (fixed for
    #   this batch: 30 µm electrodes, 200 µm centre-to-centre). Range 100 - 700.
    "electrode_pitch_um":   200.0,
    # min_electrodes : minimum electrodes with a first spike to fit a wave.
    #   Range 4 - 20; higher = stricter, fewer but more reliable events.
    "min_electrodes":       6,
    # min_fit_r2 : minimum plane-fit R^2 for an event to count as a coherent wave.
    #   Range 0.0 - 0.9; higher rejects incoherent/noisy events.
    "min_fit_r2":           0.30,
    # window_ms / window_step_ms : sliding-window length / step (ms) for the
    #   windowed method. Range 50 - 500 / 20 - 250.
    "window_ms":            100.0,
    "window_step_ms":       50.0,
    # window_min_active_frac : fraction of electrodes that must fire in a window
    #   for it to be analysed. Range 0.1 - 0.7.
    "window_min_active_frac": 0.30,
    # sweep_angle_deg : nominal stimulation-sweep direction, math convention
    #   (0 = +x/right, 90 = +y/up). Bottom-right -> top-left = 135. Range 0 - 360.
    "sweep_angle_deg":      135.0,
    # max_maps : example LAT activation maps drawn per region. Range 1 - 3.
    "max_maps":             1,
    # max_events_per_region : cap on how many propagation events (sliding windows OR
    #   laser triggers) are fitted per region; a long 1-hour / high-rate recording can
    #   otherwise present 100k+ events. Uniformly subsampled above this. Range 500-20000.
    "max_events_per_region": 4000,
}

# per-figure on/off (mirrors orig.PLOT_ENABLE; GUI "Plot outputs" tab can gate these)
SP_ENABLE = {
    "psd_firing_bursting":   True,
    "propagation_bursts":    True,
    "propagation_windows":   True,
    "propagation_digital":   True,   # laser-locked; only drawn if digital events exist
    "cv_field":              True,
}

# local conduction-velocity field (spkit-style flow map, computed from spike LATs)
CVFIELD_PARAMS = {
    # cv_field_mode : which events feed the averaged activation map. bursts | windows.
    "cv_field_mode":        "bursts",
    # cv_neighbor_factor : local-fit radius as a multiple of electrode pitch. A
    #   value of 1.6 gathers the 8 nearest neighbours. Range 1.2 - 2.5.
    "cv_neighbor_factor":   1.6,
    # cv_min_neighbors : electrodes needed inside that radius to fit a local plane.
    #   Range 3 - 6; higher = smoother, fewer vectors.
    "cv_min_neighbors":     4,
    # cv_local_min_r2 : minimum local plane-fit R^2 to keep a vector. Range 0.0-0.6.
    "cv_local_min_r2":      0.20,
}

DPI = 200


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _regcol(reg, i):
    try:
        return orig._region_color(reg, i)
    except Exception:
        return plt.cm.tab10(i % 10)[:3]


def _ordered_regions(regions):
    known = [r for r in orig.REGION_ORDER if r in regions]
    return known + sorted(r for r in regions if r not in orig.REGION_ORDER)


import re as _re

def sweep_speed_from_stem(stem):
    """Nominal laser-sweep speed (µm/s) parsed from the recording filename.

    The filename prefix is the speed at which the laser BAR swept across the
    sample (set by the Polygon trigger rate) — e.g. '200um...' = 200 µm/s,
    '4000um_2_D3' = 4000 µm/s. This is the stimulus's absolute, ground-truth
    sweep speed (hand-measured across the 4 mm sample), independent of the
    electrode pitch or of the measured neural conduction velocity.

    Returns (value_um_s, label) — label is a display string in friendly units —
    or (None, '') when the name has no speed prefix (e.g. Control recordings)."""
    m = _re.match(r"\s*(\d+(?:\.\d+)?)\s*um", str(stem), _re.IGNORECASE)
    if not m:
        return None, ""
    v = float(m.group(1))
    lab = (f"{v / 1000.0:.3g} mm/s ({v:.0f} µm/s)" if v >= 1000
           else f"{v:.0f} µm/s")
    return v, lab


def _sweep_suffix(stem):
    """Suptitle fragment noting the nominal sweep speed, or '' for Control."""
    v, lab = sweep_speed_from_stem(stem)
    return f"  ·  nominal laser sweep {lab}" if v is not None else ""


# ==============================================================================
# 1)  PSD of firing + bursting (population), per region
# ==============================================================================
def _population_rate(unit, bin_s):
    """Array-summed spikes binned at bin_s -> (rate[/bin], fs)."""
    dur = float(unit["duration"])
    n = max(2, int(np.floor(dur / bin_s)))
    edges = np.linspace(0, n * bin_s, n + 1)
    allsp = np.concatenate([np.asarray(s, float) for s in unit["spikes"]
                            if len(s)]) if any(len(s) for s in unit["spikes"]) \
        else np.empty(0)
    rate, _ = np.histogram(allsp[(allsp >= 0) & (allsp < n * bin_s)], bins=edges)
    return rate.astype(float), 1.0 / bin_s


def _burst_signal(res, dur, bin_s):
    """Network-burst envelope (1 during a network burst) at bin_s -> (env, fs)."""
    n = max(2, int(np.floor(dur / bin_s)))
    env = np.zeros(n)
    for b in res.get("net_bursts", []) or []:
        i0 = int(np.clip(b["start"] / bin_s, 0, n - 1))
        i1 = int(np.clip(b["end"] / bin_s, 0, n - 1))
        env[i0:i1 + 1] = 1.0
    return env, 1.0 / bin_s


def _welch(x, fs):
    from scipy.signal import welch
    if x.size < 8 or np.allclose(x, x.mean()):
        return np.array([]), np.array([])
    nper = int(min(SPECTRAL_PARAMS["psd_welch_nperseg"], x.size))
    f, p = welch(x - x.mean(), fs=fs, nperseg=nper)
    return f, p


def fig_psd(stem, region_units, out_dir):
    """2-panel population PSD (firing, bursting) overlaying pre/stim/post."""
    regions = _ordered_regions(list(region_units.keys()))
    bin_s = SPECTRAL_PARAMS["psd_bin_s"]
    fmax = SPECTRAL_PARAMS["psd_fmax_hz"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    peaks = {}
    for i, reg in enumerate(regions):
        unit, res = region_units[reg]
        col = _regcol(reg, i)
        rate, fs = _population_rate(unit, bin_s)
        f, p = _welch(rate, fs)
        if f.size:
            m = f <= fmax
            a1.plot(f[m], p[m], color=col, lw=1.6, label=reg)
            if m.any():
                fp = f[m][1:][np.argmax(p[m][1:])] if m.sum() > 1 else 0.0
                peaks.setdefault(reg, {})["firing"] = fp
        env, fs2 = _burst_signal(res, float(unit["duration"]), bin_s)
        f2, p2 = _welch(env, fs2)
        if f2.size:
            m2 = f2 <= fmax
            a2.plot(f2[m2], p2[m2], color=col, lw=1.6, label=reg)
            if m2.sum() > 1:
                peaks.setdefault(reg, {})["bursting"] = f2[m2][1:][np.argmax(p2[m2][1:])]
    for ax, ttl in ((a1, "Firing PSD (population)"),
                    (a2, "Bursting PSD (network-burst envelope)")):
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Power")
        ax.set_title(ttl); ax.set_xlim(0, fmax)
        if SPECTRAL_PARAMS["psd_logy"]:
            ax.set_yscale("log")
        ax.legend(title="region", fontsize=8)
        orig._aesthetics(ax)
    txt = "  ".join(f"{r}: firing {peaks.get(r, {}).get('firing', float('nan')):.2f} Hz, "
                    f"burst {peaks.get(r, {}).get('bursting', float('nan')):.2f} Hz"
                    for r in regions)
    fig.suptitle(f"Spontaneous firing / bursting spectrum — {stem}\n{txt}", fontsize=10)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "psd_firing_bursting.png"))


# ==============================================================================
# 2)  Propagation direction + speed (LAT plane-fit conduction velocity)
# ==============================================================================
def _electrode_pos(labels):
    """(labels_with_coords, pos_um Nx2) from MCS grid coords * pitch."""
    pitch = PROP_PARAMS["electrode_pitch_um"]
    keep, pos = [], []
    for l in labels:
        c = orig.mcs_coord(l)
        if c is not None:
            keep.append(l); pos.append((c[0] * pitch, c[1] * pitch))
    return keep, np.array(pos, float)


def _spikes64(unit):
    """[(label, float64 sorted spike array), ...] for one unit, converted ONCE.
    _lat_in_window used to call np.asarray(s, float) on every electrode for
    every event: on a 60-min day-5 recording that is ~700k conversions of
    float32 trains per figure (~940 s per file). The float64 values are the
    same ones the old per-call conversion produced, so every latency is
    identical."""
    out = []
    for l, s in zip(unit["labels"], unit["spikes"]):
        s = np.asarray(s, float)
        if s.size:
            out.append((l, s))
    return out


def _lat_in_window(unit, t0, t1, spikes64=None):
    """First-spike LAT (ms since t0) per electrode with a spike in [t0, t1).
    Uses searchsorted (spike trains are sorted by io.window_unit) so cost per
    window is O(log n) not O(n) — important on long, high-firing recordings.
    `spikes64` is the _spikes64(unit) list; callers that loop over many events
    pass it so the trains are converted once instead of per call."""
    if spikes64 is None:
        spikes64 = _spikes64(unit)
    lat = {}
    for l, s in spikes64:
        i = int(np.searchsorted(s, t0, side="left"))
        if i < s.size and s[i] < t1:
            lat[l] = (s[i] - t0) * 1000.0
    return lat


def _fit_wave(lat_map):
    """Plane-fit LAT(x,y) -> propagation direction (deg) + CV (mm/s) + R^2.
    Returns None if fewer than min_electrodes / poor fit / degenerate gradient."""
    labels = list(lat_map.keys())
    keep, pos = _electrode_pos(labels)
    if len(keep) < PROP_PARAMS["min_electrodes"]:
        return None
    lat = np.array([lat_map[l] for l in keep], float)
    A = np.column_stack([pos[:, 0], pos[:, 1], np.ones(len(lat))])
    coef, *_ = np.linalg.lstsq(A, lat, rcond=None)
    a, b, _c = coef
    pred = A @ coef
    ss_res = float(np.sum((lat - pred) ** 2))
    ss_tot = float(np.sum((lat - lat.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    g = float(np.hypot(a, b))                     # ms/µm
    if g <= 0 or r2 < PROP_PARAMS["min_fit_r2"]:
        return None
    return {"theta": float(np.degrees(np.arctan2(b, a)) % 360.0),
            "cv": 1.0 / g,                        # µm/ms == mm/s
            "r2": r2, "n": len(keep), "lat": lat_map}


def _burst_events(unit, res):
    return [(b["start"], b["end"]) for b in (res.get("net_bursts", []) or [])
            if b["end"] > b["start"]]


def _subsample_events(out):
    """Uniformly subsample an event list to PROP_PARAMS['max_events_per_region'] so a
    long / high-trigger-rate recording can't blow up (a few thousand events give the
    same direction/CV statistics)."""
    cap = int(PROP_PARAMS.get("max_events_per_region", 4000))
    n = len(out)
    if cap and n > cap:
        idx = np.linspace(0, n - 1, cap).astype(int)
        print(f"  [propagation] {n} events -> subsampled to {cap} (max_events_per_region)")
        return [out[i] for i in idx]
    return out


def _window_events(unit):
    dur = float(unit["duration"])
    w = PROP_PARAMS["window_ms"] / 1000.0
    step = PROP_PARAMS["window_step_ms"] / 1000.0
    t = 0.0; out = []
    while t + w <= dur + 1e-9:
        out.append((t, t + w)); t += step
    return _subsample_events(out)


def _digital_events(unit, laser_rel):
    """Stimulus-locked windows for each laser ON time (s, rel). Each window ends at
    the NEXT trigger (or window_ms, whichever is sooner) so it never straddles two
    evoked waves — a straddling window fits a spurious REVERSED direction because the
    tail of one wave (fires early-in-window) and the head of the next (fires late-in-
    window) put high LAT at the wave's start. Subsampled to a cap for long/high-rate
    recordings."""
    if laser_rel is None:
        return []
    w = PROP_PARAMS["window_ms"] / 1000.0
    dur = float(unit["duration"])
    on = np.sort(np.asarray(laser_rel, float))
    on = on[(on >= 0.0) & (on < dur)]
    if on.size == 0:
        return []
    nxt = np.empty(on.size); nxt[:-1] = on[1:]; nxt[-1] = dur
    out = [(float(t), float(min(t + w, e, dur))) for t, e in zip(on, nxt)
           if min(t + w, e, dur) - t > 1e-4]
    return _subsample_events(out)


def _events_for_mode(unit, res, mode, laser_rel=None):
    if mode == "bursts":
        return _burst_events(unit, res)
    if mode == "digital":
        return _digital_events(unit, laser_rel)
    return _window_events(unit)


def _events_for_region(unit, res, mode, laser_rel=None):
    """List of accepted wave fits for one region."""
    events = _events_for_mode(unit, res, mode, laser_rel)
    n_elec = max(1, len(unit["labels"]))
    min_active = (PROP_PARAMS["window_min_active_frac"] * n_elec
                  if mode == "windows" else PROP_PARAMS["min_electrodes"])
    fits = []
    sp64 = _spikes64(unit)
    for (t0, t1) in events:
        lat = _lat_in_window(unit, t0, t1, sp64)
        if len(lat) < min_active:
            continue
        fit = _fit_wave(lat)
        if fit is not None:
            fits.append(fit)
    return fits


def _resultant(thetas_deg):
    """Circular mean direction (deg) and resultant length R in [0,1]."""
    if len(thetas_deg) == 0:
        return float("nan"), 0.0
    a = np.radians(thetas_deg)
    C, S = np.mean(np.cos(a)), np.mean(np.sin(a))
    return float(np.degrees(np.arctan2(S, C)) % 360.0), float(np.hypot(C, S))


def _alignment(thetas_deg, sweep_deg):
    """Mean cos(theta - sweep) in [-1,1]: +1 = aligned with sweep, -1 = opposite."""
    if len(thetas_deg) == 0:
        return float("nan")
    return float(np.mean(np.cos(np.radians(np.asarray(thetas_deg) - sweep_deg))))


def _draw_lat_map(ax, lat_map, title):
    """Electrode grid coloured by LAT (early=blue -> late=red), arrow = gradient."""
    labels = list(lat_map.keys())
    keep, pos = _electrode_pos(labels)
    if len(keep) < 3:
        ax.axis("off"); return
    lat = np.array([lat_map[l] for l in keep])
    pitch = PROP_PARAMS["electrode_pitch_um"]
    sc = ax.scatter(pos[:, 0] / pitch, pos[:, 1] / pitch, c=lat, cmap="turbo",
                    s=90, edgecolors="w", lw=0.4)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="LAT (ms)")
    fit = _fit_wave(lat_map)
    if fit is not None:
        cx, cy = pos[:, 0].mean() / pitch, pos[:, 1].mean() / pitch
        dx, dy = np.cos(np.radians(fit["theta"])), np.sin(np.radians(fit["theta"]))
        ax.annotate("", xy=(cx + 1.6 * dx, cy + 1.6 * dy), xytext=(cx - 1.6 * dx, cy - 1.6 * dy),
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=2))
    ax.set_aspect("equal"); ax.set_title(title, fontsize=9); ax.axis("off")


def fig_propagation(stem, region_units, out_dir, mode, laser_by_region=None):
    """Activation map + direction rose + CV distribution + sweep alignment, per
    region. mode = 'bursts' | 'windows' | 'digital'."""
    regions = _ordered_regions(list(region_units.keys()))
    sweep = PROP_PARAMS["sweep_angle_deg"]
    lr = laser_by_region or {}
    nreg = len(regions)
    fig = plt.figure(figsize=(3.6 * nreg + 2.2, 8.4))
    gs = fig.add_gridspec(3, nreg, height_ratios=[1.1, 1.2, 0.9])
    for ci, reg in enumerate(regions):
        unit, res = region_units[reg]
        fits = _events_for_region(unit, res, mode, lr.get(reg))
        thetas = [f["theta"] for f in fits]
        cvs = [f["cv"] for f in fits]
        # AVERAGED activation map across ALL events (shows every active electrode, not
        # one sparse event) + its global plane fit = the robust "aggregate" direction.
        mean_lat, _cnt = _region_activation_map(unit, res, mode, lr.get(reg))
        agg = _fit_wave(mean_lat) if mean_lat else None
        # row 0: mean activation map
        axm = fig.add_subplot(gs[0, ci])
        if mean_lat and len(mean_lat) >= 3:
            r2txt = f", R²={agg['r2']:.2f}" if agg else ""
            _draw_lat_map(axm, mean_lat,
                          f"{reg} — mean activation ({len(mean_lat)} elec{r2txt})")
        else:
            axm.text(0.5, 0.5, f"{reg}\nno events", ha="center", va="center",
                     transform=axm.transAxes, color="0.5"); axm.axis("off")
        # row 1: per-event rose + AGGREGATE arrow (black, robust) and per-event mean (grey)
        axr = fig.add_subplot(gs[1, ci], projection="polar")
        if thetas:
            bins = np.linspace(0, 2 * np.pi, 25)
            axr.hist(np.radians(thetas), bins=bins, color=_regcol(reg, ci), alpha=0.55)
        top = axr.get_ylim()[1]
        if thetas:
            mth, R = _resultant(thetas)
            axr.annotate("", xy=(np.radians(mth), top * R), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="-|>", color="0.55", lw=1.6))
        if agg is not None:
            axr.annotate("", xy=(np.radians(agg["theta"]), top * 0.92), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="-|>", color="k", lw=2.6))
        axr.plot([np.radians(sweep)] * 2, [0, top], "--", color="crimson", lw=1.6)
        axr.plot([np.radians((sweep + 180) % 360)] * 2, [0, top], ":",
                 color="crimson", lw=1.0)
        axr.set_theta_zero_location("E"); axr.set_theta_direction(1)
        aggtxt = f"agg {agg['theta']:.0f}°" if agg else "agg —"
        axr.set_title(f"{reg}  (n={len(thetas)} events; black=aggregate)\n{aggtxt}",
                      fontsize=8, pad=12)
        # row 2: CV distribution + AGGREGATE alignment
        axc = fig.add_subplot(gs[2, ci])
        if cvs:
            axc.hist(cvs, bins=min(20, max(4, len(cvs))), color=_regcol(reg, ci), alpha=0.8)
            axc.axvline(np.median(cvs), color="k", ls="--", lw=1)
        al = (float(np.cos(np.radians(agg["theta"] - sweep))) if agg
              else _alignment(thetas, sweep))
        cvtxt = f"CV med {np.median(cvs):.0f} mm/s" if cvs else "CV —"
        aggcv = f" · agg CV {agg['cv']:.0f}" if agg else ""
        axc.set_title(f"{cvtxt}{aggcv} | align {al:+.2f}", fontsize=8)
        axc.set_xlabel("conduction velocity (mm/s)"); axc.set_ylabel("count")
        orig._aesthetics(axc)
    which = {"bursts": "network bursts",
             "digital": f"laser-locked {int(PROP_PARAMS['window_ms'])} ms windows"}.get(
        mode, f"{int(PROP_PARAMS['window_ms'])} ms windows")
    fig.suptitle(f"Signal propagation — {stem}{_sweep_suffix(stem)}  ({which}); "
                 f"red dashed = sweep axis ({sweep:.0f}° BR→TL). Black arrow = "
                 f"aggregate (mean-map) direction, grey = per-event mean.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, os.path.join(out_dir, f"propagation_{mode}.png"))


# ==============================================================================
# 3)  Local conduction-velocity FIELD, per region (spkit-style flow map)
# ------------------------------------------------------------------------------
# Where fig_propagation collapses each event to ONE global direction/speed, this
# builds a per-region AVERAGED activation-time surface (mean relative LAT per
# electrode across events) and fits a LOCAL plane in each electrode's neighbourhood
# to get a velocity VECTOR there -> a flow map that reveals spatial heterogeneity
# (e.g. a memory effect confined to one corridor of the array) that a single global
# fit averages away. Same gradient math as spkit.mea.compute_cv, on spike LATs.
# ==============================================================================
def _region_activation_map(unit, res, mode, laser_rel=None):
    """Mean relative first-spike LAT (ms) per electrode, averaged over events."""
    events = _events_for_mode(unit, res, mode, laser_rel)
    n_elec = max(1, len(unit["labels"]))
    min_active = (PROP_PARAMS["window_min_active_frac"] * n_elec
                  if mode == "windows" else PROP_PARAMS["min_electrodes"])
    acc, cnt = {}, {}
    sp64 = _spikes64(unit)
    for (t0, t1) in events:
        lat = _lat_in_window(unit, t0, t1, sp64)
        if len(lat) < min_active:
            continue
        m = min(lat.values())                    # align each event to its first spike
        for l, v in lat.items():
            acc[l] = acc.get(l, 0.0) + (v - m)
            cnt[l] = cnt.get(l, 0) + 1
    mean_lat = {l: acc[l] / cnt[l] for l in acc if cnt[l] > 0}
    return mean_lat, cnt


def _local_cv_field(mean_lat):
    """Local plane fit in each electrode neighbourhood -> per-electrode velocity
    vector. Returns (keep, pos_um, lat, vec_pos, vec_theta_deg, vec_cv_mm_s)."""
    labels = list(mean_lat.keys())
    keep, pos = _electrode_pos(labels)
    if len(keep) < CVFIELD_PARAMS["cv_min_neighbors"]:
        return keep, pos, np.array([]), np.empty((0, 2)), np.array([]), np.array([])
    lat = np.array([mean_lat[l] for l in keep], float)
    pitch = PROP_PARAMS["electrode_pitch_um"]
    R = CVFIELD_PARAMS["cv_neighbor_factor"] * pitch
    vp, vth, vcv = [], [], []
    for i in range(len(keep)):
        d = np.hypot(pos[:, 0] - pos[i, 0], pos[:, 1] - pos[i, 1])
        idx = np.where(d <= R)[0]
        if len(idx) < CVFIELD_PARAMS["cv_min_neighbors"]:
            continue
        P, L = pos[idx], lat[idx]
        A = np.column_stack([P[:, 0], P[:, 1], np.ones(len(L))])
        coef, *_ = np.linalg.lstsq(A, L, rcond=None)
        a, b, _c = coef
        g = float(np.hypot(a, b))
        if g <= 0:
            continue
        pred = A @ coef
        sst = float(np.sum((L - L.mean()) ** 2))
        r2 = 1 - float(np.sum((L - pred) ** 2)) / sst if sst > 0 else 0.0
        if r2 < CVFIELD_PARAMS["cv_local_min_r2"]:
            continue
        vp.append(pos[i]); vth.append(np.degrees(np.arctan2(b, a)) % 360.0)
        vcv.append(1.0 / g)                       # µm/ms == mm/s
    return keep, pos, lat, np.array(vp), np.array(vth), np.array(vcv)


def fig_cv_field(stem, region_units, out_dir, laser_by_region=None):
    """One flow-map panel per region: electrodes coloured by mean activation time,
    with a local conduction-velocity vector (arrow) at each electrode coloured by
    local CV. Sweep axis + per-region median CV / alignment annotated."""
    from matplotlib import cm
    from matplotlib.colors import Normalize
    cmap = plt.get_cmap("viridis")
    regions = _ordered_regions(list(region_units.keys()))
    mode = CVFIELD_PARAMS["cv_field_mode"]
    lr = laser_by_region or {}
    if mode == "digital" and not any(lr.get(r) is not None and len(lr.get(r)) for r in regions):
        mode = "bursts"                       # no laser events -> safe fallback
    sweep = PROP_PARAMS["sweep_angle_deg"]
    pitch = PROP_PARAMS["electrode_pitch_um"]
    fields = {}
    for reg in regions:
        u, res = region_units[reg]
        mean_lat, _ = _region_activation_map(u, res, mode, lr.get(reg))
        fields[reg] = _local_cv_field(mean_lat) if mean_lat else (
            [], np.empty((0, 2)), np.array([]), np.empty((0, 2)),
            np.array([]), np.array([]))
    allcv = np.concatenate([f[5] for f in fields.values() if f[5].size] or [np.array([1.0])])
    cvmax = float(np.percentile(allcv, 95)) if allcv.size else 1.0
    norm = Normalize(vmin=0.0, vmax=max(cvmax, 1e-6))
    nreg = len(regions)
    fig, axes = plt.subplots(1, nreg, figsize=(4.4 * nreg + 1.4, 4.9), squeeze=False)
    fig.subplots_adjust(left=0.02, right=0.89, top=0.85, bottom=0.04, wspace=0.06)
    for ci, reg in enumerate(regions):
        ax = axes[0][ci]
        keep, pos, lat, vp, vth, vcv = fields[reg]
        if len(keep) < 3 or vp.shape[0] == 0:
            ax.text(0.5, 0.5, f"{reg}\nno coherent field", ha="center", va="center",
                    transform=ax.transAxes, color="0.5"); ax.axis("off"); continue
        gx, gy = pos[:, 0] / pitch, pos[:, 1] / pitch
        ax.scatter(gx, gy, c=lat, cmap="Greys", s=120, alpha=0.35,
                   edgecolors="0.6", lw=0.3, zorder=1)
        vx, vy = vp[:, 0] / pitch, vp[:, 1] / pitch
        u = np.cos(np.radians(vth)); v = np.sin(np.radians(vth))
        colors = cmap(norm(vcv))
        ax.quiver(vx, vy, u, v, color=colors, angles="xy", scale_units="xy",
                  scale=1.7, width=0.010, pivot="middle", zorder=3)
        # sweep-axis reference (lower-left corner)
        cx, cy = gx.min() - 0.2, gy.min() - 0.2
        sdx, sdy = np.cos(np.radians(sweep)), np.sin(np.radians(sweep))
        ax.annotate("", xy=(cx + 1.1 * sdx, cy + 1.1 * sdy), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="-|>", color="crimson", lw=1.8))
        mth, Rres = _resultant(list(vth))
        al = _alignment(list(vth), sweep)
        ax.set_title(f"{reg}  ·  CV med {np.median(vcv):.0f} mm/s\n"
                     f"mean dir {mth:.0f}°  ·  align {al:+.2f}  (n={vp.shape[0]})",
                     fontsize=9)
        ax.set_aspect("equal"); ax.axis("off")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cax = fig.add_axes([0.915, 0.12, 0.014, 0.64])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("local conduction velocity (mm/s)")
    which = {"bursts": "network bursts", "digital": "laser-locked"}.get(
        mode, f"{int(PROP_PARAMS['window_ms'])} ms windows")
    fig.suptitle(f"Local conduction-velocity field — {stem}{_sweep_suffix(stem)}  "
                 f"({which}); red arrow = sweep axis ({sweep:.0f}° BR→TL).",
                 fontsize=11, y=0.97)
    _save(fig, os.path.join(out_dir, "cv_field.png"))


# ==============================================================================
# DRIVER  --  called by mea_nap_activity.run_stage2 per stem
# ==============================================================================
def render_spectral_propagation(stem, units, res_by_region, out_dir,
                                laser_by_region=None):
    """out_dir = Plots/<stem>/. Builds region_units {region:(unit,res)} then writes
    the PSD + propagation figures + local CV field. Each figure is isolated.
    laser_by_region : optional {region: array of laser ON times (s, window-relative)}
    from the MCS Digital Event Recorder -> enables the stimulus-locked figure."""
    region_units = {}
    res_map = {u["region"]: u for u in units}
    for reg, res in res_by_region.items():
        u = res_map.get(reg)
        if u is not None:
            region_units[reg] = (u, res)
    if not region_units:
        return
    lr = laser_by_region or {}
    has_laser = any(lr.get(r) is not None and len(lr.get(r)) for r in region_units)
    if SP_ENABLE.get("psd_firing_bursting", True):
        _safe(fig_psd, stem, region_units, out_dir)
    if SP_ENABLE.get("propagation_bursts", True):
        _safe(fig_propagation, stem, region_units, out_dir, "bursts")
    if SP_ENABLE.get("propagation_windows", True):
        _safe(fig_propagation, stem, region_units, out_dir, "windows")
    if SP_ENABLE.get("propagation_digital", True) and has_laser:
        _safe(fig_propagation, stem, region_units, out_dir, "digital", lr)
    if SP_ENABLE.get("cv_field", True):
        _safe(fig_cv_field, stem, region_units, out_dir, lr)


def _safe(fn, *a, **k):
    try:
        fn(*a, **k)
    except Exception as e:
        print(f"  [spectral/propagation warn] {getattr(fn, '__name__', fn)}: {e!r}")


# ==============================================================================
# SELF-TEST  --  a burst propagating along bottom-right -> top-left (135°)
# ==============================================================================
def _synthetic_region(stem, region, seed=0, prop_angle=135.0, cv_mm_s=200.0,
                      burst_rate_hz=0.5):
    rng = np.random.default_rng(seed)
    dur = 120.0
    labels = orig._mcs60_labels()
    pitch = PROP_PARAMS["electrode_pitch_um"]
    pos = {l: np.array(orig.mcs_coord(l)) * pitch for l in labels}
    # burst onsets
    n_burst = max(2, int(burst_rate_hz * dur))
    onsets = np.sort(rng.uniform(2, dur - 2, n_burst))
    dvec = np.array([np.cos(np.radians(prop_angle)), np.sin(np.radians(prop_angle))])
    spikes = {l: [] for l in labels}
    for on in onsets:
        for l in labels:
            proj = float(pos[l] @ dvec)                       # µm along direction
            lat_ms = proj / cv_mm_s                           # ms (cv in µm/ms)
            lat_ms -= (min(pos[k] @ dvec for k in labels)) / cv_mm_s
            t = on + lat_ms / 1000.0 + rng.normal(0, 0.002)
            spikes[l].append(t)
            # a couple of extra spikes in-burst
            spikes[l].extend(on + lat_ms / 1000.0 + rng.uniform(0, 0.03, 2))
    # sprinkle background spikes
    for l in labels:
        spikes[l].extend(rng.uniform(0, dur, int(0.3 * dur)))
        spikes[l] = np.sort(np.array(spikes[l], float))
    unit = {"stem": stem, "region": region, "age": 14, "group": "Control",
            "duration": dur, "labels": labels, "spikes": [spikes[l] for l in labels]}
    net_bursts = [{"start": float(o - 0.02), "end": float(o + 0.06),
                   "duration": 0.08, "n_channels": len(labels)} for o in onsets]
    res = {"fr": np.array([len(s) / dur for s in unit["spikes"]]),
           "per": {}, "net_bursts": net_bursts}
    return unit, res


if __name__ == "__main__":
    out = os.path.join(os.getcwd(), "_sp_selftest", "MEA_TEST")
    # pre: random direction; stim/post: along 135° (BR->TL), post faster (STDP-ish)
    upre, rpre = _synthetic_region("MEA_TEST", "pre", seed=1, prop_angle=40, cv_mm_s=150)
    ustim, rstim = _synthetic_region("MEA_TEST", "stim", seed=2, prop_angle=135, cv_mm_s=180)
    upost, rpost = _synthetic_region("MEA_TEST", "post", seed=3, prop_angle=135, cv_mm_s=260)
    units = [upre, ustim, upost]
    res_by_region = {"pre": rpre, "stim": rstim, "post": rpost}
    render_spectral_propagation("MEA_TEST", units, res_by_region, out)
    # report detected mean directions
    for reg in ("pre", "stim", "post"):
        u = {"pre": upre, "stim": ustim, "post": upost}[reg]
        r = res_by_region[reg]
        fits = _events_for_region(u, r, "bursts")
        mth, R = _resultant([f["theta"] for f in fits])
        cv = np.median([f["cv"] for f in fits]) if fits else float("nan")
        print(f"  {reg}: n={len(fits)} mean_dir={mth:.0f}° R={R:.2f} CV_med={cv:.0f} mm/s")
    print("SELFTEST done ->", out)
