#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Network-burst propagation latency  --  does burst travel carry the trained
 direction?
================================================================================
The phase lock-in asks whether spikes FOLLOW the stimulus cycle (seconds); this
module asks a complementary question: in which direction do the fast network
bursts themselves TRAVEL (tens of ms)? After sweep training the bursts' leading
edge visibly lags toward the top-left (a ~50 ms corner-to-corner delay was seen
by eye on MEA 100926 day 5); a 50 ms traversal is only ~6 degrees of phase at a
3.2 s sweep period, so the lock-in cannot see it.

Method, per Batch.csv row (window):
  1. Discrete fast bursts = excursions of the pooled 10 ms firing rate above
     `burst_rate_factor` x median, shorter than `burst_max_dur_s` (the same
     events PHASE_LOCKIN's exclude_fast_bursts removes; the ISI_N detector is
     unusable here -- it covers ~100 % of these dense recordings).
  2. Per burst: first-spike latency per electrode inside the event (+ tail_s),
     then a least-squares plane LAT(x, y) -> direction, speed, R^2 (the same
     construction as the spectral module's wave fit).
  3. Per window: circular-mean direction over the fitted bursts, resultant R,
     alignment = cos(mean direction - sweep axis), median speed, and the
     corner-to-corner delay that speed implies along the sweep diagonal.

Outputs under <PLOT_DIR>/Burst Latency/:
  burst_latency_<group>.png   rows = samples (training days), cols = regions;
                              polar rose of per-burst directions, black mean
                              arrow, crimson dashed sweep axis
  burst_latency_summary.png   per-condition panels: alignment with the sweep
                              axis by training day, pre vs post
  burst_latency_summary.csv   one row per sample x region
================================================================================
"""

import os
import re as _re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mea_nap_io as io
import mea_nap_phase_lockin as pl          # fast_burst_mask, day strip, PARULA
import mea_nap_original_plots as orig      # mcs_coord (electrode geometry)

BURST_LATENCY = {
    # enable : master switch; the GUI runs this after the Phase Lock-In and
    #   writes to the "Burst Latency" folder. True | False.
    "enable":               False,
    # burst_rate_factor / burst_max_dur_s : the fast-burst event definition,
    #   same meaning as the PHASE_LOCKIN filter (pooled 10 ms rate threshold
    #   x median; events longer than the max are not bursts). Ranges 2 - 10 /
    #   0.1 - 1.0.
    "burst_rate_factor":    4.0,
    "burst_max_dur_s":      0.3,
    # tail_s : latency window extends this far past the event end so late
    #   electrodes on the trailing edge still get a first spike. Range 0 - 0.2.
    "tail_s":               0.05,
    # min_lat_electrodes : electrodes with a first spike needed to fit a burst's
    #   latency plane. Range 6 - 40.
    "min_lat_electrodes":   10,
    # min_fit_r2 : minimum plane-fit R^2 for a burst to count as travelling
    #   (incoherent bursts are dropped, not forced into a direction). Range 0 - 0.9.
    "min_fit_r2":           0.30,
    # min_fits : fitted bursts a window needs before its mean direction is drawn
    #   in the summary figure (rows with fewer still reach the CSV). Range 1 - 100.
    "min_fits":             10,
    # pitch_um / sweep_angle_deg : electrode pitch and the trained axis the
    #   alignment is scored against. Ranges 100 - 700 / 0 - 360.
    "pitch_um":             200.0,
    "sweep_angle_deg":      135.0,
    # direction_window_s : bin width (s) for the burst-direction-over-time curve
    #   (burst_direction_<group>.png, the burst analogue of the phase lock-in's
    #   convergence plot). Each bin averages the per-burst convergences
    #   50*(1+cos(theta_b - sweep)) of the travelling bursts inside it. Range
    #   30 - 600.
    "direction_window_s":   60.0,
    # min_bursts_per_bin : fitted bursts a bin needs to be drawn at FULL opacity;
    #   bins with fewer (down to 1) are still drawn, faded, so sparse stretches
    #   stay visible but readable as low-confidence. Bins with ZERO travelling
    #   bursts have no direction at all and remain gaps. Range 1 - 20.
    "min_bursts_per_bin":   3,
}

DPI = 200


# ==============================================================================
# per-burst latency plane
# ==============================================================================
def _positions(labels, pitch):
    keep, pos = [], []
    for l in labels:
        c = orig.mcs_coord(l)
        if c is not None:
            keep.append(l)
            pos.append((c[0] * pitch, c[1] * pitch))
    return keep, np.asarray(pos, float)


def _fit_latency_plane(lat_ms, labels, pitch, min_n, min_r2):
    """Least-squares plane LAT(x,y). Returns dict(theta_deg, cv_um_ms, r2, n)
    or None. Same construction as mea_nap_spectral_propagation._fit_wave."""
    keep, pos = _positions(labels, pitch)
    lat = np.array([lat_ms[l] for l in keep if l in lat_ms], float)
    pos = np.array([p for l, p in zip(keep, pos) if l in lat_ms], float)
    if lat.size < min_n:
        return None
    A = np.column_stack([pos[:, 0], pos[:, 1], np.ones(lat.size)])
    coef, *_ = np.linalg.lstsq(A, lat, rcond=None)
    a, b, _c = coef
    pred = A @ coef
    sst = float(np.sum((lat - lat.mean()) ** 2))
    r2 = 1 - float(np.sum((lat - pred) ** 2)) / sst if sst > 0 else 0.0
    g = float(np.hypot(a, b))                  # ms / um
    if g <= 0 or r2 < min_r2:
        return None
    return {"theta_deg": float(np.degrees(np.arctan2(b, a)) % 360.0),
            "cv_um_ms": 1.0 / g, "r2": r2, "n": int(lat.size)}


def _fit_latency_radial(lat_ms, labels, pitch, min_n, min_r2):
    """Radial-group fit: least-squares LINE latency(r), r = distance from the
    ring centre (the spiral centre-out ordering collapsed to its 1-D
    coordinate). An outward-travelling burst gives a clean positive slope.
    Returns dict(corr, r2, speed_um_ms, span_delay_ms, n) or None.
    corr = sign(slope) * sqrt(R^2) in [-1, +1]: +1 clean outward, -1 inward."""
    cx, cy = pl._RADIAL_CENTRE_CELLS
    keep, pos = _positions(labels, pitch)
    sel = [(l, q) for l, q in zip(keep, pos) if l in lat_ms]
    if len(sel) < min_n:
        return None
    r = np.array([np.hypot(q[0] - cx * pitch, q[1] - cy * pitch)
                  for _l, q in sel])
    lat = np.array([lat_ms[l] for l, _q in sel], float)
    A = np.column_stack([r, np.ones(r.size)])
    coef, *_ = np.linalg.lstsq(A, lat, rcond=None)
    a, _c = coef
    pred = A @ coef
    sst = float(np.sum((lat - lat.mean()) ** 2))
    r2 = 1 - float(np.sum((lat - pred) ** 2)) / sst if sst > 0 else 0.0
    if r2 < min_r2 or abs(a) <= 0:
        return None
    return {"corr": float(np.sign(a) * np.sqrt(r2)), "r2": float(r2),
            "speed_um_ms": float(1.0 / abs(a)),
            "span_delay_ms": float(abs(a) * (r.max() - r.min())),
            "n": int(r.size)}


def _window_stats(unit, p):
    """All per-burst fits + the window summary for one Batch.csv unit."""
    dur = float(unit["duration"])
    events = pl.fast_burst_mask(unit["spikes"], dur,
                                enter_k=float(p["burst_rate_factor"]),
                                max_dur_s=float(p["burst_max_dur_s"]))
    sp64 = [(l, np.asarray(s, float))
            for l, s in zip(unit["labels"], unit["spikes"]) if len(s)]
    tail = float(p["tail_s"])
    radial = pl._is_radial_group(unit.get("group", ""))
    fits = []
    for (a, b) in events:
        lat = {}
        for l, s in sp64:
            i = int(np.searchsorted(s, a, side="left"))
            if i < s.size and s[i] < b + tail:
                lat[l] = (s[i] - a) * 1000.0
        if radial:
            f = _fit_latency_radial(lat, unit["labels"], p["pitch_um"],
                                    int(p["min_lat_electrodes"]),
                                    float(p["min_fit_r2"]))
        else:
            f = _fit_latency_plane(lat, unit["labels"], p["pitch_um"],
                                   int(p["min_lat_electrodes"]),
                                   float(p["min_fit_r2"]))
        if f is not None:
            f["t_s"] = 0.5 * (a + b)          # event midpoint, window-relative
            fits.append(f)
    st = {"n_events": len(events), "n_fits": len(fits),
          "model": "radial" if radial else "plane"}
    if fits and radial:
        corr = np.array([f["corr"] for f in fits])
        st.update(mean_dir_deg=None, R=None,
                  alignment=float(corr.mean()),
                  outward_frac=float((corr > 0).mean()),
                  median_cv_um_ms=float(np.median([f["speed_um_ms"]
                                                   for f in fits])),
                  diag_delay_ms=float(np.median([f["span_delay_ms"]
                                                 for f in fits])))
    elif fits:
        th = np.radians([f["theta_deg"] for f in fits])
        Rv = np.exp(1j * th).mean()
        sweep = float(p["sweep_angle_deg"])
        diag = 7 * p["pitch_um"] * (abs(np.cos(np.radians(sweep)))
                                    + abs(np.sin(np.radians(sweep))))
        cv = float(np.median([f["cv_um_ms"] for f in fits]))
        st.update(mean_dir_deg=float(np.degrees(np.angle(Rv))) % 360.0,
                  R=float(abs(Rv)),
                  alignment=float(np.cos(np.angle(Rv) - np.radians(sweep))),
                  median_cv_um_ms=cv,
                  diag_delay_ms=(diag / cv if cv > 0 else float("nan")))
    return st, fits


# ==============================================================================
# figures
# ==============================================================================
def _rose_panel(ax, fits, st, sweep_deg, min_fits):
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_facecolor("#FAFAFA")
    ax.set_yticklabels([])
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["0°", "90°", "180°", "270°"],
                       fontsize=6.0, color="#777")
    ax.tick_params(axis="x", pad=-4.0)
    ax.grid(True, color="#CCC", lw=0.5)
    if not fits:
        ax.text(0.5, 0.5, "no travelling bursts", ha="center", va="center",
                fontsize=7.5, color="#888", style="italic",
                transform=ax.transAxes)
        return
    th = np.radians([f["theta_deg"] for f in fits])
    cnt, edges = np.histogram(th, bins=18, range=(0, 2 * np.pi))
    top = max(cnt.max(), 1)
    ax.bar(0.5 * (edges[:-1] + edges[1:]), cnt / top,
           width=(2 * np.pi / 18) * 0.95, bottom=0.0, color="#B9C6D2",
           alpha=0.6, edgecolor="none", zorder=2)
    ax.set_ylim(0, 1.08)
    sr = np.radians(sweep_deg)
    ax.plot([sr, sr], [0, 1.05], "--", color="crimson", lw=1.4, zorder=4)
    if st.get("R") is not None:
        ax.annotate("", xy=(np.radians(st["mean_dir_deg"]), min(st["R"], 1.0)),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", lw=2.2, color="black",
                                    mutation_scale=14), zorder=6)
        low = st["n_fits"] < int(min_fits)
        ax.text(1.02, 1.02,
                f"n={st['n_fits']}  R={st['R']:.2f}\n"
                f"align {st['alignment']:+.2f}\n"
                f"diag {st['diag_delay_ms']:.0f} ms" + ("  (few!)" if low else ""),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
                color="#B03A2E" if low else "#555",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                          alpha=0.75))


def _radial_hist_panel(ax, fits, st, min_fits):
    """Radial-group replacement for the direction rose: histogram of each
    burst's signed latency-radius correlation (+1 = clean outward slope)."""
    ax.set_facecolor("#FAFAFA")
    ax.tick_params(labelsize=7, colors="#555")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.axvline(0, color="#CCC", lw=0.8)
    if not fits:
        ax.text(0.5, 0.5, "no travelling bursts", ha="center", va="center",
                fontsize=7.5, color="#888", style="italic",
                transform=ax.transAxes)
        ax.set_yticks([])
        return
    corr = np.array([f["corr"] for f in fits])
    ax.hist(corr, bins=np.linspace(-1, 1, 21), color="#B9C6D2",
            edgecolor="none", zorder=2)
    ax.axvline(float(corr.mean()), color="black", lw=2.0, zorder=4)
    ax.axvline(1.0, color="crimson", lw=1.2, ls="--", zorder=3)
    low = st.get("n_fits", 0) < int(min_fits)
    ax.text(0.02, 0.98,
            f"n={st['n_fits']}  outward {100*st.get('outward_frac', 0):.0f}%\n"
            f"mean corr {st.get('alignment', 0):+.2f}\n"
            f"radial delay {st.get('diag_delay_ms', float('nan')):.0f} ms"
            + ("  (few!)" if low else ""),
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
            color="#B03A2E" if low else "#555",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                      alpha=0.75))
    ax.set_xlabel("latency–radius corr (+1 = outward)", fontsize=7,
                  color="#555")


def _fig_group(group, samples, p, out_dir):
    """samples : [(stem, {region: (stats, fits)}, age), ...]"""
    samples = sorted(samples, key=lambda t: (t[2] if t[2] is not None else 1e9,
                                             t[0]))
    regions = []
    for r in ("pre", "stim", "post"):
        if any(r in d for _s, d, _a in samples):
            regions.append(r)
    for _s, d, _a in samples:
        for r in d:
            if r not in regions:
                regions.append(r)
    if not samples or not regions:
        return
    nr, nc = len(samples), len(regions)
    fig = plt.figure(figsize=(3.0 * nc + 2.4, 3.3 * nr), constrained_layout=True)
    gs = fig.add_gridspec(nr, nc + 1, width_ratios=[0.20] + [1] * nc)
    for i, (stem, d, age) in enumerate(samples):
        pl._day_strip(fig.add_subplot(gs[i, 0]), i, nr, stem, age)
        for j, region in enumerate(regions):
            radial = pl._is_radial_group(group)
            ax = fig.add_subplot(gs[i, j + 1], polar=not radial)
            st, fits, _win = d.get(region, ({}, [], (0.0, 0.0)))
            try:
                if radial:
                    _radial_hist_panel(ax, fits, st, p["min_fits"])
                else:
                    _rose_panel(ax, fits, st, p["sweep_angle_deg"],
                                p["min_fits"])
            except Exception as e:                      # never lose the figure
                ax.text(0.5, 0.5, f"panel failed\n{e!r}"[:60], fontsize=7,
                        ha="center", va="center", color="crimson",
                        transform=ax.transAxes)
            if i == 0:
                ax.set_title(region, fontsize=12, fontweight="bold", pad=12)
    if pl._is_radial_group(group):
        title = (f"Network-burst propagation — group {group}   (rows = "
                 f"training days, youngest at top; histogram = per-burst "
                 f"latency–radius correlation from the spiral (centre-out) "
                 f"ordering, +1 = clean outward slope; black line = mean, red "
                 f"dashed = outward; fits with R² ≥ "
                 f"{p['min_fit_r2']:g})")
    else:
        title = (f"Network-burst propagation — group {group}   (rows = "
                 f"training days, youngest at top; rose = per-burst "
                 f"latency-plane directions, black arrow = circular mean "
                 f"(length = consistency R), red dashed = sweep "
                 f"{p['sweep_angle_deg']:.0f}°; bursts with plane "
                 f"R² ≥ {p['min_fit_r2']:g})")
    fig.suptitle(title, fontsize=11)
    pl._save(fig, os.path.join(out_dir, f"burst_latency_{group}.png"))


def _burst_timecourse(regions, p):
    """Burst-direction convergence over the session, shaped like the phase
    lock-in's series dicts so pl._draw_direction_axes / pl._mean_timecourse can
    draw and average it: {"bins", "bands", "window_s", "mode": "linear"}.
    regions : {region: (stats, fits, win)}. Each bin averages the per-burst
    convergences 50*(1+cos(theta_b - sweep)); the SEM is across bursts and the
    line colour carries the bin's direction consistency R."""
    win_s = float(p.get("direction_window_s", 60.0) or 0.0)
    if win_s <= 0:
        return None
    sweep = float(p["sweep_angle_deg"])
    min_n = int(p.get("min_bursts_per_bin", 3))
    bins, bands, events = [], [], []
    for region, (st, fits, win) in regions.items():
        t_off, t_end = float(win[0]), float(win[1])
        dur = t_end - t_off
        if dur <= 0:
            continue
        bands.append((t_off, t_end, region))
        if not fits:
            continue
        t = np.array([f["t_s"] for f in fits])
        if "corr" in fits[0]:                 # radial model: spiral regression
            conv = 50.0 * (1.0 + np.array([f["corr"] for f in fits]))
            th = np.where(conv >= 50.0, 0.0, 180.0)   # outward / inward
        else:
            th = np.array([f["theta_deg"] for f in fits])
            conv = 50.0 * (1.0 + np.cos(np.radians(th - sweep)))
        for f, c in zip(fits, conv):
            events.append({"t_s": t_off + float(f["t_s"]), "convergence_pct":
                           float(c), "r2": float(f["r2"]), "region": region})
        for b in range(int(np.floor(dur / win_s))):
            sel = (t >= b * win_s) & (t < (b + 1) * win_s)
            n = int(sel.sum())
            if n < 1:
                continue                      # no travelling burst = no direction
            Rv = np.exp(1j * np.radians(np.asarray(th, float)[sel])).mean()
            bins.append({"t_mid_s": t_off + (b + 0.5) * win_s,
                         "direction_deg": float(np.degrees(np.angle(Rv))),
                         "convergence_pct": float(conv[sel].mean()),
                         "sem": (float(conv[sel].std(ddof=1) / np.sqrt(n))
                                 if n > 1 else None),
                         "coherence": float(abs(Rv)),   # consistency R (CSV/debug)
                         "weak": bool(n < min_n),
                         "n_locked": n, "region": region})
    bands.sort(key=lambda x: x[0])
    bins.sort(key=lambda d: d["t_mid_s"])
    events.sort(key=lambda d: d["t_s"])
    radial_any = any(("corr" in f) for (_st, fits, _w) in regions.values()
                     for f in fits)
    return {"bins": bins, "bands": bands, "window_s": win_s,
            "mode": "radial" if radial_any else "linear", "events": events}


def _draw_direction_axes_plain(ax, series, p, label_bands=False):
    """Burst-travel convergence row: one plain colour, raw unsmoothed line.
    Bins flagged weak (fewer than min_bursts_per_bin travelling bursts) and the
    segments touching them are drawn faded so sparse stretches read as
    low-confidence instead of disappearing."""
    sweep = float(p["sweep_angle_deg"])
    LINE = plt.get_cmap("Blues")(0.92)
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
        return
    t_lo = min(b[0] for b in bands) / 60.0
    t_hi = max(b[1] for b in bands) / 60.0
    ax.set_xlim(t_lo, t_hi)
    ax.set_ylim(-4.0, 114.0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("convergence on training angle", fontsize=8.5, color="#555")
    ax.grid(axis="y", color="#E2E2E2", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for (bt0, bt1, region) in bands:
        ax.axvspan(bt0 / 60.0, bt1 / 60.0, color=pl._region_fill(region), lw=0,
                   zorder=1)
        ax.axvline(bt0 / 60.0, color="#BFCBD8", lw=0.8, zorder=2)
        if label_bands:
            ax.text(0.5 * (bt0 + bt1) / 60.0, 112.0, str(region),
                    ha="center", va="top", fontsize=8.5, fontweight="bold",
                    color=pl._region_ink(region), zorder=5)
    radial = (series or {}).get("mode") == "radial"
    ax.axhline(100.0, color="crimson", lw=1.2, ls="--", alpha=0.85, zorder=3)
    ax.text(t_hi, 100.0, (" outward" if radial else f" sweep {sweep:.0f}°"),
            fontsize=7, color="crimson", va="bottom", ha="right", zorder=5)
    if radial:
        ax.set_ylabel("outward correlation (50% = none)", fontsize=8.5,
                      color="#555")
    events = (series or {}).get("events") or []
    if not events:
        ax.text(0.5, 0.5, "no travelling bursts", ha="center", va="center",
                fontsize=9, color="#888", style="italic",
                transform=ax.transAxes, zorder=6)
        return
    # one dot per burst at its actual time; opacity = that burst's latency-plane
    # R^2, rescaled from [min_fit_r2, 1] to [0.15, 0.95], so a barely-accepted
    # fit is faint and a clean plane is solid. No binning, no line.
    t = np.array([e["t_s"] for e in events], float) / 60.0
    y = np.clip(np.array([e["convergence_pct"] for e in events], float), 0, 100)
    r2 = np.array([e["r2"] for e in events], float)
    r2lo = float(p.get("min_fit_r2", 0.30))
    a = 0.15 + 0.80 * np.clip((r2 - r2lo) / max(1.0 - r2lo, 1e-9), 0.0, 1.0)
    base = np.array(LINE, float)
    cols = np.tile(base, (t.size, 1))
    cols[:, 3] = a
    ax.scatter(t, y, s=11, c=cols, edgecolors="none", zorder=5)


def _mean_timecourse(members, win_s, min_n):
    """Average replicate series bin-by-bin (same (region, bin-index) alignment
    as the phase lock-in's) while summing the burst counts, so the mean bin can
    be flagged weak when its members held few travelling bursts in total."""
    acc, t0s, durs = {}, {}, {}
    for sr in members:
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
            acc.setdefault((r, k), []).append(
                (b["convergence_pct"], int(b.get("n_locked", 0))))
    regions = sorted(t0s, key=lambda r: float(np.median(t0s[r])))
    bands = [(float(np.median(t0s[r])),
              float(np.median(t0s[r]) + np.median(durs[r])), r)
             for r in regions]
    span = {r: (a, b) for (a, b, r) in bands}
    bins = []
    for (r, k), vals in acc.items():
        a, b = span[r]
        tm = a + (k + 0.5) * win_s
        if not (a <= tm <= b):
            continue
        conv = np.array([v[0] for v in vals], float)
        n_bursts = int(sum(v[1] for v in vals))
        n = conv.size
        bins.append({"t_mid_s": tm, "region": r, "n": n,
                     "convergence_pct": float(conv.mean()),
                     "sem": (float(conv.std(ddof=1) / np.sqrt(n)) if n > 1
                             else None),
                     "coherence": 0.0,
                     "weak": bool(n_bursts < min_n),
                     "n_locked": n_bursts})
    bins.sort(key=lambda d: d["t_mid_s"])
    return {"bins": bins, "bands": bands, "window_s": win_s, "mode": "linear"}


def _fig_group_direction(group, samples, p, out_dir):
    """burst_direction_<group>.png — rows = samples, plain raw lines.
    samples : [(stem, series, age), ...]"""
    samples = [t for t in samples if t[1] and t[1].get("bands")]
    samples = sorted(samples, key=lambda t: (t[2] if t[2] is not None else 1e9,
                                             t[0]))
    if not samples:
        return
    nr = len(samples)
    win_s = float(p.get("direction_window_s", 60.0) or 60.0)
    fig = plt.figure(figsize=(13.0, 2.7 * nr + 0.9), constrained_layout=True)
    gs = fig.add_gridspec(nr, 2, width_ratios=[0.18, 1])
    for i, (stem, series, age) in enumerate(samples):
        pl._day_strip(fig.add_subplot(gs[i, 0]), i, nr, stem, age)
        ax = fig.add_subplot(gs[i, 1])
        try:
            _draw_direction_axes_plain(ax, series, p, label_bands=(i == 0))
        except Exception as e:                       # never lose the figure
            ax.text(0.5, 0.5, f"row failed\n{e!r}"[:70], fontsize=7,
                    ha="center", va="center", color="crimson",
                    transform=ax.transAxes)
        if i == nr - 1:
            ax.set_xlabel("session time (min)", fontsize=9, color="#555")
    if any(s[1] and s[1].get("mode") == "radial" for s in samples):
        score = ("the spiral latency–radius regression "
                 "(100% = clean outward slope, 0% = inward)")
    else:
        score = f"sweep {float(p['sweep_angle_deg']):.0f}°"
    fig.suptitle(
        f"Burst-travel convergence over time — group {group}   (rows = "
        f"training days, youngest at top; each dot = ONE travelling burst at "
        f"its actual time, scored on {score}; opacity = that burst's "
        f"fit R², faint = barely accepted)",
        fontsize=11)
    pl._save(fig, os.path.join(out_dir, f"burst_direction_{group}.png"))


def _fig_direction_means(dir_samples, p, out_dir):
    """burst_direction_mean_<condition>.png — day rows averaged across
    replicate samples via pl._mean_timecourse."""
    win_s = float(p.get("direction_window_s", 60.0) or 60.0)
    by_cond = {}
    for grp, items in dir_samples.items():
        cond = _condition_of(grp)
        for stem, series, age in items:
            if not series or not series.get("bands"):
                continue
            day = pl._training_day(stem, age)
            by_cond.setdefault(cond, {}).setdefault(day, []).append(
                (series, age))
    written = 0
    for cond, days in by_cond.items():
        if max(len(v) for v in days.values()) < 2:
            continue
        order = sorted(days, key=lambda d: (isinstance(d, float), d))
        nr = len(order)
        fig = plt.figure(figsize=(13.0, 2.7 * nr + 0.9),
                         constrained_layout=True)
        gs = fig.add_gridspec(nr, 2, width_ratios=[0.18, 1])
        min_n = int(p.get("min_bursts_per_bin", 3))
        for i, day in enumerate(order):
            members = days[day]
            mean = _mean_timecourse([m[0] for m in members], win_s, min_n)
            mean["events"] = sorted(
                (e for m in members for e in (m[0].get("events") or [])),
                key=lambda d: d["t_s"])
            pl._mean_day_strip(fig.add_subplot(gs[i, 0]), i, nr, day,
                               len(members), [m[1] for m in members])
            ax = fig.add_subplot(gs[i, 1])
            try:
                _draw_direction_axes_plain(ax, mean, p, label_bands=(i == 0))
            except Exception as e:
                ax.text(0.5, 0.5, f"row failed\n{e!r}"[:70], fontsize=7,
                        ha="center", va="center", color="crimson",
                        transform=ax.transAxes)
            if i == nr - 1:
                ax.set_xlabel("session time (min)", fontsize=9, color="#555")
        if pl._is_radial_group(cond):
            score = ("the spiral latency–radius regression "
                     "(100% = clean outward slope, 0% = inward)")
        else:
            score = f"sweep {float(p['sweep_angle_deg']):.0f}°"
        fig.suptitle(
            f"Burst-travel convergence over time — {cond}, all samples "
            f"pooled   (rows = training days; each dot = ONE travelling burst "
            f"at its actual time, every replicate overlaid, scored on {score}; "
            f"opacity = that burst's fit R²)",
            fontsize=11)
        pl._save(fig, os.path.join(out_dir,
                                   f"burst_direction_mean_{cond}.png"))
        written += 1
    return written


def _condition_of(group):
    g = str(group or "").strip()
    return _re.sub(r"[\s_-]*\d+$", "", g) or g


def _fig_summary(rows, p, out_dir):
    """rows : list of dicts (the CSV rows). One panel per condition."""
    conds = sorted({_condition_of(r["group"]) for r in rows})
    if not conds:
        return
    blu = plt.get_cmap("Blues")
    fig, axes = plt.subplots(1, len(conds), figsize=(4.4 * len(conds) + 0.8, 4.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    days = sorted({r["day"] for r in rows if r["day"] is not None})
    for ax, cond in zip(axes, conds):
        ax.set_facecolor("#FAFAFA")
        ax.axhline(0, color="#CCC", lw=0.8)
        ax.axhline(1, color="crimson", lw=1.1, ls="--", alpha=0.8)
        series = [("pre", 0.45), ("post", 0.95), ("stim", 0.70)]
        for region, shade in series:
            xs, ys = [], []
            for r in rows:
                if (_condition_of(r["group"]) != cond or r["region"] != region
                        or r.get("alignment") is None
                        or r["n_fits"] < int(p["min_fits"])
                        or r["day"] is None):
                    continue
                off = {"pre": -0.22, "stim": 0.0, "post": 0.22}[region]
                xs.append(r["day"] + off)
                ys.append(r["alignment"])
            if not xs:
                continue
            ax.plot(xs, ys, "o", ms=6, color=blu(shade), mec="#333", mew=0.4,
                    alpha=0.9, label=region)
        ax.set_title(cond, fontsize=11)
        ax.set_xlabel("training day")
        if days:
            ax.set_xticks(days)
        ax.set_ylim(-1.15, 1.25)
    axes[0].set_ylabel("burst-direction alignment with sweep "
                       f"{p['sweep_angle_deg']:.0f}°\n(cos deviation)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        f"Network-burst propagation vs the trained axis — each dot = one "
        f"sample's circular-mean burst direction (≥ {int(p['min_fits'])} "
        f"fitted bursts)", fontsize=10)
    fig.tight_layout()
    pl._save(fig, os.path.join(out_dir, "burst_latency_summary.png"))


def _write_csv(rows, out_dir):
    import csv
    cols = ["stem", "region", "group", "day", "age", "model", "n_events",
            "n_fits", "mean_dir_deg", "R", "alignment", "outward_frac",
            "median_cv_um_ms", "diag_delay_ms"]
    path = os.path.join(out_dir, "burst_latency_summary.csv")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v)
                        for k, v in r.items() if k in cols})
    io._log(f"  [BurstLat] summary CSV -> {path}")


# ==============================================================================
# driver
# ==============================================================================
def run_burst_latency():
    """Analyse every Batch.csv row enabled for analysis; write figures + CSV to
    the 'Burst Latency' folder under PLOT_DIR."""
    p = BURST_LATENCY
    csv_path = os.path.join(io.CONFIG["DATA_DIR"],
                            io.CONFIG.get("BATCH_CSV_NAME", "Batch.csv"))
    if not os.path.exists(csv_path):
        io._log(f"[BurstLat] Batch.csv not found at {csv_path}; nothing to do.")
        return
    rows_in = io.filter_rows_by_group(io.parse_batch(csv_path), "analyze")
    if not rows_in:
        io._log("[BurstLat] no rows to analyse.")
        return
    out_dir = os.path.join(os.path.normpath(io.CONFIG["PLOT_DIR"]),
                           "Burst Latency")
    io._log(f"[BurstLat] network-burst propagation -> {out_dir}")
    by_group = {}
    csv_rows = []
    for u in io.iter_batch_units(rows_in):
        io.check_abort()
        st, fits = _window_stats(u, p)
        m = _re.search(r"_D(\d+)", u["stem"])
        day = int(m.group(1)) if m else None
        by_group.setdefault(u["group"], {}).setdefault(
            u["stem"], ({}, u.get("age")))[0][u["region"]] = (
                st, fits, u.get("win", (0.0, float(u["duration"]))))
        csv_rows.append({"stem": u["stem"], "region": u["region"],
                         "group": u["group"], "day": day, "age": u.get("age"),
                         **st})
        if st.get("R") is not None:
            msg = (f"dir {st['mean_dir_deg']:5.1f}  R {st['R']:.2f}  align "
                   f"{st['alignment']:+.2f}  diag {st['diag_delay_ms']:5.1f} ms")
        elif st.get("alignment") is not None:
            msg = (f"outward corr {st['alignment']:+.2f}  outward "
                   f"{100*st['outward_frac']:.0f}%  radial delay "
                   f"{st['diag_delay_ms']:5.1f} ms")
        else:
            msg = "too few travelling bursts"
        io._log(f"  [BurstLat] {u['stem']} [{u['region']}]: "
                f"{st['n_events']} burst(s), {st['n_fits']} fit(s)  {msg}")
    dir_samples = {}
    for group, stems in by_group.items():
        samples = [(stem, d, age) for stem, (d, age) in stems.items()]
        _fig_group(group, samples, p, out_dir)
        for stem, (d, age) in stems.items():
            try:
                series = _burst_timecourse(d, p)
            except Exception as e:
                series = None
                io._log(f"  [BurstLat] {stem}: time course failed ({e!r})")
            dir_samples.setdefault(group, []).append((stem, series, age))
        _fig_group_direction(group, dir_samples[group], p, out_dir)
    _write_csv(csv_rows, out_dir)
    _fig_summary(csv_rows, p, out_dir)
    n_mean = _fig_direction_means(dir_samples, p, out_dir)
    io._log(f"[BurstLat] done - {len(by_group)} group figure(s), "
            f"{len(by_group)} direction curve(s), {n_mean} mean curve(s) + summary.")


# ==============================================================================
# self-test: synthetic travelling bursts must recover the planted direction
# ==============================================================================
def _selftest():
    rng = np.random.default_rng(0)
    labels = orig._mcs60_labels()
    p = dict(BURST_LATENCY)
    pitch = p["pitch_um"]
    pos = {l: np.array(orig.mcs_coord(l)) * pitch for l in labels}
    d = np.array([np.cos(np.radians(135.0)), np.sin(np.radians(135.0))])
    proj = {l: float(pos[l] @ d) for l in labels}
    p0 = min(proj.values()); span = max(proj.values()) - p0
    dur = 60.0
    spikes = {l: [] for l in labels}
    for k in range(40):                       # 40 bursts, 50 ms corner-to-corner
        t0 = 1.0 + k * 1.4
        for l in labels:
            lat = (proj[l] - p0) / span * 0.050
            for j in range(3):
                spikes[l].append(t0 + lat + rng.normal(0, 0.002) + j * 0.004)
    for l in labels:                          # 2 Hz background
        spikes[l].extend(rng.uniform(0, dur, int(2 * dur)))
    unit = {"labels": labels,
            "spikes": [np.sort(np.array(spikes[l])) for l in labels],
            "duration": dur}
    # radial case: bursts whose latency grows with radius must give corr ~ +1
    cx, cy = pl._RADIAL_CENTRE_CELLS
    rr = {l: float(np.hypot(pos[l][0] - cx * pitch, pos[l][1] - cy * pitch))
          for l in labels}
    rmax = max(rr.values())
    for outward, want in ((True, 1.0), (False, -1.0)):
        rs = {l: [] for l in labels}
        for k in range(40):
            t0 = 1.0 + k * 1.4
            for l in labels:
                lat = (rr[l] / rmax if outward else 1 - rr[l] / rmax) * 0.050
                for j in range(3):
                    rs[l].append(t0 + lat + rng.normal(0, 0.002) + j * 0.004)
        for l in labels:
            rs[l].extend(rng.uniform(0, dur, int(2 * dur)))
        runit = {"labels": labels, "group": "radial1",
                 "spikes": [np.sort(np.array(rs[l])) for l in labels],
                 "duration": dur}
        rst, _rf = _window_stats(runit, p)
        print(f"  radial {'outward' if outward else 'inward '}: mean corr "
              f"{rst['alignment']:+.2f} (want {want:+.0f}), outward "
              f"{100*rst['outward_frac']:.0f}%, fits {rst['n_fits']}")
        # background spikes inside the burst window corrupt a few first-spike
        # latencies, so the mean |corr| of the planted train is ~0.7, not 1.0;
        # the outward FRACTION is the robust check.
        frac_ok = (rst["outward_frac"] > 0.8) if want > 0 else (rst["outward_frac"] < 0.2)
        assert rst["n_fits"] >= 30 and abs(rst["alignment"] - want) < 0.45 and frac_ok

    st, fits = _window_stats(unit, p)
    series = _burst_timecourse({"stim": (st, fits, (0.0, dur))}, p)
    tb = [b["convergence_pct"] for b in series["bins"]]
    print(f"  time course: {len(tb)} bin(s), convergence mean "
          f"{np.mean(tb):.1f}% (planted ~100%)")
    dev = (st["mean_dir_deg"] - 135.0 + 180) % 360 - 180
    ok = (st["n_fits"] >= 30 and abs(dev) < 10 and st["R"] > 0.7
          and abs(st["diag_delay_ms"] - 50.0) < 15.0
          and len(tb) >= 1 and np.mean(tb) > 90.0)
    print(f"  bursts fitted {st['n_fits']}/40  mean dir {st['mean_dir_deg']:.1f} "
          f"(planted 135)  R {st['R']:.2f}  diag delay {st['diag_delay_ms']:.1f} ms "
          f"(planted 50)")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    _selftest()
