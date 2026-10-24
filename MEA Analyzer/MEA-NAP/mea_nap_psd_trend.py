#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 PSD trends  --  did stimulation change the spontaneous bursting rhythm?
================================================================================
Per Batch.csv window this module computes the population-rate power spectrum
(Welch, like the per-recording PSD figure in mea_nap_spectral_propagation) and
two robust burst-rhythm numbers from the discrete fast-burst events (the same
pooled-rate events the phase lock-in's filter and the Burst Latency module use):

  burst_rate_per_min : fast-burst events per minute of window
  burst_freq_hz      : 1 / median inter-burst interval
  psd_peak_hz        : frequency of the largest population-rate PSD peak inside
                       [peak_lo_hz, peak_hi_hz] (the slow bursting band)

and compares them ACROSS TRAINING DAYS, pre vs stim vs post, per condition.

Outputs under <PLOT_DIR>/PSD Trends/:
  psd_overlay_<condition>.png  mean population-rate PSD per day (Blues ramp,
                               dark = later day), one panel per region
  psd_trend_summary.png        burst rate / burst frequency / PSD peak by day,
                               one column per condition, pre vs stim vs post
  psd_trend_summary.csv        one row per sample x region
================================================================================
"""

import os
import re as _re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mea_nap_io as io
import mea_nap_phase_lockin as pl          # fast_burst_mask, _save

PSD_TREND = {
    # enable : master switch; runs after the Burst Latency section and writes to
    #   the "PSD Trends" folder. True | False.
    "enable":            False,
    # bin_s : population-rate bin (s); sampling rate = 1/bin, so 0.05 s sees
    #   rhythms up to 10 Hz. Range 0.02 - 0.2.
    "bin_s":             0.05,
    # welch_nperseg : Welch segment length (samples of the binned rate). 2048
    #   at 20 Hz = 102 s per segment -> 0.01 Hz resolution; shortened
    #   automatically for windows that hold fewer samples. Range 256 - 8192.
    "welch_nperseg":     2048,
    # fmax_hz : highest frequency drawn in the overlay. Range 0.5 - 10.
    "fmax_hz":           2.0,
    # peak_lo_hz / peak_hi_hz : band searched for the dominant slow (bursting)
    #   PSD peak. Ranges 0.005 - 1 / 0.05 - 5.
    "peak_lo_hz":        0.02,
    "peak_hi_hz":        2.0,
    # burst_rate_factor / burst_max_dur_s : fast-burst event definition (same
    #   meaning as PHASE_LOCKIN / BURST_LATENCY). Ranges 2 - 10 / 0.1 - 1.0.
    "burst_rate_factor": 4.0,
    "burst_max_dur_s":   0.3,
}


# ==============================================================================
# per-window numbers
# ==============================================================================
def _population_rate(unit, bin_s):
    dur = float(unit["duration"])
    n = max(2, int(np.floor(dur / bin_s)))
    allsp = [np.asarray(s, float) for s in unit["spikes"] if len(s)]
    t = np.concatenate(allsp) if allsp else np.empty(0)
    rate, _ = np.histogram(t[(t >= 0) & (t < n * bin_s)],
                           bins=n, range=(0, n * bin_s))
    return rate.astype(float), 1.0 / bin_s


def window_psd_stats(unit, p):
    """(stats dict, (f, psd) of the population rate) for one unit."""
    from scipy.signal import welch
    dur = float(unit["duration"])
    rate, fs = _population_rate(unit, float(p["bin_s"]))
    nper = int(min(int(p["welch_nperseg"]), rate.size))
    if rate.size < 8 or np.allclose(rate, rate.mean()):
        f, psd = np.empty(0), np.empty(0)
    else:
        f, psd = welch(rate - rate.mean(), fs=fs, nperseg=nper)
    events = pl.fast_burst_mask(unit["spikes"], dur,
                                enter_k=float(p["burst_rate_factor"]),
                                max_dur_s=float(p["burst_max_dur_s"]))
    st = {"duration_s": dur, "n_bursts": len(events),
          "burst_rate_per_min": len(events) / dur * 60.0 if dur > 0 else np.nan}
    if len(events) > 2:
        ibi = np.diff([0.5 * (a + b) for a, b in events])
        med = float(np.median(ibi))
        st["median_ibi_s"] = med
        st["burst_freq_hz"] = (1.0 / med) if med > 0 else np.nan
    else:
        st["median_ibi_s"] = np.nan
        st["burst_freq_hz"] = np.nan
    if f.size:
        band = (f >= float(p["peak_lo_hz"])) & (f <= float(p["peak_hi_hz"]))
        if band.any():
            bi = np.where(band)[0]
            st["psd_peak_hz"] = float(f[bi[np.argmax(psd[bi])]])
        else:
            st["psd_peak_hz"] = np.nan
    else:
        st["psd_peak_hz"] = np.nan
    return st, (f, psd)


# ==============================================================================
# figures
# ==============================================================================
def _condition_of(group):
    g = str(group or "").strip()
    return _re.sub(r"[\s_-]*\d+$", "", g) or g


def _fig_overlays(psds, p, out_dir):
    """psds : {(condition, region, day): [(f, psd), ...]} -> one figure per
    condition, one panel per region, one Blues-shaded mean line per day."""
    blu = plt.get_cmap("Blues")
    conds = sorted({c for (c, _r, _d) in psds})
    for cond in conds:
        regions = [r for r in ("pre", "stim", "post")
                   if any(c == cond and r == rr for (c, rr, _d) in psds)]
        regions += sorted({r for (c, r, _d) in psds
                           if c == cond and r not in regions})
        days = sorted({d for (c, _r, d) in psds if c == cond})
        if not regions or not days:
            continue
        fig, axes = plt.subplots(1, len(regions),
                                 figsize=(4.4 * len(regions) + 0.6, 4.0),
                                 sharey=True, squeeze=False)
        axes = axes[0]
        for ax, region in zip(axes, regions):
            ax.set_facecolor("#FAFAFA")
            for di, day in enumerate(days):
                entries = psds.get((cond, region, day), [])
                curves = [(f, s) for (f, s) in entries if f.size]
                if not curves:
                    continue
                # average on the shortest common grid
                nmin = min(f.size for f, _s in curves)
                f = curves[0][0][:nmin]
                m = np.mean([s[:nmin] for _f, s in curves], axis=0)
                sel = f <= float(p["fmax_hz"])
                ax.plot(f[sel], m[sel], lw=1.6,
                        color=blu(0.35 + 0.55 * di / max(1, len(days) - 1)),
                        label=f"day {day}")
            ax.set_yscale("log")
            ax.set_xlim(0, float(p["fmax_hz"]))
            ax.set_xlabel("frequency (Hz)")
            ax.set_title(region, fontsize=11)
            ax.grid(color="#E2E2E2", lw=0.6)
            ax.set_axisbelow(True)
        axes[0].set_ylabel("population-rate power")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Population-rate PSD by training day — {cond}   "
                     f"(mean across samples; darker = later day; Welch on the "
                     f"{p['bin_s']*1000:.0f} ms-binned array firing rate)",
                     fontsize=10)
        fig.tight_layout()
        pl._save(fig, os.path.join(out_dir, f"psd_overlay_{cond}.png"))


def _fig_summary(rows, p, out_dir):
    blu = plt.get_cmap("Blues")
    conds = sorted({_condition_of(r["group"]) for r in rows})
    metrics = [("burst_rate_per_min", "fast bursts / min"),
               ("burst_freq_hz", "burst frequency (Hz)\n(1 / median IBI)"),
               ("psd_peak_hz", "dominant PSD peak (Hz)")]
    if not conds:
        return
    fig, axes = plt.subplots(len(metrics), len(conds),
                             figsize=(4.0 * len(conds) + 0.8,
                                      2.9 * len(metrics)),
                             sharex="col", squeeze=False)
    days = sorted({r["day"] for r in rows if r["day"] is not None})
    for mi, (key, label) in enumerate(metrics):
        for ci, cond in enumerate(conds):
            ax = axes[mi][ci]
            ax.set_facecolor("#FAFAFA")
            ax.grid(axis="y", color="#E2E2E2", lw=0.6)
            ax.set_axisbelow(True)
            for region, shade, off in (("pre", 0.45, -0.22),
                                       ("stim", 0.70, 0.0),
                                       ("post", 0.95, 0.22),
                                       ("Control", 0.70, 0.0)):
                xs, ys = [], []
                for r in rows:
                    v = r.get(key)
                    if (_condition_of(r["group"]) != cond
                            or r["region"] != region or r["day"] is None
                            or v is None or not np.isfinite(v)):
                        continue
                    xs.append(r["day"] + off)
                    ys.append(v)
                if not xs:
                    continue
                ax.plot(xs, ys, "o", ms=5.5, color=blu(shade), mec="#333",
                        mew=0.4, alpha=0.9,
                        label=region if mi == 0 and ci == 0 else None)
                import collections
                by_day = collections.defaultdict(list)
                for x, y in zip(xs, ys):
                    by_day[round(x - off)].append(y)
                for d, vals in by_day.items():
                    ax.plot([d + off - 0.12, d + off + 0.12],
                            [np.mean(vals)] * 2, color=blu(shade), lw=2.2)
            if mi == 0:
                ax.set_title(cond, fontsize=11)
            if mi == len(metrics) - 1:
                ax.set_xlabel("training day")
                if days:
                    ax.set_xticks(days)
            if ci == 0:
                ax.set_ylabel(label, fontsize=9)
    axes[0][0].legend(fontsize=8, loc="best")
    fig.suptitle("Spontaneous bursting rhythm across training days — each "
                 "dot = one sample window, bar = day mean", fontsize=11)
    fig.tight_layout()
    pl._save(fig, os.path.join(out_dir, "psd_trend_summary.png"))


def _write_csv(rows, out_dir):
    import csv
    cols = ["stem", "region", "group", "day", "age", "duration_s", "n_bursts",
            "burst_rate_per_min", "median_ibi_s", "burst_freq_hz",
            "psd_peak_hz"]
    path = os.path.join(out_dir, "psd_trend_summary.csv")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items() if k in cols})
    io._log(f"  [PSDTrend] summary CSV -> {path}")


# ==============================================================================
# driver
# ==============================================================================
def run_psd_trend():
    """Analyse every Batch.csv row enabled for analysis; write figures + CSV to
    the 'PSD Trends' folder under PLOT_DIR."""
    p = PSD_TREND
    csv_path = os.path.join(io.CONFIG["DATA_DIR"],
                            io.CONFIG.get("BATCH_CSV_NAME", "Batch.csv"))
    if not os.path.exists(csv_path):
        io._log(f"[PSDTrend] Batch.csv not found at {csv_path}; nothing to do.")
        return
    rows_in = io.filter_rows_by_group(io.parse_batch(csv_path), "analyze")
    if not rows_in:
        io._log("[PSDTrend] no rows to analyse.")
        return
    out_dir = os.path.join(os.path.normpath(io.CONFIG["PLOT_DIR"]), "PSD Trends")
    io._log(f"[PSDTrend] bursting-rhythm trends -> {out_dir}")
    rows, psds = [], {}
    for u in io.iter_batch_units(rows_in):
        io.check_abort()
        st, (f, psd) = window_psd_stats(u, p)
        m = _re.search(r"_D(\d+)", u["stem"])
        day = int(m.group(1)) if m else None
        rows.append({"stem": u["stem"], "region": u["region"],
                     "group": u["group"], "day": day, "age": u.get("age"),
                     **st})
        if day is not None:
            psds.setdefault((_condition_of(u["group"]), u["region"], day),
                            []).append((f, psd))
        io._log(f"  [PSDTrend] {u['stem']} [{u['region']}]: "
                f"{st['n_bursts']} burst(s), {st['burst_rate_per_min']:.1f}/min, "
                f"IBI-freq {st['burst_freq_hz']:.3f} Hz, PSD peak "
                f"{st['psd_peak_hz']:.3f} Hz"
                if np.isfinite(st.get("burst_freq_hz", np.nan)) else
                f"  [PSDTrend] {u['stem']} [{u['region']}]: "
                f"{st['n_bursts']} burst(s) (too few for a rhythm)")
    _write_csv(rows, out_dir)
    _fig_overlays(psds, p, out_dir)
    _fig_summary(rows, p, out_dir)
    io._log("[PSDTrend] done.")


# ==============================================================================
# self-test: synthetic rhythmic bursting must recover the planted frequency
# ==============================================================================
def _selftest():
    import mea_nap_original_plots as orig
    rng = np.random.default_rng(1)
    labels = orig._mcs60_labels()
    dur = 300.0
    f_burst = 0.5                                   # one burst every 2 s
    spikes = {l: [] for l in labels}
    for k in range(int(dur * f_burst)):
        t0 = k / f_burst + rng.normal(0, 0.05)
        for l in labels:
            for j in range(3):
                spikes[l].append(t0 + rng.uniform(0, 0.05))
    for l in labels:
        spikes[l].extend(rng.uniform(0, dur, int(1.0 * dur)))
    unit = {"labels": labels,
            "spikes": [np.sort(np.array(spikes[l])) for l in labels],
            "duration": dur}
    st, (f, psd) = window_psd_stats(unit, PSD_TREND)
    ok = (abs(st["burst_freq_hz"] - f_burst) < 0.1
          and abs(st["psd_peak_hz"] - f_burst) < 0.1
          and abs(st["burst_rate_per_min"] - f_burst * 60) < 6)
    print(f"  planted {f_burst} Hz bursting -> rate {st['burst_rate_per_min']:.1f}/min, "
          f"IBI-freq {st['burst_freq_hz']:.3f} Hz, PSD peak {st['psd_peak_hz']:.3f} Hz")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    _selftest()
