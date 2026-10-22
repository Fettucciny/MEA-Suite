"""Test whether the MEA headstage is DC-coupled or AC-coupled (high-pass).

Why this matters
----------------
A real solar cell under sustained illumination produces a sustained (step) voltage:
it goes up at light-ON, HOLDS a plateau while the light is on, and steps back down
at light-OFF. If the headstage / acquisition is AC-coupled with a high-pass corner
f_c, that filter DIFFERENTIATES the step: the plateau cannot pass, so a sustained
input is displayed as a sharp transient that decays back to baseline with
    tau = 1 / (2*pi*f_c).
For a 300 Hz high-pass, tau = 0.53 ms -- so a truly sustained photovoltage would
show up as a ~0.5 ms spike that decays to zero, with an equal-and-opposite spike at
light-OFF. That is numerically close to the fast onset transient we see in the STA,
so it must be ruled out before treating the transient as the real cell response.

This script answers the question two independent ways:

(A) HARDWARE METADATA (definitive if present).
    MCS HDF5 files record the acquisition filter per channel in the InfoChannel
    compound dataset (HighPassFilterType / HighPassFilterCutOff, LowPassFilter...).
    A HighPass cutoff of 0 / "DC" / "NoFilter" == DC-coupled hardware. A non-zero
    cutoff (e.g. 300, 100, 10, 1 Hz) == AC-coupled at that corner.

(B) EMPIRICAL STEP RESPONSE on the RAW (un-filtered) trace.
    We build the flood-onset-triggered average of the RAW calibrated voltage
    (NO software band-pass) on the strongest-responding electrodes, over a window
    that spans the whole illumination pulse. Then:
      * plateau ratio = (level held during mid-illumination) / (onset-peak level).
          ~0  -> the level decays away  -> AC-coupled (differentiated step).
          ~1  -> the level is held      -> DC-coupled (true sustained step).
      * onset decay tau -> corner frequency f_c = 1/(2*pi*tau). Compared to 300 Hz.
      * OFF transient: an equal-and-OPPOSITE spike at light-OFF is the capacitor-
          discharge signature of an AC (high-pass) system; a DC system just steps
          back down with no overshoot.
      * net integral over one full ON+OFF cycle: ~0 for a high-pass (zero DC gain),
          non-zero for DC coupling.

    The empirical test is important even if the metadata says DC: if the hardware is
    DC-coupled but the raw STA still shows only a decaying transient, then it is the
    downstream *software* 300 Hz filter (not the hardware) that is manufacturing the
    0.5 ms spike -- which is exactly the "filter the peak first, then detect" concern
    raised earlier. Conversely, if the raw trace holds a plateau, the sustained
    photovoltage is real in the recording and only the filtered view hides it.

Run (from the MEA Solar Cell Analyzer folder, with your venv python):
  "%USERPROFILE%\\Nextcloud2\\MEA\\MEA Analyzer\\.venv\\Scripts\\python.exe" verify_coupling.py
  ... or pass a stem / full .h5 path:
  python verify_coupling.py "2026-07-21T14-22-461 Solar Cell Laser"

Outputs (small, safe to stage back): coupling_diag.npz and coupling_diag.png
"""
import os
import sys
import numpy as np

# numpy 2.x renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config          # noqa
import loaders         # noqa
import peaks as pk     # noqa

# -- defaults (override the stem/path on the command line) -------------------
DATA_DIR = r"C:\Users\Yikan\Nextcloud2\MEA Data\MEA 071826\H5"
STEM = "1 Solar Cell Laser"   # solar-cell recording (1 cell)

PRE_MS = 5.0          # baseline window before each light onset
TAIL_MS = 15.0        # extra window after light-OFF to capture the OFF transient
N_TOP = 6             # how many strongest responders to analyse
SETTLE_MS = 2.0       # skip this long after the onset peak before reading a plateau


# ---------------------------------------------------------------------------
# (A) hardware filter metadata straight out of the MCS HDF5
# ---------------------------------------------------------------------------
def read_hw_filter(h5_path):
    """Best-effort read of the acquisition filter recorded in the MCS file.
    Returns a list of dicts (one per InfoChannel row found), each with any of
    HighPassFilterType/HighPassFilterCutOff/HighPassFilterOrder + LowPass...,
    plus a plain-text interpretation. Empty list if nothing filter-like is found."""
    out = []
    try:
        import h5py
    except Exception as exc:
        print("  [metadata] h5py not importable (%r) -- skipping hardware read" % exc)
        return out

    def _dec(v):
        if isinstance(v, bytes):
            try:
                return v.decode("utf-8", "replace")
            except Exception:
                return repr(v)
        return v

    info_rows = []
    try:
        with h5py.File(h5_path, "r") as f:
            # Walk the whole tree; collect any compound dataset that has a field
            # name containing 'filter', and any attribute mentioning filter/coupling.
            def visit(name, obj):
                try:
                    dt = getattr(obj, "dtype", None)
                    names = list(dt.names) if (dt is not None and dt.names) else []
                except Exception:
                    names = []
                if names and any("filter" in n.lower() for n in names):
                    try:
                        data = obj[()]
                        for row in np.atleast_1d(data):
                            rec = {}
                            for n in names:
                                nl = n.lower()
                                if ("filter" in nl or "label" in nl
                                        or "channelid" in nl):
                                    rec[n] = _dec(row[n])
                            rec["_path"] = name
                            info_rows.append(rec)
                    except Exception:
                        pass
                # attributes
                try:
                    for k, v in obj.attrs.items():
                        if any(s in k.lower()
                               for s in ("filter", "coupl", "highpass", "lowpass")):
                            out.append({"attr": k, "value": _dec(v), "_path": name})
                except Exception:
                    pass
            f.visititems(visit)
            try:
                for k, v in f.attrs.items():
                    if any(s in k.lower()
                           for s in ("filter", "coupl", "highpass", "lowpass")):
                        out.append({"attr": k, "value": _dec(v), "_path": "/"})
            except Exception:
                pass
    except Exception as exc:
        print("  [metadata] could not open H5 for metadata (%r)" % exc)
        return out

    # de-duplicate InfoChannel rows (they are identical across channels)
    seen = set()
    for rec in info_rows:
        key = tuple(sorted((k, str(v)) for k, v in rec.items() if k != "_path"
                           and "label" not in k.lower()
                           and "channelid" not in k.lower()))
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def _interpret_hp(val):
    """Turn a HighPass cutoff value into a coupling verdict + numeric Hz (or None)."""
    if val is None:
        return "unknown", None
    s = str(val).strip().lower()
    if s in ("", "0", "0.0", "nofilter", "no filter", "none", "dc", "-1"):
        return "DC-coupled (no hardware high-pass)", 0.0
    try:
        hz = float(s)
    except Exception:
        return "unrecognised value %r" % val, None
    if hz <= 0:
        return "DC-coupled (no hardware high-pass)", 0.0
    return "AC-coupled, high-pass corner ~ %.3g Hz" % hz, hz


# ---------------------------------------------------------------------------
# (B) empirical step-response shape on the RAW trace
# ---------------------------------------------------------------------------
def _tau_from_decay(t_ms, y, t_peak_ms, plateau, peak_val):
    """Time constant of the onset transient's decay toward the plateau.
    Returns (tau_ms_1e, tau_ms_fit). tau_1e is the model-free 1/e crossing; tau_fit
    is a least-squares single-exponential fit if scipy is available (else NaN)."""
    amp = peak_val - plateau
    tau_1e = float("nan")
    if abs(amp) > 1e-9:
        target = plateau + amp / np.e
        m = t_ms > t_peak_ms
        tt, yy = t_ms[m], y[m]
        # first sample where the signal has decayed past the 1/e level
        if amp > 0:
            cross = np.where(yy <= target)[0]
        else:
            cross = np.where(yy >= target)[0]
        if cross.size:
            tau_1e = float(tt[cross[0]] - t_peak_ms)
    tau_fit = float("nan")
    try:
        from scipy.optimize import curve_fit
        m = (t_ms >= t_peak_ms)
        tt = t_ms[m] - t_peak_ms
        yy = y[m]

        def model(x, a, tau, c):
            return a * np.exp(-x / tau) + c
        p0 = [peak_val - plateau, max(0.2, tau_1e if np.isfinite(tau_1e) else 1.0),
              plateau]
        popt, _ = curve_fit(model, tt, yy, p0=p0, maxfev=20000)
        if popt[1] > 0:
            tau_fit = float(popt[1])
    except Exception:
        pass
    return tau_1e, tau_fit


def analyse_electrode(t_ms, sta, on_dur_ms):
    """Characterise one raw STA waveform. Returns a dict of shape metrics."""
    pre_mask = t_ms < 0
    base = float(np.median(sta[pre_mask])) if pre_mask.any() else 0.0
    y = sta - base

    # onset transient: largest |excursion| within [0, min(on_dur, settle window)]
    on_win = (t_ms >= 0) & (t_ms <= min(on_dur_ms, 5.0) + 0.5)
    if not on_win.any():
        on_win = (t_ms >= 0) & (t_ms <= 3.0)
    i_pk = np.argmax(np.abs(np.where(on_win, y, 0.0)))
    t_peak = float(t_ms[i_pk])
    peak_val = float(y[i_pk])

    # plateau: median over the middle of the illumination window, after settling
    plat_lo = t_peak + SETTLE_MS
    plat_hi = on_dur_ms - 1.0
    plateau = float("nan")
    if plat_hi > plat_lo + 0.5:
        pm = (t_ms >= plat_lo) & (t_ms <= plat_hi)
        if pm.any():
            plateau = float(np.median(y[pm]))
    plateau_eff = plateau if np.isfinite(plateau) else 0.0
    plateau_ratio = (plateau_eff / peak_val) if abs(peak_val) > 1e-9 else float("nan")

    # OFF transient: largest |excursion| in a window just after light-OFF
    off_lo, off_hi = on_dur_ms - 0.5, on_dur_ms + 5.0
    om = (t_ms >= off_lo) & (t_ms <= off_hi)
    off_val = float("nan")
    if om.any():
        j = np.argmax(np.abs(y[om]))
        off_val = float(y[om][j])

    # decay time constant of the onset transient -> corner frequency
    tau_1e, tau_fit = _tau_from_decay(t_ms, y, t_peak, plateau_eff, peak_val)
    tau_use = tau_fit if np.isfinite(tau_fit) else tau_1e
    fc = (1.0 / (2.0 * np.pi * (tau_use * 1e-3))) if (np.isfinite(tau_use)
                                                      and tau_use > 0) else float("nan")

    # net integral over one full ON+OFF cycle (trapezoid, mV*ms) -> ~0 for high-pass
    cyc = (t_ms >= 0) & (t_ms <= on_dur_ms + min(on_dur_ms, TAIL_MS))
    net_int = float(_trapz(y[cyc], t_ms[cyc])) if cyc.sum() > 2 else float("nan")
    on_int = float(_trapz(y[(t_ms >= 0) & (t_ms <= on_dur_ms)],
                            t_ms[(t_ms >= 0) & (t_ms <= on_dur_ms)]))

    return dict(base=base, t_peak_ms=t_peak, peak_uv=peak_val, plateau_uv=plateau,
                plateau_ratio=plateau_ratio, off_uv=off_val,
                off_over_on=(off_val / peak_val) if abs(peak_val) > 1e-9 else np.nan,
                tau_1e_ms=tau_1e, tau_fit_ms=tau_fit, fc_hz=fc,
                net_integral=net_int, on_integral=on_int)


def main():
    # resolve which recording to open
    arg = sys.argv[1] if len(sys.argv) > 1 else STEM
    if arg.lower().endswith(".h5") and os.path.isabs(arg):
        path = arg
    elif os.path.isabs(arg):
        path = arg + ".h5"
    else:
        path = os.path.join(DATA_DIR, arg + (".h5" if not arg.lower().endswith(".h5")
                                             else ""))
    print("=" * 74)
    print("HEADSTAGE COUPLING TEST")
    print("opening:", path)
    if not os.path.exists(path):
        print("  !! file not found. Pass a stem or full .h5 path as the argument.")
        return

    # ---- (A) hardware metadata --------------------------------------------
    print("\n(A) HARDWARE ACQUISITION FILTER (from the MCS file metadata)")
    hw = read_hw_filter(path)
    hp_hz_meta = None
    if not hw:
        print("  no filter/coupling fields found in the file metadata.")
    else:
        for rec in hw:
            if "attr" in rec:
                print("  attr %-28s = %s   [%s]" % (rec["attr"], rec["value"],
                                                    rec.get("_path", "")))
                continue
            # InfoChannel-style row
            hp = None
            for k, v in rec.items():
                if k == "_path":
                    continue
                print("  %-30s = %s" % (k, v))
                if "highpass" in k.lower() and "cut" in k.lower():
                    hp = v
                elif hp is None and "highpass" in k.lower() and "type" in k.lower():
                    pass
            if hp is not None:
                verdict, hz = _interpret_hp(hp)
                hp_hz_meta = hz
                print("    -> HIGH-PASS reads as: %s" % verdict)
            print("    (path: %s)" % rec.get("_path", ""))

    # ---- load raw + laser events ------------------------------------------
    print("\n(B) EMPIRICAL STEP RESPONSE (raw, un-filtered STA)")
    raw = loaders.RawRecording(path)
    fs = raw.fs
    print("  fs=%.0f Hz  dur=%.1f s  n_ch=%d" % (fs, raw.duration, len(raw.labels)))
    ev = loaders.laser_events(path)
    on = np.asarray(ev.get("on_s", []), float)
    off = ev.get("off_s")
    off = np.asarray(off, float) if off is not None and np.size(off) else None
    print("  laser onsets: n=%d  source=%s" % (on.size, ev.get("source")))
    if on.size == 0:
        print("  !! no laser events -> cannot run the empirical test.")
        raw.free()
        return

    # stimulus timing: single-pulse ON duration + pulse spacing
    if off is not None and off.size:
        n = min(on.size, off.size)
        durs = (off[:n] - on[:n])
        durs = durs[(durs > 0) & (durs < 5.0)]
        on_dur_ms = float(np.median(durs) * 1000.0) if durs.size else 20.0
        print("  single-pulse ON duration: median=%.2f ms  (n=%d, range %.2f-%.2f)"
              % (on_dur_ms, durs.size,
                 (durs.min() * 1000 if durs.size else 0),
                 (durs.max() * 1000 if durs.size else 0)))
    else:
        on_dur_ms = 20.0
        print("  no OFF events; assuming a %.0f ms illumination window." % on_dur_ms)
    if on.size > 1:
        gaps = np.diff(np.sort(on))
        print("  inter-onset gap: median=%.4f s  min=%.4f s" %
              (float(np.median(gaps)), float(gaps.min())))
    starts = pk.train_starts(on, 0.25)
    print("  -> %d pulse onsets grouped into %d trains (~%s s apart)"
          % (on.size, starts.size,
             ("%.2f" % float(np.median(np.diff(starts)))) if starts.size > 1 else "-"))

    # window: pre-baseline + full pulse + tail to see the OFF transient
    post_ms = on_dur_ms + TAIL_MS
    # rank electrodes by short-window p2p, keep the strongest
    scored = []
    for lab in raw.labels:
        _a, _t, _n, p2p = pk.flood_triggered_average(
            raw.channel(lab), fs, on, pre_ms=PRE_MS, post_ms=min(post_ms, 40.0))
        scored.append((float(p2p), lab))
    scored.sort(key=lambda r: -r[0])
    top = [l for _p, l in scored[:N_TOP]]
    print("  strongest responders:", [(l, round(p, 1)) for p, l in scored[:N_TOP]])

    # full raw STA on the top responders (triggering on individual pulse onsets)
    waves, metrics, t_ms = [], [], None
    for lab in top:
        sta, tt, nseg, _p2p = pk.flood_triggered_average(
            raw.channel(lab), fs, on, pre_ms=PRE_MS, post_ms=post_ms)
        t_ms = tt
        waves.append(sta)
        metrics.append(analyse_electrode(tt, sta, on_dur_ms))
    raw.free()
    waves = np.array(waves)

    # ---- report per-electrode shape ---------------------------------------
    print("\n  per-electrode step-response shape (raw trace, baseline-subtracted):")
    hdr = ("    %-5s %9s %9s %9s %8s %9s %9s %8s" %
           ("elec", "peak(uV)", "plat(uV)", "plat/pk", "off/pk",
            "tau(ms)", "fc(Hz)", "netInt"))
    print(hdr)
    for lab, m in zip(top, metrics):
        print("    %-5s %9.1f %9s %9.3f %8.2f %9s %9s %8.1f" % (
            lab, m["peak_uv"],
            ("%.1f" % m["plateau_uv"]) if np.isfinite(m["plateau_uv"]) else "n/a",
            m["plateau_ratio"] if np.isfinite(m["plateau_ratio"]) else float("nan"),
            m["off_over_on"] if np.isfinite(m["off_over_on"]) else float("nan"),
            ("%.3f" % (m["tau_fit_ms"] if np.isfinite(m["tau_fit_ms"])
                       else m["tau_1e_ms"]))
            if np.isfinite(m["tau_fit_ms"]) or np.isfinite(m["tau_1e_ms"]) else "n/a",
            ("%.0f" % m["fc_hz"]) if np.isfinite(m["fc_hz"]) else "n/a",
            m["net_integral"]))

    # ---- aggregate verdict -------------------------------------------------
    pr = np.array([m["plateau_ratio"] for m in metrics], float)
    fc = np.array([m["fc_hz"] for m in metrics], float)
    offr = np.array([m["off_over_on"] for m in metrics], float)
    med_pr = float(np.nanmedian(np.abs(pr)))
    med_fc = float(np.nanmedian(fc))
    med_off = float(np.nanmedian(offr))

    print("\n  SUMMARY (median over responders):")
    print("    |plateau / onset-peak|     = %.3f" % med_pr)
    print("    onset decay tau            = %.3f ms  -> corner f_c ~ %s Hz" % (
        (1.0 / (2 * np.pi * med_fc) * 1e3) if np.isfinite(med_fc) and med_fc > 0
        else float("nan"),
        ("%.0f" % med_fc) if np.isfinite(med_fc) else "n/a"))
    print("    OFF transient / ON peak    = %.2f  (near -1 = capacitive discharge)"
          % med_off)

    print("\n  VERDICT")
    # metadata verdict
    if hp_hz_meta is not None:
        if hp_hz_meta == 0.0:
            print("   * Metadata: hardware is DC-coupled (no acquisition high-pass).")
        else:
            print("   * Metadata: hardware is AC-coupled, high-pass ~%.3g Hz."
                  % hp_hz_meta)
    else:
        print("   * Metadata: no hardware high-pass field was found in the file.")

    # empirical verdict
    resolvable_fc = 1.0 / (2 * np.pi * (on_dur_ms * 1e-3))   # lowest corner the
    # illumination window can reveal as a decay
    if np.isfinite(med_pr) and med_pr > 0.3:
        print("   * Raw trace: the response HOLDS a plateau during illumination "
              "(|plateau/peak|=%.2f)." % med_pr)
        print("     -> The recording itself is effectively DC over this timescale; "
              "any acquisition")
        print("        high-pass corner is below ~%.1f Hz. A sustained photovoltage "
              "is REAL in the" % resolvable_fc)
        print("        raw data. The 0.5 ms transient seen after analysis is produced "
              "by the SOFTWARE")
        print("        300 Hz filter, not by the headstage.")
    elif np.isfinite(med_fc) and med_fc > 0:
        print("   * Raw trace: the response DECAYS to baseline during illumination "
              "(tau=%.2f ms)." % (1.0 / (2 * np.pi * med_fc) * 1e3))
        print("     -> This is the AC-coupled (high-pass) signature, corner f_c ~ "
              "%.0f Hz." % med_fc)
        near300 = 0.5 * 300 <= med_fc <= 2.0 * 300
        print("        %s the ~300 Hz hypothesis (0.53 ms differentiation)."
              % ("CONSISTENT with" if near300 else "NOT a match to"))
        if np.isfinite(med_off) and med_off < -0.3:
            print("        The opposite-polarity OFF transient (off/on=%.2f) confirms "
                  "capacitive coupling." % med_off)
    else:
        print("   * Raw trace: shape inconclusive (no clean plateau and no clean "
              "decay resolved).")
        print("     Illumination window (%.1f ms) can only reveal corners above "
              "~%.1f Hz." % (on_dur_ms, resolvable_fc))
    print("=" * 74)

    # ---- save small diagnostics -------------------------------------------
    out_npz = os.path.join(HERE, "coupling_diag.npz")
    np.savez_compressed(
        out_npz, t_ms=t_ms.astype(np.float32), waves=waves.astype(np.float32),
        labels=np.array(top), on_dur_ms=float(on_dur_ms), fs=float(fs),
        plateau_ratio=pr.astype(np.float32), fc_hz=fc.astype(np.float32),
        off_over_on=offr.astype(np.float32),
        hp_hz_meta=(-1.0 if hp_hz_meta is None else float(hp_hz_meta)))
    print("WROTE", out_npz, "(%.0f KB)" % (os.path.getsize(out_npz) / 1024.0))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
        for lab, sta, m in zip(top, waves, metrics):
            y = sta - m["base"]
            a1.plot(t_ms, y, lw=1.2, label=lab)
        a1.axvspan(0, on_dur_ms, color="gold", alpha=0.15, label="light ON")
        a1.axhline(0, color="k", lw=0.6)
        a1.set_xlabel("time from light onset (ms)")
        a1.set_ylabel("raw voltage (uV, baseline-subtracted)")
        a1.set_title("Full step response (raw, un-filtered)")
        a1.legend(fontsize=7, ncol=2)
        # zoom on the onset transient to read the decay
        for lab, sta, m in zip(top, waves, metrics):
            y = sta - m["base"]
            a2.plot(t_ms, y, lw=1.2, label=lab)
        a2.axvspan(0, on_dur_ms, color="gold", alpha=0.15)
        a2.axhline(0, color="k", lw=0.6)
        a2.set_xlim(-PRE_MS, min(on_dur_ms, 8.0))
        a2.set_xlabel("time from light onset (ms)")
        a2.set_title("Onset zoom: decay-to-baseline (AC) vs held plateau (DC)")
        a2.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        out_png = os.path.join(HERE, "coupling_diag.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print("WROTE", out_png, "(%.0f KB)" % (os.path.getsize(out_png) / 1024.0))
    except Exception as exc:
        print("  (plot skipped: %r)" % exc)


if __name__ == "__main__":
    main()
