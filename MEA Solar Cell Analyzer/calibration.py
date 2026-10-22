"""Calibration report: plots of the learned per-condition peak-shape wavelets
(mean +/- error band) and the bior1.5 spike-detection wavelet + its scales.

Writes PNGs + a summary into "<out_root>/Calibration Report/". Uses a headless
matplotlib backend, so it works whether or not a display is present.
"""
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import templates as tpl

REPORT_DIRNAME = "Calibration Report"

CLASS_COLORS = {
    tpl.SOLAR: "#1f9bd6", tpl.PIEZO: "#e8791f", tpl.BIO: "#3ab24a",
}
CLASS_TITLE = {
    tpl.SOLAR: "Solar-cell voltage",
    tpl.PIEZO: "Piezoelectric laser",
    tpl.BIO: "Biological spike",
}


def _report_dir(out_root):
    d = os.path.join(out_root, REPORT_DIRNAME)
    os.makedirs(d, exist_ok=True)
    return d


def plot_template(name, t, path, xlim=None):
    """One class: mean waveform (µV) with a +/-1 SD error band."""
    color = CLASS_COLORS.get(name, "#444444")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(t.t_ms, t.mean_uv, color=color, lw=2.2, label="mean")
    if t.std_uv.size == t.mean_uv.size and np.any(t.std_uv > 0):
        ax.fill_between(t.t_ms, t.mean_uv - t.std_uv, t.mean_uv + t.std_uv,
                        color=color, alpha=0.25, lw=0, label="±1 SD")
    onset = getattr(t, "aligned", "peak") == "onset"
    ax.axvline(0, color="0.7", lw=0.8, ls="--")
    ax.axhline(0, color="0.85", lw=0.8)
    ax.set_xlabel("time from flood onset (ms)" if onset else "time from peak (ms)")
    ax.set_ylabel("amplitude (µV)")
    if onset:                                # whole flood train (250 ms, ~10 pulses)
        ax.set_title("%s — whole %.0f ms flood train, %d electrode(s), p-p≈%.1f µV"
                     % (CLASS_TITLE.get(name, name),
                        float(t.t_ms[-1] - t.t_ms[0]) if t.t_ms.size else 0.0,
                        t.n, t.amp_med))
    else:
        ax.set_title("%s wavelet — n=%d, |amp|≈%.1f µV, width≈%.2f ms"
                     % (CLASS_TITLE.get(name, name), t.n, t.amp_med, t.width_med))
    # a peak template shares the common short axis; an onset (250 ms) template
    # auto-scales to its own full span so it doesn't squash the peak plots.
    if xlim is not None and not onset:
        ax.set_xlim(*xlim)
    ax.legend(loc="best", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)


def plot_event(name, t, path):
    """Flood-onset-aligned event waveform (both ON+OFF transients) with ±1 SD,
    for a solar/piezo class that carries an event average."""
    color = CLASS_COLORS.get(name, "#444444")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(t.event_t_ms, t.event_uv, color=color, lw=2.2, label="mean")
    if t.event_std is not None and t.event_std.size == t.event_uv.size \
            and np.any(t.event_std > 0):
        ax.fill_between(t.event_t_ms, t.event_uv - t.event_std,
                        t.event_uv + t.event_std, color=color, alpha=0.25, lw=0,
                        label="±1 SD")
    ax.axvline(0, color="0.7", lw=0.8, ls="--")
    ax.axhline(0, color="0.85", lw=0.8)
    ax.set_xlabel("time from flood onset (ms)")
    ax.set_ylabel("amplitude (µV)")
    p2p = (float(t.event_uv.max() - t.event_uv.min()) if t.event_uv.size else 0.0)
    ax.set_title("%s flood event — %d electrode(s), p-p≈%.1f µV (ON+OFF transients)"
                 % (CLASS_TITLE.get(name, name), t.event_n, p2p))
    ax.legend(loc="best", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)


def plot_overlay(ts, path, xlim=None):
    """All class wavelets on one axis, amplitude-normalised for shape comparison."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    any_plotted = False
    for name in ts.names():
        t = ts.get(name)
        if t is None or t.n == 0:
            continue
        if getattr(t, "aligned", "peak") == "onset":
            continue                         # 250 ms train has its own plot; the
            #                                  peak-shape overlay compares short peaks
        ax.plot(t.t_ms, t.mean_norm, lw=2.2,
                color=CLASS_COLORS.get(name, "#444444"),
                label="%s (n=%d)" % (CLASS_TITLE.get(name, name), t.n))
        any_plotted = True
    ax.axvline(0, color="0.7", lw=0.8, ls="--")
    ax.set_xlabel("time (ms)  [peak- or onset-aligned per class]")
    ax.set_ylabel("normalised amplitude")
    ax.set_title("Learned wavelets — normalised overlay")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if any_plotted:
        ax.legend(loc="best", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)


def _mother_wavelet(info):
    """(x, psi) of the bior1.5 analysis wavelet, from the loaders if available
    else reconstructed with PyWavelets."""
    if info and info.get("psi_d") is not None and info.get("x") is not None:
        return np.asarray(info["x"], float), np.asarray(info["psi_d"], float)
    try:
        import pywt
        w = pywt.Wavelet("bior1.5")
        phi_d, psi_d, phi_r, psi_r, x = w.wavefun(level=8)
        return np.asarray(x, float), np.asarray(psi_d, float)
    except Exception:
        return None, None


def plot_detection_wavelet(info, path):
    """The spike-detection wavelet calibration: the bior1.5 mother wavelet and the
    same wavelet at each CWT scale the detector uses for this sampling rate."""
    x, psi = _mother_wavelet(info)
    fs = float((info or {}).get("fs") or 20000.0)
    scales = (info or {}).get("scales")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    if x is not None:
        axes[0].plot(x, psi, color="#5b5bd6", lw=2)
    axes[0].axhline(0, color="0.85", lw=0.8)
    axes[0].set_title("bior1.5 analysis wavelet")
    axes[0].set_xlabel("wavelet support (a.u.)")
    axes[0].set_ylabel("ψ")

    # the wavelet stretched to each detection scale, in ms (matches the detector:
    # ψ(t/scale)/sqrt(scale), sign-negated so a negative-going AP reads as a peak)
    if x is not None and scales is not None and len(scales):
        x_range = float(x[-1] - x[0])
        for s in np.asarray(scales, float):
            n_wvl = max(3, int(round(x_range * s)))
            t_fine = np.linspace(x[0], x[-1], n_wvl)
            psi_s = np.interp(t_fine, x, psi) / np.sqrt(s)
            t_ms = (np.arange(n_wvl) - n_wvl / 2.0) / fs * 1000.0
            width_ms = n_wvl / fs * 1000.0
            axes[1].plot(t_ms, -psi_s, lw=1.8,
                         label="scale %.1f  (~%.2f ms)" % (s, width_ms))
        axes[1].legend(loc="best", fontsize=8, frameon=False)
        axes[1].set_title("Detection wavelet at each scale (fs=%.0f Hz)" % fs)
    else:
        axes[1].set_title("Detection scales unavailable (loaders not loaded)")
    axes[1].axhline(0, color="0.85", lw=0.8)
    axes[1].axvline(0, color="0.85", lw=0.8, ls="--")
    axes[1].set_xlabel("time (ms)")
    axes[1].set_ylabel("ψ (negated)")
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)


def _summary_md(ts, info, sources, files):
    lines = ["# Spike detection & wavelet calibration report", ""]
    p = (info or {}).get("params") or {}
    if p:
        lines += ["## Detection parameters (validated bior1.5 detector)", "",
                  "| parameter | value |", "|---|---|",
                  "| wavelet | %s |" % p.get("wname", "bior1.5"),
                  "| bandpass | %s–%s Hz |" % (p.get("bp_low_hz", "?"),
                                               p.get("bp_high_hz", "?")),
                  "| spike-width range | %s ms |" % (p.get("wid_ms", "?"),),
                  "| CWT scales (ns) | %s |" % (p.get("ns", "?"),),
                  "| Bayesian cost L | %s |" % (p.get("L", "?"),),
                  "| sampling rate | %.0f Hz |" % float((info or {}).get("fs")
                                                        or 20000.0), ""]
    sc = (info or {}).get("scales")
    if sc is not None and len(sc):
        lines += ["Detection CWT scales: " +
                  ", ".join("%.2f" % s for s in np.asarray(sc, float)), ""]
    lines += ["## Per-condition wavelets (mean ± error range)", "",
              "| condition | class | n peaks | median |amp| (µV) | median width (ms) |",
              "|---|---|---|---|---|"]
    for name in ts.names():
        t = ts.get(name)
        if t is None:
            continue
        src = sources.get(name, "")
        lines.append("| %s | %s | %d | %.1f | %.3f |"
                     % (CLASS_TITLE.get(name, name) + (" (%s)" % src if src else ""),
                        name, t.n, t.amp_med, t.width_med))
    lines += ["", "## Figures", ""]
    for f in files:
        lines.append("- `%s`" % os.path.basename(f))
    lines.append("")
    return "\n".join(lines)


def generate_report(templateset, out_root, wavelet_info=None, sources=None):
    """Write the calibration report. Returns (report_dir, [file paths])."""
    rep = _report_dir(out_root)
    sources = sources or {}
    files = []

    # common x-limits for the PEAK templates only (the 250 ms onset trains auto-
    # scale to their own span, so they don't stretch the short peak axes)
    ts = [templateset.get(nm) for nm in templateset.names()
          if templateset.get(nm) is not None and templateset.get(nm).n > 0
          and getattr(templateset.get(nm), "aligned", "peak") == "peak"]
    xlim = None
    if ts:
        xlim = (min(float(t.t_ms[0]) for t in ts),
                max(float(t.t_ms[-1]) for t in ts))

    for name in templateset.names():
        t = templateset.get(name)
        if t is None or t.n == 0:
            continue
        path = os.path.join(rep, "wavelet_%s.png" % name)
        plot_template(name, t, path, xlim=xlim)
        files.append(path)
        # solar/piezo also get the flood-onset EVENT waveform (both transients)
        if getattr(t, "has_event", False):
            epath = os.path.join(rep, "event_%s.png" % name)
            plot_event(name, t, epath)
            files.append(epath)

    if templateset.names():
        overlay = os.path.join(rep, "wavelets_overlay.png")
        plot_overlay(templateset, overlay, xlim=xlim)
        files.append(overlay)

    detpath = os.path.join(rep, "detection_wavelet.png")
    plot_detection_wavelet(wavelet_info, detpath)
    files.append(detpath)

    summary = os.path.join(rep, "calibration_summary.md")
    with open(summary, "w", encoding="utf-8") as fh:
        fh.write(_summary_md(templateset, wavelet_info, sources, files))
    files.append(summary)
    return rep, files
