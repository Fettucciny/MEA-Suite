"""
radial_proto.py -- PROTOTYPE ONLY, nothing here is committed to the module.

Demonstrates the radial (center-out) matched filter proposed in chat, on
synthetic data with known ground truth, and compares it against the existing
plane-wave fit on both a radial and a linear stimulus.

Per-electrode phase is computed by the module's OWN electrode_locking(), i.e.
equation 2 is untouched. Only the spatial model differs.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import mea_nap_phase_lockin as plk
import mea_nap_original_plots as orig

PITCH = plk.PHASE_LOCKIN["pitch_um"]
TRUE_CENTER = (4.5 * PITCH, 4.5 * PITCH)      # middle of 44/45/54/55


# ---------------------------------------------------------------- synthetic
def synth_radial(v_um_s=700.0, period_s=2.64, dur_s=120.0, seed=11,
                 center=TRUE_CENTER, jitter_frac=0.02, bg_hz=2.0,
                 outward=True):
    """Thin ring leaving `center` once per cycle at constant speed v."""
    rng = np.random.default_rng(seed)
    labels = orig._mcs60_labels()
    pos = {l: np.array(orig.mcs_coord(l), float) * PITCH for l in labels}
    r = {l: float(np.hypot(*(pos[l] - np.asarray(center)))) for l in labels}
    r_max = max(r.values())
    spikes = {}
    n_cycles = int(dur_s / period_s)
    for l in labels:
        s = []
        # time-of-arrival of the wavefront at this electrode, within the cycle
        rr = r[l] if outward else (r_max - r[l])
        t_in_cycle = rr / v_um_s
        if t_in_cycle < period_s:
            for c in range(n_cycles):
                s.append(c * period_s + t_in_cycle
                         + rng.normal(0, jitter_frac * period_s))
        s.extend(rng.uniform(0, dur_s, int(bg_hz * dur_s)))
        spikes[l] = np.sort(np.array(s, float))
    return spikes, dur_s


# ------------------------------------------------- radial matched filter
def _matched_power_radial(w, z, x, y, cx, cy, kr):
    """Same normalisation as the module's _matched_power: |acc| / sum(w)."""
    r = np.hypot(x - cx, y - cy)
    return abs((w * z * np.exp(-1j * kr * r)).sum()) / w.sum()


def fit_radial_wave(locks, period_s, pitch_um, margin_cells=2.0,
                    n_centre=29, n_kr=161, refine=True):
    """Grid-search (centre, radial wavenumber) maximising radial coherence.

    kr > 0  -> phase grows with radius -> OUTWARD (centre fires first)
    kr < 0  -> INWARD
    Returns None if there are too few locked electrodes, mirroring
    fit_plane_wave's >=6 requirement.
    """
    if len(locks) < 6:
        return None
    w = np.array([np.sqrt(l.n) for l in locks])
    z = np.array([l.z for l in locks])
    x = np.array([l.x_um for l in locks])
    y = np.array([l.y_um for l in locks])

    lo_x, hi_x = x.min() - margin_cells * pitch_um, x.max() + margin_cells * pitch_um
    lo_y, hi_y = y.min() - margin_cells * pitch_um, y.max() + margin_cells * pitch_um
    cxs = np.linspace(lo_x, hi_x, n_centre)
    cys = np.linspace(lo_y, hi_y, n_centre)
    k_max = np.pi / pitch_um                      # radial Nyquist
    krs = np.linspace(-k_max, k_max, n_kr)

    CX, CY = np.meshgrid(cxs, cys, indexing="ij")
    R = np.hypot(x[None, None, :] - CX[..., None],
                 y[None, None, :] - CY[..., None])         # (nc, nc, ne)
    wz = (w * z)[None, None, None, :]
    phase = np.exp(-1j * krs[None, None, :, None] * R[:, :, None, :])
    P = np.abs((wz * phase).sum(axis=-1)) / w.sum()        # (nc, nc, nkr)

    i0, j0, k0 = np.unravel_index(np.argmax(P), P.shape)
    cx0, cy0, kr0 = cxs[i0], cys[j0], krs[k0]
    coh = float(P[i0, j0, k0])
    on_edge = (i0 in (0, n_centre - 1)) or (j0 in (0, n_centre - 1))

    if refine:                                   # local polish, same box shape
        dx = cxs[1] - cxs[0]; dk = krs[1] - krs[0]
        fx = np.linspace(cx0 - dx, cx0 + dx, 9)
        fy = np.linspace(cy0 - dx, cy0 + dx, 9)
        fk = np.linspace(kr0 - dk, kr0 + dk, 21)
        fx = np.clip(fx, lo_x, hi_x); fy = np.clip(fy, lo_y, hi_y)
        best = (coh, cx0, cy0, kr0)
        for a in fx:
            for b in fy:
                for kk in fk:
                    pv = _matched_power_radial(w, z, x, y, a, b, kk)
                    if pv > best[0]:
                        best = (pv, a, b, kk)
        coh, cx0, cy0, kr0 = best

    # The radial model NESTS the planar one (centre -> infinity), so its raw
    # coherence is almost always >= planar and cannot by itself decide between
    # them. The discriminating question is whether the best centre is physically
    # inside the array: re-maximise with the centre confined to the electrode
    # footprint and report that separately.
    inx = (cxs >= x.min()) & (cxs <= x.max())
    iny = (cys >= y.min()) & (cys <= y.max())
    Pin = P[np.ix_(inx, iny, np.ones(n_kr, bool))]
    ii, jj, kk = np.unravel_index(np.argmax(Pin), Pin.shape)
    coh_in = float(Pin[ii, jj, kk])
    cin = (float(cxs[inx][ii]), float(cys[iny][jj]))

    kmag = abs(kr0)
    speed = (2 * np.pi / (period_s * kmag)) if kmag > 1e-12 else np.inf
    r_fit = np.hypot(x - cx0, y - cy0)
    span = float(kmag * (r_fit.max() - r_fit.min()))
    return {"centre_um": (float(cx0), float(cy0)), "k_r": float(kr0),
            "outward": bool(kr0 > 0), "speed_um_per_s": float(speed),
            "wavelength_um": float(2 * np.pi / kmag) if kmag > 1e-12 else np.inf,
            "coherence": float(coh), "phase_span_rad": span,
            "centre_on_bound": bool(on_edge), "n_electrodes": len(locks),
            "coherence_centre_inside": coh_in, "centre_inside_um": cin,
            "centre_inside_array": bool(x.min() <= cx0 <= x.max()
                                        and y.min() <= cy0 <= y.max()),
            "_grid": (cxs, cys, P.max(axis=-1))}


def centre_from_gradients(locks, pitch_um):
    """Closed-form centre: each electrode's phase-gradient direction defines a
    line through the centre; return the least-squares intersection of those
    lines. No grid search -- used as a seed and an independent cross-check.
    """
    gx, gy, Z = plk._phase_field(locks, pitch_um)
    dZy, dZx = np.gradient(Z, gy, gx)
    den = np.abs(Z) ** 2 + 1e-12
    U = np.imag(np.conj(Z) * dZx) / den
    V = np.imag(np.conj(Z) * dZy) / den
    mag = np.abs(Z)
    keep = mag > 0.25 * mag.max()
    X, Y = np.meshgrid(gx, gy)
    px, py = X[keep], Y[keep]
    u, v = U[keep], V[keep]
    nrm = np.hypot(u, v); ok = nrm > 1e-12
    px, py, u, v, nrm = px[ok], py[ok], u[ok], v[ok], nrm[ok]
    u, v = u / nrm, v / nrm
    # minimise sum of squared perpendicular distances to each line
    nx, ny = -v, u                                   # line normals
    A = np.stack([nx, ny], axis=1)
    b = nx * px + ny * py
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return (float(sol[0] * pitch_um), float(sol[1] * pitch_um))


# ------------------------------------------------------------------ report
def analyse(name, spikes, dur, period_s, p):
    lxy = plk._label_xy(list(spikes.keys()), PITCH)
    locks = plk.electrode_locking(spikes, 0.0, dur, period_s, lxy, p["min_spikes"])
    planar = plk.fit_plane_wave(locks, period_s, PITCH)
    radial = fit_radial_wave(locks, period_s, PITCH)
    return {"name": name, "locks": locks, "planar": planar, "radial": radial,
            "period_s": period_s}


def main():
    p = plk.PHASE_LOCKIN
    T = 2.64
    out_dir = os.path.join(os.getcwd(), "_radial_proto")
    os.makedirs(out_dir, exist_ok=True)

    sp_r, dur_r = synth_radial(v_um_s=700.0, period_s=T, seed=11)
    sp_l, dur_l = plk._synthetic_region(prop_angle=135.0, period_s=T,
                                        dur_s=120.0, seed=12, locked=True)

    A = analyse("radial stimulus", sp_r, dur_r, T, p)
    B = analyse("linear BR->TL stimulus", sp_l, dur_l, T, p)

    print(f"\nground truth centre = ({TRUE_CENTER[0]:.0f}, {TRUE_CENTER[1]:.0f}) um"
          f"   true ring speed = 700 um/s   T = {T:.2f} s\n")
    print(f"{'dataset':<24}{'planar':>8}{'radial':>8}{'radial|in':>10}"
          f"{'centre in array':>17}{'  verdict':<11}{'fitted centre':>19}"
          f"{'speed':>12}")
    print("-" * 112)
    for res in (A, B):
        pl = res["planar"]; ra = res["radial"]
        verdict = ("RADIAL" if (ra["centre_inside_array"]
                                and ra["coherence_centre_inside"] > pl["coherence"])
                   else "PLANAR")
        c = ra["centre_um"]
        print(f"{res['name']:<24}{pl['coherence']:>8.3f}{ra['coherence']:>8.3f}"
              f"{ra['coherence_centre_inside']:>10.3f}"
              f"{str(ra['centre_inside_array']):>17}  {verdict:<9}"
              f"{f'({c[0]:.0f}, {c[1]:.0f})':>19}"
              f"{ra['speed_um_per_s']:>9.0f} um/s")
    ra = A["radial"]
    err = np.hypot(ra["centre_um"][0] - TRUE_CENTER[0],
                   ra["centre_um"][1] - TRUE_CENTER[1])
    gc = centre_from_gradients(A["locks"], PITCH)
    gerr = np.hypot(gc[0] - TRUE_CENTER[0], gc[1] - TRUE_CENTER[1])
    print(f"\nradial dataset: centre error = {err:.0f} um "
          f"({err / PITCH:.2f} electrode pitches), "
          f"speed error = {ra['speed_um_per_s'] - 700.0:+.0f} um/s")
    print(f"gradient-line closed-form centre = ({gc[0]:.0f}, {gc[1]:.0f}) um"
          f"  -> error {gerr:.0f} um ({gerr / PITCH:.2f} pitches)")
    print(f"planar fit on the radial data reports heading "
          f"{A['planar']['direction_deg']:+.0f} deg at coherence "
          f"{A['planar']['coherence']:.3f} -- a confident-looking artefact")
    print(f"centre on search boundary: {ra['centre_on_bound']}  "
          f"(True would mean 'indistinguishable from planar')")

    figure(A, B, out_dir)
    print(f"\nfigure -> {os.path.join(out_dir, 'radial_proto.png')}")


def figure(A, B, out_dir):
    p = plk.PHASE_LOCKIN
    locks = A["locks"]; ra = A["radial"]
    cx, cy = ra["centre_um"]
    x = np.array([l.x_um for l in locks]); y = np.array([l.y_um for l in locks])
    st = np.array([l.strength for l in locks])
    th = np.array([l.phase for l in locks])
    r = np.hypot(x - cx, y - cy)

    fig = plt.figure(figsize=(15.5, 4.4), constrained_layout=True)
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1.15, 1])

    # (a) electrode map, colour = preferred phase, iso-phase rings
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor("#FAFAFA"); ax.set_aspect("equal")
    sc = ax.scatter(x / PITCH, y / PITCH, c=(th % (2 * np.pi)), cmap="twilight",
                    s=95, edgecolors="k", linewidths=0.4, zorder=3)
    lam = ra["wavelength_um"]
    for m in range(1, 7):
        ax.add_patch(plt.Circle((cx / PITCH, cy / PITCH), m * lam / 4 / PITCH,
                                fill=False, color="#555", lw=0.8, ls=":",
                                alpha=0.75, zorder=2))
    ax.plot(cx / PITCH, cy / PITCH, "x", color="crimson", ms=11, mew=2.4, zorder=5)
    ax.plot(TRUE_CENTER[0] / PITCH, TRUE_CENTER[1] / PITCH, "+", color="black",
            ms=13, mew=1.6, zorder=5)
    ax.set_xlim(0.3, 8.7); ax.set_ylim(0.3, 8.7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(a) preferred phase + fitted centre", fontsize=10)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046); cb.ax.tick_params(labelsize=6)
    cb.set_label("preferred phase (rad)", fontsize=7)

    # (b) the diagnostic: phase vs radius
    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor("#FAFAFA")
    thu = np.unwrap(th[np.argsort(r)])
    rs = np.sort(r)
    thu = thu - thu[0]
    ax.scatter(rs, thu, c=st / st.max(), cmap=plk.PARULA, norm=plk._NORM01,
               s=42, edgecolors="white", linewidths=0.5, zorder=3)
    rr = np.linspace(0, rs.max(), 50)
    ax.plot(rr, abs(ra["k_r"]) * rr, color="crimson", lw=1.6, ls="--", zorder=4,
            label=f"fit, slope $k_r$={abs(ra['k_r']):.2e} rad/µm")
    ax.set_xlabel("radius from fitted centre (µm)", fontsize=8.5)
    ax.set_ylabel("unwrapped preferred phase (rad)", fontsize=8.5)
    ax.tick_params(labelsize=8); ax.grid(color="#E2E2E2", lw=0.7)
    ax.set_axisbelow(True); ax.legend(fontsize=7, loc="upper left")
    ax.set_title("(b) phase vs radius — the key diagnostic", fontsize=10)

    # (c) centre likelihood surface
    ax = fig.add_subplot(gs[0, 2])
    cxs, cys, Pm = ra["_grid"]
    im = ax.pcolormesh(cxs / PITCH, cys / PITCH, Pm.T, cmap=plk.PARULA,
                       shading="auto")
    ax.plot(cx / PITCH, cy / PITCH, "x", color="crimson", ms=11, mew=2.4)
    ax.plot(TRUE_CENTER[0] / PITCH, TRUE_CENTER[1] / PITCH, "+", color="white",
            ms=13, mew=1.8)
    ax.set_aspect("equal"); ax.tick_params(labelsize=8)
    ax.set_title("(c) coherence vs candidate centre", fontsize=10)
    ax.set_xlabel("x (electrode pitches)", fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046); cb.ax.tick_params(labelsize=6)
    cb.set_label("max over $k_r$ of radial coherence", fontsize=7)

    # (d) model comparison
    ax = fig.add_subplot(gs[0, 3])
    ax.set_facecolor("#FAFAFA")
    names = ["radial\nstimulus", "linear\nstimulus"]
    pl = [A["planar"]["coherence"], B["planar"]["coherence"]]
    rd = [A["radial"]["coherence_centre_inside"],
          B["radial"]["coherence_centre_inside"]]
    xi = np.arange(2); wdt = 0.36
    b1 = ax.bar(xi - wdt / 2, pl, wdt, label="plane-wave fit",
                color=plt.get_cmap("Blues")(0.45), edgecolor="white", lw=1.2)
    b2 = ax.bar(xi + wdt / 2, rd, wdt, label="radial fit (centre in array)",
                color=plt.get_cmap("Blues")(0.85), edgecolor="white", lw=1.2)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=7.5, color="#555")
    ax.set_xticks(xi); ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("coherence (same scale)", fontsize=8.5)
    ax.tick_params(labelsize=8); ax.legend(fontsize=7.5, frameon=False)
    ax.grid(axis="y", color="#E2E2E2", lw=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("(d) model selection", fontsize=10)

    fig.suptitle("Radial (centre-out) lock-in — PROTOTYPE on synthetic data   "
                 "(black + = true centre at 900, 900 µm; red × = fitted)",
                 fontsize=11)
    fig.savefig(os.path.join(out_dir, "radial_proto.png"), dpi=170,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
