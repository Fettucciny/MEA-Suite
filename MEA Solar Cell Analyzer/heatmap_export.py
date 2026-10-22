"""Render Filter-tab electrode heatmaps to a PNG that mirrors the in-app map.

Dark theme (so the solar-cell field overlay reads like the on-screen map instead of
washing out on white), robust 95th-percentile OR log colour scaling, electrode
numbers on a translucent dark chip, artifact-subtracted electrodes ringed in gold,
an optional spheroid outline, an optional solar-cell field background, and the
placed solar cells outlined with the MAX voltage detected on their nearest
electrodes written on top of each cell. Can emit
a single panel or a pre/stim/post triptych (field only on the stim panel).
Headless (Agg); saves at 600 DPI.
"""
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import geometry as geo

_BG = "#15171c"          # dark panel background (matches the pyqtgraph map)
_FG = "#e6e6e6"          # light text/ticks
_SOLAR = "#3cc8ff"       # solar-cell outline (same cyan as the on-screen map)
_SOLAR_TXT = "#fbbf24"   # amber — this program's accent, for the voltage number


def fmt_uv(v):
    """Human-readable voltage for an annotation: µV below 1 mV, mV above."""
    v = float(v)
    if not np.isfinite(v):
        return "n/a"
    a = abs(v)
    if a >= 1000.0:
        return "%.2f mV" % (v / 1000.0)
    if a >= 100.0:
        return "%.0f µV" % v
    return "%.1f µV" % v


def _draw_solar_cells(ax, cells, values, lim, labels=None, unit_fmt=fmt_uv,
                      plateaus=None):
    """Outline each placed solar cell and write its max detected voltage on top.

    cells    : list of geometry.SolarCell.
    values   : parallel list of scalars (µV) — the largest response measured on the
               electrodes nearest that cell. None/NaN draws the outline only.
    labels   : optional parallel list of electrode labels (which electrode held the
               max), shown as a small second line.
    plateaus : optional parallel list of the SUSTAINED photovoltage (µV) held during
               illumination near each cell, shown as a small third line ("plat …").
    """
    if not cells:
        return
    vals = list(values or [])
    labs = list(labels or [])
    plats = list(plateaus or [])
    for i, cell in enumerate(cells):
        try:
            c = np.asarray(cell.corners_grid(), float)
        except Exception:
            continue
        poly = np.vstack([c, c[0]])
        ax.plot(poly[:, 0], poly[:, 1], color=_SOLAR, lw=1.8, alpha=0.95, zorder=7)
        v = vals[i] if i < len(vals) else None
        if v is None or not np.isfinite(float(v)):
            continue
        txt = unit_fmt(v)
        el = labs[i] if i < len(labs) else None
        if el:
            txt += "\n@ %s" % el
        pl = plats[i] if i < len(plats) else None
        if pl is not None and np.isfinite(float(pl)):
            txt += "\nplat %s" % unit_fmt(float(pl))
        # sit just above the cell's top edge; flip below if that would clip out of
        # the square crop.
        cx = float(np.mean(c[:, 0]))
        top, bot = float(np.max(c[:, 1])), float(np.min(c[:, 1]))
        y, va = top + 0.12, "bottom"
        if y > lim[1] - 0.45:
            y, va = bot - 0.12, "top"
        ax.text(cx, y, txt, ha="center", va=va, color=_SOLAR_TXT,
                fontsize=7.5, fontweight="bold", zorder=8, linespacing=1.1,
                bbox=dict(boxstyle="round,pad=0.18", fc=(0, 0, 0, 0.82),
                          ec=_SOLAR, lw=0.7))


def _xy_vals(labels, values_by_label):
    xs, ys, vs, labs = [], [], [], []
    for l in labels:
        c = geo.mcs_coord(str(l))
        if c is None:
            continue
        xs.append(c[0])
        ys.append(c[1])
        vs.append(float(values_by_label.get(l, np.nan)))
        labs.append(str(l))
    return np.array(xs), np.array(ys), np.array(vs, float), labs


def _make_norm(all_vals, log):
    """A shared colour Normalize across panels (so regions are comparable)."""
    fin = np.asarray(all_vals, float)
    fin = fin[np.isfinite(fin)]
    if log:
        pos = fin[fin > 0]
        if pos.size:
            vlo = float(pos.min())
            vhi = float(np.percentile(pos, 95)) if pos.size >= 8 else float(pos.max())
            if vhi <= vlo:
                vhi = vlo * 10.0
            return mcolors.LogNorm(vmin=vlo, vmax=vhi), vlo
        return mcolors.LogNorm(vmin=1e-3, vmax=1.0), 1e-3
    if fin.size:
        vlo = float(fin.min())
        vhi = float(np.percentile(fin, 95)) if fin.size >= 8 else float(fin.max())
        if vhi <= vlo:
            vhi = vlo + 1.0
    else:
        vlo, vhi = 0.0, 1.0
    return mcolors.Normalize(vmin=vlo, vmax=vhi, clip=True), vlo


def _square_limits(labels, pad=0.7, extra_pts=None):
    """A single square (x==y span) data window covering the electrode grid, so each
    panel is cropped to a square and the grid fills it. `extra_pts` (Nx2 grid
    coords, e.g. solar-cell corners) widens the window so cells placed off to the
    side of the array stay inside the crop."""
    xs, ys = [], []
    for l in labels:
        c = geo.mcs_coord(str(l))
        if c is not None:
            xs.append(c[0])
            ys.append(c[1])
    if not xs:
        return (0.0, 1.0)
    lo = min(min(xs), min(ys)) - pad
    hi = max(max(xs), max(ys)) + pad
    if extra_pts is not None and len(extra_pts):
        p = np.asarray(extra_pts, float).reshape(-1, 2)
        if p.size:
            lo = min(lo, float(p.min()) - pad)
            hi = max(hi, float(p.max()) + pad)
    return (lo, hi)


def _draw_panel(ax, labels, values_by_label, norm, floor, log, marks,
                field, spheroid_outline, title, lim,
                solar_cells=None, cell_values=None, cell_labels=None,
                cell_plateaus=None):
    ax.set_facecolor(_BG)
    xs, ys, vs, labs = _xy_vals(labels, values_by_label)
    if field is not None:
        Z, ext = field
        Z = np.asarray(Z, float)
        fin = Z[np.isfinite(Z)]
        if fin.size:
            flo = float(np.nanpercentile(fin, 2))
            fhi = float(np.nanpercentile(fin, 98))
            if fhi <= flo:
                fhi = flo + 1.0
        else:
            flo, fhi = 0.0, 1.0
        ax.imshow(Z, extent=[ext[0], ext[1], ext[2], ext[3]], origin="lower",
                  cmap="magma", alpha=0.85, aspect="equal", zorder=0,
                  vmin=flo, vmax=fhi, interpolation="bilinear")
    fin = np.isfinite(vs)
    off = ~fin
    if off.any():                            # grounded / off electrodes -> gray
        ax.scatter(xs[off], ys[off], c="#4a4d55", s=360, edgecolors="k",
                   linewidths=0.8, zorder=3)
    pv = np.where(vs > 0, vs, floor) if log else vs
    sc = ax.scatter(xs[fin], ys[fin], c=pv[fin], cmap="viridis", norm=norm,
                    s=360, edgecolors="k", linewidths=0.8, zorder=3)
    for x, y, l in zip(xs, ys, labs):
        ax.text(x, y, l, ha="center", va="center", color="white",
                fontsize=6.0, fontweight="bold", zorder=4,
                bbox=dict(boxstyle="round,pad=0.12", fc=(0, 0, 0, 0.55), ec="none"))
    mk = np.array([(x, y) for x, y, l in zip(xs, ys, labs) if l in (marks or ())],
                  float)
    if mk.size:
        ax.scatter(mk[:, 0], mk[:, 1], s=660, facecolors="none",
                   edgecolors="gold", linewidths=2.2, zorder=5)
    if spheroid_outline is not None and len(spheroid_outline) >= 3:
        o = np.asarray(spheroid_outline, float)
        o = np.vstack([o, o[0]])
        ax.plot(o[:, 0], o[:, 1], color="#ff78c8", lw=2.0, zorder=6)
    _draw_solar_cells(ax, solar_cells, cell_values, lim, labels=cell_labels,
                      plateaus=cell_plateaus)
    ax.set_aspect("equal")
    # square crop; NO y-flip — mcs_coord already puts row 1 (y = 8) on top, matching
    # the on-screen GUI map.
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, color=_FG, fontsize=11)
    for s in ax.spines.values():
        s.set_color("#3a3d44")
    return sc


def save_region_heatmaps(path, regions, suptitle="", marks=None,
                         spheroid_outline=None, log=False,
                         value_label="rate (Hz)", dpi=600,
                         solar_cells=None, cell_values=None, cell_labels=None,
                         cell_plateaus=None):
    """Render a pre/stim/post triptych (or any list) sharing one colour scale.

    regions     : list of (name, values_by_label, field_or_None). `field` is an
                  optional (Z, extent_grid) background drawn only on that panel
                  (e.g. stim only).
    solar_cells : optional list of geometry.SolarCell to outline on every panel.
    cell_values : optional parallel list of the MAX voltage (µV) detected on the
                  electrodes nearest each cell — written on top of that cell.
    cell_labels : optional parallel list of the electrode label each max came from.
    Returns the written path.
    """
    labels = sorted({l for _n, v, _f in regions for l in v},
                    key=lambda s: (len(str(s)), str(s)))
    all_vals = [v.get(l, np.nan) for _n, v, _f in regions for l in labels]
    norm, floor = _make_norm(all_vals, log)

    n = len(regions)
    corners = []
    for cell in (solar_cells or []):
        try:
            corners.append(np.asarray(cell.corners_grid(), float))
        except Exception:
            pass
    lim = _square_limits(labels,
                         extra_pts=np.vstack(corners) if corners else None)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 6.0), facecolor=_BG)
    if n == 1:
        axes = [axes]
    sc = None
    for ax, (name, vals, field) in zip(axes, regions):
        sc = _draw_panel(ax, labels, vals, norm, floor, log, marks, field,
                         spheroid_outline, name, lim,
                         solar_cells=solar_cells, cell_values=cell_values,
                         cell_labels=cell_labels, cell_plateaus=cell_plateaus)
    if suptitle:
        fig.suptitle(suptitle, color=_FG, fontsize=13)
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
        cb.set_label(value_label + ("  (log)" if log else "  (linear, ≤95th pct)"),
                     color=_FG)
        cb.ax.yaxis.set_tick_params(color=_FG)
        plt.setp(cb.ax.get_yticklabels(), color=_FG)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=_BG)
    plt.close(fig)
    return path


def save_electrode_heatmap(path, labels, values_by_label, title="",
                           marks=None, field=None, spheroid_outline=None,
                           log=False, value_label="rate (Hz)", dpi=600,
                           solar_cells=None, cell_values=None, cell_labels=None,
                           cell_plateaus=None):
    """Single-panel heatmap (dark theme). Kept for callers that want one map."""
    return save_region_heatmaps(
        path, [(title, values_by_label, field)], suptitle="", marks=marks,
        spheroid_outline=spheroid_outline, log=log, value_label=value_label,
        dpi=dpi, solar_cells=solar_cells, cell_values=cell_values,
        cell_labels=cell_labels, cell_plateaus=cell_plateaus)
