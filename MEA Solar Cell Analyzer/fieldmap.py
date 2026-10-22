"""Effective electric-field radius and shape of a solar cell (pure numpy/scipy).

From a SOLAR-ONLY recording (laser on the solar cell, no culture), each
electrode's flood-triggered-average peak-to-peak amplitude measures how strong
the cell's voltage field is at that electrode. Mapping that amplitude against the
electrode's distance from the square cell gives:

  * a decay length constant lambda (exponential fit),
  * an EFFECTIVE RADIUS = distance at which the field drops to a chosen floor,
  * ANISOTROPY: because the electrodes contact two opposite edges, the field
    reaches farther ALONG the electrode axis than ACROSS it -- fitted separately,
  * a 2-D interpolated field map + iso-amplitude contour (the field "shape").

Works for any square size because the cell geometry (side, orientation) is a
parameter, and distance is measured to the square's edge.
"""
import numpy as np

try:
    from scipy.optimize import curve_fit
    from scipy.interpolate import griddata
    _HAVE_SCIPY = True
except Exception:                       # pragma: no cover
    _HAVE_SCIPY = False


def _exp_model(d, a0, lam, base):
    return a0 * np.exp(-d / np.maximum(lam, 1e-6)) + base


def _fit_exp(dist, resp):
    """Fit resp = a0*exp(-d/lam)+base. Returns (a0, lam, base) or a log-linear
    fallback if curve_fit fails/unavailable."""
    dist = np.asarray(dist, float)
    resp = np.asarray(resp, float)
    a0_0 = max(resp.max() - resp.min(), 1e-3)
    base0 = float(np.min(resp))
    span = max(dist.max() - dist.min(), 1.0)
    if _HAVE_SCIPY and dist.size >= 4:
        try:
            popt, _ = curve_fit(
                _exp_model, dist, resp,
                p0=[a0_0, span / 2.0, base0],
                bounds=([0, 1.0, -abs(base0) - a0_0],
                        [10 * a0_0 + 1, 10 * span, resp.max() + 1]),
                maxfev=10000)
            return float(popt[0]), float(popt[1]), float(popt[2])
        except Exception:
            pass
    # fallback: log-linear on baseline-subtracted amplitude
    y = resp - base0
    y = np.maximum(y, 1e-6)
    A = np.vstack([np.ones_like(dist), -dist]).T
    coef, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
    lam = 1.0 / coef[1] if coef[1] > 1e-9 else span
    return float(np.exp(coef[0])), float(lam), base0


class FieldMapResult:
    def __init__(self, labels, resp, dist, along, across, fit, r_eff, r_1e,
                 lam_along, lam_across, floor_uv, cells, nearest=None):
        self.labels = list(labels)
        self.resp = np.asarray(resp, float)     # STA p2p per electrode (µV)
        self.dist = np.asarray(dist, float)     # µm to nearest cell edge
        self.along = np.asarray(along, float)   # µm along nearest cell's axis
        self.across = np.asarray(across, float)
        # index of the solar cell each electrode was assigned to (-1 = none)
        self.nearest = (np.full(len(self.labels), -1, int) if nearest is None
                        else np.asarray(nearest, int))
        self.fit = fit                          # (a0, lam, base)
        self.r_eff = float(r_eff)               # effective radius (µm) at floor
        self.r_1e = float(r_1e)                 # 1/e length (µm) = lam
        self.lam_along = float(lam_along)
        self.lam_across = float(lam_across)
        self.floor_uv = float(floor_uv)
        self.cells = list(cells)                # solar cells used (1 or 2+)

    @property
    def n_cells(self):
        return len(self.cells)

    @property
    def peak_uv(self):
        return float(np.max(self.resp)) if self.resp.size else 0.0

    @property
    def mean_uv(self):
        return float(np.mean(self.resp)) if self.resp.size else 0.0

    @property
    def total_uv(self):
        return float(np.sum(self.resp)) if self.resp.size else 0.0

    @property
    def cell_peaks(self):
        """Largest response (µV) on each cell's OWN electrodes — one value per
        entry of `self.cells`, using the nearest-cell assignment already made when
        the map was computed. NaN for a cell that claimed no electrode."""
        out = [float("nan")] * len(self.cells)
        for i, k in enumerate(self.nearest):
            k = int(k)
            if k < 0 or k >= len(out):
                continue
            v = float(self.resp[i])
            if not np.isfinite(v):
                continue
            if not np.isfinite(out[k]) or v > out[k]:
                out[k] = v
        return out

    def cell_peak_labels(self):
        """The electrode label holding each cell's peak (None where there is no
        assigned electrode). Parallel to `cell_peaks`."""
        best = [float("nan")] * len(self.cells)
        labs = [None] * len(self.cells)
        for i, k in enumerate(self.nearest):
            k = int(k)
            if k < 0 or k >= len(labs):
                continue
            v = float(self.resp[i])
            if not np.isfinite(v):
                continue
            if not np.isfinite(best[k]) or v > best[k]:
                best[k] = v
                labs[k] = self.labels[i]
        return labs

    @property
    def anisotropy(self):
        """lambda_along / lambda_across (>1 => field elongated along the axis)."""
        if self.lam_across <= 0:
            return float("nan")
        return self.lam_along / self.lam_across

    def interpolate(self, array, n=120, pad_grid=1.0):
        """Interpolated response over the array plane for a heatmap.
        Returns (Z, extent=(x0,x1,y0,y1)) in GRID units. Z is masked (nan) outside
        the convex hull of the electrodes."""
        # positions of the electrodes we actually HAVE a response for (may be < 60
        # if the recording dropped a reference/grounded channel) — must match resp.
        pts = np.array([array.position_grid(l) for l in self.labels], float)
        x0, y0 = pts.min(0) - pad_grid
        x1, y1 = pts.max(0) + pad_grid
        gx, gy = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
        if _HAVE_SCIPY:
            Z = griddata(pts, self.resp, (gx, gy), method="cubic")
        else:                              # nearest fallback
            Z = np.zeros_like(gx)
            for i in range(gx.shape[0]):
                for j in range(gx.shape[1]):
                    d2 = ((pts[:, 0] - gx[i, j]) ** 2
                          + (pts[:, 1] - gy[i, j]) ** 2)
                    Z[i, j] = self.resp[np.argmin(d2)]
        return Z, (x0, x1, y0, y1)


def model_field_grid(array, cells, amp0_uv, lam_um, n=140, pad_grid=1.5):
    """Smooth MODELED solar-cell field magnitude over the array plane.

    field(x) = amp0_uv * exp(-d_edge(x) / lam_um), where d_edge is the µm distance
    from x to the NEAREST cell's edge (0 inside a cell). A clean physical picture of
    the field a cell of that output voltage emits — usable even on a culture sample
    with no dedicated solar-only recording. Returns (Z (n,n) in µV, extent in GRID
    units matching the electrode map)."""
    import geometry as _geo
    pts = np.array([array.position_grid(l) for l in array.labels], float)
    x0, y0 = pts.min(0) - pad_grid
    x1, y1 = pts.max(0) + pad_grid
    gx, gy = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    P = np.column_stack([gx.ravel(), gy.ravel()])
    dist, _ = _geo.nearest_cell_distance_um(list(cells), P)   # µm to nearest edge
    Z = float(amp0_uv) * np.exp(-dist / max(float(lam_um), 1e-6))
    return Z.reshape(gx.shape), (x0, x1, y0, y1)


def _dir_length_constant(coord_abs, dist, resp, frac=0.6):
    """Length constant for electrodes whose displacement is predominantly along
    one axis (|coord| dominates). Falls back to the global fit if too few."""
    coord_abs = np.asarray(coord_abs, float)
    other = dist  # approx; used only to gate 'predominantly along'
    sel = coord_abs >= frac * np.maximum(dist, 1e-6)
    if sel.sum() >= 4:
        _, lam, _ = _fit_exp(dist[sel], resp[sel])
        return lam
    _, lam, _ = _fit_exp(dist, resp)
    return lam


def compute_field_map(array, response_by_label, cells, floor_uv=None,
                      noise_by_label=None, floor_k=3.0, baseline_by_label=None):
    """Build a FieldMapResult for ONE or MORE solar cells.

    array             : geometry.ElectrodeArray.
    response_by_label : label -> flood-evoked p2p amplitude (µV) (from sta_bank).
    cells             : list of geometry.SolarCell (1 for single-cell, 2 for the
                        amplification study). Distance is measured to the NEAREST
                        cell; anisotropy uses that cell's electrode axis.
    baseline_by_label : optional label -> baseline p2p (µV), e.g. the laser-only
                        (piezo) response; subtracted so the map is the pure solar
                        field. Clamped at 0.
    floor_uv          : field 'edge' amplitude threshold (µV). If None, uses
                        floor_k x median electrode noise (or 5 µV if no noise given).
    """
    import geometry as _geo
    cells = list(cells or [])
    if not cells:
        raise ValueError("No solar cell placed for this recording.")
    labels = [l for l in array.labels if l in response_by_label]
    resp = np.array([response_by_label[l] for l in labels], float)
    if baseline_by_label:
        base = np.array([baseline_by_label.get(l, 0.0) for l in labels], float)
        resp = np.maximum(resp - base, 0.0)
    grid = np.array([array.position_grid(l) for l in labels], float)
    dist, nearest = _geo.nearest_cell_distance_um(cells, grid)
    along = np.zeros(len(labels))
    across = np.zeros(len(labels))
    for i in range(len(labels)):
        ac = cells[int(nearest[i])].axis_coords_um([grid[i]])[0]
        along[i], across[i] = ac[0], ac[1]

    if floor_uv is None:
        if noise_by_label:
            med_noise = float(np.median([noise_by_label.get(l, np.nan)
                                         for l in labels]))
            floor_uv = floor_k * (med_noise if np.isfinite(med_noise) else 2.0)
        else:
            floor_uv = 5.0

    a0, lam, base = _fit_exp(dist, resp)
    # effective radius: where the fitted (baseline-subtracted) field == floor
    amp0 = max(a0, 1e-6)
    if floor_uv < amp0:
        r_eff = lam * np.log(amp0 / floor_uv)
    else:
        r_eff = 0.0
    r_eff = float(max(r_eff, 0.0))

    lam_along = _dir_length_constant(np.abs(along), dist, resp)
    lam_across = _dir_length_constant(np.abs(across), dist, resp)

    return FieldMapResult(labels, resp, dist, along, across, (a0, lam, base),
                          r_eff, lam, lam_along, lam_across, floor_uv, cells,
                          nearest=nearest)
