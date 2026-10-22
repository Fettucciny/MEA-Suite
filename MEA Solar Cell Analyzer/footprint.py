"""Spheroid footprint from the recording (pure numpy/scipy).

After artifact removal, each electrode has a biological spike rate (and mean spike
amplitude). Mapped onto the array these give the spheroid's electrophysiological
FOOTPRINT -- which electrodes the soma cluster actually drives. In this prep the
neurites usually don't spike, only the soma, so the footprint marks where somata
sit close enough to an electrode to be recorded.

Reported:
  * active-electrode map + footprint centroid and spread (radius of gyration),
  * SPIKING REACH: the farthest an active electrode is from the (imaging) soma
    centre -- the empirical 'minimum proximity' at which a soma still drives an
    electrode,
  * spike rate vs distance from the soma centre (to compare against imaging soma
    location / neurite density),
  * a 2-D interpolated footprint map + contour (the footprint shape).
"""
import numpy as np

try:
    from scipy.interpolate import griddata
    _HAVE_SCIPY = True
except Exception:                       # pragma: no cover
    _HAVE_SCIPY = False


class FootprintResult:
    def __init__(self, labels, rate, amp, active, centroid_grid, spread_um,
                 reach_um, dist_um, spheroid):
        self.labels = list(labels)
        self.rate = np.asarray(rate, float)      # Hz per electrode
        self.amp = np.asarray(amp, float)        # mean |spike amp| (µV) per elec
        self.active = np.asarray(active, bool)
        self.centroid_grid = (None if centroid_grid is None
                              else np.asarray(centroid_grid, float))
        self.spread_um = float(spread_um)
        self.reach_um = float(reach_um)          # max active-electrode distance
        self.dist_um = (None if dist_um is None else np.asarray(dist_um, float))
        self.spheroid = spheroid

    def n_active(self):
        return int(self.active.sum())

    def rate_vs_distance(self):
        """(distance_um, rate) for active electrodes, sorted by distance."""
        if self.dist_um is None:
            return np.empty(0), np.empty(0)
        d = self.dist_um[self.active]
        r = self.rate[self.active]
        o = np.argsort(d)
        return d[o], r[o]

    def interpolate(self, array, n=120, pad_grid=1.0, metric="rate"):
        # positions matching self.labels (may be < 60), so pts and vals align.
        pts = np.array([array.position_grid(l) for l in self.labels], float)
        vals = self.rate if metric == "rate" else self.amp
        x0, y0 = pts.min(0) - pad_grid
        x1, y1 = pts.max(0) + pad_grid
        gx, gy = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
        if _HAVE_SCIPY:
            Z = griddata(pts, vals, (gx, gy), method="cubic")
        else:
            Z = np.zeros_like(gx)
            for i in range(gx.shape[0]):
                for j in range(gx.shape[1]):
                    d2 = ((pts[:, 0] - gx[i, j]) ** 2
                          + (pts[:, 1] - gy[i, j]) ** 2)
                    Z[i, j] = vals[np.argmin(d2)]
        return Z, (x0, x1, y0, y1)


def compute_footprint(array, rate_by_label, amp_by_label=None, spheroid=None,
                      active_rate_hz=0.1, active_frac_of_max=0.05):
    """Build a FootprintResult.

    array           : geometry.ElectrodeArray.
    rate_by_label   : label -> biological spike rate (Hz) (FilterResult.rate_map).
    amp_by_label    : label -> mean spike amplitude (µV), optional.
    spheroid        : geometry.Spheroid (for distance/reach), optional.
    active_*        : an electrode is 'active' if rate >= active_rate_hz AND
                      >= active_frac_of_max x the max electrode rate.
    """
    labels = [l for l in array.labels if l in rate_by_label]
    rate = np.array([rate_by_label[l] for l in labels], float)
    amp = np.array([(amp_by_label or {}).get(l, 0.0) for l in labels], float)
    grid = np.array([array.position_grid(l) for l in labels], float)

    rmax = rate.max() if rate.size else 0.0
    active = (rate >= active_rate_hz) & (rate >= active_frac_of_max * max(rmax, 1e-9))

    centroid_grid, spread_um, reach_um, dist_um = None, 0.0, 0.0, None
    if active.any():
        w = rate[active]
        pa = grid[active]
        centroid_grid = np.average(pa, axis=0, weights=w)
        d_c = np.linalg.norm((pa - centroid_grid) * array.pitch_um, axis=1)
        spread_um = float(np.sqrt(np.average(d_c ** 2, weights=w)))
    if spheroid is not None:
        dist_um = spheroid.distance_um(grid)
        if active.any():
            reach_um = float(np.max(dist_um[active]))

    return FootprintResult(labels, rate, amp, active, centroid_grid, spread_um,
                           reach_um, dist_um, spheroid)
