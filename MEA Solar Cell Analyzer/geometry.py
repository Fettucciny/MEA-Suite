"""Electrode-array geometry, solar-cell and spheroid models (pure numpy).

Coordinate frames
-----------------
* GRID units -- the same frame the electrode map draws in and the user clicks in.
  An MCS label "CR" (column C, row R) sits at (x, y) = (C, 9 - R), so row 1 is at
  the TOP. This matches mea_nap_original_plots.mcs_coord exactly, so anything laid
  out here lines up with the MEA-NAP heatmaps and the raster viewer's map.
* MICRON units -- grid units * PITCH_UM (default 200 µm, MCS 60MEA200). Distances
  and the solar-cell field radius are reported in microns.

The solar cell is a SQUARE block. Electrodes contact two OPPOSITE edges of the
square, so current flows across the cell along one axis (the "electrode axis").
`orientation_deg` rotates the square in the plane; `electrode_axis_deg` is the
direction of that current-flow axis (defaults to orientation). This lets the
field analysis separate decay ALONG the axis from decay ACROSS it (a rotated
rectangle field is anisotropic).
"""
import numpy as np

DEFAULT_PITCH_UM = 200.0
_CORNERS = {"11", "18", "81", "88"}

# The MCS 8x8 grid centre in GRID units (cols 1-8 -> mean 4.5; y = 9-row -> 4.5).
ARRAY_CENTER_GRID = (4.5, 4.5)
# Diameter (µm) of the field-of-view circle drawn on the map, centred on the
# array, so solar cells placed off to the sides are still visible/placeable.
FOV_DIAMETER_UM = 3000.0


def grid_from_um(x_um, y_um, pitch_um, center_grid=ARRAY_CENTER_GRID):
    """Grid coords from a µm offset relative to the array centre (+x right, +y up)."""
    return (center_grid[0] + x_um / pitch_um, center_grid[1] + y_um / pitch_um)


def um_from_grid(gx, gy, pitch_um, center_grid=ARRAY_CENTER_GRID):
    """µm offset from the array centre for a grid coordinate."""
    return ((gx - center_grid[0]) * pitch_um, (gy - center_grid[1]) * pitch_um)


def mcs_coord(label):
    """(x, y) grid position for an MCS label, y flipped (row 1 on top); None for
    non-grid labels. Identical to mea_nap_original_plots.mcs_coord."""
    s = str(label).strip()
    if len(s) == 2 and s.isdigit():
        col, row = int(s[0]), int(s[1])
        if 1 <= col <= 8 and 1 <= row <= 8:
            return float(col), float(9 - row)
    return None


def mcs60_labels():
    """All 60 MCS 8x8-grid electrode labels (four corners removed)."""
    return [f"{c}{r}" for c in range(1, 9) for r in range(1, 9)
            if f"{c}{r}" not in _CORNERS]


class ElectrodeArray:
    """The 60 usable MCS electrodes with grid + micron positions."""

    def __init__(self, labels=None, pitch_um=DEFAULT_PITCH_UM):
        self.pitch_um = float(pitch_um)
        self.labels = list(labels) if labels is not None else mcs60_labels()
        self.labels = [l for l in self.labels if mcs_coord(l) is not None]
        self.grid = np.array([mcs_coord(l) for l in self.labels], float)  # (n,2)
        self._index = {l: i for i, l in enumerate(self.labels)}

    @property
    def um(self):
        """(n, 2) electrode positions in microns."""
        return self.grid * self.pitch_um

    def index(self, label):
        return self._index[str(label)]

    def position_grid(self, label):
        return self.grid[self._index[str(label)]]

    def position_um(self, label):
        return self.grid[self._index[str(label)]] * self.pitch_um


def _rot(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


class SolarCell:
    """A square solar-cell block placed on the array.

    center_grid       : (x, y) in grid units (where the user clicked).
    side_um           : side length of the square (microns).
    orientation_deg   : rotation of the square in the plane.
    electrode_axis_deg: direction of the electrode current-flow axis (the normal
                        to the two electrode-contacted edges). Defaults to
                        orientation_deg. Used to separate along-axis vs across-axis
                        field decay.
    """

    def __init__(self, center_grid, side_um, orientation_deg=0.0,
                 electrode_axis_deg=None, pitch_um=DEFAULT_PITCH_UM):
        self.center_grid = np.asarray(center_grid, float)
        self.side_um = float(side_um)
        self.orientation_deg = float(orientation_deg)
        self.electrode_axis_deg = (float(orientation_deg)
                                   if electrode_axis_deg is None
                                   else float(electrode_axis_deg))
        self.pitch_um = float(pitch_um)

    @property
    def center_um(self):
        return self.center_grid * self.pitch_um

    def _to_um(self, pts_grid):
        return np.atleast_2d(np.asarray(pts_grid, float)) * self.pitch_um

    def corners_grid(self):
        """4 corner points of the square in grid units (for drawing)."""
        h = self.side_um / 2.0
        local = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
        world_um = local @ _rot(self.orientation_deg).T + self.center_um
        return world_um / self.pitch_um

    def signed_distance_um(self, pts_grid):
        """Signed distance (µm) from each point to the square boundary: negative
        inside, 0 on the edge, positive outside (the true exterior distance to a
        rotated box)."""
        pts_um = self._to_um(pts_grid)
        local = (pts_um - self.center_um) @ _rot(self.orientation_deg)  # inverse rot
        h = self.side_um / 2.0
        q = np.abs(local) - h
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside

    def distance_um(self, pts_grid):
        """Exterior distance (µm); 0 for points inside the square."""
        return np.maximum(self.signed_distance_um(pts_grid), 0.0)

    def contains(self, pts_grid):
        return self.signed_distance_um(pts_grid) <= 0.0

    def axis_coords_um(self, pts_grid):
        """Each point's (along, across) coordinates (µm) in the electrode-axis
        frame, relative to the cell centre. 'along' is the current-flow axis; a
        larger |along| than |across| decay length => the field reaches farther
        along the electrode axis (expected for a two-edge-contacted square)."""
        pts_um = self._to_um(pts_grid)
        rel = (pts_um - self.center_um) @ _rot(self.electrode_axis_deg)
        return rel  # column 0 = along axis, column 1 = across


class Spheroid:
    """A neuron spheroid seeded on the array (soma cluster + neurite halo).

    center_grid : (x, y) grid units (imaging soma centre, or best guess).
    radius_um   : nominal soma-cluster radius (microns); the footprint analysis
                  measures the ACTUAL spiking extent from the recording.
    """

    def __init__(self, center_grid, radius_um=200.0, pitch_um=DEFAULT_PITCH_UM):
        self.center_grid = np.asarray(center_grid, float)
        self.radius_um = float(radius_um)
        self.pitch_um = float(pitch_um)

    @property
    def center_um(self):
        return self.center_grid * self.pitch_um

    def distance_um(self, pts_grid):
        """Distance (µm) from each point to the spheroid EDGE (0 inside the disc)."""
        pts_um = np.atleast_2d(np.asarray(pts_grid, float)) * self.pitch_um
        d = np.linalg.norm(pts_um - self.center_um, axis=1) - self.radius_um
        return np.maximum(d, 0.0)

    def outline_grid(self, n=64):
        """Closed circle outline in grid units (for drawing)."""
        th = np.linspace(0, 2 * np.pi, n)
        r = self.radius_um / self.pitch_um
        return np.column_stack([self.center_grid[0] + r * np.cos(th),
                                self.center_grid[1] + r * np.sin(th)])


def _polygon_centroid(poly):
    """Area-weighted centroid of a closed polygon (Nx2). Falls back to the mean
    for degenerate (zero-area) polygons."""
    p = np.asarray(poly, float)
    x, y = p[:, 0], p[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y2 - x2 * y
    A = cross.sum() / 2.0
    if abs(A) < 1e-9:
        return p.mean(axis=0)
    cx = ((x + x2) * cross).sum() / (6 * A)
    cy = ((y + y2) * cross).sum() / (6 * A)
    return np.array([cx, cy])


def _point_in_polygon(pt, poly):
    """Ray-casting point-in-polygon test."""
    x, y = pt
    p = np.asarray(poly, float)
    inside = False
    n = len(p)
    j = n - 1
    for i in range(n):
        xi, yi = p[i]
        xj, yj = p[j]
        if ((yi > y) != (yj > y)) and \
                (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _dist_point_to_polygon(pt, poly):
    """Min distance from a point to a closed polygon's edges (grid units)."""
    p = np.asarray(poly, float)
    a = p
    b = np.roll(p, -1, axis=0)
    ab = b - a
    t = np.clip(((pt - a) * ab).sum(1) / ((ab * ab).sum(1) + 1e-12), 0, 1)
    proj = a + t[:, None] * ab
    return float(np.min(np.linalg.norm(pt - proj, axis=1)))


def fit_closed_curve(points_grid, n=200):
    """Smooth closed curve through perimeter points (in click order).

    Uses a periodic cubic spline when possible so an irregular, hand-clicked
    outline is smoothed without reordering the points; falls back to the raw
    closed polygon for <4 points or if scipy is unavailable."""
    p = np.asarray(points_grid, float).reshape(-1, 2)
    if len(p) < 3:
        return p
    try:
        from scipy.interpolate import splprep, splev
        k = min(3, len(p) - 1)
        tck, _u = splprep([p[:, 0], p[:, 1]], s=0, per=True, k=k)
        un = np.linspace(0, 1, n)
        xo, yo = splev(un, tck)
        return np.column_stack([xo, yo])
    except Exception:
        return np.vstack([p, p[:1]])


class SpheroidOutline:
    """A spheroid whose footprint is a hand-traced perimeter (irregular shape).

    points_grid : perimeter points (grid units) in the order the user clicked.
    The smoothed closed curve is fitted through them; distances are measured to
    that curve (0 inside)."""

    def __init__(self, points_grid, pitch_um=DEFAULT_PITCH_UM, smooth=True):
        self.points_grid = np.asarray(points_grid, float).reshape(-1, 2)
        self.pitch_um = float(pitch_um)
        self.smooth = smooth

    @property
    def center_grid(self):
        # Use the clicked polygon (stable, symmetric) rather than the smoothed
        # curve, which can wobble when only a few points are clicked.
        if len(self.points_grid) == 0:
            return np.array(ARRAY_CENTER_GRID, float)
        if len(self.points_grid) < 3:
            return self.points_grid.mean(axis=0)
        return _polygon_centroid(self.points_grid)

    @property
    def center_um(self):
        return self.center_grid * self.pitch_um

    def outline_grid(self, n=200):
        return fit_closed_curve(self.points_grid, n) if self.smooth \
            else self.points_grid

    def distance_um(self, pts_grid):
        """Distance (µm) from each point to the outline (0 inside)."""
        pts = np.atleast_2d(np.asarray(pts_grid, float))
        if len(self.points_grid) < 3:
            # not enough points for an area: distance to the nearest clicked point
            if len(self.points_grid) == 0:
                return np.zeros(len(pts))
            d = np.min(np.linalg.norm(
                pts[:, None, :] - self.points_grid[None, :, :], axis=2), axis=1)
            return d * self.pitch_um
        poly = self.outline_grid()
        out = np.zeros(len(pts))
        for i, q in enumerate(pts):
            if _point_in_polygon(q, poly):
                out[i] = 0.0
            else:
                out[i] = _dist_point_to_polygon(q, poly) * self.pitch_um
        return out


def nearest_cell_distance_um(cells, pts_grid):
    """For each point, the distance (µm) to the NEAREST solar cell and that cell's
    index. Returns (distances, indices). Empty cells -> inf / -1."""
    pts = np.atleast_2d(np.asarray(pts_grid, float))
    if not cells:
        return np.full(len(pts), np.inf), np.full(len(pts), -1, int)
    D = np.column_stack([c.distance_um(pts) for c in cells])   # (n, ncells)
    idx = np.argmin(D, axis=1)
    return D[np.arange(len(pts)), idx], idx


def max_by_cell(cells, values_by_label, array, max_dist_um=None,
                by_magnitude=False):
    """Assign every electrode to its NEAREST solar cell, then take the LARGEST
    value measured on each cell's own electrodes.

    This is what the maps annotate: with 3 cells in the field of view, every
    electrode is claimed by whichever cell it sits closest to (distance to the
    square's EDGE, so an electrode under a cell has distance 0), and each cell is
    labelled with the biggest value among the electrodes it claimed.

    cells           : list of SolarCell.
    values_by_label : label -> scalar (e.g. flood-STA peak-to-peak in µV). Labels
                      missing from the array, or with non-finite values, are
                      skipped.
    array           : ElectrodeArray supplying the electrode positions.
    max_dist_um     : optional cut-off — electrodes farther than this from their
                      nearest cell's edge are not counted. None (default) lets
                      every electrode belong to some cell.
    by_magnitude    : compare by |value| instead of signed value (so a signed
                      quantity like the sustained plateau reports the electrode with
                      the strongest deflection of EITHER polarity, keeping its sign).

    Returns one dict per cell, in the same order as `cells`:
        {"value": µV (nan if nothing qualified), "label": electrode or None,
         "dist_um": that electrode's distance to the cell edge, "n": how many
         electrodes were assigned to this cell}
    """
    cells = list(cells or [])
    out = [{"value": float("nan"), "label": None, "dist_um": float("nan"), "n": 0}
           for _ in cells]
    if not cells or not values_by_label:
        return out
    labels = [l for l in array.labels if l in values_by_label]
    if not labels:
        return out
    grid = np.array([array.position_grid(l) for l in labels], float)
    dist, nearest = nearest_cell_distance_um(cells, grid)
    for i, lab in enumerate(labels):
        v = float(values_by_label[lab])
        if not np.isfinite(v):
            continue
        if max_dist_um is not None and dist[i] > float(max_dist_um):
            continue
        rec = out[int(nearest[i])]
        rec["n"] += 1
        better = (abs(v) > abs(rec["value"])) if by_magnitude else (v > rec["value"])
        if not np.isfinite(rec["value"]) or better:
            rec["value"] = v
            rec["label"] = str(lab)
            rec["dist_um"] = float(dist[i])
    return out


def electrode_distances_um(array, obj):
    """Distance (µm) from every electrode in `array` to a SolarCell or Spheroid,
    returned as a dict label -> distance and a parallel numpy array."""
    d = obj.distance_um(array.grid)
    return {lab: float(d[i]) for i, lab in enumerate(array.labels)}, d
