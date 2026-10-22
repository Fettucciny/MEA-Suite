"""Clickable MCS 8x8 electrode-map widget (PyQt6 / pyqtgraph).

Draws the 60-electrode grid (same layout as the raster viewer). Click modes let
the user place the SOLAR CELL (a square, with its orientation) and the SPHEROID
(a circle) at ANY continuous coordinate -- including off-grid / between electrodes
-- and set the microscope-image overlay centre. A 3 mm field-of-view circle is
drawn centred on the array so cells placed off to the sides stay visible.

An optional microscope image of the MEA surface can be shown behind the grid and
aligned (scale / rotate / move / opacity / flip) so the electrodes overlay the
real sites.

It can also colour electrodes by a per-electrode scalar (heatmap), ring a subset
(active / flood-driven), and write each solar cell's MAX detected voltage on top
of that cell (`set_solars(..., values=...)`) — every electrode belongs to the cell
it is nearest to, so N cells get N independent numbers.

Signals (all emit continuous GRID coords):
  solarPlaced(x, y) / spheroidPlaced(x, y) / imagePlaced(x, y)
  electrodeClicked(label)
"""
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

import geometry as geo


def _fmt_uv(v):
    """Human-readable voltage for the on-cell annotation: µV below 1 mV, mV above.
    Kept identical to heatmap_export.fmt_uv so the screen and the PNG agree."""
    v = float(v)
    if not np.isfinite(v):
        return "n/a"
    a = abs(v)
    if a >= 1000.0:
        return "%.2f mV" % (v / 1000.0)
    if a >= 100.0:
        return "%.0f µV" % v
    return "%.1f µV" % v


def _named_lut(name):
    for getter in ("getFromMatplotlib", "get"):
        try:
            cmap = getattr(pg.colormap, getter)(name)
            if cmap is not None:
                return cmap.getLookupTable(0.0, 1.0, 256)
        except Exception:
            pass
    return None


def _viridis_lut():
    return _named_lut("viridis")


def _magma_lut():
    """Colormap for the solar-cell field overlay — distinct from the viridis
    electrode dots so the background field reads separately."""
    lut = _named_lut("magma")
    return lut if lut is not None else _named_lut("inferno")


class ElectrodeMap(QtWidgets.QWidget):
    solarPlaced = QtCore.pyqtSignal(float, float)
    spheroidPlaced = QtCore.pyqtSignal(float, float)
    imagePlaced = QtCore.pyqtSignal(float, float)
    electrodeClicked = QtCore.pyqtSignal(str)

    def __init__(self, pitch_um=geo.DEFAULT_PITCH_UM, fov=False, parent=None):
        super().__init__(parent)
        self.array = geo.ElectrodeArray(pitch_um=pitch_um)
        self.fov = bool(fov)               # show 3mm field + allow off-grid placement
        self.mode = "inspect"              # inspect | solar | spheroid | image
        self._spheroid = None
        self._lut = _viridis_lut()
        self._heat = None
        self._marks = set()
        self._log = False
        self._img_arr = None               # microscope image (H,W[,C])
        self._img_params = dict(um_per_px=1.0, rotation_deg=0.0, cx_um=0.0,
                                cy_um=0.0, opacity=0.6, flip_y=True)
        self._build()

    # -- construction -------------------------------------------------------
    def _build(self):
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        self.pi = pi = self.plot.getPlotItem()
        pi.setMenuEnabled(False)
        pi.hideButtons()
        pi.hideAxis("left")
        pi.hideAxis("bottom")
        pi.setAspectLocked(True)
        pi.getViewBox().setMouseEnabled(x=False, y=False)
        self._set_range()

        # microscope image (lowest layer)
        self.bg_image = pg.ImageItem()
        self.bg_image.setZValue(-20)
        pi.addItem(self.bg_image)

        # solar-cell electric-field overlay (above the microscope image, below the
        # electrodes) — a modeled/measured field heatmap drawn as the background.
        try:
            self.field_image = pg.ImageItem(axisOrder="row-major")
        except Exception:
            self.field_image = pg.ImageItem()
        self.field_image.setZValue(-15)
        self._field_lut = _magma_lut()
        if self._field_lut is not None:
            self.field_image.setLookupTable(self._field_lut)
        pi.addItem(self.field_image)

        # 3mm field-of-view ring
        self.fov_ring = pg.PlotCurveItem(pen=pg.mkPen(120, 130, 145, 120, width=1,
                                                      style=QtCore.Qt.PenStyle.DashLine))
        self.fov_ring.setZValue(-5)
        pi.addItem(self.fov_ring)
        self._draw_fov()

        self.bg = pg.ScatterPlotItem(size=24, pen=pg.mkPen(70, 70, 70),
                                     brush=pg.mkBrush(45, 48, 55))
        self.dots = pg.ScatterPlotItem(pen=pg.mkPen(0, 0, 0))
        self.marks = pg.ScatterPlotItem(pen=pg.mkPen(255, 210, 60, 235, width=3),
                                        brush=None, size=34)
        pi.addItem(self.bg)
        pi.addItem(self.marks)
        pi.addItem(self.dots)

        # solar cells: all non-selected outlines, the selected outline (brighter),
        # and all electrode-axis lines. Multiple cells are concatenated with NaN
        # breaks (connect='finite').
        self._solars = []
        self._solar_sel = -1
        self._solar_vals = []       # max voltage (µV) detected near each cell
        self._solar_val_labels = [] # which electrode that max came from
        self._solar_plats = []      # sustained plateau (µV) per cell (optional)
        self.solar_texts = []       # pooled TextItems, one per drawn value
        self.solar_curve = pg.PlotCurveItem(pen=pg.mkPen(60, 200, 255, 180, width=2))
        self.solar_sel = pg.PlotCurveItem(pen=pg.mkPen(120, 235, 255, 255, width=3))
        self.solar_axis = pg.PlotCurveItem(pen=pg.mkPen(60, 200, 255, 150, width=1,
                                                        style=QtCore.Qt.PenStyle.DashLine))
        self.spheroid_curve = pg.PlotCurveItem(
            pen=pg.mkPen(255, 120, 200, 235, width=2))
        self.sph_points = pg.ScatterPlotItem(
            size=8, pen=pg.mkPen(255, 120, 200, 255),
            brush=pg.mkBrush(255, 120, 200, 200))
        for it in (self.solar_curve, self.solar_sel, self.solar_axis,
                   self.spheroid_curve, self.sph_points):
            it.setZValue(30)
            pi.addItem(it)

        # electrode-number labels: omitted on the placement (fov) map so they don't
        # obscure the microscope image / real pads.
        self.texts = {}
        if not self.fov:
            mf = QtGui.QFont()
            mf.setPointSize(8)
            mf.setBold(True)
            for lab in self.array.labels:
                xy = geo.mcs_coord(lab)
                # white glyphs on a translucent dark chip -> legible on BOTH the
                # light (yellow) and dark (purple) ends of the viridis scale and on
                # the microscope image.
                t = pg.TextItem(anchor=(0.5, 0.5), color=(255, 255, 255),
                                fill=pg.mkBrush(0, 0, 0, 130))
                try:
                    t.textItem.setFont(mf)
                except Exception:
                    pass
                t.setText(lab)
                t.setPos(xy[0], xy[1])
                t.setZValue(40)
                pi.addItem(t)
                self.texts[lab] = t

        self.plot.scene().sigMouseClicked.connect(self._on_click)
        v.addWidget(self.plot)
        self._refresh_dots()

    def _set_range(self):
        if self.fov:
            r = 0.5 * geo.FOV_DIAMETER_UM / self.array.pitch_um
            cx, cy = geo.ARRAY_CENTER_GRID
            self.pi.getViewBox().setRange(
                xRange=(cx - r - 0.5, cx + r + 0.5),
                yRange=(cy - r - 0.5, cy + r + 0.5), padding=0)
        else:
            self.pi.getViewBox().setRange(xRange=(0.3, 8.7), yRange=(0.3, 8.7),
                                          padding=0)

    def _draw_fov(self):
        if not self.fov:
            self.fov_ring.clear()
            return
        r = 0.5 * geo.FOV_DIAMETER_UM / self.array.pitch_um
        cx, cy = geo.ARRAY_CENTER_GRID
        th = np.linspace(0, 2 * np.pi, 128)
        self.fov_ring.setData(cx + r * np.cos(th), cy + r * np.sin(th))

    # -- public API ---------------------------------------------------------
    def set_mode(self, mode):
        self.mode = mode

    def set_pitch(self, pitch_um):
        self.array = geo.ElectrodeArray(pitch_um=pitch_um)
        for cell in self._solars:
            cell.pitch_um = pitch_um
        if self._spheroid is not None:
            self._spheroid.pitch_um = pitch_um
        self._set_range()
        self._draw_fov()
        self._redraw_solars()
        self._redraw_spheroid()
        self._apply_image_tf()

    def set_solars(self, cells, selected=-1, values=None, value_labels=None,
                   plateaus=None):
        """Draw a list of solar cells; `selected` (index) is highlighted.

        values       : optional parallel list — the MAX voltage (µV) detected on
                       the electrodes nearest each cell. Drawn on top of the cell.
        value_labels : optional parallel list of the electrode label each max came
                       from (shown as a small second line).
        plateaus     : optional parallel list — the SUSTAINED photovoltage (µV) held
                       during illumination near each cell, shown as a small third
                       line ("plat …") beneath the edge-spike max.
        """
        self._solars = list(cells or [])
        self._solar_sel = selected
        self._solar_vals = list(values or [])
        self._solar_val_labels = list(value_labels or [])
        self._solar_plats = list(plateaus or [])
        self._redraw_solars()

    def set_solar(self, solar):
        """Backward-compatible single-cell setter."""
        self.set_solars([solar] if solar is not None else [],
                        0 if solar is not None else -1)

    def set_solar_values(self, values, value_labels=None, plateaus=None):
        """Update just the per-cell max-voltage annotations (cells unchanged)."""
        self._solar_vals = list(values or [])
        self._solar_val_labels = list(value_labels or [])
        if plateaus is not None:
            self._solar_plats = list(plateaus)
        self._redraw_solars()

    def set_spheroid(self, spheroid):
        self._spheroid = spheroid
        self._redraw_spheroid()

    def set_spheroid_points(self, points_grid):
        """Show the hand-clicked perimeter points (list of grid coords)."""
        pts = list(points_grid or [])
        self.sph_points.setData([{"pos": (float(x), float(y))} for x, y in pts])

    def set_heatmap(self, values_by_label, marks=None, log=False):
        self._heat = dict(values_by_label) if values_by_label else None
        self._marks = set(marks or [])
        self._log = bool(log)
        self._refresh_dots()

    def clear_heatmap(self):
        self._heat = None
        self._marks = set()
        self._refresh_dots()

    # -- solar-cell electric-field overlay ----------------------------------
    def set_field_overlay(self, Z, extent_grid, opacity=0.55):
        """Draw a solar-cell field heatmap behind the electrodes.

        Z            : 2-D array (ny, nx) of field magnitude (µV).
        extent_grid  : (x0, x1, y0, y1) in GRID units (electrode coordinate frame).
        """
        if Z is None:
            self.field_image.clear()
            return
        Z = np.asarray(Z, float)
        finite = Z[np.isfinite(Z)]
        if finite.size:
            lo = float(np.nanpercentile(finite, 2))
            hi = float(np.nanpercentile(finite, 98))
            if hi <= lo:
                hi = lo + 1.0
        else:
            lo, hi = 0.0, 1.0
        img = np.where(np.isfinite(Z), Z, lo)
        self.field_image.setImage(img, levels=(lo, hi))
        x0, x1, y0, y1 = extent_grid
        self.field_image.setRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
        self.field_image.setOpacity(float(opacity))

    def clear_field_overlay(self):
        self.field_image.clear()

    # -- microscope image ---------------------------------------------------
    def set_image_array(self, arr):
        """Show a microscope image (numpy H×W×{3,4} or H×W). None clears it."""
        self._img_arr = arr
        if arr is None:
            self.bg_image.clear()
            return
        self.bg_image.setImage(arr, autoLevels=True)
        self._apply_image_tf()

    def set_image_params(self, um_per_px=None, rotation_deg=None, cx_um=None,
                         cy_um=None, opacity=None, flip_y=None):
        for k, val in (("um_per_px", um_per_px), ("rotation_deg", rotation_deg),
                       ("cx_um", cx_um), ("cy_um", cy_um), ("opacity", opacity),
                       ("flip_y", flip_y)):
            if val is not None:
                self._img_params[k] = val
        self._apply_image_tf()

    def _apply_image_tf(self):
        if self._img_arr is None:
            return
        p = self._img_params
        h, w = self._img_arr.shape[:2]
        s = float(p["um_per_px"]) / self.array.pitch_um     # grid units per pixel
        cgx, cgy = geo.grid_from_um(p["cx_um"], p["cy_um"], self.array.pitch_um)
        tr = QtGui.QTransform()
        tr.translate(cgx, cgy)
        tr.rotate(float(p["rotation_deg"]))
        tr.scale(s, -s if p["flip_y"] else s)               # flip Y so it reads upright
        tr.translate(-w / 2.0, -h / 2.0)
        self.bg_image.setTransform(tr)
        self.bg_image.setOpacity(float(p["opacity"]))

    def image_center_um(self):
        return self._img_params["cx_um"], self._img_params["cy_um"]

    # -- drawing ------------------------------------------------------------
    def _color_for(self, frac):
        if self._lut is not None:
            i = int(np.clip(frac, 0, 1) * 255)
            r, g, b = self._lut[i][:3]
            return pg.mkBrush(int(r), int(g), int(b))
        val = int(40 + 200 * np.clip(frac, 0, 1))
        return pg.mkBrush(val, val, 60)

    def _frac_fn(self):
        """Return a value->[0,1] colour-fraction function for the current heatmap,
        honouring the log / linear-95th-percentile scaling mode."""
        vals = np.array([self._heat.get(l, np.nan) for l in self.array.labels],
                        float)
        finite = vals[np.isfinite(vals)]
        if not finite.size:
            return lambda v: 0.0
        if self._log:
            pos = finite[finite > 0]
            if pos.size == 0:
                return lambda v: 0.0
            lo = float(np.log10(pos.min()))
            hi = (float(np.log10(np.percentile(pos, 95))) if pos.size >= 8
                  else float(np.log10(pos.max())))
            if hi <= lo:
                hi = lo + 1.0

            def frac(v):
                if not np.isfinite(v) or v <= 0:
                    return 0.0
                return (float(np.log10(v)) - lo) / (hi - lo)
            return frac
        # linear, robust 95th-percentile upper limit so a couple of hyperactive
        # electrodes can't wash out the footprint.
        vmin = float(finite.min())
        if finite.size >= 8:
            vmax = float(np.percentile(finite, 95))
            if vmax <= vmin:
                vmax = float(finite.max())
        else:
            vmax = float(finite.max())
        rng = (vmax - vmin) or 1.0
        return lambda v: (float(v) - vmin) / rng

    def _refresh_dots(self):
        spots = []
        frac_fn = self._frac_fn() if self._heat else None
        for lab in self.array.labels:
            xy = geo.mcs_coord(lab)
            if self._heat and np.isfinite(self._heat.get(lab, np.nan)):
                frac = frac_fn(self._heat[lab])
                brush, size, pen = self._color_for(frac), 28, pg.mkPen(0, 0, 0)
            elif self._heat:
                brush, size, pen = pg.mkBrush(40, 42, 48), 22, pg.mkPen(0, 0, 0)
            elif self.fov:
                # placement map: 60%-transparent, smaller ring so the real pads /
                # microscope image show through.
                brush, size = pg.mkBrush(235, 238, 245, 100), 12
                pen = pg.mkPen(120, 130, 145, 120)
            else:
                brush, size, pen = pg.mkBrush(220, 222, 228), 24, pg.mkPen(0, 0, 0)
            spots.append({"pos": xy, "data": lab, "brush": brush, "size": size,
                          "pen": pen})
        self.dots.setData(spots)
        self.marks.setData([{"pos": geo.mcs_coord(l), "data": l}
                            for l in self._marks if geo.mcs_coord(l)])

    def _redraw_solars(self):
        xs, ys, sx, sy, axs, ays = [], [], [], [], [], []
        for i, cell in enumerate(self._solars):
            c = cell.corners_grid()
            cx = list(np.append(c[:, 0], c[0, 0]))
            cy = list(np.append(c[:, 1], c[0, 1]))
            cen = cell.center_grid
            half = (cell.side_um / 2.0) / cell.pitch_um
            ang = np.deg2rad(cell.electrode_axis_deg)
            dx, dy = np.cos(ang) * half, np.sin(ang) * half
            if i == self._solar_sel:
                sx += cx + [np.nan]
                sy += cy + [np.nan]
            else:
                xs += cx + [np.nan]
                ys += cy + [np.nan]
            axs += [cen[0] - dx, cen[0] + dx, np.nan]
            ays += [cen[1] - dy, cen[1] + dy, np.nan]
        self.solar_curve.setData(xs, ys, connect="finite")
        self.solar_sel.setData(sx, sy, connect="finite")
        self.solar_axis.setData(axs, ays, connect="finite")
        self._redraw_solar_values()

    def _redraw_solar_values(self):
        """Write each cell's max detected voltage on top of that cell (amber chip,
        this program's accent). Cells with no value keep just their outline."""
        wanted = []
        for i, cell in enumerate(self._solars):
            v = self._solar_vals[i] if i < len(self._solar_vals) else None
            if v is None or not np.isfinite(float(v)):
                continue
            txt = _fmt_uv(v)
            el = (self._solar_val_labels[i]
                  if i < len(self._solar_val_labels) else None)
            if el:
                txt += "\n@ %s" % el
            pl = self._solar_plats[i] if i < len(self._solar_plats) else None
            if pl is not None and np.isfinite(float(pl)):
                txt += "\nplat %s" % _fmt_uv(float(pl))
            c = np.asarray(cell.corners_grid(), float)
            wanted.append((float(np.mean(c[:, 0])), float(np.max(c[:, 1])) + 0.28,
                           txt))
        while len(self.solar_texts) < len(wanted):
            t = pg.TextItem(anchor=(0.5, 1.0), color=(251, 191, 36),
                            fill=pg.mkBrush(0, 0, 0, 185),
                            border=pg.mkPen(60, 200, 255, 200))
            f = QtGui.QFont()
            f.setPointSize(9)
            f.setBold(True)
            try:
                t.textItem.setFont(f)
            except Exception:
                pass
            t.setZValue(45)
            self.pi.addItem(t)
            self.solar_texts.append(t)
        for t, (x, y, txt) in zip(self.solar_texts, wanted):
            t.setText(txt)
            t.setPos(x, y)
            t.setVisible(True)
        for t in self.solar_texts[len(wanted):]:
            t.setVisible(False)

    def _redraw_spheroid(self):
        if self._spheroid is None:
            self.spheroid_curve.clear()
            return
        try:
            c = np.asarray(self._spheroid.outline_grid())
        except Exception:
            self.spheroid_curve.clear()
            return
        if c.size < 4:
            self.spheroid_curve.clear()
            return
        self.spheroid_curve.setData(np.append(c[:, 0], c[0, 0]),
                                    np.append(c[:, 1], c[0, 1]))

    # -- interaction --------------------------------------------------------
    def _nearest_label(self, x, y, max_d2=0.36):
        best, best_d2 = None, max_d2
        for lab in self.array.labels:
            xy = geo.mcs_coord(lab)
            d2 = (xy[0] - x) ** 2 + (xy[1] - y) ** 2
            if d2 < best_d2:
                best, best_d2 = lab, d2
        return best

    def _on_click(self, ev):
        vb = self.pi.getViewBox()
        p = vb.mapSceneToView(ev.scenePos())
        x, y = float(p.x()), float(p.y())
        if self.mode == "solar":
            self.solarPlaced.emit(x, y)
        elif self.mode == "spheroid":
            self.spheroidPlaced.emit(x, y)
        elif self.mode == "image":
            self.imagePlaced.emit(x, y)
        else:
            lab = self._nearest_label(x, y)
            if lab:
                self.electrodeClicked.emit(lab)
