"""Interactive raster viewer (PyQt6 + pyqtgraph) — MEA Explorer.

Two-resolution raster:
  * OVERVIEW (zoomed out): binned spikes/bin rate image, ~1 bin per pixel.
  * DETAIL   (zoomed in) : real spike ticks per electrode.
Auto by visible spike count (with hysteresis), or forced via the View selector.

An electrode map on the right shows the MCS 8x8 grid (square, scales with the
window). Left-click an electrode to cycle it through 3 highlight-color groups
(none -> 1 -> 2 -> 3 -> none); right-click or Ctrl-click shows/hides it. Each
group is drawn as a translucent band on its raster rows, and the legend above
the raster (labels editable under the map) names the groups.

Look comes from the shared MEA-Suite theme (mea_theme, accent "explorer").
"""
import os
import sys
import json
import math

# Use PyQt6 for pyqtgraph too, so the whole app runs on one binding.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")

import numpy as np  # noqa: E402

try:
    from PyQt6 import QtWidgets, QtCore, QtGui      # noqa: E402
    import pyqtgraph as pg                          # noqa: E402
except Exception as exc:  # pragma: no cover
    sys.stderr.write(
        "\nCould not import PyQt6 / pyqtgraph.\n"
        "The shared MEA .venv needs both — run setup.bat (or "
        "`pip install PyQt6 pyqtgraph` into the venv).\n"
        "Original error: {}\n".format(exc)
    )
    raise

import mea_theme                                    # noqa: E402  shared MEA-Suite theme

# PyQt6 scoped-enum / moved-class forms.
QAction = QtGui.QAction
YES_BUTTON = QtWidgets.QMessageBox.StandardButton.Yes
RIGHT_BUTTON = QtCore.Qt.MouseButton.RightButton
CTRL_MOD = QtCore.Qt.KeyboardModifier.ControlModifier
SHIFT_MOD = QtCore.Qt.KeyboardModifier.ShiftModifier

import lod                                          # noqa: E402
import export                                       # noqa: E402
import explorer                                     # noqa: E402
from datasource import (                            # noqa: E402
    Recording, list_recordings, set_spike_dir, get_spike_dir,
    set_data_dir, electrode_xy, grid_labels)

ACCENT = "explorer"                                 # emerald (#34D399)

# Dark pyqtgraph background/foreground + antialias + row-major images, from the
# shared theme. Set at import so every plot widget picks it up.
mea_theme.style_pyqtgraph(ACCENT)

# Three highlight groups: distinct categorical colours, readable over parula and
# on the electrode-map dots (functional data colours, not theme chrome).
HL_COLORS = [(255, 145, 30), (236, 64, 150), (0, 200, 230)]
DEFAULT_GROUP_NAMES = ["Group 1", "Group 2", "Group 3"]


def _toolbar_qss():
    """Supplemental stylesheet for the QToolBar, built from theme tokens so the
    toolbar matches the suite (mea_theme.apply_qt doesn't style QToolBar)."""
    A = mea_theme.accent(ACCENT)
    return (
        "QToolBar {{ background: {panel}; border: 0px; padding: 5px; spacing: 4px; }}"
        "QToolBar QLabel {{ color: {muted}; }}"
        "QToolButton {{ padding: 5px 12px; border-radius: 8px; border: 0px;"
        " color: {text}; }}"
        "QToolButton:hover {{ background: {hi}; }}"
        "QToolButton:pressed {{ background: {acc}; color: #0A0E16; }}"
        "QToolButton:checked {{ background: {acc}; color: #0A0E16; }}"
        "QToolBar::separator {{ background: {line}; width: 1px; margin: 4px 6px; }}"
    ).format(panel=mea_theme.PANEL, muted=mea_theme.TEXT_MUTED,
             text=mea_theme.TEXT, hi=mea_theme.PANEL_HI,
             acc=A, line=mea_theme.LINE)


SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "mea_explorer_settings.json")


def load_settings():
    """UI preferences from the last session ({} if none / unreadable)."""
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_settings(data):
    """Write UI preferences atomically (temp + replace), never fatally."""
    try:
        tmp = SETTINGS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, SETTINGS_FILE)
    except Exception:
        pass


def _install_excepthook():
    """With the windowless launcher there is no console, so surface crashes via
    a dialog and a log file next to the app instead of a hidden stderr."""
    import traceback
    logpath = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "error.log"))

    def hook(exctype, value, tb):
        msg = "".join(traceback.format_exception(exctype, value, tb))
        try:
            with open(logpath, "a", encoding="utf-8") as fh:
                fh.write(msg + "\n")
        except Exception:
            pass
        try:
            QtWidgets.QMessageBox.critical(
                None, "MEA Explorer error",
                msg[-3000:] + "\n\nLogged to:\n" + logpath)
        except Exception:
            pass
        sys.__excepthook__(exctype, value, tb)

    sys.excepthook = hook


class _ReorderList(QtWidgets.QListWidget):
    """QListWidget that emits `dropped` after an internal drag-reorder."""
    dropped = QtCore.pyqtSignal()

    def dropEvent(self, e):
        super().dropEvent(e)
        self.dropped.emit()


class ArrangeDialog(QtWidgets.QDialog):
    """Modeless dialog: drag electrodes to reorder the raster y-axis, apply
    preset orders, and save/load the arrangement as a JSON file. Applies live
    via apply_cb(list_of_labels)."""

    def __init__(self, parent, order, orig_order, xy_of, apply_cb):
        super().__init__(parent)
        self.setWindowTitle("Arrange electrodes")
        self.resize(300, 540)
        self.xy_of = xy_of
        self.orig_order = list(orig_order)
        self.apply_cb = apply_cb
        self._suppress = False

        v = QtWidgets.QVBoxLayout(self)
        cap = QtWidgets.QLabel(
            "Drag to reorder — the top of the list is the top row of the "
            "raster. Order applies live.")
        cap.setWordWrap(True)
        cap.setStyleSheet("color: {};".format(mea_theme.TEXT_MUTED))
        v.addWidget(cap)

        self.list = _ReorderList()
        self.list.setDragDropMode(
            QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.dropped.connect(self._apply)
        self._fill(order)
        v.addWidget(self.list, 1)

        rb = QtWidgets.QHBoxLayout()
        for txt, fn in (("Move up", self._up), ("Move down", self._down)):
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(fn)
            rb.addWidget(b)
        v.addLayout(rb)

        grid = QtWidgets.QGridLayout()
        for i, (txt, fn) in enumerate((
                ("Spike-file order", self._preset_orig),
                ("By label", self._preset_label),
                ("By array (row)", self._preset_array),
                ("Reverse", self._preset_reverse))):
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(fn)
            grid.addWidget(b, i // 2, i % 2)
        v.addLayout(grid)

        io_ = QtWidgets.QHBoxLayout()
        for txt, fn in (("Save…", self._save), ("Load…", self._load)):
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(fn)
            io_.addWidget(b)
        v.addLayout(io_)

        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.close)
        v.addWidget(close)

    def _fill(self, order):
        self._suppress = True
        self.list.clear()
        self.list.addItems([str(l) for l in order])
        self._suppress = False

    def current_order(self):
        return [self.list.item(i).text() for i in range(self.list.count())]

    def set_order(self, order, source=None):
        self._fill(order)
        self._apply(source)

    def _apply(self, source=None):
        if not self._suppress:
            self.apply_cb(self.current_order(), source)

    def _move(self, delta):
        rows = sorted(self.list.row(it) for it in self.list.selectedItems())
        if not rows:
            return
        if delta < 0 and rows[0] == 0:
            return
        if delta > 0 and rows[-1] == self.list.count() - 1:
            return
        if delta > 0:
            rows = list(reversed(rows))
        self._suppress = True
        for r in rows:
            it = self.list.takeItem(r)
            self.list.insertItem(r + delta, it)
            it.setSelected(True)
        self._suppress = False
        self._apply()

    def _up(self):
        self._move(-1)

    def _down(self):
        self._move(1)

    def _preset_orig(self):
        cur = set(self.current_order())
        self.set_order([l for l in self.orig_order if l in cur])

    def _preset_label(self):
        def key(l):
            return (0, int(l)) if l.isdigit() else (1, l)
        self.set_order(sorted(self.current_order(), key=key))

    def _preset_array(self):
        def key(l):
            xy = self.xy_of(l)
            if xy is None:
                return (99, 99)
            return (9 - xy[1], xy[0])              # row (top->bottom), then col
        self.set_order(sorted(self.current_order(), key=key))

    def _preset_reverse(self):
        self.set_order(list(reversed(self.current_order())))

    def _save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save electrode arrangement", "electrode_order.json",
            "JSON (*.json)")
        if not path:
            return
        data = {"format": "mea-explorer-electrode-order", "version": 1,
                "order": self.current_order()}
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(exc))

    def _load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load electrode arrangement", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            order = data.get("order") if isinstance(data, dict) else data
            if not isinstance(order, list):
                raise ValueError("file has no 'order' list")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
            return
        present = set(self.current_order())
        new = [str(l) for l in order if str(l) in present]
        for l in self.current_order():                 # keep any not in the file
            if l not in new:
                new.append(l)
        if not new:
            QtWidgets.QMessageBox.warning(
                self, "No matching electrodes",
                "None of the electrodes in that file are in this recording.")
            return
        self.set_order(new, source=os.path.basename(path))


class TimeAxisItem(pg.AxisItem):
    """Time axis whose unit follows the zoom level.

    In "auto" mode the unit is picked from the visible span, biased toward
    milliseconds wherever the numbers stay readable:

        span >= 10 s     -> s
        10 ms .. 10 s    -> ms      (preferred)
        span <  10 ms    -> us

    The unit can also be pinned via set_unit_mode("s"|"ms"|"us").
    """

    #  mode -> (multiplier applied to seconds, suffix shown)
    UNITS = {"s": (1.0, "s"), "ms": (1e3, "ms"), "us": (1e6, "µs")}

    # class-level defaults: AxisItem.__init__ calls setRange() (and so
    # _sync_label) before instance attributes would otherwise exist.
    unit_mode = "auto"
    use_offset = True
    _unit = None
    _state = None

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.unit_mode = "auto"
        self.use_offset = True
        self._unit = None
        self._state = None
        self.enableAutoSIPrefix(False)      # we do our own unit handling
        self._sync_label()

    # -- unit choice --------------------------------------------------------
    def set_unit_mode(self, mode):
        """mode: 'auto' | 's' | 'ms' | 'us'."""
        self.unit_mode = mode if mode in ("auto",) + tuple(self.UNITS) else "auto"
        self._unit = None                   # force a relabel
        self._state = None
        self._sync_label()
        self.picture = None
        self.update()

    def current_unit(self):
        if self.unit_mode != "auto":
            return self.unit_mode
        try:
            lo, hi = self.range
            span = abs(float(hi) - float(lo))
        except Exception:
            span = 1.0
        if span <= 0:
            return "ms"
        if span < 0.01:                     # under 10 ms
            return "us"
        if span < 10.0:                     # under 10 s -> ms (preferred)
            return "ms"
        return "s"

    def current_offset(self):
        """Zero point subtracted from the ticks so deep zooms stay readable.

        Zooming to a 20 ms window at t = 10 s would otherwise print ticks like
        '10000, 10010, 10020' ms. When the absolute time dwarfs the visible
        span we subtract a round offset and state it in the axis label, the
        way matplotlib does. Returns 0.0 when the raw numbers are fine.
        """
        if not self.use_offset:
            return 0.0
        try:
            lo, hi = self.range
            lo = float(lo)
            span = abs(float(hi) - lo)
        except Exception:
            return 0.0
        if span <= 0 or abs(lo) < span * 20:
            return 0.0
        mag = 10.0 ** math.floor(math.log10(span))
        return math.floor(lo / mag) * mag

    @staticmethod
    def _fmt_offset(off):
        """Offset shown in the label, in whichever unit reads cleanly."""
        for factor, suffix in ((1.0, "s"), (1e3, "ms"), (1e6, "µs")):
            v = off * factor
            if abs(v) >= 1:
                txt = ("{:.3f}".format(v)).rstrip("0").rstrip(".")
                return "{} {}".format(txt, suffix)
        return "{:g} s".format(off)

    def _sync_label(self):
        u = self.current_unit()
        off = self.current_offset()
        if (u, off) != self._state:
            self._state = (u, off)
            self._unit = u
            if off:
                text = "Time ({}, +{})".format(self.UNITS[u][1],
                                               self._fmt_offset(off))
            else:
                text = "Time ({})".format(self.UNITS[u][1])
            self.setLabel(text=text)

    # -- pyqtgraph hooks ----------------------------------------------------
    def setRange(self, mn, mx):
        super().setRange(mn, mx)
        self._sync_label()                  # zoom changed -> maybe a new unit

    def tickStrings(self, values, scale, spacing):
        factor, _ = self.UNITS[self.current_unit()]
        off = self.current_offset()
        step = abs(spacing) * factor
        if step >= 100:
            dec = 0
        elif step >= 10:
            dec = 0
        elif step >= 1:
            dec = 0
        elif step >= 0.1:
            dec = 1
        elif step >= 0.01:
            dec = 2
        else:
            dec = 3
        fmt = "{:." + str(dec) + "f}"
        return [fmt.format((float(v) - off) * factor) for v in values]


class ExportImageDialog(QtWidgets.QDialog):
    """Fonts, electrode-label visibility and output size for an image export."""

    def __init__(self, parent, opts, natural):
        super().__init__(parent)
        self.setWindowTitle("Export image")
        self._nat_w, self._nat_h = natural
        form = QtWidgets.QFormLayout(self)

        cap = QtWidgets.QLabel(
            "These apply to the exported image only — the on-screen view is "
            "left unchanged.")
        cap.setWordWrap(True)
        cap.setStyleSheet("color: {};".format(mea_theme.TEXT_MUTED))
        form.addRow(cap)

        self.sp_label = QtWidgets.QSpinBox()
        self.sp_label.setRange(4, 72)
        self.sp_label.setSuffix(" pt")
        self.sp_label.setValue(int(opts.get("label_pt", 11)))
        form.addRow("Axis label size", self.sp_label)

        self.sp_tick = QtWidgets.QSpinBox()
        self.sp_tick.setRange(4, 72)
        self.sp_tick.setSuffix(" pt")
        self.sp_tick.setValue(int(opts.get("tick_pt", 9)))
        form.addRow("Tick mark size", self.sp_tick)

        self.cb_hide_y = QtWidgets.QCheckBox("Hide electrode tick labels")
        self.cb_hide_y.setToolTip(
            "Leave the electrode numbers off the y-axis in the exported "
            "image (the 'Electrode' axis label is kept).")
        self.cb_hide_y.setChecked(bool(opts.get("hide_y_ticks", False)))
        form.addRow("", self.cb_hide_y)

        self.sp_width = QtWidgets.QSpinBox()
        self.sp_width.setRange(100, 20000)
        self.sp_width.setSingleStep(100)
        self.sp_width.setSuffix(" px")
        self.sp_width.setValue(int(opts.get("width_px")
                                   or round(self._nat_w)))
        self.sp_width.setToolTip(
            "Output width. Everything scales together, so the layout is "
            "unchanged and only the resolution goes up.")
        form.addRow("Output width", self.sp_width)

        self.sp_dpi = QtWidgets.QSpinBox()
        self.sp_dpi.setRange(36, 1200)
        self.sp_dpi.setSingleStep(10)
        self.sp_dpi.setSuffix(" dpi")
        self.sp_dpi.setValue(int(opts.get("dpi", 300)))
        self.sp_dpi.setToolTip(
            "Written into the file so Word / Illustrator place it at the "
            "right physical size. Journals usually ask for 300–600.")
        form.addRow("Resolution", self.sp_dpi)

        self.hint = QtWidgets.QLabel()
        self.hint.setStyleSheet("color: {};".format(mea_theme.TEXT_MUTED))
        form.addRow("", self.hint)

        row = QtWidgets.QHBoxLayout()
        for txt, w in (("Screen size", None), ("1×", 1), ("2×", 2), ("4×", 4)):
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(
                lambda _=False, m=w: self.sp_width.setValue(
                    int(round(self._nat_w * (m or 1)))))
            row.addWidget(b)
        form.addRow("Preset", row)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btns.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Export…")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

        self.sp_width.valueChanged.connect(self._update_hint)
        self.sp_dpi.valueChanged.connect(self._update_hint)
        self._update_hint()

    def _update_hint(self, *a):
        w = self.sp_width.value()
        h = int(round(w * self._nat_h / max(1.0, self._nat_w)))
        dpi = max(1, self.sp_dpi.value())
        self.hint.setText(
            "{} × {} px  ·  {:.2f} × {:.2f} in  ({:.0f} × {:.0f} mm) "
            "at {} dpi".format(w, h, w / dpi, h / dpi,
                               w / dpi * 25.4, h / dpi * 25.4, dpi))

    def values(self):
        return {
            "label_pt": self.sp_label.value(),
            "tick_pt": self.sp_tick.value(),
            "hide_y_ticks": self.cb_hide_y.isChecked(),
            "width_px": self.sp_width.value(),
            "dpi": self.sp_dpi.value(),
        }


class _ScaleSlider(QtWidgets.QSlider):
    """Slider that reports double-clicks (used to reset the heatmap scale)."""
    doubleClicked = QtCore.pyqtSignal()

    def mouseDoubleClickEvent(self, ev):
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(ev)


class _MapPanel(QtWidgets.QWidget):
    """Panel whose designated child (the electrode map) is kept square: its
    height is tied to its current width, so it scales with the window."""
    square_widget = None

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        w = self.square_widget
        if w is not None:
            ww = w.width()
            if ww > 0 and w.height() != ww:
                w.setFixedHeight(ww)


class RasterWindow(QtWidgets.QMainWindow):
    def __init__(self, rec, stem, apply_qc=True):
        super().__init__()
        self.rec = rec
        self.stem = stem
        self.apply_qc = apply_qc
        self._mode = None
        self.view_mode = "auto"                   # auto | heatmap | ticks
        self._trigger_lines = []
        self._hl_bands = []
        self.vis = {}                             # label -> bool (shown)
        self.highlight = {}                       # label -> group 1..3
        self.order = []                           # display order (labels, top->bottom)
        self._lab2ch = {}                         # label -> channel index
        self._arrange_dlg = None
        self._auto_vmax = None                    # last auto colour ceiling
        self._loading = True                      # suppress saves while building

        # -- preferences remembered between sessions ------------------------
        s = load_settings()
        self.view_mode = str(s.get("view_mode", "auto")).lower()
        if self.view_mode not in ("auto", "heatmap", "ticks"):
            self.view_mode = "auto"
        self._time_unit = str(s.get("time_unit", "auto")).lower()
        if self._time_unit not in ("auto", "s", "ms", "us"):
            self._time_unit = "auto"
        self.show_triggers = bool(s.get("show_triggers", True))
        try:
            self.heat_gain = min(4.0, max(0.1, float(s.get("heat_gain", 1.0))))
        except Exception:
            self.heat_gain = 1.0
        self.export_label_pt = int(s.get("export_label_pt", 11) or 11)
        self.export_tick_pt = int(s.get("export_tick_pt", 9) or 9)
        self.export_hide_y_ticks = bool(s.get("export_hide_y_ticks", False))
        try:
            wpx = s.get("export_width_px")
            self.export_width_px = int(wpx) if wpx else None
        except Exception:
            self.export_width_px = None
        self.export_dpi = int(s.get("export_dpi", 300) or 300)
        saved_order = s.get("electrode_order")
        self._saved_order = ([str(l) for l in saved_order]
                             if isinstance(saved_order, list) else [])
        self._order_source = s.get("electrode_order_source") or None
        ex = s.get("explorer")
        self.explorer_opts = dict(ex) if isinstance(ex, dict) else {}
        self._explorer = None                 # single-electrode window
        self._current_electrode = None        # last double-clicked row
        self.setWindowTitle("MEA Explorer  —  {}".format(stem))
        self.resize(1500, 860)

        self._build_ui()
        self._sync_scale_enabled()
        self._install_recording(rec)
        self._loading = False

    # -- UI construction ----------------------------------------------------
    def _build_ui(self):
        self._make_toolbar()

        self.glw = pg.GraphicsLayoutWidget()
        self.time_axis = TimeAxisItem(orientation="bottom")
        self.raster = self.glw.addPlot(row=0, col=0,
                                       axisItems={"bottom": self.time_axis})
        self.raster.setLabel("left", "Electrode")
        self.raster.getViewBox().invertY(True)   # row 0 at the top
        self.raster.setMouseEnabled(x=True, y=True)

        self.image = pg.ImageItem()
        self.image.setLookupTable(mea_theme.parula_lut())   # suite colormap
        self.raster.addItem(self.image)

        self.scatter = pg.ScatterPlotItem(
            pen=None, brush=pg.mkBrush(mea_theme.TEXT),
            size=6, symbol="|", pxMode=True,
        )
        self.raster.addItem(self.scatter)

        selcol = QtGui.QColor(mea_theme.accent(ACCENT))
        selcol.setAlpha(45)
        self.region = pg.LinearRegionItem(brush=pg.mkBrush(selcol))
        self.region.setZValue(10)
        self.region.hide()

        self.time_axis.set_unit_mode(self._time_unit)     # restored preference
        self.wave_time_axis = TimeAxisItem(orientation="bottom")
        self.wave = self.glw.addPlot(row=1, col=0,
                                     axisItems={"bottom": self.wave_time_axis})
        self.wave.setLabel("left", "Voltage", units="uV")
        self.wave_time_axis.set_unit_mode(self._time_unit)
        self.wave.setXLink(self.raster)
        self.wave.setMaximumHeight(200)
        self.wave.hide()
        self.wave_curve = pg.PlotCurveItem(pen=pg.mkPen("#F2C879", width=1))
        self.wave.addItem(self.wave_curve)

        # left = legend on top of the raster/waveform graphics
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(2)
        lv.addWidget(self._build_legend())
        lv.addWidget(self.glw, 1)

        split = QtWidgets.QSplitter()
        split.addWidget(left)
        split.addWidget(self._build_map_panel())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([1100, 340])
        self.setCentralWidget(split)

        self._redraw_timer = QtCore.QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(40)
        self._redraw_timer.timeout.connect(self._redraw)
        self.raster.sigXRangeChanged.connect(self._schedule_redraw)

        self.raster.scene().sigMouseClicked.connect(self._on_click)
        self.status = self.statusBar()

    def _make_toolbar(self):
        tb = self.addToolBar("main")
        tb.setMovable(False)

        def act(text, slot):
            a = QAction(text, self)
            a.triggered.connect(slot)
            tb.addAction(a)
            return a

        act("Open folder…", self.open_folder)
        act("Fit all", self.fit_all)
        act("Arrange…", self.open_arrange)
        a = act("Explore electrode…", lambda: self.open_explorer())
        a.setToolTip("Filtered trace, spikes and bursts of one electrode.\n"
                     "Opens the last double-clicked electrode; Shift-click "
                     "the map to pick one.")
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(" View: "))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Auto", "Heatmap", "Ticks"])
        self.mode_combo.setToolTip(
            "Auto: heatmap when dense, ticks when sparse.\n"
            "Heatmap: always the firing-rate image.\n"
            "Ticks: always individual spikes.")
        # restore before connecting, so loading a preference is not a "change"
        self.mode_combo.setCurrentText(self.view_mode.capitalize())
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        tb.addWidget(self.mode_combo)
        tb.addWidget(QtWidgets.QLabel("  Time: "))
        self.time_combo = QtWidgets.QComboBox()
        self.time_combo.addItems(["Auto", "s", "ms", "µs"])
        self.time_combo.setToolTip(
            "Time-axis unit.\n"
            "Auto: follows the zoom level (ms wherever it reads well).\n"
            "s / ms / µs: pin the unit.")
        self.time_combo.setCurrentText(
            {"auto": "Auto", "s": "s", "ms": "ms", "us": "µs"}[self._time_unit])
        self.time_combo.currentTextChanged.connect(self._on_time_unit_changed)
        tb.addWidget(self.time_combo)
        tb.addSeparator()
        self.act_select = act("Selection", self.toggle_selection)
        self.act_select.setCheckable(True)
        self.act_trig = QAction("Triggers", self)
        self.act_trig.setCheckable(True)
        self.act_trig.setChecked(self.show_triggers)
        self.act_trig.setToolTip("Show / hide the laser (digital-trigger) time lines")
        self.act_trig.triggered.connect(self._toggle_triggers)
        tb.addAction(self.act_trig)
        act("Export image…", self.export_image_dialog)
        act("Export spikes…", self.export_spikes_dialog)
        tb.addSeparator()
        self.act_qc = QAction("QC (drop auto-ground)", self)
        self.act_qc.setCheckable(True)
        self.act_qc.setChecked(self.apply_qc)
        self.act_qc.triggered.connect(self.toggle_qc)
        tb.addAction(self.act_qc)
        act("Clear waveform", self.clear_waveform)

    def _swatch(self, color, size=14):
        sw = QtWidgets.QLabel()
        sw.setFixedSize(size, size)
        sw.setStyleSheet(
            "background: rgb({},{},{}); border: 1px solid {}; "
            "border-radius: 3px;".format(color[0], color[1], color[2],
                                         mea_theme.PLOT_BG))
        return sw

    def _build_legend(self):
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(10, 3, 10, 3)
        h.setSpacing(6)
        cap = QtWidgets.QLabel("Highlight groups:")
        cap.setStyleSheet("color: {};".format(mea_theme.TEXT_MUTED))
        h.addWidget(cap)
        self.legend_labels = []
        for i, color in enumerate(HL_COLORS):
            h.addSpacing(8)
            h.addWidget(self._swatch(color))
            lbl = QtWidgets.QLabel(DEFAULT_GROUP_NAMES[i])
            self.legend_labels.append(lbl)
            h.addWidget(lbl)
        h.addStretch(1)
        w.setMaximumHeight(30)
        return w

    def _build_map_panel(self):
        panel = _MapPanel()
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(6, 6, 6, 6)

        title = QtWidgets.QLabel("Electrode map")
        title.setStyleSheet(
            "font-weight: 600; font-size: 11pt; color: {};".format(mea_theme.TEXT))
        v.addWidget(title)
        hint = QtWidgets.QLabel(
            "Left-click: cycle highlight color\n"
            "Right-click / Ctrl-click: show / hide\n"
            "Shift-click: explore electrode")
        hint.setStyleSheet(
            "color: {}; font-size: 8pt;".format(mea_theme.TEXT_MUTED))
        v.addWidget(hint)

        self.map = pg.PlotWidget()
        pi = self.map.getPlotItem()
        pi.setMenuEnabled(False)
        pi.hideButtons()
        pi.hideAxis("left")
        pi.hideAxis("bottom")
        pi.setAspectLocked(True)
        pi.getViewBox().setMouseEnabled(x=False, y=False)
        pi.getViewBox().setRange(xRange=(0.3, 8.7), yRange=(0.3, 8.7), padding=0)
        self.map_bg = pg.ScatterPlotItem(
            size=15, pen=pg.mkPen(mea_theme.LINE), brush=pg.mkBrush(mea_theme.PANEL))
        self.map_scatter = pg.ScatterPlotItem(pen=pg.mkPen(mea_theme.PLOT_BG))
        pi.addItem(self.map_bg)
        pi.addItem(self.map_scatter)

        self.map_texts = {}
        mf = QtGui.QFont()
        mf.setPointSize(7)
        for lab in grid_labels():
            xy = electrode_xy(lab)
            if xy is None:
                continue
            t = pg.TextItem(anchor=(0.5, 0.5))
            try:
                t.textItem.setFont(mf)
            except Exception:
                pass
            t.setText(lab)
            t.setPos(xy[0], xy[1])
            t.setZValue(20)
            pi.addItem(t)
            self.map_texts[lab] = t

        self.map.scene().sigMouseClicked.connect(self._on_map_click)
        v.addWidget(self.map)
        panel.square_widget = self.map            # keep the map square

        # -- heatmap colour-scale slider -----------------------------------
        v.addSpacing(6)
        srow = QtWidgets.QHBoxLayout()
        scap = QtWidgets.QLabel("Heatmap scale")
        scap.setStyleSheet("color: {};".format(mea_theme.TEXT_MUTED))
        srow.addWidget(scap)
        srow.addStretch(1)
        self.scale_value = QtWidgets.QLabel()
        self.scale_value.setStyleSheet("color: {};".format(mea_theme.TEXT))
        srow.addWidget(self.scale_value)
        v.addLayout(srow)

        self.scale_slider = _ScaleSlider(QtCore.Qt.Orientation.Horizontal)
        self.scale_slider.setRange(10, 400)       # percent of the auto ceiling
        self.scale_slider.setSingleStep(5)
        self.scale_slider.setPageStep(25)
        self.scale_slider.setTickInterval(50)
        self.scale_slider.setTickPosition(
            QtWidgets.QSlider.TickPosition.TicksBelow)
        self.scale_slider.setToolTip(
            "Colour ceiling of the heatmap, as a % of the automatic one.\n"
            "Lower = brighter (weak activity shows up);\n"
            "higher = dimmer (only the busiest bins light up).\n"
            "Double-click to reset to 100%.")
        self.scale_slider.setValue(int(round(self.heat_gain * 100)))
        self._update_scale_label()
        self.scale_slider.valueChanged.connect(self._on_heat_scale)
        self.scale_slider.sliderReleased.connect(self._save_ui_settings)
        self.scale_slider.doubleClicked.connect(self._reset_heat_scale)
        v.addWidget(self.scale_slider)
        v.addSpacing(4)

        grid = QtWidgets.QGridLayout()
        for i, (text, slot) in enumerate((
                ("Show all", self.electrodes_show_all),
                ("Hide all", self.electrodes_hide_all),
                ("Clear highlight", self.highlight_clear),
                ("Show only highlighted", self.electrodes_only_highlighted))):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            grid.addWidget(b, i // 2, i % 2)
        v.addLayout(grid)

        v.addSpacing(4)
        lcap = QtWidgets.QLabel("Legend labels:")
        lcap.setStyleSheet("color: {};".format(mea_theme.TEXT_MUTED))
        v.addWidget(lcap)
        self.group_edits = []
        for i, color in enumerate(HL_COLORS):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(self._swatch(color))
            le = QtWidgets.QLineEdit(DEFAULT_GROUP_NAMES[i])
            le.textChanged.connect(
                lambda text, idx=i: self._on_group_label_changed(idx, text))
            row.addWidget(le)
            v.addLayout(row)
            self.group_edits.append(le)

        v.addStretch(1)
        panel.setMinimumWidth(240)
        return panel

    def _on_group_label_changed(self, idx, text):
        self.legend_labels[idx].setText(text.strip() or DEFAULT_GROUP_NAMES[idx])

    # -- recording lifecycle -----------------------------------------------
    def _install_recording(self, rec):
        self._close_explorer()                     # it holds the old recording
        self.rec = rec
        self._mode = None
        self.vis = {lab: True for lab in rec.labels}
        self.highlight = {}
        self._lab2ch = {lab: i for i, lab in enumerate(rec.labels)}
        self.order = list(rec.labels)
        # re-apply the remembered arrangement to this recording
        restored = 0
        if self._saved_order:
            self.order = self._reconcile_order(self._saved_order)
            restored = sum(1 for lab in self._saved_order
                           if lab in self._lab2ch)
        if self._arrange_dlg is not None:
            self._arrange_dlg.close()
            self._arrange_dlg = None
        self._clear_highlight_bands()
        self._setup_map()
        self.raster.setLimits(xMin=0, xMax=rec.duration,
                              yMin=0, yMax=max(1, rec.n_channels))
        self.fit_all()
        on = rec.trigger_on_times()
        if restored:
            note = "  Electrode order restored{} ({}/{} matched).".format(
                " from " + self._order_source if self._order_source else "",
                restored, rec.n_channels)
        else:
            note = "  Double-click a channel to load its waveform."
        self.status.showMessage(
            "{} — {} channels, {:.1f}s, {} triggers.{}".format(
                self.stem, rec.n_channels, rec.duration, on.size, note))

    def vis_idx(self):
        """Channel indices in the current display order, filtered to shown."""
        return [self._lab2ch[lab] for lab in self.order
                if self.vis.get(lab, True) and lab in self._lab2ch]

    # -- electrode map ------------------------------------------------------
    def _setup_map(self):
        recset = set(self.rec.labels)
        bg = []
        for lab in grid_labels():
            xy = electrode_xy(lab)
            if xy is not None and lab not in recset:
                bg.append({"pos": xy, "data": lab})
        self.map_bg.setData(bg)
        self._refresh_map()

    def _refresh_map(self):
        spots = []
        for lab in self.rec.labels:
            xy = electrode_xy(lab)
            if xy is None:
                continue
            g = self.highlight.get(lab, 0)
            if g:
                brush, size = pg.mkBrush(*HL_COLORS[g - 1]), 22
            elif self.vis.get(lab, True):
                brush, size = pg.mkBrush(mea_theme.TEXT), 17
            else:
                brush, size = pg.mkBrush(mea_theme.LINE), 14
            spots.append({"pos": xy, "data": lab, "brush": brush,
                          "size": size, "pen": pg.mkPen(mea_theme.PLOT_BG)})
        self.map_scatter.setData(spots)

        recset = set(self.rec.labels)
        for lab, t in self.map_texts.items():
            if self.highlight.get(lab, 0):
                t.setColor(pg.mkColor("#0A0E16"))          # on colored dot
            elif lab in recset and self.vis.get(lab, True):
                t.setColor(pg.mkColor(mea_theme.PLOT_BG))  # on light dot
            elif lab in recset:
                t.setColor(pg.mkColor(mea_theme.TEXT_MUTED))  # on hidden dot
            else:
                t.setColor(pg.mkColor(mea_theme.TEXT_DIM))    # background pos

    def _nearest_label(self, x, y, max_d2=0.36):
        best, best_d2 = None, max_d2
        for lab in self.rec.labels:
            xy = electrode_xy(lab)
            if xy is None:
                continue
            d2 = (xy[0] - x) ** 2 + (xy[1] - y) ** 2
            if d2 < best_d2:
                best, best_d2 = lab, d2
        return best

    def _on_map_click(self, ev):
        vb = self.map.getPlotItem().getViewBox()
        p = vb.mapSceneToView(ev.scenePos())
        lab = self._nearest_label(p.x(), p.y())
        if lab is None:
            return
        mods = ev.modifiers() if hasattr(ev, "modifiers") else 0
        if bool(mods & SHIFT_MOD) and ev.button() != RIGHT_BUTTON:
            self.open_explorer(lab)
            return
        toggle_visibility = (ev.button() == RIGHT_BUTTON) or bool(mods & CTRL_MOD)
        if toggle_visibility:
            self.vis[lab] = not self.vis.get(lab, True)
        else:
            nxt = (self.highlight.get(lab, 0) + 1) % 4    # none->1->2->3->none
            if nxt == 0:
                self.highlight.pop(lab, None)
            else:
                self.highlight[lab] = nxt
        self._refresh_map()
        self._redraw()

    def electrodes_show_all(self):
        for lab in self.rec.labels:
            self.vis[lab] = True
        self._refresh_map()
        self._redraw()

    def electrodes_hide_all(self):
        for lab in self.rec.labels:
            self.vis[lab] = False
        self._refresh_map()
        self._redraw()

    def electrodes_only_highlighted(self):
        if not self.highlight:
            return
        hl = set(self.highlight)
        for lab in self.rec.labels:
            self.vis[lab] = lab in hl
        self._refresh_map()
        self._redraw()

    def highlight_clear(self):
        self.highlight.clear()
        self._refresh_map()
        self._redraw()

    def _clear_highlight_bands(self):
        for it in self._hl_bands:
            self.raster.removeItem(it)
        self._hl_bands = []

    def _refresh_highlight_bands(self, idx):
        self._clear_highlight_bands()
        if not self.highlight:
            return
        labels = self.rec.labels
        for row, ch in enumerate(idx):
            g = self.highlight.get(labels[ch], 0)
            if g:
                c = HL_COLORS[g - 1]
                band = pg.LinearRegionItem(
                    values=(row, row + 1), orientation="horizontal",
                    brush=pg.mkBrush(c[0], c[1], c[2], 60), movable=False)
                band.setZValue(3)
                for ln in band.lines:
                    ln.setPen(pg.mkPen(c[0], c[1], c[2], 130))
                self.raster.addItem(band)
                self._hl_bands.append(band)

    # -- redraw / level-of-detail ------------------------------------------
    def _schedule_redraw(self):
        self._redraw_timer.start()

    def _current_xrange(self):
        (x0, x1), _ = self.raster.viewRange()
        x0 = max(0.0, float(x0))
        x1 = min(float(self.rec.duration), float(x1))
        if x1 <= x0:
            x1 = x0 + 1e-3
        return x0, x1

    def _redraw(self):
        rec = self.rec
        t0, t1 = self._current_xrange()
        idx = self.vis_idx()
        n_vis = len(idx)

        ax = self.raster.getAxis("left")
        self.raster.setLimits(yMin=0, yMax=max(1, n_vis))
        if n_vis == 0:
            self.image.hide()
            self.scatter.hide()
            self._clear_highlight_bands()
            ax.setTicks([[], []])
            self.status.showMessage(
                "All electrodes hidden — enable some on the electrode map.")
            return
        self.raster.setYRange(0, n_vis, padding=0)
        ax.setTicks([[(r + 0.5, self.rec.labels[ch])
                      for r, ch in enumerate(idx)], []])

        vb = self.raster.getViewBox()
        width_px = max(1, int(vb.geometry().width()))
        vc = rec.visible_count(t0, t1, idx)
        if self.view_mode == "heatmap":
            mode = "overview"
        elif self.view_mode == "ticks":
            mode = "detail"
        else:
            mode = lod.choose_mode(self._mode, vc)

        decimated = False
        if mode == "overview":
            nbins = lod.bin_count(width_px)
            M = rec.window_counts(t0, t1, nbins, idx)     # (n_vis, nbins)
            self.image.setImage(M, autoLevels=False)
            auto_vmax = max(float(np.percentile(M, 99.0)), 1.0) if M.size else 1.0
            self._auto_vmax = auto_vmax                    # for the scale slider
            vmax = max(auto_vmax * self.heat_gain, 1e-6)   # slider-scaled
            self.image.setLevels((0.0, vmax))
            self.image.setRect(QtCore.QRectF(t0, 0.0, t1 - t0, float(n_vis)))
            self.image.show()
            self.scatter.hide()
        else:
            windows = rec.window_spikes(t0, t1, idx)
            n_rows = len(windows)
            # Split the tick budget evenly across visible rows; any row over its
            # share is uniformly subsampled so every electrode keeps a
            # representative row at any zoom, with the total still bounded.
            per_row_cap = max(1, lod.MAX_TICKS // max(1, n_rows))
            xs, ys = [], []
            for row, tw in enumerate(windows):
                if tw.size == 0:
                    continue
                if tw.size > per_row_cap:
                    step = int(np.ceil(tw.size / per_row_cap))
                    tw = tw[::step]
                    decimated = True
                xs.append(tw)
                ys.append(np.full(tw.size, row + 0.5, dtype=np.float64))
            x = np.concatenate(xs) if xs else np.empty(0)
            y = np.concatenate(ys) if ys else np.empty(0)
            self.scatter.setData(x=x, y=y)
            self.scatter.show()
            self.image.hide()

        self._mode = mode
        self._draw_triggers(t0, t1)
        self._refresh_highlight_bands(idx)
        note = ("  |  ticks decimated to ~{} total".format(lod.MAX_TICKS)
                if decimated else "")
        forced = "" if self.view_mode == "auto" else " (forced)"
        hid = self.rec.n_channels - n_vis
        hidmsg = "  |  {} hidden".format(hid) if hid else ""
        self.status.showMessage(
            "{}  |  {:.3f}–{:.3f}s  |  {} spikes visible  |  {} mode{}{}{}".format(
                self.stem, t0, t1, vc, mode, forced, note, hidmsg))

    def _draw_triggers(self, t0, t1):
        for ln in self._trigger_lines:
            self.raster.removeItem(ln)
        self._trigger_lines = []
        if not self.show_triggers:
            return
        on = self.rec.trigger_on_times()
        if on.size == 0:
            return
        vis = on[(on >= t0) & (on <= t1)]
        if vis.size > 800:
            return
        pen = pg.mkPen("#FBBF24", width=1)               # amber trigger marks
        for t in vis:
            ln = pg.InfiniteLine(pos=float(t), angle=90, pen=pen, movable=False)
            ln.setZValue(5)
            self.raster.addItem(ln)
            self._trigger_lines.append(ln)

    # -- actions ------------------------------------------------------------
    def _on_mode_changed(self, text):
        self.view_mode = text.lower()             # auto | heatmap | ticks
        self._sync_scale_enabled()
        self._redraw()
        self._save_ui_settings()

    def _sync_scale_enabled(self):
        """The colour scale only means anything when a heatmap can appear."""
        on = self.view_mode != "ticks"
        for w in (self.scale_slider, self.scale_value):
            w.setEnabled(on)
        self.scale_slider.setToolTip(
            "Colour ceiling of the heatmap, as a % of the automatic one.\n"
            "Lower = brighter (weak activity shows up);\n"
            "higher = dimmer (only the busiest bins light up).\n"
            "Double-click to reset to 100%."
            if on else "Only applies to the heatmap view.")

    def _on_time_unit_changed(self, text):
        mode = {"Auto": "auto", "s": "s", "ms": "ms", "µs": "us"}.get(text, "auto")
        self._time_unit = mode
        for ax in (self.time_axis, self.wave_time_axis):
            ax.set_unit_mode(mode)
        self._save_ui_settings()

    def _update_scale_label(self):
        pct = int(round(self.heat_gain * 100))
        self.scale_value.setText(
            "{}%".format(pct) + ("  (auto)" if pct == 100 else ""))

    def _on_heat_scale(self, value):
        self.heat_gain = max(0.1, value / 100.0)
        self._update_scale_label()
        # Dragging the slider only moves the colour ceiling — re-binning the
        # spikes for every step would recompute the whole histogram hundreds
        # of times, so reuse the cached auto ceiling when the image is up.
        if self._mode == "overview" and self._auto_vmax:
            self.image.setLevels(
                (0.0, max(self._auto_vmax * self.heat_gain, 1e-6)))
        else:
            self._redraw()

    def _reset_heat_scale(self):
        self.scale_slider.setValue(100)
        self._save_ui_settings()

    def _save_ui_settings(self):
        """Persist the toolbar/scale preferences for the next launch."""
        if getattr(self, "_loading", False):
            return
        data = load_settings()
        data.update({
            "view_mode": self.view_mode,
            "time_unit": self._time_unit,
            "show_triggers": bool(self.show_triggers),
            "heat_gain": round(float(self.heat_gain), 3),
            "export_label_pt": int(self.export_label_pt),
            "export_tick_pt": int(self.export_tick_pt),
            "export_hide_y_ticks": bool(self.export_hide_y_ticks),
            "export_width_px": (int(self.export_width_px)
                                if self.export_width_px else None),
            "export_dpi": int(self.export_dpi),
            "electrode_order": list(self._saved_order or []),
            "electrode_order_source": self._order_source,
            "explorer": dict(self.explorer_opts or {}),
        })
        try:
            data["spike_dir"] = get_spike_dir()
        except Exception:
            pass
        save_settings(data)

    def closeEvent(self, ev):
        self._save_ui_settings()
        self._close_explorer()
        super().closeEvent(ev)

    # -- single-electrode explorer -----------------------------------------
    def open_explorer(self, label=None):
        """Open (or retarget) the electrode explorer window."""
        if label is None:
            label = self._current_electrode
        if label is None or not self.rec.has_label(label):
            idx = self.vis_idx()
            label = (self.rec.labels[idx[0]] if idx
                     else (self.rec.labels[0] if self.rec.labels else None))
        if label is None:
            return
        if self._explorer is not None:
            self._explorer.set_electrode(label)
            self._explorer.showNormal()
            self._explorer.raise_()
            self._explorer.activateWindow()
            return
        w = explorer.ElectrodeExplorer(
            self, self.rec, label, list(self.order), TimeAxisItem,
            time_unit=self._time_unit, opts=self.explorer_opts,
            show_triggers=self.show_triggers,
            on_raw_located=self._on_raw_located,
            on_opts_changed=self._on_explorer_opts)
        w.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        w.destroyed.connect(self._on_explorer_closed)
        _set_window_icon(w)
        self._explorer = w
        w.show()

    def _on_explorer_closed(self, *a):
        self._explorer = None

    def _close_explorer(self):
        w, self._explorer = getattr(self, "_explorer", None), None
        if w is not None:
            try:
                w.destroyed.disconnect(self._on_explorer_closed)
            except Exception:
                pass
            w.close()

    def _on_explorer_opts(self, opts):
        self.explorer_opts = dict(opts)
        self._save_ui_settings()

    def _on_raw_located(self, data_dir):
        """User pointed the explorer at the raw .h5: remember the folder and
        refresh what depends on it (duration, trigger lines)."""
        self.raster.setLimits(xMax=self.rec.duration)
        data = load_settings()
        data["data_dir"] = data_dir
        save_settings(data)
        self._redraw()

    def fit_all(self):
        self.raster.setXRange(0, self.rec.duration, padding=0)
        self._redraw()

    def _toggle_triggers(self, checked):
        self.show_triggers = checked
        self._redraw()
        self._save_ui_settings()

    def open_arrange(self):
        if self._arrange_dlg is not None:
            self._arrange_dlg.raise_()
            self._arrange_dlg.activateWindow()
            return
        dlg = ArrangeDialog(self, list(self.order), list(self.rec.labels),
                            electrode_xy, self._set_order)
        dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.destroyed.connect(self._on_arrange_closed)
        self._arrange_dlg = dlg
        dlg.show()

    def _on_arrange_closed(self, *a):
        self._arrange_dlg = None

    def _reconcile_order(self, labels):
        """`labels` restricted to this recording, with any of its electrodes
        that the list doesn't mention appended in spike-file order."""
        recset = set(self.rec.labels)
        seen = []
        for lab in labels:
            lab = str(lab)
            if lab in recset and lab not in seen:
                seen.append(lab)
        for lab in self.rec.labels:
            if lab not in seen:
                seen.append(lab)
        return seen

    def _merged_order_pref(self):
        """What to remember: the display order, but keeping electrodes from
        the previous preference that this recording happens not to have (QC
        drops a few, recordings differ) anchored next to the same neighbour,
        so switching recordings doesn't quietly forget their position."""
        out = list(self.order)
        known = set(out)
        prev = self._saved_order or []
        for i, lab in enumerate(prev):
            if lab in known:
                continue
            anchor = next((prev[j] for j in range(i - 1, -1, -1)
                           if prev[j] in known), None)
            out.insert(0 if anchor is None else out.index(anchor) + 1, lab)
            known.add(lab)
        return out

    def _set_order(self, labels, source=None):
        """Set the raster display order and remember it for next launch.

        `source` is the file an order was loaded from (None when the user
        dragged / used a preset, which clears the remembered filename).
        """
        self.order = self._reconcile_order(labels)
        self._order_source = source
        self._saved_order = self._merged_order_pref()
        self._save_ui_settings()
        self._redraw()

    def toggle_selection(self, checked):
        if checked:
            t0, t1 = self._current_xrange()
            span = t1 - t0
            self.region.setRegion((t0 + 0.25 * span, t0 + 0.75 * span))
            self.raster.addItem(self.region)
            self.region.show()
        else:
            self.region.hide()
            self.raster.removeItem(self.region)

    def toggle_qc(self, checked):
        self.apply_qc = checked
        old_order = list(self.order)
        try:
            rec = Recording(self.stem, apply_qc=checked, drop_reference=True)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Reload failed", str(exc))
            return
        self._install_recording(rec)
        self._set_order(old_order)                 # keep the manual order

    def _selection_range(self):
        if self.act_select.isChecked() and self.region.isVisible():
            a, b = self.region.getRegion()
            return float(min(a, b)), float(max(a, b))
        return self._current_xrange()

    def export_image_dialog(self):
        opts = {"label_pt": self.export_label_pt,
                "tick_pt": self.export_tick_pt,
                "hide_y_ticks": self.export_hide_y_ticks,
                "width_px": self.export_width_px,
                "dpi": self.export_dpi}
        dlg = ExportImageDialog(self, opts, export.natural_size(self.raster))
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        v = dlg.values()
        self.export_label_pt = v["label_pt"]
        self.export_tick_pt = v["tick_pt"]
        self.export_hide_y_ticks = v["hide_y_ticks"]
        self.export_width_px = v["width_px"]
        self.export_dpi = v["dpi"]
        self._save_ui_settings()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export raster image", self.stem + "_raster.png",
            "PNG image (*.png);;SVG image (*.svg)")
        if not path:
            return
        # A vector export writes one element per spike, so a dense tick view
        # can produce a file no editor will open comfortably.
        if path.lower().endswith(".svg") and self._mode == "detail":
            n = len(self.scatter.getData()[0] or [])
            if n > 20000:
                ans = QtWidgets.QMessageBox.question(
                    self, "Very large SVG",
                    "This view has {:,} spike marks; as SVG that is roughly "
                    "{:.0f} MB and slow to open.\n\n"
                    "Export it anyway? (Heatmap view or PNG stays small.)"
                    .format(n, n * 550 / 1e6))
                if ans != YES_BUTTON:
                    return
        try:
            export.export_image(
                self.raster, path,
                label_pt=self.export_label_pt,
                tick_pt=self.export_tick_pt,
                width_px=self.export_width_px,
                dpi=self.export_dpi,
                hide_tick_values=(("left",) if self.export_hide_y_ticks else ()))
            self.status.showMessage(
                "Saved image ({} px @ {} dpi, {}/{} pt{}): {}".format(
                    self.export_width_px, self.export_dpi,
                    self.export_label_pt, self.export_tick_pt,
                    ", no electrode labels" if self.export_hide_y_ticks else "",
                    path))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export failed", str(exc))

    def export_spikes_dialog(self):
        t0, t1 = self._selection_range()
        path, sel = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export spikes in [{:.3f}, {:.3f}] s".format(t0, t1),
            "{}_{:.0f}-{:.0f}s.npz".format(self.stem, t0, t1),
            "NumPy archive (*.npz);;CSV file (*.csv)")
        if not path:
            return
        try:
            if path.lower().endswith(".csv"):
                export.export_spikes_csv(self.rec, t0, t1, path)
            else:
                export.export_spikes_npz(self.rec, t0, t1, path)
            self.status.showMessage("Saved spikes [{:.3f}, {:.3f}]s: {}".format(
                t0, t1, path))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export failed", str(exc))

    # -- waveform on double-click ------------------------------------------
    def _on_click(self, ev):
        if not ev.double():
            return
        vb = self.raster.getViewBox()
        pt = vb.mapSceneToView(ev.scenePos())
        idx = self.vis_idx()
        row = int(np.floor(pt.y()))
        if row < 0 or row >= len(idx):
            return
        label = self.rec.labels[idx[row]]
        self._current_electrode = label
        t0, t1 = self._current_xrange()
        if (t1 - t0) > 60.0:
            ans = QtWidgets.QMessageBox.question(
                self, "Load raw voltage?",
                "This loads the whole recording into RAM the first time and the "
                "window is {:.0f}s wide. Continue?".format(t1 - t0))
            if ans != YES_BUTTON:
                return
        self.status.showMessage("Loading waveform for {}…".format(label))
        QtWidgets.QApplication.processEvents()
        try:
            t, uv = self.rec.channel_waveform(label, t0, t1)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Waveform failed", str(exc))
            return
        self.wave_curve.setData(t, uv)
        self.wave.setLabel("left", "{} (uV)".format(label))
        self.wave.show()
        self.status.showMessage("Waveform: {}  ({:.3f}–{:.3f}s)".format(
            label, t0, t1))

    def clear_waveform(self):
        self.wave_curve.clear()
        self.wave.hide()
        self.rec.free_raw()
        self.status.showMessage("Waveform cleared; raw buffer freed.")

    def open_folder(self):
        """Pick a different spike-data folder and load a recording from it."""
        stem = _choose_spike_dir_and_recording(self, get_spike_dir())
        if stem is None:
            return
        try:
            rec = Recording(stem, apply_qc=self.apply_qc, drop_reference=True)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
            return
        self.stem = stem
        self.setWindowTitle("MEA Explorer  —  {}".format(stem))
        self._install_recording(rec)


def _pick_recording(stems, parent=None):
    """Choose one stem from a non-empty list (auto-selects if only one)."""
    if not stems:
        return None
    if len(stems) == 1:
        return stems[0]
    stem, ok = QtWidgets.QInputDialog.getItem(
        parent, "Choose a recording", "Detected recordings:", stems, 0, False)
    return stem if ok else None


def _choose_spike_dir_and_recording(parent=None, start=None):
    """Folder picker -> recording picker loop. Returns a stem or None."""
    start = start or ""
    while True:
        d = QtWidgets.QFileDialog.getExistingDirectory(
            parent, "Select the spike-data folder to browse", start)
        if not d:
            return None
        start = d
        set_spike_dir(d)
        stems = list_recordings()
        if not stems:
            QtWidgets.QMessageBox.warning(
                parent, "No detected spikes",
                "No *_spikes.npz files were found in:\n{}\n\n"
                "Pick a folder that contains detected spike files "
                "(run Stage-1 detection first if there are none).".format(d))
            continue
        stem = _pick_recording(stems, parent)
        if stem is None:
            continue
        return stem


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    stem = None
    apply_qc = True
    for a in argv:
        if a in ("--no-qc", "--noqc"):
            apply_qc = False
        elif a.startswith("--spike-dir="):
            set_spike_dir(a.split("=", 1)[1])
        elif not a.startswith("-"):
            stem = a

    QtWidgets.QApplication.setApplicationName("MEA Explorer")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    try:
        mea_theme.apply_qt(app, ACCENT)                 # shared MEA-Suite theme
        app.setStyleSheet(app.styleSheet() + _toolbar_qss())
    except Exception:
        pass
    _set_app_icon(app)
    _install_excepthook()

    # Reopen where the last session left off (unless overridden on the CLI).
    if not any(a.startswith("--spike-dir=") for a in argv):
        last = load_settings().get("spike_dir")
        if last and os.path.isdir(last):
            set_spike_dir(last)
    # Raw-file folder picked by hand in the explorer last time; the .h5 is
    # otherwise found next to the spike data (<root>/H5), so this is only
    # a first place to look.
    raw = load_settings().get("data_dir")
    if raw and os.path.isdir(raw):
        set_data_dir(raw)

    if stem is None:
        stem = _choose_spike_dir_and_recording(None, get_spike_dir())
        if stem is None:
            return 0
    try:
        rec = Recording(stem, apply_qc=apply_qc, drop_reference=True)
    except Exception as exc:
        QtWidgets.QMessageBox.critical(None, "Load failed", str(exc))
        return 1

    win = RasterWindow(rec, stem, apply_qc=apply_qc)
    _set_window_icon(win)
    win.show()
    return app.exec()


def _icon_path():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "MEA Explorer.ico")
    return p if os.path.exists(p) else None


def _set_app_icon(app):
    p = _icon_path()
    if p:
        try:
            app.setWindowIcon(QtGui.QIcon(p))
        except Exception:
            pass


def _set_window_icon(win):
    p = _icon_path()
    if p:
        try:
            win.setWindowIcon(QtGui.QIcon(p))
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
