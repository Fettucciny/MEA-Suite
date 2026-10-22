"""Single-electrode explorer: filtered trace on top, that electrode's spikes
below, and burst detection with the pipeline's own ISI_N (Bakkum) code.

Opened from the main window (toolbar "Explore…", or Shift-click an electrode
on the map). Everything heavy runs off the GUI thread:

  * burst detection on the full spike train (fast for real electrodes, but a
    very active one over an hour can take seconds), and
  * the raw recording, which io.MCSFile loads into memory in full the first
    time — that can be many GB.

The filtered trace is only drawn once the view is ≤ MAX_TRACE_SPAN_S wide;
wider than that the spike row and burst bars still give the overview.
"""
import os
import threading

import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

import mea_theme
import export
from datasource import detect_bursts

MAX_TRACE_SPAN_S = 30.0      # widest window the filtered trace is drawn for
MAX_TICKS = 50000            # spike marks drawn at once (uniformly subsampled)
MAX_TRACE_REGIONS = 200      # burst shades drawn behind the trace at once
MAX_TRIGGER_LINES = 200
BIG_RAW_BYTES = 4 * 1024 ** 3   # ask before loading a raw file this large

ACCENT = mea_theme.accent("explorer")
YES = QtWidgets.QMessageBox.StandardButton.Yes
ALIGN_R = (QtCore.Qt.AlignmentFlag.AlignRight
           | QtCore.Qt.AlignmentFlag.AlignVCenter)


def _rgba(hex_colour, alpha):
    c = QtGui.QColor(hex_colour)
    c.setAlpha(int(alpha))
    return c


class _Jobs(QtCore.QObject):
    """Runs callables on worker threads and reports back on the GUI thread.

    Each job kind keeps a token; a newer request of the same kind makes older
    results stale, so a slow job can never overwrite a newer one.
    """
    done = QtCore.pyqtSignal(str, int, object, object)   # kind, token, res, err

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tok = {}

    def submit(self, kind, fn, *args):
        tok = self._tok.get(kind, 0) + 1
        self._tok[kind] = tok

        def run():
            try:
                res, err = fn(*args), None
            except Exception as exc:          # reported in the GUI
                res, err = None, exc
            try:
                self.done.emit(kind, tok, res, err)
            except RuntimeError:              # window closed meanwhile
                pass

        threading.Thread(target=run, daemon=True).start()
        return tok

    def cancel(self, kind):
        self._tok[kind] = self._tok.get(kind, 0) + 1

    def is_current(self, kind, tok):
        return self._tok.get(kind) == tok


class BurstModel(QtCore.QAbstractTableModel):
    """Table of detected bursts; scales to thousands of rows."""
    HEAD = ["#", "Start (s)", "Dur. (ms)", "Spikes", "Rate (Hz)"]
    TIPS = ["Burst number", "Burst start time", "Burst duration",
            "Spikes in the burst", "Firing rate within the burst"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.r = None

    def set_result(self, r):
        self.beginResetModel()
        self.r = r
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid() or self.r is None:
            return 0
        return int(self.r["t_start"].size)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.HEAD)

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if orientation != QtCore.Qt.Orientation.Horizontal:
            return None
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return self.HEAD[section]
        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            return self.TIPS[section]
        return None

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or self.r is None:
            return None
        i, c = index.row(), index.column()
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            r = self.r
            if c == 0:
                return str(i + 1)
            if c == 1:
                return "{:.3f}".format(r["t_start"][i])
            if c == 2:
                return "{:.1f}".format(r["duration_ms"][i])
            if c == 3:
                return str(int(r["size"][i]))
            v = r["in_burst_fr"][i]
            return "—" if not np.isfinite(v) else "{:.1f}".format(v)
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
            return ALIGN_R.value
        return None


class ElectrodeExplorer(QtWidgets.QMainWindow):
    """One electrode: filtered trace, spikes, bursts."""

    def __init__(self, parent, rec, label, order, time_axis_cls,
                 time_unit="auto", opts=None, show_triggers=True,
                 on_raw_located=None, on_opts_changed=None):
        super().__init__(parent)
        self.rec = rec
        self.order = [l for l in order if rec.has_label(l)] or list(rec.labels)
        self.label = str(label) if rec.has_label(label) else self.order[0]
        self.show_triggers = bool(show_triggers)
        self._on_raw_located = on_raw_located
        self._on_opts_changed = on_opts_changed
        opts = dict(opts or {})
        self.min_spikes = int(opts.get("min_spikes", 5) or 5)
        self.auto_th = bool(opts.get("auto_threshold", True))
        self.manual_ms = float(opts.get("manual_ms", 100.0) or 100.0)
        self.trace_on = bool(opts.get("show_trace", True))

        self.result = None              # last detect_bursts() result
        self._trace = None              # last filtered_window() result
        self._raw_ok = None             # user's answer to the big-file prompt
        self._loading_raw = False
        self._touched = False           # user has moved the view
        self._regions, self._trig_lines = [], []

        self.jobs = _Jobs(self)
        self.jobs.done.connect(self._job_done)

        self.setWindowTitle("Electrode explorer — {}".format(rec.stem))
        self.resize(1360, 760)
        self._build_ui(time_axis_cls, time_unit)

        self._view_timer = QtCore.QTimer(self)
        self._view_timer.setSingleShot(True)
        self._view_timer.setInterval(80)
        self._view_timer.timeout.connect(self._refresh_view)
        self._detect_timer = QtCore.QTimer(self)
        self._detect_timer.setSingleShot(True)
        self._detect_timer.setInterval(150)
        self._detect_timer.timeout.connect(self._detect)

        self.spike_plot.setXRange(0.0, min(10.0, rec.duration), padding=0)
        self.spike_plot.sigXRangeChanged.connect(self._on_range_changed)
        self.set_electrode(self.label)

    # ======================================================================
    # UI
    # ======================================================================
    def _build_ui(self, time_axis_cls, time_unit):
        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(8, 6, 8, 4)
        outer.setSpacing(6)
        outer.addLayout(self._build_topbar())

        split = QtWidgets.QSplitter()
        split.addWidget(self._build_plots(time_axis_cls, time_unit))
        split.addWidget(self._build_side())
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)
        split.setSizes([1000, 360])
        outer.addWidget(split, 1)
        self.setCentralWidget(central)
        self.status = self.statusBar()

    def _build_topbar(self):
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Electrode"))
        self.combo = QtWidgets.QComboBox()
        self.combo.setMinimumWidth(170)
        for lab in self.order:
            self.combo.addItem("{}   ·   {} spikes".format(
                lab, self.rec.spike_times_of(lab).size), lab)
        self.combo.currentIndexChanged.connect(
            lambda i: self.set_electrode(self.combo.itemData(i)))
        row.addWidget(self.combo)
        for txt, tip, d in (("◀", "Previous electrode", -1),
                            ("▶", "Next electrode", +1)):
            b = QtWidgets.QPushButton(txt)
            b.setToolTip(tip)
            b.setFixedWidth(36)
            b.clicked.connect(lambda _=False, d=d: self._step_electrode(d))
            row.addWidget(b)
        row.addSpacing(14)
        for txt, tip, fn in (
                ("Fit all", "Show the whole recording", self.fit_all),
                ("◀ Burst", "Previous burst", lambda: self._step_burst(-1)),
                ("Burst ▶", "Next burst", lambda: self._step_burst(+1))):
            b = QtWidgets.QPushButton(txt)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addSpacing(14)
        self.cb_trace = QtWidgets.QCheckBox("Filtered trace")
        self.cb_trace.setChecked(self.trace_on)
        self.cb_trace.toggled.connect(self._on_trace_toggled)
        row.addWidget(self.cb_trace)
        self.band_label = QtWidgets.QLabel()
        self.band_label.setProperty("muted", True)
        self.band_label.setStyleSheet("color: {};".format(mea_theme.TEXT_MUTED))
        row.addWidget(self.band_label)
        row.addStretch(1)
        self.locate_btn = QtWidgets.QPushButton("Locate raw .h5…")
        self.locate_btn.setToolTip(
            "Point to this recording's raw file if it isn't found next to "
            "the spike data.")
        self.locate_btn.clicked.connect(self._locate_raw)
        self.locate_btn.hide()
        row.addWidget(self.locate_btn)
        return row

    def _build_plots(self, time_axis_cls, time_unit):
        self.glw = pg.GraphicsLayoutWidget()

        # -- filtered trace (top) -------------------------------------------
        self.trace_plot = self.glw.addPlot(row=0, col=0)
        self.trace_plot.setLabel("left", "Filtered (µV)")
        self.trace_plot.hideAxis("bottom")
        self.trace_plot.setMenuEnabled(False)
        self.trace_plot.setMouseEnabled(x=True, y=False)
        self.trace_curve = pg.PlotCurveItem(
            pen=pg.mkPen(mea_theme.PLOT_FG, width=1))
        self.trace_curve.setZValue(5)
        self.trace_plot.addItem(self.trace_curve)
        self.mark_iso = pg.ScatterPlotItem(
            size=6, pen=None, brush=pg.mkBrush(mea_theme.TEXT_MUTED))
        self.mark_bst = pg.ScatterPlotItem(
            size=7, pen=None, brush=pg.mkBrush(ACCENT))
        for it in (self.mark_iso, self.mark_bst):
            it.setZValue(6)
            self.trace_plot.addItem(it)
        self.hint = pg.TextItem("", color=mea_theme.TEXT_MUTED, anchor=(0.5, 0.5))
        self.hint.setZValue(20)
        self.trace_plot.addItem(self.hint, ignoreBounds=True)

        # -- spikes + bursts (bottom) ---------------------------------------
        self.glw.nextRow()
        self.time_axis = time_axis_cls(orientation="bottom")
        try:
            self.time_axis.set_unit_mode(time_unit)
        except Exception:
            pass
        self.spike_plot = self.glw.addPlot(
            row=1, col=0, axisItems={"bottom": self.time_axis})
        self.spike_plot.setLabel("left", "Spikes")
        self.spike_plot.getAxis("left").setTicks([[], []])
        self.spike_plot.setMenuEnabled(False)
        self.spike_plot.setMouseEnabled(x=True, y=False)
        self.spike_plot.setYRange(0, 1, padding=0)
        self.spike_plot.setMaximumHeight(130)
        self.spike_plot.setMinimumHeight(110)
        self.trace_plot.setXLink(self.spike_plot)
        self.burst_bars = pg.BarGraphItem(
            x0=[], width=[], y0=0.08, height=0.84,
            brush=pg.mkBrush(_rgba(ACCENT, 70)), pen=pg.mkPen(None))
        self.spike_plot.addItem(self.burst_bars)
        self.tick_iso = pg.ScatterPlotItem(
            symbol="|", size=16, pen=pg.mkPen(mea_theme.TEXT_MUTED, width=1.2))
        self.tick_bst = pg.ScatterPlotItem(
            symbol="|", size=16, pen=pg.mkPen(ACCENT, width=1.4))
        for it in (self.tick_iso, self.tick_bst):
            it.setZValue(5)
            self.spike_plot.addItem(it)
        self.spike_plot.getViewBox().setLimits(xMin=0, xMax=self.rec.duration)
        self.trace_plot.getViewBox().setLimits(xMin=0, xMax=self.rec.duration)
        return self.glw

    def _card(self, title):
        box = QtWidgets.QGroupBox(title)
        lay = QtWidgets.QFormLayout(box)
        lay.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        return box, lay

    def _build_side(self):
        side = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(side)
        v.setContentsMargins(4, 0, 0, 0)

        # -- detection parameters -------------------------------------------
        box, f = self._card("Burst detection — ISI_N (Bakkum)")
        self.sp_n = QtWidgets.QSpinBox()
        self.sp_n.setRange(3, 200)
        self.sp_n.setValue(self.min_spikes)
        self.sp_n.setToolTip(
            "N: a burst needs at least N spikes, and every N consecutive "
            "spikes must fall within the ISI_N threshold.")
        self.sp_n.valueChanged.connect(self._params_changed)
        f.addRow("Min spikes (N)", self.sp_n)

        trow = QtWidgets.QHBoxLayout()
        self.rb_auto = QtWidgets.QRadioButton("Automatic")
        self.rb_manual = QtWidgets.QRadioButton("Manual")
        self.rb_auto.setToolTip(
            "Valley of the ISI_N histogram (Bakkum et al.), capped at 100 ms "
            "— exactly what the MEA-NAP pipeline uses.")
        (self.rb_auto if self.auto_th else self.rb_manual).setChecked(True)
        self.rb_auto.toggled.connect(self._params_changed)
        trow.addWidget(self.rb_auto)
        trow.addWidget(self.rb_manual)
        f.addRow("ISI_N threshold", trow)
        self.sp_ms = QtWidgets.QDoubleSpinBox()
        self.sp_ms.setRange(0.5, 2000.0)
        self.sp_ms.setDecimals(1)
        self.sp_ms.setSingleStep(5.0)
        self.sp_ms.setSuffix(" ms")
        self.sp_ms.setValue(self.manual_ms)
        self.sp_ms.setEnabled(not self.auto_th)
        self.sp_ms.valueChanged.connect(self._params_changed)
        f.addRow("Manual value", self.sp_ms)
        self.th_used = QtWidgets.QLabel("—")
        f.addRow("Threshold used", self.th_used)
        v.addWidget(box)

        # -- summary ---------------------------------------------------------
        box, f = self._card("Summary")
        self.stats = {}
        for key, name in (("spikes", "Spikes"), ("bursts", "Bursts"),
                          ("rate", "Burst rate"), ("dur", "Mean duration"),
                          ("size", "Mean spikes / burst"),
                          ("fr", "Mean rate in burst"),
                          ("isi_in", "Mean ISI within"),
                          ("isi_out", "Mean ISI outside"),
                          ("frac", "Spikes in bursts")):
            lab = QtWidgets.QLabel("—")
            lab.setAlignment(ALIGN_R)
            self.stats[key] = lab
            f.addRow(name, lab)
        v.addWidget(box)

        # -- burst table -----------------------------------------------------
        box = QtWidgets.QGroupBox("Bursts — click to zoom")
        bl = QtWidgets.QVBoxLayout(box)
        self.model = BurstModel(self)
        self.table = QtWidgets.QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().hide()
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.clicked.connect(lambda ix: self.zoom_to_burst(ix.row()))
        bl.addWidget(self.table)
        self.export_btn = QtWidgets.QPushButton("Export bursts…")
        self.export_btn.setProperty("primary", True)
        self.export_btn.clicked.connect(self.export_bursts)
        bl.addWidget(self.export_btn)
        v.addWidget(box, 1)
        side.setMinimumWidth(320)
        return side

    # ======================================================================
    # electrode / parameters
    # ======================================================================
    def set_electrode(self, label):
        label = str(label)
        if not self.rec.has_label(label):
            return
        self.label = label
        i = self.combo.findData(label)
        if i >= 0 and i != self.combo.currentIndex():
            self.combo.blockSignals(True)
            self.combo.setCurrentIndex(i)
            self.combo.blockSignals(False)
        self.setWindowTitle("Electrode {} — {}".format(label, self.rec.stem))
        self.result = None
        self._trace = None
        self.model.set_result(None)
        self._clear_trace()
        self._detect()
        self._refresh_view()

    def _step_electrode(self, d):
        if not self.order:
            return
        i = (self.order.index(self.label) + d) % len(self.order)
        self.set_electrode(self.order[i])

    def _params_changed(self, *a):
        self.min_spikes = self.sp_n.value()
        self.auto_th = self.rb_auto.isChecked()
        self.manual_ms = self.sp_ms.value()
        self.sp_ms.setEnabled(not self.auto_th)
        self._emit_opts()
        self._detect_timer.start()

    def opts(self):
        return {"min_spikes": self.min_spikes, "auto_threshold": self.auto_th,
                "manual_ms": self.manual_ms, "show_trace": self.trace_on}

    def _emit_opts(self):
        if self._on_opts_changed:
            try:
                self._on_opts_changed(self.opts())
            except Exception:
                pass

    # ======================================================================
    # burst detection (worker thread)
    # ======================================================================
    def _detect(self):
        th = "automatic" if self.auto_th else self.manual_ms / 1000.0
        self.status.showMessage("Detecting bursts on {}…".format(self.label))
        self.jobs.submit("bursts", detect_bursts,
                         self.rec.spike_times_of(self.label), self.rec.fs,
                         self.min_spikes, th, self.rec.duration)

    def _apply_bursts(self, r):
        self.result = r
        self.model.set_result(r)
        nb = int(r["t_start"].size)
        if np.isfinite(r["isin_th"]):
            self.th_used.setText("{:.1f} ms  ({})".format(
                r["isin_th"] * 1000.0, "automatic" if r["auto"] else "manual"))
        else:
            self.th_used.setText("— (fewer than N spikes)")

        def fmt(v, spec, unit=""):
            return "—" if v is None or not np.isfinite(v) else (spec.format(v) + unit)

        s = self.stats
        s["spikes"].setText("{:,}".format(r["n_spikes"]))
        s["bursts"].setText("{:,}".format(nb))
        s["rate"].setText(fmt(r["burst_rate_per_min"] if nb else 0.0, "{:.2f}", " /min"))
        s["dur"].setText(fmt(r["mean_burst_dur_ms"], "{:.1f}", " ms"))
        s["size"].setText(fmt(r["mean_spikes_per_burst"], "{:.1f}"))
        s["fr"].setText(fmt(r["mean_in_burst_fr_hz"], "{:.1f}", " Hz"))
        s["isi_in"].setText(fmt(r["mean_isi_within_ms"], "{:.1f}", " ms"))
        s["isi_out"].setText(fmt(r["mean_isi_outside_ms"], "{:.0f}", " ms"))
        s["frac"].setText(fmt(100.0 * r["frac_in_bursts"] if nb else 0.0, "{:.1f}", " %"))
        self.export_btn.setEnabled(nb > 0)

        self.burst_bars.setOpts(x0=r["t_start"],
                                width=np.maximum(r["t_end"] - r["t_start"],
                                                 0.002),
                                y0=0.08, height=0.84)
        self.status.showMessage(
            "Electrode {}: {:,} spikes, {:,} bursts.".format(
                self.label, r["n_spikes"], nb))
        if nb and not self._touched:          # first look: show a burst
            self.zoom_to_burst(0, touch=False)
        else:
            self._refresh_view()

    # ======================================================================
    # view
    # ======================================================================
    def _on_range_changed(self, *a):
        self._view_timer.start()

    def _xrange(self):
        (x0, x1), _ = self.spike_plot.viewRange()
        x0 = max(0.0, float(x0))
        x1 = min(float(self.rec.duration), float(x1))
        return x0, max(x1, x0 + 1e-4)

    def fit_all(self):
        self._touched = True
        self.spike_plot.setXRange(0.0, self.rec.duration, padding=0)

    def zoom_to_burst(self, i, touch=True):
        r = self.result
        if r is None or not (0 <= i < r["t_start"].size):
            return
        if touch:
            self._touched = True
        a, b = float(r["t_start"][i]), float(r["t_end"][i])
        span = max(0.5, 4.0 * (b - a))
        c = 0.5 * (a + b)
        self._burst_cursor = i
        self.spike_plot.setXRange(max(0.0, c - span / 2),
                                  min(self.rec.duration, c + span / 2), padding=0)
        ix = self.model.index(i, 0)
        self.table.selectRow(i)
        self.table.scrollTo(ix)

    def _step_burst(self, d):
        r = self.result
        if r is None or not r["t_start"].size:
            return
        t0, t1 = self._xrange()
        c = 0.5 * (t0 + t1)
        mid = 0.5 * (r["t_start"] + r["t_end"])
        if d > 0:
            nxt = np.nonzero(mid > c + 1e-6)[0]
            i = int(nxt[0]) if nxt.size else int(mid.size - 1)
        else:
            prv = np.nonzero(mid < c - 1e-6)[0]
            i = int(prv[-1]) if prv.size else 0
        self.zoom_to_burst(i)

    def _refresh_view(self):
        t0, t1 = self._xrange()
        self._draw_ticks(t0, t1)
        self._draw_regions(t0, t1)
        self._draw_triggers(t0, t1)
        self._request_trace(t0, t1)

    def _draw_ticks(self, t0, t1):
        t = self.rec.spike_times_of(self.label)
        lo, hi = np.searchsorted(t, [t0, t1])
        tv = t[lo:hi]
        if self.result is not None and self.result["spike_times"].size == t.size:
            inb = self.result["spike_burst"][lo:hi] >= 0
        else:
            inb = np.zeros(tv.size, dtype=bool)
        if tv.size > MAX_TICKS:                       # keep it responsive
            step = int(np.ceil(tv.size / MAX_TICKS))
            tv, inb = tv[::step], inb[::step]
        self.tick_iso.setData(x=tv[~inb], y=np.full(int((~inb).sum()), 0.5))
        self.tick_bst.setData(x=tv[inb], y=np.full(int(inb.sum()), 0.5))

    def _draw_regions(self, t0, t1):
        for it in self._regions:
            self.trace_plot.removeItem(it)
        self._regions = []
        r = self.result
        if r is None or not r["t_start"].size:
            return
        sel = np.nonzero((r["t_end"] >= t0) & (r["t_start"] <= t1))[0]
        if sel.size > MAX_TRACE_REGIONS:
            return                                    # the spike row shows them
        brush = pg.mkBrush(_rgba(ACCENT, 34))
        for i in sel:
            reg = pg.LinearRegionItem(
                values=(float(r["t_start"][i]), float(r["t_end"][i])),
                movable=False, brush=brush, pen=pg.mkPen(None))
            reg.setZValue(-10)
            for ln in reg.lines:
                ln.setPen(pg.mkPen(_rgba(ACCENT, 110), width=1))
            self.trace_plot.addItem(reg)
            self._regions.append(reg)

    def _draw_triggers(self, t0, t1):
        for plot, ln in self._trig_lines:
            plot.removeItem(ln)
        self._trig_lines = []
        if not self.show_triggers:
            return
        on = self.rec.trigger_on_times()
        vis = on[(on >= t0) & (on <= t1)] if on.size else on
        if vis.size == 0 or vis.size > MAX_TRIGGER_LINES:
            return
        pen = pg.mkPen("#FBBF24", width=1)
        for t in vis:
            for plot in (self.trace_plot, self.spike_plot):
                ln = pg.InfiniteLine(pos=float(t), angle=90, pen=pen, movable=False)
                ln.setZValue(8)
                plot.addItem(ln)
                self._trig_lines.append((plot, ln))

    # ======================================================================
    # filtered trace (worker thread)
    # ======================================================================
    def _on_trace_toggled(self, on):
        self.trace_on = bool(on)
        if on:
            self._raw_ok = None               # allow the size prompt again
        self._emit_opts()
        self._refresh_view()

    def _clear_trace(self):
        self.trace_curve.setData([], [])
        self.mark_iso.setData([], [])
        self.mark_bst.setData([], [])

    def _show_hint(self, text):
        self._clear_trace()
        self.hint.setText(text)
        t0, t1 = self._xrange()
        self.trace_plot.setYRange(-1, 1, padding=0)
        self.hint.setPos(0.5 * (t0 + t1), 0.0)

    def _request_trace(self, t0, t1):
        lo, hi = self.rec.filter_band()
        self.band_label.setText("bandpass {:.0f}–{:.0f} Hz (detection settings)"
                                .format(lo, hi))
        if not self.trace_on:
            self.jobs.cancel("trace")
            self._show_hint("Filtered trace off")
            return
        if self.rec.h5_path is None:
            self.locate_btn.show()
            self._show_hint("Raw file '{}.h5' not found — use “Locate raw .h5…”"
                            .format(self.rec.stem))
            return
        self.locate_btn.hide()
        if t1 - t0 > MAX_TRACE_SPAN_S:
            self.jobs.cancel("trace")
            self._show_hint("Zoom in to ≤ {:.0f} s to see the filtered trace"
                            .format(MAX_TRACE_SPAN_S))
            return
        if not self._confirm_raw_load():
            self.cb_trace.blockSignals(True)
            self.cb_trace.setChecked(False)
            self.cb_trace.blockSignals(False)
            self.trace_on = False
            self._show_hint("Filtered trace off")
            return
        if not self.rec.raw_loaded():
            self._loading_raw = True
            self._show_hint("Loading raw recording ({:.1f} GB)…".format(
                self.rec.raw_file_size() / 1e9))
        width = max(800, int(self.trace_plot.getViewBox().width()))
        self.jobs.submit("trace", self.rec.filtered_window,
                         self.label, t0, t1, width)

    def _confirm_raw_load(self):
        if self.rec.raw_loaded():
            return True
        size = self.rec.raw_file_size()
        if size < BIG_RAW_BYTES:
            return True
        if self._raw_ok is None:
            ans = QtWidgets.QMessageBox.question(
                self, "Load raw recording?",
                "The filtered trace needs the raw recording, which is loaded "
                "into memory in full ({:.1f} GB) the first time.\n\nLoad it?"
                .format(size / 1e9))
            self._raw_ok = (ans == YES)
        return self._raw_ok

    def _apply_trace(self, w):
        self._loading_raw = False
        self._trace = w
        self.hint.setText("")
        if w["t"].size == 0:
            self._clear_trace()
            return
        self.trace_curve.setData(w["t"], w["y"])
        # mark the detected spikes on the trace, at their filtered value
        full, s0, fs = w["full"], w["s0"], w["fs"]
        t = self.rec.spike_times_of(self.label)
        a, b = s0 / fs, (s0 + full.size) / fs
        lo, hi = np.searchsorted(t, [a, b])
        ts = t[lo:hi]
        k = np.clip(np.round(ts * fs).astype(np.int64) - s0, 0, full.size - 1)
        ys = full[k] if full.size else np.empty(0)
        if self.result is not None and self.result["spike_times"].size == t.size:
            inb = self.result["spike_burst"][lo:hi] >= 0
        else:
            inb = np.zeros(ts.size, dtype=bool)
        self.mark_iso.setData(x=ts[~inb], y=ys[~inb])
        self.mark_bst.setData(x=ts[inb], y=ys[inb])
        lo_y, hi_y = float(np.min(w["y"])), float(np.max(w["y"]))
        pad = 0.08 * max(hi_y - lo_y, 1e-3)
        self.trace_plot.setYRange(lo_y - pad, hi_y + pad, padding=0)

    # ======================================================================
    # results from worker threads
    # ======================================================================
    def _job_done(self, kind, tok, res, err):
        if not self.jobs.is_current(kind, tok):
            return                                    # superseded
        if err is not None:
            if kind == "trace":
                self._loading_raw = False
                self._show_hint("Could not read the raw trace:\n{}".format(err))
            else:
                self.status.showMessage("Burst detection failed: {}".format(err))
            return
        if kind == "bursts":
            self._apply_bursts(res)
        elif kind == "trace":
            self._apply_trace(res)

    # ======================================================================
    # raw file location / export
    # ======================================================================
    def _locate_raw(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Locate raw recording for '{}'".format(self.rec.stem),
            "", "MCS HDF5 (*.h5);;All files (*)")
        if not path:
            return
        want = self.rec.stem + ".h5"
        if os.path.basename(path).lower() != want.lower():
            ans = QtWidgets.QMessageBox.question(
                self, "Different file name",
                "The spikes came from '{}', but you picked '{}'.\n\n"
                "Use it anyway?".format(want, os.path.basename(path)))
            if ans != YES:
                return
        self.rec.set_raw_path(path)
        if self._on_raw_located:
            try:
                self._on_raw_located(os.path.dirname(path))
            except Exception:
                pass
        self._refresh_view()

    def export_bursts(self):
        r = self.result
        if r is None or not r["t_start"].size:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export bursts",
            "{}_{}_bursts.csv".format(self.rec.stem, self.label),
            "CSV file (*.csv)")
        if not path:
            return
        try:
            export.export_bursts_csv(self.rec.stem, self.label, r, path)
            self.status.showMessage("Saved {:,} bursts: {}".format(
                int(r["t_start"].size), path))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export failed", str(exc))

    def closeEvent(self, ev):
        self.jobs.cancel("bursts")
        self.jobs.cancel("trace")
        super().closeEvent(ev)
