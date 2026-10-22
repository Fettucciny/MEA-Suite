"""MEA Solar Cell Analyzer -- main GUI (PyQt6 / pyqtgraph).

Tabs:
  Setup       folders, array pitch, per-recording CONDITION tags, and the
              clickable electrode map to place the solar cell + spheroid.
  Peaks       learn templates from the tagged recordings, then detect & classify
              peaks (solar / piezo / biological) on any recording; inspect a
              single electrode's trace with flood markers + coloured peaks.
  Templates   the learned solar/piezo/bio peak-shape waveforms; save / load.
  Filter      artifact-subtract a sample and re-detect pure biological spikes;
              export the cleaned spike train.
  Field map   solar-cell effective field radius / shape from a solar-only file.
  Footprint   spheroid footprint from biological spikes + spiking-reach metrics.

Only the analysis MATH lives in the loader-free, unit-tested modules; this file
is UI + wiring. Heavy data operations reuse the validated MEA-NAP loaders and run
on a worker thread.
"""
import json
import os
import sys
import traceback

# PyQt6 first (the migrated suite binding), PyQt5 as a dev/legacy fallback. The GUI
# imports Qt through pyqtgraph.Qt, so this env var selects the binding for the whole app.
if "PYQTGRAPH_QT_LIB" not in os.environ:
    for _binding in ("PyQt6", "PyQt5"):
        try:
            __import__(_binding)
            os.environ["PYQTGRAPH_QT_LIB"] = _binding
            break
        except Exception:
            continue

import numpy as np

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
except Exception as exc:  # pragma: no cover
    sys.stderr.write(
        "\nCould not import pyqtgraph / a Qt binding.\n"
        "Add pyqtgraph + PyQt6 to the shared MEA .venv:\n"
        "  ..\\.venv\\Scripts\\python.exe -m pip install pyqtgraph PyQt6\n"
        "Original error: {}\n".format(exc))
    raise

import mea_theme                              # shared MEA-Suite look-and-feel
import geometry as geo
import project as proj
import templates as tpl
import classify as clf
import pipeline
from electrode_map import ElectrodeMap

mea_theme.style_pyqtgraph("solar")           # dark suite plot bg + antialias

try:                                       # QMessageBox.Yes moved between Qt5/Qt6
    _YES = QtWidgets.QMessageBox.StandardButton.Yes
except AttributeError:
    _YES = QtWidgets.QMessageBox.Yes

CLASS_COLORS = {
    clf.SOLAR: (60, 200, 255), clf.PIEZO: (255, 150, 40),
    clf.BIO: (120, 230, 130), "unknown": (150, 150, 150),
}

def _install_excepthook():
    logpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "error.log"))

    def hook(exctype, value, tb):
        msg = "".join(traceback.format_exception(exctype, value, tb))
        try:
            with open(logpath, "a", encoding="utf-8") as fh:
                fh.write(msg + "\n")
        except Exception:
            pass
        try:
            QtWidgets.QMessageBox.critical(
                None, "MEA Solar Cell Analyzer error",
                msg[-3000:] + "\n\nLogged to:\n" + logpath)
        except Exception:
            pass
        sys.__excepthook__(exctype, value, tb)

    sys.excepthook = hook


def _viridis_lut():
    for getter in ("getFromMatplotlib", "get"):
        try:
            cmap = getattr(pg.colormap, getter)("viridis")
            if cmap is not None:
                return cmap.getLookupTable(0.0, 1.0, 256)
        except Exception:
            pass
    return None


# ===========================================================================
class Worker(QtCore.QThread):
    """Run one function(progress_cb) off the GUI thread."""
    progress = QtCore.pyqtSignal(int, int, str)
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        def prog(d, t, msg=""):
            self.progress.emit(int(d), int(t), str(msg))
        try:
            self.done.emit(self.fn(prog))
        except Exception:
            self.failed.emit(traceback.format_exc())


# ===========================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._settings = _load_settings()
        self._last_dir = self._settings.get("last_dir", "")
        self.project = proj.Project()
        if self._settings.get("pitch_um"):        # remember the last pitch used
            self.project.pitch_um = float(self._settings["pitch_um"])
        self.templates = tpl.TemplateSet()
        self._tpl_info = {}            # class -> "stem: N peaks" (report sources)
        self._raw_cache = {}           # stem -> RawRecording (heavy; freed on switch)
        self._classify = None          # (results, counts, stem)
        self._filter = None            # (FilterResult, stem, dur)
        self._field = None
        self._worker = None
        self._lut = _viridis_lut()
        self._solar_stem = None        # recording currently being placed on
        self._solar_sel = -1           # index of the selected cell
        self._sph_points = []          # spheroid perimeter points (grid coords)
        self._img_arr = None
        self.setWindowTitle("MEA Solar Cell Analyzer")
        self.resize(1550, 900)
        self._build()
        # apply the remembered image scale as the default, and remember future edits
        if self._settings.get("um_per_px"):
            _silent(self.img_scale, float(self._settings["um_per_px"]))
        self.img_scale.valueChanged.connect(
            lambda v: self._remember("um_per_px", float(v)))
        self._refresh_all()

    # -- construction -------------------------------------------------------
    def _build(self):
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(self._build_setup_tab(), "Setup")
        self.tabs.addTab(self._build_peaks_tab(), "Peaks && Classify")
        self.tabs.addTab(self._build_templates_tab(), "Templates")
        self.tabs.addTab(self._build_filter_tab(), "Filter")
        self.tabs.addTab(self._build_field_tab(), "Field map")
        self.tabs.addTab(self._build_footprint_tab(), "Footprint")
        self.status = self.statusBar()
        self.prog = QtWidgets.QProgressBar()
        self.prog.setMaximumWidth(320)
        self.prog.setVisible(False)
        self.status.addPermanentWidget(self.prog)

    # ---- Setup tab --------------------------------------------------------
    def _build_setup_tab(self):
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)

        # left column: folders + recording table + project buttons
        left = QtWidgets.QVBoxLayout()
        fld = QtWidgets.QGroupBox("Data")
        fl = QtWidgets.QGridLayout(fld)
        self.data_edit = QtWidgets.QLineEdit()
        self.data_edit.setPlaceholderText("Folder with the .h5 files and Batch.csv")
        b_data = QtWidgets.QPushButton("Browse…")
        b_data.clicked.connect(self._pick_data_dir)
        fl.addWidget(QtWidgets.QLabel("Data folder"), 0, 0)
        fl.addWidget(self.data_edit, 0, 1)
        fl.addWidget(b_data, 0, 2)
        self.spike_edit = QtWidgets.QLineEdit()
        self.spike_edit.setPlaceholderText(
            "where THIS app saves cleaned biological spikes (Filter tab export)")
        b_spk = QtWidgets.QPushButton("Browse…")
        b_spk.clicked.connect(self._pick_spike_dir)
        fl.addWidget(QtWidgets.QLabel("Cleaned-spikes output"), 1, 0)
        fl.addWidget(self.spike_edit, 1, 1)
        fl.addWidget(b_spk, 1, 2)
        self.pitch_spin = QtWidgets.QDoubleSpinBox()
        self.pitch_spin.setRange(1, 5000)
        self.pitch_spin.setValue(self.project.pitch_um)
        self.pitch_spin.setSuffix(" µm")
        self.pitch_spin.valueChanged.connect(self._on_pitch)
        fl.addWidget(QtWidgets.QLabel("Electrode pitch"), 2, 0)
        fl.addWidget(self.pitch_spin, 2, 1)
        # flood stimulation-train length (drives the artifact window / subtraction)
        self.stim_train_spin = QtWidgets.QDoubleSpinBox()
        self.stim_train_spin.setRange(1, 5000)
        self.stim_train_spin.setValue(self.project.analysis.get("stim_train_ms", 250.0))
        self.stim_train_spin.setSuffix(" ms")
        self.stim_train_spin.setToolTip(
            "Duration of the whole flood pulse-train — the artifact spans this "
            "window. Set from the flood.seq via 'Load flood .seq'.")
        self.stim_train_spin.valueChanged.connect(
            lambda v: self.project.analysis.__setitem__("stim_train_ms", float(v)))
        b_seq = QtWidgets.QPushButton("Load flood .seq…")
        b_seq.clicked.connect(self._load_seq)
        fl.addWidget(QtWidgets.QLabel("Flood train length"), 3, 0)
        fl.addWidget(self.stim_train_spin, 3, 1)
        fl.addWidget(b_seq, 3, 2)
        self.stim_auto = QtWidgets.QCheckBox(
            "Auto-detect stim duration from the data (any pulse count/width)")
        self.stim_auto.setToolTip(
            "Measure each recording's actual artifact duration from the flood-evoked "
            "signal, instead of the fixed length above. Handles single pulses, "
            "variable trains and mixed protocols with no manual setting.")
        self.stim_auto.setChecked(
            self.project.analysis.get("stim_span_mode", "auto") == "auto")
        self.stim_auto.toggled.connect(self._toggle_stim_auto)
        self.stim_train_spin.setEnabled(not self.stim_auto.isChecked())
        fl.addWidget(self.stim_auto, 4, 0, 1, 3)
        left.addWidget(fld)

        self.rec_table = QtWidgets.QTableWidget(0, 3)
        self.rec_table.setHorizontalHeaderLabels(["Recording", "Batch info",
                                                  "Condition"])
        self.rec_table.horizontalHeader().setStretchLastSection(True)
        self.rec_table.verticalHeader().setVisible(False)
        left.addWidget(QtWidgets.QLabel("Recordings (tag each one's condition):"))
        left.addWidget(self.rec_table, 1)

        pb = QtWidgets.QHBoxLayout()
        for text, slot in (("New project", self._new_project),
                           ("Open project…", self._open_project),
                           ("Save project…", self._save_project)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            pb.addWidget(b)
        left.addLayout(pb)

        rb = QtWidgets.QHBoxLayout()
        b_reload = QtWidgets.QPushButton("↻ Reload scripts (restart)")
        b_reload.setToolTip("Re-launch the app so newly pushed code takes effect, "
                            "keeping the current project + templates. Checks the "
                            "scripts compile first.")
        b_reload.clicked.connect(self._reload_restart)
        rb.addWidget(b_reload)
        rb.addStretch(1)
        left.addLayout(rb)

        # right column: electrode map + placement controls
        right = QtWidgets.QVBoxLayout()
        self.map = ElectrodeMap(pitch_um=self.project.pitch_um, fov=True)
        self.map.solarPlaced.connect(self._on_solar_placed)
        self.map.spheroidPlaced.connect(self._on_spheroid_placed)
        self.map.imagePlaced.connect(self._on_image_placed)
        hint = QtWidgets.QLabel(
            "Coordinates are µm from the array centre (dashed ring = 3 mm field). "
            "Click a mode then click the map, or type coordinates below.")
        hint.setStyleSheet("color:#9aa0a8;")
        right.addWidget(hint)
        mode_row = QtWidgets.QHBoxLayout()
        self.mode_group = QtWidgets.QButtonGroup(self)
        for i, (text, mode) in enumerate((("Inspect", "inspect"),
                                          ("Place solar cell", "solar"),
                                          ("Outline spheroid", "spheroid"),
                                          ("Align image", "image"))):
            b = QtWidgets.QPushButton(text)
            b.setCheckable(True)
            if i == 0:
                b.setChecked(True)
            b.clicked.connect(lambda _c, m=mode: self.map.set_mode(m))
            self.mode_group.addButton(b)
            mode_row.addWidget(b)
        right.addLayout(mode_row)
        right.addWidget(self.map, 1)

        def _spin(lo, hi, val, suffix, step=1.0):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSuffix(suffix)
            s.setSingleStep(step)
            return s

        # solar controls: per-recording list of cells (0 / 1 / 2+); each with its
        # own centre, size, rotation and electrode axis.
        sc = QtWidgets.QGroupBox("Solar cells (per recording; electrodes on two "
                                 "opposite edges)")
        sg = QtWidgets.QGridLayout(sc)
        self.solar_rec = QtWidgets.QComboBox()
        self.solar_rec.currentIndexChanged.connect(self._on_solar_rec_changed)
        self.solar_cells_combo = QtWidgets.QComboBox()
        self.solar_cells_combo.currentIndexChanged.connect(self._on_solar_sel_changed)
        b_addcell = QtWidgets.QPushButton("Add cell")
        b_addcell.clicked.connect(self._solar_add_cell)
        b_delcell = QtWidgets.QPushButton("Remove cell")
        b_delcell.clicked.connect(self._solar_del_cell)
        sg.addWidget(QtWidgets.QLabel("For recording"), 0, 0)
        sg.addWidget(self.solar_rec, 0, 1, 1, 3)
        sg.addWidget(QtWidgets.QLabel("Cell"), 1, 0)
        sg.addWidget(self.solar_cells_combo, 1, 1)
        sg.addWidget(b_addcell, 1, 2)
        sg.addWidget(b_delcell, 1, 3)
        self.solar_cx = _spin(-3000, 3000, 0, " µm x", 1)
        self.solar_cy = _spin(-3000, 3000, 0, " µm y", 1)
        self.solar_side = _spin(10, 5000, 400, " µm side", 10)
        self.solar_orient = _spin(-180, 180, 0, " ° square", 0.5)
        self.solar_orient.setDecimals(1)
        self.solar_axis = _spin(-180, 180, 0, " ° e-axis", 0.5)
        self.solar_axis.setDecimals(1)
        for wdg in (self.solar_cx, self.solar_cy, self.solar_side,
                    self.solar_orient, self.solar_axis):
            wdg.valueChanged.connect(self._solar_edited)
        sg.addWidget(QtWidgets.QLabel("Centre X"), 2, 0)
        sg.addWidget(self.solar_cx, 2, 1)
        sg.addWidget(QtWidgets.QLabel("Centre Y"), 2, 2)
        sg.addWidget(self.solar_cy, 2, 3)
        sg.addWidget(QtWidgets.QLabel("Side"), 3, 0)
        sg.addWidget(self.solar_side, 3, 1)
        sg.addWidget(QtWidgets.QLabel("Rotation"), 3, 2)
        sg.addWidget(self.solar_orient, 3, 3)
        sg.addWidget(QtWidgets.QLabel("Electrode axis"), 4, 0)
        sg.addWidget(self.solar_axis, 4, 1)
        self.solar_lbl = QtWidgets.QLabel("pick a recording, then Add cell / click "
                                          "the map")
        self.solar_lbl.setStyleSheet("color:#9aa0a8;")
        self.solar_lbl.setWordWrap(True)
        sg.addWidget(self.solar_lbl, 4, 2, 1, 2)
        right.addWidget(sc)

        # spheroid controls (hand-traced perimeter outline)
        sp = QtWidgets.QGroupBox("Spheroid outline")
        spg = QtWidgets.QGridLayout(sp)
        sphint = QtWidgets.QLabel(
            "Choose 'Outline spheroid', then click around the perimeter; the "
            "closed curve fits your points (in click order).")
        sphint.setWordWrap(True)
        sphint.setStyleSheet("color:#9aa0a8;")
        spg.addWidget(sphint, 0, 0, 1, 3)
        b_undo = QtWidgets.QPushButton("Undo point")
        b_undo.clicked.connect(self._sph_undo)
        b_clear = QtWidgets.QPushButton("Clear outline")
        b_clear.clicked.connect(self._sph_clear)
        self.sph_smooth = QtWidgets.QCheckBox("Smooth curve")
        self.sph_smooth.setChecked(True)
        self.sph_smooth.stateChanged.connect(lambda _s: self._rebuild_spheroid())
        spg.addWidget(b_undo, 1, 0)
        spg.addWidget(b_clear, 1, 1)
        spg.addWidget(self.sph_smooth, 1, 2)
        self.sph_lbl = QtWidgets.QLabel("no points")
        self.sph_lbl.setStyleSheet("color:#9aa0a8;")
        spg.addWidget(self.sph_lbl, 2, 0, 1, 3)
        right.addWidget(sp)

        # microscope-image overlay controls
        ig = QtWidgets.QGroupBox("Microscope image overlay (align to the electrodes)")
        igl = QtWidgets.QGridLayout(ig)
        b_img = QtWidgets.QPushButton("Load image…")
        b_img.clicked.connect(self._load_image)
        b_fit = QtWidgets.QPushButton("Fit to array")
        b_fit.clicked.connect(self._fit_image)
        b_clr = QtWidgets.QPushButton("Clear")
        b_clr.clicked.connect(self._clear_image)
        self.img_scale = _spin(0.0001, 1000, 2.0, " µm/px", 0.01)
        self.img_scale.setDecimals(4)
        self.img_rot = _spin(-180, 180, 0, " °", 0.1)   # 0.1° per arrow-key step
        self.img_rot.setDecimals(1)
        self.img_cx = _spin(-3000, 3000, 0, " µm x", 1)  # 1 µm per arrow-key step
        self.img_cy = _spin(-3000, 3000, 0, " µm y", 1)
        self.img_opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.img_opacity.setRange(0, 100)
        self.img_opacity.setValue(60)
        self.img_flip = QtWidgets.QCheckBox("Flip Y")
        self.img_flip.setChecked(True)
        for wdg in (self.img_scale, self.img_rot, self.img_cx, self.img_cy):
            wdg.valueChanged.connect(self._apply_image)
        self.img_opacity.valueChanged.connect(self._apply_image)
        self.img_flip.stateChanged.connect(self._apply_image)
        igl.addWidget(b_img, 0, 0)
        igl.addWidget(b_fit, 0, 1)
        igl.addWidget(b_clr, 0, 2)
        igl.addWidget(self.img_flip, 0, 3)
        igl.addWidget(QtWidgets.QLabel("Scale"), 1, 0)
        igl.addWidget(self.img_scale, 1, 1)
        igl.addWidget(QtWidgets.QLabel("Rotation"), 1, 2)
        igl.addWidget(self.img_rot, 1, 3)
        igl.addWidget(QtWidgets.QLabel("Centre X"), 2, 0)
        igl.addWidget(self.img_cx, 2, 1)
        igl.addWidget(QtWidgets.QLabel("Centre Y"), 2, 2)
        igl.addWidget(self.img_cy, 2, 3)
        igl.addWidget(QtWidgets.QLabel("Opacity"), 3, 0)
        igl.addWidget(self.img_opacity, 3, 1, 1, 3)
        self.img_lbl = QtWidgets.QLabel("no image loaded")
        self.img_lbl.setStyleSheet("color:#9aa0a8;")
        igl.addWidget(self.img_lbl, 4, 0, 1, 4)
        right.addWidget(ig)

        h.addLayout(left, 3)
        h.addLayout(right, 4)
        return w

    # ---- Peaks tab --------------------------------------------------------
    def _build_peaks_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        bar = QtWidgets.QHBoxLayout()
        self.peaks_rec = QtWidgets.QComboBox()
        self.peaks_region = QtWidgets.QComboBox()
        self.peaks_region.addItems(["whole", "pre", "stim", "post"])
        self.b_learn = QtWidgets.QPushButton("Learn templates")
        self.b_learn.clicked.connect(self._do_learn_templates)
        self.b_classify = QtWidgets.QPushButton("Detect && classify")
        self.b_classify.clicked.connect(self._do_classify)
        bar.addWidget(QtWidgets.QLabel("Recording"))
        bar.addWidget(self.peaks_rec, 1)
        bar.addWidget(QtWidgets.QLabel("Region"))
        bar.addWidget(self.peaks_region)
        bar.addWidget(self.b_classify)
        v.addLayout(bar)

        # explicit template-source recordings (so SOLAR comes from the 1-cell file)
        srcbar = QtWidgets.QHBoxLayout()
        self.src_solar = QtWidgets.QComboBox()
        self.src_piezo = QtWidgets.QComboBox()
        self.src_bio = QtWidgets.QComboBox()
        srcbar.addWidget(QtWidgets.QLabel("Learn templates from —  Solar (1 cell):"))
        srcbar.addWidget(self.src_solar, 1)
        srcbar.addWidget(QtWidgets.QLabel("Piezo (laser only):"))
        srcbar.addWidget(self.src_piezo, 1)
        srcbar.addWidget(QtWidgets.QLabel("Biological:"))
        srcbar.addWidget(self.src_bio, 1)
        srcbar.addWidget(self.b_learn)
        v.addLayout(srcbar)

        self.class_counts = QtWidgets.QLabel("No classification yet.")
        self.class_counts.setStyleSheet("color:#c3c7cf;")
        v.addWidget(self.class_counts)

        split = QtWidgets.QSplitter()
        # left: map coloured by biological count
        lw = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(lw)
        _pl = QtWidgets.QLabel("Classified biological peaks / electrode\n"
                               "(broadband detector — classification check).\n"
                               "Ringed = artifact-dominated electrodes.")
        _pl.setStyleSheet("color:#9aa0a8;")
        lv.addWidget(_pl)
        self.peaks_map = ElectrodeMap(pitch_um=self.project.pitch_um)
        lv.addWidget(self.peaks_map)
        split.addWidget(lw)
        # right: single-electrode trace inspector
        rw = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(rw)
        row = QtWidgets.QHBoxLayout()
        self.trace_elec = QtWidgets.QComboBox()
        self.trace_t0 = QtWidgets.QDoubleSpinBox()
        self.trace_t0.setRange(0, 1e6)
        self.trace_t0.setSuffix(" s")
        self.trace_dur = QtWidgets.QDoubleSpinBox()
        self.trace_dur.setRange(0.05, 60)
        self.trace_dur.setValue(2.0)
        self.trace_dur.setSuffix(" s")
        b_show = QtWidgets.QPushButton("Show trace")
        b_show.clicked.connect(self._show_trace)
        row.addWidget(QtWidgets.QLabel("Electrode"))
        row.addWidget(self.trace_elec)
        row.addWidget(QtWidgets.QLabel("t0"))
        row.addWidget(self.trace_t0)
        row.addWidget(QtWidgets.QLabel("window"))
        row.addWidget(self.trace_dur)
        row.addWidget(b_show)
        rv.addLayout(row)
        self.trace_plot = pg.PlotWidget()
        self.trace_plot.setLabel("bottom", "Time", units="s")
        self.trace_plot.setLabel("left", "Voltage", units="µV")
        rv.addWidget(self.trace_plot, 1)
        self.sta_plot = pg.PlotWidget()
        self.sta_plot.setLabel("bottom", "Time from flood", units="ms")
        self.sta_plot.setLabel("left", "µV")
        self.sta_plot.setMaximumHeight(180)
        rv.addWidget(self.sta_plot)
        split.addWidget(rw)
        split.setSizes([500, 900])
        v.addWidget(split, 1)
        return w

    # ---- Templates tab ----------------------------------------------------
    def _build_templates_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        bar = QtWidgets.QHBoxLayout()
        b_load = QtWidgets.QPushButton("Load templates…")
        b_load.clicked.connect(self._load_templates)
        b_save = QtWidgets.QPushButton("Save templates…")
        b_save.clicked.connect(self._save_templates)
        b_report = QtWidgets.QPushButton("Calibration report…")
        b_report.setToolTip("Write PNGs of each condition's wavelet (mean ± error "
                            "range) and the bior1.5 detection wavelet + scales to "
                            "a 'Calibration Report' subfolder of the output dir.")
        b_report.clicked.connect(self._do_calibration_report)
        bar.addWidget(b_load)
        bar.addWidget(b_save)
        bar.addWidget(b_report)
        bar.addStretch(1)
        v.addLayout(bar)
        self.tpl_info = QtWidgets.QLabel("No templates learned yet.")
        v.addWidget(self.tpl_info)
        self.tpl_glw = pg.GraphicsLayoutWidget()
        v.addWidget(self.tpl_glw, 1)
        return w

    # ---- Filter tab -------------------------------------------------------
    def _build_filter_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        bar = QtWidgets.QHBoxLayout()
        self.filter_rec = QtWidgets.QComboBox()
        self.filter_region = QtWidgets.QComboBox()
        self.filter_region.addItems(["whole", "pre", "stim", "post"])
        b_run = QtWidgets.QPushButton("Subtract artifacts && re-detect")
        b_run.clicked.connect(self._do_filter)
        b_exp = QtWidgets.QPushButton("Export for MEA Analyzer…")
        b_exp.setToolTip("Write this sample's cleaned biological spikes as "
                         "<stem>_spikes.npz (+ .mat) in the exact MEA-NAP format, "
                         "so MEA Analyzer / the raster viewer read and plot them.")
        b_exp.clicked.connect(self._export_clean)
        b_batch = QtWidgets.QPushButton("Batch filter folder…")
        b_batch.setToolTip("Filter EVERY recording in the data folder and write one "
                           "MEA-NAP <stem>_spikes.npz per recording.")
        b_batch.clicked.connect(self._do_batch_filter)
        self.filter_tighten = QtWidgets.QCheckBox("Tighten (200–3000 Hz + 5× thr)")
        self.filter_tighten.setToolTip(
            "AND the bior1.5 detections with a 2nd-order Butterworth 200–3000 Hz "
            "band-pass + 5× robust-noise amplitude threshold (MEA-NAP style), to "
            "reject low-amplitude non-biological detections.")
        self.filter_tighten.setChecked(
            bool(self.project.analysis.get("prefilter_enabled", True)))
        self.filter_tighten.toggled.connect(self._toggle_prefilter)
        bar.addWidget(QtWidgets.QLabel("Sample"))
        bar.addWidget(self.filter_rec, 1)
        bar.addWidget(QtWidgets.QLabel("Region"))
        bar.addWidget(self.filter_region)
        bar.addWidget(self.filter_tighten)
        bar.addWidget(b_run)
        bar.addWidget(b_exp)
        bar.addWidget(b_batch)
        v.addLayout(bar)
        # ---- plot output options -----------------------------------------
        obar = QtWidgets.QHBoxLayout()
        obar.addWidget(QtWidgets.QLabel("Metric"))
        self.filter_metric = QtWidgets.QComboBox()
        self.filter_metric.addItem("Spike rate", "spike")
        self.filter_metric.addItem("Burst rate", "burst")
        self.filter_metric.setToolTip("Colour the live map by biological spike rate "
                                      "(Hz) or burst rate (bursts/min). Both plots are "
                                      "saved on each run regardless.")
        obar.addWidget(self.filter_metric)
        obar.addSpacing(8)
        obar.addWidget(QtWidgets.QLabel("Plot:"))
        self.opt_rings = QtWidgets.QCheckBox("Artifact rings")
        self.opt_rings.setChecked(True)
        self.opt_log = QtWidgets.QCheckBox("Log colour")
        self.opt_log.setToolTip("Log colour scale (off = linear, capped at the "
                                "95th percentile).")
        self.opt_field = QtWidgets.QCheckBox("Solar field")
        self.opt_field.setToolTip("Overlay the solar-cell electric field "
                                  "(needs a placed cell). In the saved plot it is "
                                  "drawn on the STIM panel only.")
        self.opt_sph = QtWidgets.QCheckBox("Spheroid")
        self.opt_sph.setToolTip("Overlay the traced spheroid outline.")
        self.filter_fieldmode = QtWidgets.QComboBox()
        self.filter_fieldmode.addItem("modeled", "modeled")
        self.filter_fieldmode.addItem("measured", "measured")
        self.filter_fieldmode.setToolTip("Modeled = smooth field from the cell + "
                                         "measured amplitude; Measured = interpolated "
                                         "electrode response.")
        b_saveplot = QtWidgets.QPushButton("Save plot ⭳")
        b_saveplot.setToolTip("Write the pre/stim/post heatmap plot to "
                              "Output/Filtered Signal with the current options.")
        b_saveplot.clicked.connect(self._save_plot_clicked)
        for wdg in (self.opt_rings, self.opt_log, self.opt_field, self.opt_sph):
            wdg.toggled.connect(self._render_filter_map)
            obar.addWidget(wdg)
        self.filter_metric.currentIndexChanged.connect(self._render_filter_map)
        obar.addWidget(QtWidgets.QLabel("field:"))
        obar.addWidget(self.filter_fieldmode)
        self.filter_fieldmode.currentIndexChanged.connect(self._render_filter_map)
        obar.addStretch(1)
        obar.addWidget(b_saveplot)
        v.addLayout(obar)
        self.filter_info = QtWidgets.QLabel("No filtering run yet.")
        v.addWidget(self.filter_info)
        split = QtWidgets.QSplitter()
        lw = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(lw)
        _fl = QtWidgets.QLabel("Biological spike RATE (Hz) — validated wavelet\n"
                               "detector on the artifact-removed trace (the "
                               "biological output).\nRinged = artifact-subtracted "
                               "electrodes.")
        _fl.setStyleSheet("color:#9aa0a8;")
        lv.addWidget(_fl)
        self.filter_map = ElectrodeMap(pitch_um=self.project.pitch_um)
        lv.addWidget(self.filter_map)
        split.addWidget(lw)
        rw = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(rw)
        rv.addWidget(QtWidgets.QLabel("Per-electrode biological spike counts"))
        self.filter_plot = pg.PlotWidget()
        self.filter_plot.setLabel("bottom", "electrode #")
        self.filter_plot.setLabel("left", "spikes")
        rv.addWidget(self.filter_plot, 1)
        split.addWidget(rw)
        split.setSizes([500, 900])
        v.addWidget(split, 1)
        return w

    # ---- Field-map tab ----------------------------------------------------
    def _build_field_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        bar = QtWidgets.QHBoxLayout()
        self.field_rec = QtWidgets.QComboBox()
        self.field_baseline = QtWidgets.QComboBox()
        self.field_baseline.addItem("(none)", "")
        b_run = QtWidgets.QPushButton("Compute field map")
        b_run.clicked.connect(self._do_field)
        bar.addWidget(QtWidgets.QLabel("Recording (1 or 2 cells)"))
        bar.addWidget(self.field_rec, 1)
        bar.addWidget(QtWidgets.QLabel("Subtract baseline"))
        bar.addWidget(self.field_baseline, 1)
        bar.addWidget(b_run)
        v.addLayout(bar)
        self.field_info = QtWidgets.QLabel(
            "No field map computed yet. Use the recording's own placed cells; "
            "subtract a laser-only recording to isolate the pure solar field.")
        self.field_info.setWordWrap(True)
        v.addWidget(self.field_info)
        split = QtWidgets.QSplitter()
        self.field_img = pg.PlotWidget()
        self.field_img.getPlotItem().setAspectLocked(True)
        self.field_img.setLabel("bottom", "electrode column")
        self.field_img.setLabel("left", "row")
        self.field_image_item = pg.ImageItem()
        if self._lut is not None:
            self.field_image_item.setLookupTable(self._lut)
        self.field_img.addItem(self.field_image_item)
        self.field_solar = pg.PlotCurveItem(pen=pg.mkPen(255, 60, 60, 230, width=2))
        self.field_img.addItem(self.field_solar)
        # per-cell "max voltage detected" chips, created on demand in _fielded()
        self.field_cell_texts = []
        split.addWidget(self.field_img)
        self.field_decay = pg.PlotWidget()
        self.field_decay.setLabel("bottom", "distance from solar cell", units="µm")
        self.field_decay.setLabel("left", "flood response (p-p)", units="µV")
        split.addWidget(self.field_decay)
        split.setSizes([650, 750])
        v.addWidget(split, 1)
        return w

    # ---- Footprint tab ----------------------------------------------------
    def _build_footprint_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        bar = QtWidgets.QHBoxLayout()
        self.fp_rec = QtWidgets.QComboBox()
        self.fp_region = QtWidgets.QComboBox()
        self.fp_region.addItems(["whole", "pre", "stim", "post"])
        b_run = QtWidgets.QPushButton("Compute footprint")
        b_run.clicked.connect(self._do_footprint)
        bar.addWidget(QtWidgets.QLabel("Recording"))
        bar.addWidget(self.fp_rec, 1)
        bar.addWidget(QtWidgets.QLabel("Region"))
        bar.addWidget(self.fp_region)
        bar.addWidget(b_run)
        v.addLayout(bar)
        self.fp_info = QtWidgets.QLabel("No footprint computed yet.")
        v.addWidget(self.fp_info)
        split = QtWidgets.QSplitter()
        lw = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(lw)
        _fpl = QtWidgets.QLabel("Biological footprint — spike RATE (Hz), SAME "
                                "wavelet detection as the Filter tab.\n"
                                "Ringed = artifact-subtracted electrodes.")
        _fpl.setStyleSheet("color:#9aa0a8;")
        lv.addWidget(_fpl)
        self.fp_map = ElectrodeMap(pitch_um=self.project.pitch_um)
        lv.addWidget(self.fp_map)
        split.addWidget(lw)
        self.fp_decay = pg.PlotWidget()
        self.fp_decay.setLabel("bottom", "distance from spheroid centre", units="µm")
        self.fp_decay.setLabel("left", "spike rate", units="Hz")
        split.addWidget(self.fp_decay)
        split.setSizes([600, 800])
        v.addWidget(split, 1)
        return w

    # ======================================================================
    # setup-tab handlers
    # ======================================================================
    # -- persistent app settings (last folder, scale, pitch) ----------------
    def _remember(self, key, value):
        self._settings[key] = value
        _save_settings(self._settings)

    def _record_dir(self, path):
        """Remember the folder a dialog landed in, so the next one starts there."""
        if not path:
            return
        d = path if os.path.isdir(path) else os.path.dirname(path)
        if d:
            self._last_dir = d
            self._remember("last_dir", d)

    def _start_dir(self, *prefer):
        """First existing folder among the preferred ones, else the last-used one."""
        for p in prefer:
            if p and os.path.isdir(p):
                return p
        return self._last_dir or ""

    def _pick_data_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select the data folder (.h5 files + Batch.csv)",
            self._start_dir(self.project.data_dir))
        if d:
            self.project.data_dir = d
            self.data_edit.setText(d)
            self._record_dir(d)
            self._scan_recordings()

    def _pick_spike_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select the cleaned-spikes output folder",
            self._start_dir(self.project.spike_dir))
        if d:
            self.project.spike_dir = d
            self.spike_edit.setText(d)
            self._record_dir(d)

    def _load_seq(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load the flood stimulation .seq", self._start_dir(),
            "Polygon sequence (*.seq);;All files (*)")
        if not path:
            return
        self._record_dir(path)
        import seqfile
        info = seqfile.read_seq(path)
        if not info or info.get("total_ms", 0) <= 0:
            QtWidgets.QMessageBox.warning(
                self, "Could not read .seq",
                "This file isn't a recognisable Polygon .seq, or its timing "
                "couldn't be decoded. Set the flood train length manually.")
            return
        _silent(self.stim_train_spin, info["total_ms"])
        self.project.analysis["stim_train_ms"] = float(info["total_ms"])
        self.status.showMessage("Flood pattern: " + seqfile.describe(info))
        QtWidgets.QMessageBox.information(
            self, "Flood .seq loaded",
            "%s\n\nFlood train length set to %.0f ms — the artifact window and "
            "subtraction now span the whole pulse train."
            % (seqfile.describe(info), info["total_ms"]))

    def _on_pitch(self, val):
        self.project.pitch_um = float(val)
        self._remember("pitch_um", float(val))
        for m in (self.map, self.peaks_map, self.filter_map, self.fp_map):
            m.set_pitch(float(val))
        self._push_solar_map()
        if self._sph_points:
            self._rebuild_spheroid()
        elif self.project.spheroid:                    # legacy circle spheroid
            for m in self._maps():
                m.set_spheroid(self.project.spheroid_obj())
        self._apply_image()

    def _scan_recordings(self):
        """Fill the recordings table from the folder + Batch.csv."""
        data_dir = self.project.data_dir
        stems, batch_info = [], {}
        try:
            stems = pipeline.loaders.list_h5_stems(data_dir)
            csv = os.path.join(data_dir, "Batch.csv")
            if os.path.exists(csv):
                for r in pipeline.loaders.parse_batch(csv):
                    batch_info.setdefault(r["stem"], []).append(
                        f"{r['region']} {r['win'][0]:g}-{r['win'][1]:g}s "
                        f"(grp {r['group']})")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Scan failed",
                                          "Could not read the folder:\n%s" % exc)
            return
        self.rec_table.setRowCount(len(stems))
        for i, stem in enumerate(stems):
            self.rec_table.setItem(i, 0, QtWidgets.QTableWidgetItem(stem))
            info = "; ".join(batch_info.get(stem, [])) or "(not in Batch.csv)"
            self.rec_table.setItem(i, 1, QtWidgets.QTableWidgetItem(info))
            combo = QtWidgets.QComboBox()
            for c in proj.CONDITIONS:
                combo.addItem(proj.CONDITION_LABELS[c], c)
            cur = self.project.condition(stem)
            combo.setCurrentIndex(proj.CONDITIONS.index(cur)
                                  if cur in proj.CONDITIONS else len(proj.CONDITIONS) - 1)
            combo.currentIndexChanged.connect(
                lambda _i, s=stem, cb=combo:
                self.project.set_condition(s, cb.currentData()))
            self.rec_table.setCellWidget(i, 2, combo)
        self.rec_table.resizeColumnsToContents()
        self._refresh_rec_combos(stems)

    def _refresh_rec_combos(self, stems):
        for combo in (self.peaks_rec, self.filter_rec, self.field_rec, self.fp_rec):
            cur = combo.currentText()
            combo.clear()
            combo.addItems(stems)
            if cur in stems:
                combo.setCurrentText(cur)
        # laser-only baseline combo for the field map ("(none)" + all stems)
        curb = self.field_baseline.currentData()
        self.field_baseline.blockSignals(True)
        self.field_baseline.clear()
        self.field_baseline.addItem("(none)", "")
        for s in stems:
            self.field_baseline.addItem(s, s)
        i = self.field_baseline.findData(curb)
        self.field_baseline.setCurrentIndex(i if i >= 0 else 0)
        self.field_baseline.blockSignals(False)
        # template-source combos ("(auto)" + stems), preselecting the defaults
        try:
            defaults = pipeline.default_template_sources(self.project)
        except Exception:
            defaults = {}
        for cls, combo in ((tpl.SOLAR, self.src_solar),
                           (tpl.PIEZO, self.src_piezo), (tpl.BIO, self.src_bio)):
            cur = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("(auto)", "")
            for s in stems:
                combo.addItem(s, s)
            want = cur if cur in stems else defaults.get(cls, "")
            i = combo.findData(want) if want else 0
            combo.setCurrentIndex(i if i >= 0 else 0)
            combo.blockSignals(False)
        # solar-placement recording combo: preserve selection, then reload cells
        cur = self.solar_rec.currentText()
        self.solar_rec.blockSignals(True)
        self.solar_rec.clear()
        self.solar_rec.addItems(stems)
        if cur in stems:
            self.solar_rec.setCurrentText(cur)
        self.solar_rec.blockSignals(False)
        self._on_solar_rec_changed()

    def _maps(self):
        return (self.map, self.peaks_map, self.filter_map, self.fp_map)

    # -- solar cells (per-recording list; 0 / 1 / 2+ cells) -----------------
    def _current_solar_cells(self):
        return list(self.project.solar_list(self._solar_stem)) \
            if self._solar_stem else []

    def _cell_obj(self, d):
        return geo.SolarCell(
            center_grid=d["center_grid"], side_um=d["side_um"],
            orientation_deg=d.get("orientation_deg", 0.0),
            electrode_axis_deg=d.get("electrode_axis_deg"),
            pitch_um=self.project.pitch_um)

    def _push_solar_map(self):
        cells = [self._cell_obj(d) for d in self._current_solar_cells()]
        self.map.set_solars(cells, self._solar_sel)

    def _update_solar_label(self):
        cells = self._current_solar_cells()
        if not self._solar_stem:
            self.solar_lbl.setText("pick a recording, then Add cell / click the map")
        elif not cells:
            self.solar_lbl.setText("%s: no cells — Add cell or click the map"
                                   % self._solar_stem)
        else:
            self.solar_lbl.setText("%s: %d cell(s) placed  (click moves the "
                                   "selected cell)" % (self._solar_stem, len(cells)))

    def _rebuild_cells_combo(self):
        self.solar_cells_combo.blockSignals(True)
        self.solar_cells_combo.clear()
        cells = self._current_solar_cells()
        for i in range(len(cells)):
            self.solar_cells_combo.addItem("Cell %d" % (i + 1))
        if cells:
            self._solar_sel = min(max(self._solar_sel, 0), len(cells) - 1)
            self.solar_cells_combo.setCurrentIndex(self._solar_sel)
        else:
            self._solar_sel = -1
        self.solar_cells_combo.blockSignals(False)
        self._update_solar_label()

    def _load_cell_into_spins(self):
        cells = self._current_solar_cells()
        if not (0 <= self._solar_sel < len(cells)):
            return
        d = cells[self._solar_sel]
        cx, cy = geo.um_from_grid(d["center_grid"][0], d["center_grid"][1],
                                  self.project.pitch_um)
        _silent(self.solar_cx, cx)
        _silent(self.solar_cy, cy)
        _silent(self.solar_side, d["side_um"])
        _silent(self.solar_orient, d.get("orientation_deg", 0))
        _silent(self.solar_axis, d.get("electrode_axis_deg") or 0)

    def _on_solar_rec_changed(self, *a):
        self._solar_stem = self.solar_rec.currentText() or None
        self._solar_sel = len(self._current_solar_cells()) - 1
        self._rebuild_cells_combo()
        self._load_cell_into_spins()
        self._push_solar_map()

    def _on_solar_sel_changed(self, idx):
        self._solar_sel = idx
        self._load_cell_into_spins()
        self._push_solar_map()

    def _solar_add_cell(self):
        if not self._solar_stem:
            QtWidgets.QMessageBox.information(
                self, "Pick a recording",
                "Choose the recording to place cells on first "
                "(Solar cells → For recording).")
            return
        cells = self._current_solar_cells()
        grid = (geo.grid_from_um(self.solar_cx.value(), self.solar_cy.value(),
                                 self.project.pitch_um)
                if cells else geo.ARRAY_CENTER_GRID)
        axis = None if self.solar_axis.value() == 0 else self.solar_axis.value()
        cells.append(proj.Project.make_solar_dict(
            grid, self.solar_side.value() or 400, self.solar_orient.value(), axis))
        self._solar_sel = len(cells) - 1
        self.project.set_solar_list(self._solar_stem, cells)
        self._rebuild_cells_combo()
        self._load_cell_into_spins()
        self._push_solar_map()

    def _solar_del_cell(self):
        cells = self._current_solar_cells()
        if not (0 <= self._solar_sel < len(cells)):
            return
        cells.pop(self._solar_sel)
        self._solar_sel = len(cells) - 1
        self.project.set_solar_list(self._solar_stem, cells)
        self._rebuild_cells_combo()
        self._load_cell_into_spins()
        self._push_solar_map()

    def _solar_edited(self, *a):
        cells = self._current_solar_cells()
        if not (0 <= self._solar_sel < len(cells)):
            return
        grid = geo.grid_from_um(self.solar_cx.value(), self.solar_cy.value(),
                                self.project.pitch_um)
        axis = None if self.solar_axis.value() == 0 else self.solar_axis.value()
        cells[self._solar_sel] = proj.Project.make_solar_dict(
            grid, self.solar_side.value(), self.solar_orient.value(), axis)
        self.project.set_solar_list(self._solar_stem, cells)
        self._push_solar_map()
        self._update_solar_label()

    def _on_solar_placed(self, x, y):
        if not self._solar_stem:
            QtWidgets.QMessageBox.information(
                self, "Pick a recording",
                "Choose the recording to place cells on first "
                "(Solar cells → For recording).")
            return
        cells = self._current_solar_cells()
        if not (0 <= self._solar_sel < len(cells)):
            self._solar_add_cell()
            cells = self._current_solar_cells()
        d = dict(cells[self._solar_sel])
        d["center_grid"] = [float(x), float(y)]
        cells[self._solar_sel] = d
        self.project.set_solar_list(self._solar_stem, cells)
        self._load_cell_into_spins()
        self._push_solar_map()
        self._update_solar_label()

    # -- spheroid (hand-traced perimeter outline) ---------------------------
    def _on_spheroid_placed(self, x, y):
        """A perimeter point was clicked -> add it and refit the outline."""
        self._sph_points.append((float(x), float(y)))
        self._rebuild_spheroid()

    def _sph_undo(self):
        if self._sph_points:
            self._sph_points.pop()
        self._rebuild_spheroid()

    def _sph_clear(self):
        self._sph_points = []
        self.project.spheroid = None
        for m in self._maps():
            m.set_spheroid(None)
            m.set_spheroid_points([])
        self.sph_lbl.setText("no points")

    def _rebuild_spheroid(self):
        pts = self._sph_points
        if not pts:
            self._sph_clear()
            return
        self.project.set_spheroid_points(pts)
        obj = geo.SpheroidOutline(pts, pitch_um=self.project.pitch_um,
                                  smooth=self.sph_smooth.isChecked())
        for m in self._maps():
            m.set_spheroid(obj)
            m.set_spheroid_points(pts)
        if len(pts) >= 3:
            cg = obj.center_grid
            cum = geo.um_from_grid(cg[0], cg[1], self.project.pitch_um)
            self.sph_lbl.setText("%d points; centroid (%+.0f, %+.0f) µm"
                                 % (len(pts), cum[0], cum[1]))
        else:
            self.sph_lbl.setText("%d point(s) — click at least 3 to close the outline"
                                 % len(pts))

    # -- microscope image overlay -------------------------------------------
    def _load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load microscope image of the MEA surface", self._start_dir(),
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.gif)")
        if not path:
            return
        self._record_dir(path)
        try:
            arr = _load_image_array(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Could not load image",
                "%s\n\n(TIFF/other formats may need Pillow: "
                "..\\.venv\\Scripts\\python.exe -m pip install pillow)" % exc)
            return
        self._img_arr = arr
        self.map.set_image_array(arr)
        h, w = arr.shape[:2]
        # auto-read the physical scale (µm/px) from the image metadata if present
        scale_note = ""
        try:
            scale, src = _read_image_scale_um_per_px(path)
        except Exception:
            scale, src = None, None
        if scale and scale > 0:
            _silent(self.img_scale, scale)
            scale_note = "  •  scale %.4g µm/px from %s" % (scale, src)
        else:
            scale_note = "  •  no scale in metadata — set µm/px manually"
        self.img_lbl.setText("%s  (%d×%d px)%s"
                             % (os.path.basename(path), w, h, scale_note))
        self.project.set_image(
            path, um_per_px=self.img_scale.value(),
            rotation_deg=self.img_rot.value(), cx_um=self.img_cx.value(),
            cy_um=self.img_cy.value(), opacity=self.img_opacity.value() / 100.0,
            flip_y=self.img_flip.isChecked())
        self._apply_image()

    def _apply_image(self, *a):
        if self._img_arr is None:
            return
        self.map.set_image_params(
            um_per_px=self.img_scale.value(), rotation_deg=self.img_rot.value(),
            cx_um=self.img_cx.value(), cy_um=self.img_cy.value(),
            opacity=self.img_opacity.value() / 100.0,
            flip_y=self.img_flip.isChecked())
        if self.project.image:
            self.project.set_image(
                self.project.image["path"], um_per_px=self.img_scale.value(),
                rotation_deg=self.img_rot.value(), cx_um=self.img_cx.value(),
                cy_um=self.img_cy.value(),
                opacity=self.img_opacity.value() / 100.0,
                flip_y=self.img_flip.isChecked())

    def _fit_image(self):
        """Scale the image so its width spans the electrode grid (~1.4 mm) and
        centre it, as a starting point for manual alignment."""
        if self._img_arr is None:
            return
        h, w = self._img_arr.shape[:2]
        grid_span_um = 7.0 * self.project.pitch_um       # 8x8 grid edge-to-edge
        _silent(self.img_scale, grid_span_um / max(w, 1))
        _silent(self.img_cx, 0.0)
        _silent(self.img_cy, 0.0)
        _silent(self.img_rot, 0.0)
        self._apply_image()

    def _clear_image(self):
        self._img_arr = None
        self.map.set_image_array(None)
        self.project.image = None
        self.img_lbl.setText("no image loaded")

    def _on_image_placed(self, x, y):
        xum, yum = geo.um_from_grid(x, y, self.project.pitch_um)
        _silent(self.img_cx, xum)
        _silent(self.img_cy, yum)
        self._apply_image()

    # ======================================================================
    # project persistence
    # ======================================================================
    def _new_project(self):
        self.project = proj.Project()
        if self._settings.get("pitch_um"):
            self.project.pitch_um = float(self._settings["pitch_um"])
        self.templates = tpl.TemplateSet()
        self._raw_free_all()
        self._solar_stem = None
        self._solar_sel = -1
        self._sph_points = []
        self._img_arr = None
        for m in self._maps():
            m.set_solars([])
            m.set_spheroid(None)
            m.set_spheroid_points([])
            m.set_image_array(None)
            m.clear_heatmap()
        self.img_lbl.setText("no image loaded")
        self.sph_lbl.setText("no points")
        self._refresh_all()

    def _open_project(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open project", self._start_dir(),
            "Project (*.mscproj *.json)")
        if not path:
            return
        self._record_dir(path)
        try:
            self.project = proj.Project.load(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Open failed", str(exc))
            return
        if self.project.templates_path and os.path.exists(self.project.templates_path):
            try:
                self.templates = tpl.TemplateSet.load(self.project.templates_path)
            except Exception:
                pass
        self._refresh_all()

    def _save_project(self):
        default = self.project.name + ".mscproj"
        if self._last_dir:
            default = os.path.join(self._last_dir, default)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save project", default, "Project (*.mscproj)")
        if not path:
            return
        try:
            self.project.save(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(exc))
            return
        self._record_dir(path)
        self.setWindowTitle("MEA Solar Cell Analyzer — " + self.project.name)
        self.status.showMessage("Saved project: " + path)

    # -- live reload: restart the process on the freshly-pushed code ---------
    def _reload_restart(self):
        here = os.path.dirname(os.path.abspath(__file__))
        # 1) compile-check every script so a bad push can't leave you with nothing
        import py_compile
        import glob as _glob
        errs = []
        for f in sorted(_glob.glob(os.path.join(here, "*.py"))):
            try:
                py_compile.compile(f, doraise=True)
            except py_compile.PyCompileError as e:
                errs.append(str(e))
        if errs:
            QtWidgets.QMessageBox.critical(
                self, "Reload aborted — a script has an error",
                "Not restarting (your session is safe). Fix these first:\n\n"
                + "\n\n".join(errs)[:3000])
            return
        # 2) preserve the current project + learned templates to a temp folder
        import tempfile
        tmp = tempfile.gettempdir()
        proj_path = os.path.join(tmp, "_msca_reload.mscproj")
        try:
            if self.templates.names():
                tpath = os.path.join(tmp, "_msca_reload_templates.npz")
                self.templates.save(tpath)
                self.project.templates_path = tpath
            self.project.save(proj_path)
        except Exception as exc:
            if QtWidgets.QMessageBox.question(
                    self, "Could not save state",
                    "Restart anyway without preserving the current project?\n\n"
                    "(%s)" % exc) != _YES:
                return
            proj_path = ""
        # 3) launch a fresh instance (windowless via pythonw = sys.executable),
        #    then quit this one
        self._raw_free_all()
        script = os.path.join(here, "app.py")
        args = [script] + ([proj_path] if proj_path else [])
        ok = QtCore.QProcess.startDetached(sys.executable, args, here)
        if not ok:
            QtWidgets.QMessageBox.warning(
                self, "Restart failed",
                "Could not start a new instance with:\n%s\nStaying open."
                % sys.executable)
            return
        QtWidgets.QApplication.quit()

    def _refresh_all(self):
        self.data_edit.setText(self.project.data_dir)
        self.spike_edit.setText(self.project.spike_dir)
        _silent(self.pitch_spin, self.project.pitch_um)
        _silent(self.stim_train_spin,
                self.project.analysis.get("stim_train_ms", 250.0))
        for m in self._maps():
            m.set_pitch(self.project.pitch_um)
        # solar cells reload per-recording once the recording combos are filled
        # (see _refresh_rec_combos -> _on_solar_rec_changed, called from _scan).
        if self.project.spheroid and self.project.spheroid.get("points_grid"):
            self._sph_points = [tuple(p) for p in
                                self.project.spheroid["points_grid"]]
            self._rebuild_spheroid()
        elif self.project.spheroid:                    # legacy circle spheroid
            self._sph_points = []
            sp = self.project.spheroid_obj()
            for m in self._maps():
                m.set_spheroid(sp)
            self.sph_lbl.setText("legacy circle outline (re-trace to edit)")
        if self.project.image:
            im = self.project.image
            _silent(self.img_scale, im.get("um_per_px", 2.0))
            _silent(self.img_rot, im.get("rotation_deg", 0.0))
            _silent(self.img_cx, im.get("cx_um", 0.0))
            _silent(self.img_cy, im.get("cy_um", 0.0))
            self.img_opacity.blockSignals(True)
            self.img_opacity.setValue(int(im.get("opacity", 0.6) * 100))
            self.img_opacity.blockSignals(False)
            self.img_flip.blockSignals(True)
            self.img_flip.setChecked(im.get("flip_y", True))
            self.img_flip.blockSignals(False)
            path = im.get("path", "")
            if path and os.path.exists(path):
                try:
                    self._img_arr = _load_image_array(path)
                    self.map.set_image_array(self._img_arr)
                    h, w = self._img_arr.shape[:2]
                    self.img_lbl.setText("%s  (%d×%d px)" %
                                         (os.path.basename(path), w, h))
                    self._apply_image()
                except Exception:
                    self.img_lbl.setText("image not found: " + path)
        if self.project.data_dir and os.path.isdir(self.project.data_dir):
            self._scan_recordings()
        self._refresh_templates_view()

    # ======================================================================
    # heavy operations (worker thread)
    # ======================================================================
    def _busy(self, on, msg=""):
        self.prog.setVisible(on)
        for b in self.findChildren(QtWidgets.QPushButton):
            b.setEnabled(not on)
        if msg:
            self.status.showMessage(msg)
        QtWidgets.QApplication.processEvents()

    def _run(self, fn, on_done, label="Working…"):
        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.information(self, "Busy",
                                              "A task is already running.")
            return
        if not self.project.data_dir:
            QtWidgets.QMessageBox.warning(self, "No data folder",
                                          "Pick the data folder on the Setup tab first.")
            return
        self._busy(True, label)
        self.prog.setRange(0, 100)
        self.prog.setValue(0)
        self._worker = Worker(fn)
        self._worker.progress.connect(self._on_progress)
        self._worker.failed.connect(self._on_failed)

        def _finish(result):
            self._busy(False)
            try:
                on_done(result)
            except Exception:
                self._on_failed(traceback.format_exc())
        self._worker.done.connect(_finish)
        self._worker.start()

    def _on_progress(self, d, t, msg):
        self.prog.setRange(0, max(1, t))
        self.prog.setValue(d)
        if msg:
            self.status.showMessage(msg)

    def _on_failed(self, tb):
        self._busy(False)
        logpath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                               "error.log"))
        try:
            with open(logpath, "a", encoding="utf-8") as fh:
                fh.write(tb + "\n")
        except Exception:
            pass
        QtWidgets.QMessageBox.critical(self, "Task failed", tb[-2500:])

    # ---- learn templates --------------------------------------------------
    def _do_learn_templates(self):
        need = (self.project.stems_with_condition(proj.SOLAR_ONLY)
                or self.project.stems_with_condition(proj.LASER_ONLY)
                or self.project.stems_with_condition(proj.CULTURE_ONLY)
                or self.project.stems_with_condition(proj.SOLAR_CULTURE))
        if not need:
            QtWidgets.QMessageBox.warning(
                self, "Tag recordings first",
                "On the Setup tab, tag at least one recording as Solar-only, "
                "Laser-only, or Culture/Sample so templates can be learned.")
            return
        p = self.project
        try:
            defaults = pipeline.default_template_sources(p)
        except Exception:
            defaults = {}
        sources = {}
        for cls, combo in ((tpl.SOLAR, self.src_solar),
                           (tpl.PIEZO, self.src_piezo), (tpl.BIO, self.src_bio)):
            sel = combo.currentData()
            sources[cls] = sel or defaults.get(cls, "")
        if not any(sources.values()):
            QtWidgets.QMessageBox.warning(
                self, "No template sources",
                "Tag/choose at least one source recording (Solar 1-cell, "
                "Laser-only, or Culture/Sample).")
            return
        self._run(lambda prog: pipeline.learn_templates(p, sources=sources,
                                                        progress=prog),
                  self._learned, "Learning templates…")

    def _learned(self, result):
        ts, info = result
        self.templates = ts
        self._tpl_info = dict(info or {})
        self._refresh_templates_view(info)
        self.status.showMessage("Templates learned: " +
                                "; ".join(f"{k}={v}" for k, v in info.items()))

    def _refresh_templates_view(self, info=None):
        self.tpl_glw.clear()
        names = self.templates.names()
        if not names:
            self.tpl_info.setText("No templates learned yet. Use the Peaks tab "
                                  "→ Learn templates.")
            return
        # common x-range for the PEAK templates only; the 250 ms onset trains
        # (e.g. solar) auto-range to their own span so they don't squash the peaks.
        peak_ts = [self.templates.get(nm) for nm in names
                   if getattr(self.templates.get(nm), "aligned", "peak") == "peak"]
        if peak_ts:
            xmin = min(float(t.t_ms[0]) for t in peak_ts)
            xmax = max(float(t.t_ms[-1]) for t in peak_ts)
        else:
            xmin = xmax = None
        txt = []
        first_plt = None
        for i, name in enumerate(names):
            t = self.templates.get(name)
            onset = getattr(t, "aligned", "peak") == "onset"
            plt = self.tpl_glw.addPlot(row=0, col=i)
            ttl = (f"{name} — 250 ms train  (n={t.n})" if onset
                   else f"{name}  (n={t.n})")
            plt.setTitle(ttl)
            col = CLASS_COLORS.get(name, (200, 200, 200))
            plt.plot(t.t_ms, t.mean_uv, pen=pg.mkPen(*col, width=2))
            if t.std_uv.size == t.mean_uv.size:
                plt.plot(t.t_ms, t.mean_uv + t.std_uv, pen=pg.mkPen(*col, 90))
                plt.plot(t.t_ms, t.mean_uv - t.std_uv, pen=pg.mkPen(*col, 90))
            xlab = "ms from flood onset" if onset else "ms from peak"
            plt.setLabel("bottom", xlab)
            plt.setLabel("left", "µV")
            if onset or xmin is None:
                plt.enableAutoRange(axis="x")    # 250 ms train: own axis
            else:
                plt.setXRange(xmin, xmax, padding=0.02)
                if first_plt is None:
                    first_plt = plt
                else:
                    plt.setXLink(first_plt)      # keep the peak axes synced
            # second row: the flood-onset EVENT waveform (both transients) for
            # solar/piezo — display only; the top row is what classifies peaks.
            if getattr(t, "has_event", False):
                ep = self.tpl_glw.addPlot(row=1, col=i)
                ep.setTitle(f"{name} flood event")
                ep.plot(t.event_t_ms, t.event_uv, pen=pg.mkPen(*col, width=2))
                if t.event_std is not None and t.event_std.size == t.event_uv.size:
                    ep.plot(t.event_t_ms, t.event_uv + t.event_std, pen=pg.mkPen(*col, 90))
                    ep.plot(t.event_t_ms, t.event_uv - t.event_std, pen=pg.mkPen(*col, 90))
                ep.setLabel("bottom", "ms from flood onset")
                ep.setLabel("left", "µV")
            txt.append(f"{name}: n={t.n}, |amp|~{t.amp_med:.1f}µV, "
                       f"width~{t.width_med:.2f}ms" +
                       (f"  [{info[name]}]" if info and name in info else ""))
        self.tpl_info.setText(
            "Solar = whole 250 ms flood train (no single solar peak). Piezo/bio top "
            "row = peak template used for classification; piezo second row = flood "
            "event (display).   " + "   |   ".join(txt))

    # ---- classify ---------------------------------------------------------
    def _do_classify(self):
        stem = self.peaks_rec.currentText()
        if not stem:
            return
        if not self.templates.names():
            QtWidgets.QMessageBox.warning(self, "No templates",
                                          "Learn templates first.")
            return
        region = self.peaks_region.currentText()
        p = self.project

        def job(prog):
            raw = self._get_raw(stem)
            win = None if region == "whole" else \
                pipeline.region_window(p.data_dir, stem, region)
            on, off, dur = pipeline.flood_on_off(p.data_dir, stem, win)
            results, counts = pipeline.classify_recording(
                raw, on, off, dur, self.templates, p, stem, window=win,
                progress=prog)
            return results, counts, stem
        self._run(job, self._classified, "Detecting && classifying…")

    def _classified(self, result):
        results, counts, stem = result
        self._classify = (results, counts, stem)
        self.peaks_map.set_solars(self.project.solar_cells(stem))
        total = sum(counts.values())
        parts = []
        for c in (clf.SOLAR, clf.PIEZO, clf.BIO, "unknown"):
            parts.append(f"{c}: {counts.get(c, 0)}")
        self.class_counts.setText(f"{stem}: {total} peaks   |   " +
                                  "   ".join(parts))
        bio_by = {lab: int(res.mask(clf.BIO).sum())
                  for lab, res in results.items()}
        # ring convention (consistent with Filter/Footprint): rings = artifact
        # electrodes. Here that's electrodes where solar/piezo artifact peaks
        # dominate over biological ones.
        art_marks = []
        for lab, res in results.items():
            na = int(res.mask(clf.SOLAR).sum() + res.mask(clf.PIEZO).sum())
            nb = int(res.mask(clf.BIO).sum())
            if na > 0 and na >= max(3, nb):
                art_marks.append(lab)
        self.peaks_map.set_heatmap(bio_by, marks=art_marks)
        self.trace_elec.clear()
        self.trace_elec.addItems(sorted(results.keys(), key=_lab_key))

    def _show_trace(self):
        if not self._classify:
            QtWidgets.QMessageBox.information(self, "Classify first",
                                              "Run Detect && classify first.")
            return
        results, _, stem = self._classify
        lab = self.trace_elec.currentText()
        if lab not in results:
            return
        p = self.project
        t0 = self.trace_t0.value()
        dur = self.trace_dur.value()
        try:
            raw = self._get_raw(stem)
            trace, off = raw.channel_window(lab, t0, t0 + dur)
            floods, floods_off, flood_dur = pipeline.flood_on_off(
                p.data_dir, stem, (t0, t0 + dur))
        except Exception as exc:
            self._on_failed(str(exc))
            return
        import peaks as pk
        filt = pk.highpass(trace, raw.fs, p.analysis["hp_hz"])
        tt = off + np.arange(filt.size) / raw.fs
        self.trace_plot.clear()
        self.trace_plot.plot(tt, filt, pen=pg.mkPen(180, 185, 195, 200))
        for f in floods:
            self.trace_plot.addItem(pg.InfiniteLine(
                pos=float(f), angle=90, pen=pg.mkPen(0, 220, 120, 140)))
        res = results[lab]
        for c, col in CLASS_COLORS.items():
            m = res.mask(c) & (res.peaks.t >= t0) & (res.peaks.t <= t0 + dur)
            if m.any():
                xs = res.peaks.t[m]
                ys = res.peaks.amp[m]
                self.trace_plot.plot(xs, ys, pen=None, symbol="o",
                                     symbolBrush=pg.mkBrush(*col),
                                     symbolPen=None, symbolSize=9, name=c)
        # flood-onset-triggered average over the whole exposure (both transients)
        A = self.project.analysis
        post = (flood_dur * 1000.0 + A.get("event_tail_ms", 15.0) + 10.0
                if flood_dur > 0 else A["lock_post_ms"] + 10.0)
        sta, t_ms, n, p2p = pk.flood_triggered_average(
            trace, raw.fs, floods, pre_ms=5, post_ms=post, t_offset=off)
        self.sta_plot.clear()
        self.sta_plot.plot(t_ms, sta, pen=pg.mkPen(255, 150, 40, 230, width=2))
        self.sta_plot.setTitle(f"{lab}: flood-onset average (n={n}, "
                               f"p-p={p2p:.1f}µV, dur≈{flood_dur*1000:.0f}ms)")

    # ---- filter -----------------------------------------------------------
    def _do_filter(self):
        stem = self.filter_rec.currentText()
        if not stem:
            return
        region = self.filter_region.currentText()
        p = self.project

        def job(prog):
            win = None if region == "whole" else \
                pipeline.region_window(p.data_dir, stem, region)
            res, s, dur = pipeline.filter_sample(p, stem=stem, window=win,
                                                 progress=prog)
            return res, s, dur
        self._run(job, self._filtered, "Subtracting artifacts && re-detecting…")

    def _filtered(self, result):
        res, stem, dur = result
        self._filter = (res, stem, dur)
        labels = sorted(res.bio_spikes.keys(), key=_lab_key)
        counts = [len(res.bio_spikes[l]) for l in labels]
        self.filter_plot.clear()
        bg = pg.BarGraphItem(x=np.arange(len(labels)), height=counts, width=0.8,
                             brush=pg.mkBrush(120, 230, 130, 200))
        self.filter_plot.addItem(bg)
        self._render_filter_map()
        paths = self._save_filter_plot()          # saves BOTH spike + burst plots
        extra = ("  Plots → %s" % os.path.dirname(paths[0])) if paths else ""
        span = getattr(res, "stim_span_ms", None)
        span_txt = ""
        if span and self.project.analysis.get("stim_span_mode", "auto") == "auto":
            span_txt = f" Artifact window auto-detected = {span:.0f} ms."
        self.filter_info.setText(
            f"{stem}: {res.total_spikes()} biological spikes over {dur:.0f}s; "
            f"{len(res.artifact_labels)} artifact-contaminated electrode(s) "
            f"subtracted (ringed).{span_txt}{extra}")

    def _toggle_prefilter(self, on):
        self.project.analysis["prefilter_enabled"] = bool(on)

    def _toggle_stim_auto(self, on):
        self.project.analysis["stim_span_mode"] = "auto" if on else "fixed"
        self.stim_train_spin.setEnabled(not on)

    def _render_filter_map(self, *_):
        """Redraw the in-app Filter map from the current result honouring the plot
        checkboxes (rings / log colour / field overlay / spheroid)."""
        if not getattr(self, "filter_map", None) or not self._filter:
            return
        res, stem, dur = self._filter
        cells, cvals, clabs = pipeline.solar_cell_peaks(self.project, res, stem)
        _c2, cplat, _pl = pipeline.solar_cell_plateaus(self.project, res, stem)
        self.filter_map.set_solars(cells, values=cvals, value_labels=clabs,
                                   plateaus=cplat)
        self.filter_map.set_spheroid(
            self.project.spheroid_obj() if self.opt_sph.isChecked() else None)
        marks = sorted(res.artifact_labels) if self.opt_rings.isChecked() else []
        if self.filter_metric.currentData() == "burst":
            vals = pipeline.burst_rate_map(res, dur, self.project.analysis)
        else:
            vals = res.rate_map(dur)
        for lab in getattr(res, "grounded", ()):     # grounded electrodes -> OFF
            vals[lab] = float("nan")
        self.filter_map.set_heatmap(vals, marks=marks,
                                    log=self.opt_log.isChecked())
        if self.opt_field.isChecked():
            try:
                fld = pipeline.field_overlay(
                    self.project, res, stem,
                    mode=self.filter_fieldmode.currentData())
            except Exception:
                fld = None
            if fld is not None:
                self.filter_map.set_field_overlay(*fld)
            else:
                self.filter_map.clear_field_overlay()
                if not self.project.solar_cells(stem):
                    self.status.showMessage("Field overlay needs a placed solar "
                                            "cell on " + stem)
        else:
            self.filter_map.clear_field_overlay()

    def _output_root(self):
        """The 'Output' folder that holds SpikeData — where Filtered Signal lives."""
        sd = self.project.spike_dir
        if sd and os.path.basename(os.path.normpath(sd)).lower() == "spikedata":
            return os.path.dirname(os.path.normpath(sd))
        return sd or self.project.data_dir or ""

    def _save_plot_clicked(self):
        if not self._filter:
            QtWidgets.QMessageBox.information(self, "Nothing to plot",
                                              "Run the filter first.")
            return
        paths = self._save_filter_plot()
        if paths:
            self.status.showMessage("Saved %d plot(s) to %s"
                                    % (len(paths), os.path.dirname(paths[0])))

    def _save_filter_plot(self):
        """Write BOTH the spike-rate and burst-rate pre/stim/post heatmap plots to
        Output/Filtered Signal. Returns the list of written paths."""
        if not self._filter:
            return []
        paths = []
        for metric in ("spike", "burst"):
            p = self._save_one_metric_plot(metric)
            if p:
                paths.append(p)
        return paths

    def _save_one_metric_plot(self, metric):
        """One metric's pre/stim/post triptych (field on the STIM panel only),
        honouring the plot checkboxes. metric = 'spike' | 'burst'."""
        res, stem, dur = self._filter
        root = self._output_root()
        if not root:
            return None
        A = self.project.analysis
        burst = (metric == "burst")
        value_label = "burst rate (bursts/min)" if burst else "rate (Hz)"
        suffix = "_filtered_burst_heatmap.png" if burst else "_filtered_heatmap.png"
        title = "%s — biological %s" % (stem, "burst rate" if burst
                                        else "spike rate")

        def region_maps(windows):
            return (pipeline.region_burst_rate_maps(res, windows, A) if burst
                    else pipeline.region_rate_maps(res, windows))

        def whole_map():
            m = (pipeline.burst_rate_map(res, dur, A) if burst
                 else res.rate_map(dur))
            for lab in getattr(res, "grounded", ()):
                m[lab] = float("nan")
            return m
        try:
            import heatmap_export
            marks = sorted(res.artifact_labels) if self.opt_rings.isChecked() else []
            log = self.opt_log.isChecked()
            sph = None
            if self.opt_sph.isChecked() and self.project.spheroid_obj() is not None:
                try:
                    sph = np.asarray(self.project.spheroid_obj().outline_grid())
                except Exception:
                    sph = None
            field = None
            if self.opt_field.isChecked():
                try:
                    field = pipeline.field_overlay(
                        self.project, res, stem,
                        mode=self.filter_fieldmode.currentData())
                except Exception:
                    field = None
            # placed solar cells + the max voltage detected on each cell's own
            # (nearest) electrodes — drawn on top of every panel.
            try:
                cells, cvals, clabs = pipeline.solar_cell_peaks(
                    self.project, res, stem)
            except Exception:
                cells, cvals, clabs = [], [], []
            try:
                _c2, cplat, _pl = pipeline.solar_cell_plateaus(
                    self.project, res, stem)
            except Exception:
                cplat = []
            out_dir = os.path.join(root, "Filtered Signal")
            path = os.path.join(out_dir, stem + suffix)
            # pre/stim/post from the DIGITAL-EVENT trace (laser), not the region
            # column; fall back to the Batch.csv regions if there are no events.
            try:
                full_dur = float(pipeline.loaders.file_duration(
                    self.project.data_dir, stem))
            except Exception:
                full_dur = dur
            windows = pipeline.event_regions(self.project.data_dir, stem, full_dur, A)
            if not windows:
                windows = pipeline.region_windows(self.project.data_dir, stem)
            if windows:
                rmaps = region_maps(windows)
                # pre & post get NO field overlay; only stim has the laser field.
                regions = [(r.capitalize(), rmaps[r],
                            field if r == "stim" else None)
                           for r in ("pre", "stim", "post") if r in rmaps]
            else:
                regions = [("%s (%.0fs)" % (stem, dur), whole_map(), field)]
            heatmap_export.save_region_heatmaps(
                path, regions, suptitle=title, marks=marks, spheroid_outline=sph,
                log=log, value_label=value_label, solar_cells=cells,
                cell_values=cvals, cell_labels=clabs, cell_plateaus=cplat)
            return path
        except Exception:
            self._on_failed(traceback.format_exc())
            return None

    def _export_clean(self):
        if not self._filter:
            QtWidgets.QMessageBox.information(self, "Nothing to export",
                                              "Run the filter first.")
            return
        res, stem, dur = self._filter
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose the folder for the MEA-NAP spike file "
                  "(<stem>_spikes.npz)",
            self._start_dir(self.project.spike_dir))
        if not out_dir:
            return
        self._record_dir(out_dir)
        if not self.project.spike_dir:
            self.project.spike_dir = out_dir
            self.spike_edit.setText(out_dir)
        try:
            noise = {lab: res.sta_bank[lab]["noise"] for lab in res.bio_spikes
                     if lab in res.sta_bank}
            npz_path = pipeline.loaders.save_meanap_spike_file(
                out_dir, stem, res.bio_spikes, res.fs, noise_by_label=noise,
                extra_params={"artifact_labels": sorted(res.artifact_labels),
                              "duration_s": float(dur)})
            self.status.showMessage("Exported MEA-NAP spike file: " + npz_path)
            QtWidgets.QMessageBox.information(
                self, "Exported for MEA Analyzer",
                "Wrote MEA-NAP spike files (readable/plottable in MEA Analyzer):\n\n"
                "%s\n%s\n\n%d biological spikes across %d electrodes."
                % (npz_path, npz_path[:-4] + ".mat", res.total_spikes(),
                   len(res.bio_spikes)))
        except Exception:
            self._on_failed(traceback.format_exc())

    def _do_batch_filter(self):
        p = self.project
        if not p.data_dir:
            QtWidgets.QMessageBox.warning(self, "No data folder",
                                          "Pick the data folder on the Setup tab first.")
            return
        try:
            stems = pipeline.loaders.list_h5_stems(p.data_dir)
        except Exception:
            stems = []
        if not stems:
            QtWidgets.QMessageBox.warning(self, "No recordings",
                                          "No .h5 recordings found in the data folder.")
            return
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose the output folder for the MEA-NAP spike files "
                  "(one <stem>_spikes.npz per recording)",
            self._start_dir(p.spike_dir))
        if not out_dir:
            return
        self._record_dir(out_dir)
        if not p.spike_dir:
            p.spike_dir = out_dir
            self.spike_edit.setText(out_dir)
        if QtWidgets.QMessageBox.question(
                self, "Batch filter",
                "Filter all %d recording(s) in the data folder and write a "
                "MEA-NAP <stem>_spikes.npz for each into:\n\n%s\n\nThis can take a "
                "while. Proceed?" % (len(stems), out_dir),
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.Yes
                ) != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._batch_out = out_dir

        def job(prog):
            return pipeline.batch_filter(p, out_dir, data_dir=p.data_dir,
                                         stems=stems, progress=prog)
        self._run(job, self._batch_done,
                  "Batch filtering %d recording(s)…" % len(stems))

    def _batch_done(self, results):
        ok = [r for r in results if r["error"] is None]
        bad = [r for r in results if r["error"] is not None]
        total_spikes = sum(r["n_spikes"] for r in ok)
        lines = ["%s → %d spikes (%d artifact ch)"
                 % (r["stem"], r["n_spikes"], r["n_artifact_ch"]) for r in ok]
        if bad:
            lines.append("")
            lines += ["FAILED  %s: %s" % (r["stem"], r["error"]) for r in bad]
        self.filter_info.setText(
            "Batch filtered %d recording(s): %d biological spikes total, %d failed. "
            "Wrote MEA-NAP <stem>_spikes.npz to %s"
            % (len(ok), total_spikes, len(bad), self._batch_out))
        self.status.showMessage("Batch filter complete: %d ok, %d failed."
                                % (len(ok), len(bad)))
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning if bad
                    else QtWidgets.QMessageBox.Icon.Information)
        box.setWindowTitle("Batch filter complete")
        box.setText("Filtered %d of %d recording(s); wrote MEA-NAP spike files to:\n%s"
                    % (len(ok), len(results), self._batch_out))
        box.setDetailedText("\n".join(lines))
        box.exec()
        try:
            if sys.platform.startswith("win"):
                os.startfile(self._batch_out)      # noqa
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", self._batch_out])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", self._batch_out])
        except Exception:
            pass

    # ---- field map --------------------------------------------------------
    def _do_field(self):
        stem = self.field_rec.currentText()
        if not stem:
            return
        if not self.project.solar_cells(stem):
            QtWidgets.QMessageBox.warning(
                self, "No solar cell placed",
                "Place at least one solar cell on '%s' first "
                "(Setup tab → Solar cells → For recording)." % stem)
            return
        baseline = self.field_baseline.currentData() or None
        if baseline == stem:
            baseline = None
        p = self.project
        self._run(lambda prog: pipeline.field_map_from_solar(
            p, stem=stem, baseline_stem=baseline, progress=prog),
            self._fielded, "Computing field map…")

    def _draw_field_cell_values(self, fmres, plateaus=None):
        """Write each solar cell's MAX detected voltage on top of that cell in the
        Field-map image, with the SUSTAINED plateau (if available) on a small line
        beneath it. Each electrode belongs to the cell it is nearest to, so N cells
        get N independent numbers."""
        import electrode_map as _em
        vals = fmres.cell_peaks
        labs = fmres.cell_peak_labels()
        plats = list(plateaus or [])
        wanted = []
        for i, cell in enumerate(fmres.cells):
            v = vals[i] if i < len(vals) else float("nan")
            if not np.isfinite(v):
                continue
            txt = _em._fmt_uv(v)
            if i < len(labs) and labs[i]:
                txt += "\n@ %s" % labs[i]
            pl = plats[i] if i < len(plats) else None
            if pl is not None and np.isfinite(float(pl)):
                txt += "\nplat %s" % _em._fmt_uv(float(pl))
            c = np.asarray(cell.corners_grid(), float)
            wanted.append((float(np.mean(c[:, 0])),
                           float(np.max(c[:, 1])) + 0.28, txt))
        while len(self.field_cell_texts) < len(wanted):
            t = pg.TextItem(anchor=(0.5, 1.0), color=(251, 191, 36),
                            fill=pg.mkBrush(0, 0, 0, 185),
                            border=pg.mkPen(255, 60, 60, 200))
            t.setZValue(45)
            self.field_img.addItem(t)
            self.field_cell_texts.append(t)
        for t, (x, y, txt) in zip(self.field_cell_texts, wanted):
            t.setText(txt)
            t.setPos(x, y)
            t.setVisible(True)
        for t in self.field_cell_texts[len(wanted):]:
            t.setVisible(False)

    def _fielded(self, result):
        fmres, stem, bank = result
        self._field = fmres
        arr = self.project.array()
        Z, ext = fmres.interpolate(arr, n=140)
        self.field_image_item.setImage(np.nan_to_num(Z), autoLevels=True)
        self.field_image_item.setRect(QtCore.QRectF(
            ext[0], ext[2], ext[1] - ext[0], ext[3] - ext[2]))
        # draw all cells (concatenated with NaN breaks)
        xs, ys = [], []
        for cell in fmres.cells:
            c = cell.corners_grid()
            xs += list(np.append(c[:, 0], c[0, 0])) + [np.nan]
            ys += list(np.append(c[:, 1], c[0, 1])) + [np.nan]
        self.field_solar.setData(xs, ys, connect="finite")
        # sustained plateau per cell (held photovoltage), from the STA bank
        cplat = [float("nan")] * len(fmres.cells)
        try:
            plat = {lab: bank[lab]["plateau"] for lab in bank
                    if "plateau" in bank[lab] and np.isfinite(bank[lab]["plateau"])}
            if plat:
                precs = geo.max_by_cell(fmres.cells, plat, self.project.array(),
                                        by_magnitude=True)
                cplat = [r["value"] for r in precs]
        except Exception:
            pass
        self._field_cell_plateaus = cplat
        self._draw_field_cell_values(fmres, plateaus=cplat)
        self.field_decay.clear()
        self.field_decay.plot(fmres.dist, fmres.resp, pen=None, symbol="o",
                              symbolSize=7, symbolBrush=pg.mkBrush(120, 200, 255, 200))
        a0, lam, base = fmres.fit
        dmax = fmres.dist.max() if fmres.dist.size else 1.0
        dd = np.linspace(0, dmax, 100)
        self.field_decay.plot(dd, a0 * np.exp(-dd / lam) + base,
                              pen=pg.mkPen(255, 150, 40, 230, width=2))
        self.field_decay.addItem(pg.InfiniteLine(
            pos=fmres.r_eff, angle=90, pen=pg.mkPen(255, 60, 60, 200,
                                                    style=QtCore.Qt.PenStyle.DashLine)))
        base_note = ""
        if self.field_baseline.currentData():
            base_note = " (laser-only baseline subtracted)"
        _pk = fmres.cell_peaks
        _pl = fmres.cell_peak_labels()
        peak_txt = "  Max per cell: " + ", ".join(
            "#%d %s%s" % (i + 1,
                          ("%.0f µV" % v) if np.isfinite(v) else "n/a",
                          (" @%s" % _pl[i]) if _pl[i] else "")
            for i, v in enumerate(_pk)) if _pk else ""
        _plat = getattr(self, "_field_cell_plateaus", None)
        plat_txt = ""
        if _plat and any(np.isfinite(v) for v in _plat):
            plat_txt = "  Sustained (plateau) per cell: " + ", ".join(
                "#%d %s" % (i + 1, ("%.1f µV" % v) if np.isfinite(v) else "n/a")
                for i, v in enumerate(_plat))
        self.field_info.setText(
            f"{stem}: {fmres.n_cells} cell(s){base_note}. λ = {lam:.0f} µm; "
            f"effective radius (field ≥ {fmres.floor_uv:.1f} µV) = "
            f"{fmres.r_eff:.0f} µm; anisotropy ∥/⊥ = {fmres.anisotropy:.2f} "
            f"(λ∥={fmres.lam_along:.0f}, λ⊥={fmres.lam_across:.0f} µm). "
            f"Field peak {fmres.peak_uv:.1f} µV, mean {fmres.mean_uv:.1f} µV, "
            f"total {fmres.total_uv:.0f} µV — compare 1-cell vs 2-cell to see "
            f"amplification." + peak_txt + plat_txt)

    # ---- footprint --------------------------------------------------------
    def _do_footprint(self):
        stem = self.fp_rec.currentText()
        if not stem:
            return
        region = self.fp_region.currentText()
        p = self.project

        def job(prog):
            win = None if region == "whole" else \
                pipeline.region_window(p.data_dir, stem, region)
            res, s, dur = pipeline.filter_sample(p, stem=stem, window=win,
                                                 progress=prog)
            fpres = pipeline.footprint_from_filter(p, res, dur)
            return fpres, s, sorted(res.artifact_labels)
        self._run(job, self._footprinted, "Computing footprint…")

    def _footprinted(self, result):
        fpres, stem, artifact_labels = result
        arr = self.project.array()
        self.fp_map.set_solars(self.project.solar_cells(stem))
        rate_by = {lab: fpres.rate[i] for i, lab in enumerate(fpres.labels)}
        # ring convention (consistent everywhere): rings = artifact-subtracted.
        self.fp_map.set_heatmap(rate_by, marks=artifact_labels)
        self.fp_decay.clear()
        d, r = fpres.rate_vs_distance()
        if d.size:
            self.fp_decay.plot(d, r, pen=None, symbol="o", symbolSize=8,
                               symbolBrush=pg.mkBrush(120, 230, 130, 210))
        reach = (f"{fpres.reach_um:.0f} µm" if fpres.spheroid is not None
                 else "place a spheroid to measure reach")
        self.fp_info.setText(
            f"{stem}: {fpres.n_active()} active electrode(s); footprint spread "
            f"(radius of gyration) = {fpres.spread_um:.0f} µm; spiking reach from "
            f"soma centre = {reach}.")

    # ---- raw-recording cache ---------------------------------------------
    def _get_raw(self, stem):
        if stem not in self._raw_cache:
            self._raw_free_all()
            self._raw_cache[stem] = pipeline.load_raw(self.project.data_dir, stem)
        return self._raw_cache[stem]

    def _raw_free_all(self):
        for r in self._raw_cache.values():
            try:
                r.free()
            except Exception:
                pass
        self._raw_cache = {}

    # ---- templates save/load ---------------------------------------------
    def _save_templates(self):
        if not self.templates.names():
            QtWidgets.QMessageBox.information(self, "No templates",
                                              "Learn templates first.")
            return
        default = os.path.join(self._last_dir, "templates.npz") if self._last_dir \
            else "templates.npz"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save templates", default, "NumPy archive (*.npz)")
        if not path:
            return
        self._record_dir(path)
        self.templates.save(path)
        self.project.templates_path = path
        self.status.showMessage("Saved templates: " + path)

    def _load_templates(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load templates", self._start_dir(), "NumPy archive (*.npz)")
        if not path:
            return
        self._record_dir(path)
        try:
            self.templates = tpl.TemplateSet.load(path)
            self.project.templates_path = path
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
            return
        self._refresh_templates_view()

    # ---- calibration report ----------------------------------------------
    def _do_calibration_report(self):
        if not self.templates.names():
            QtWidgets.QMessageBox.information(
                self, "No templates",
                "Learn (or load) templates first — the report plots each "
                "condition's wavelet + error range.")
            return
        out_root = self.project.spike_dir
        if not (out_root and os.path.isdir(out_root)):
            out_root = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Choose the output folder for the Calibration Report",
                self._start_dir())
            if not out_root:
                return
            self._record_dir(out_root)
        # sampling rate from a learned template (else default)
        fs = 20000.0
        for name in self.templates.names():
            t = self.templates.get(name)
            if t is not None and t.fs > 0:
                fs = t.fs
                break
        try:
            info = pipeline.loaders.detection_wavelet_info(fs)
        except Exception:
            info = {"fs": fs}
        self._busy(True, "Writing calibration report…")
        try:
            import calibration
            rep, files = calibration.generate_report(
                self.templates, out_root, wavelet_info=info,
                sources=self._tpl_info)
        except Exception:
            self._busy(False)
            self._on_failed(traceback.format_exc())
            return
        self._busy(False)
        self.status.showMessage("Calibration report written to: " + rep)
        QtWidgets.QMessageBox.information(
            self, "Calibration report",
            "Wrote %d file(s) to:\n%s" % (len(files), rep))
        try:                                   # open the folder for convenience
            if sys.platform.startswith("win"):
                os.startfile(rep)              # noqa
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", rep])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", rep])
        except Exception:
            pass

    def closeEvent(self, ev):
        self._raw_free_all()
        super().closeEvent(ev)


def _lab_key(lab):
    s = str(lab)
    return (int(s) if s.isdigit() else 9999, s)


SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "settings.json")


def _load_settings():
    """App-level settings persisted across sessions (last folder, scale, pitch),
    independent of any project file."""
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_settings(d):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=2)
    except Exception:
        pass


def _silent(spin, value):
    """Set a spin box value without emitting valueChanged (avoids re-entrancy)."""
    spin.blockSignals(True)
    spin.setValue(value)
    spin.blockSignals(False)


def _load_image_array(path):
    """Load a microscope image as a numpy array (H×W×{3,4} or H×W). Tries Pillow
    (handles TIFF/JPG/PNG), then matplotlib as a fallback."""
    try:
        from PIL import Image
        im = Image.open(path)
        if im.mode not in ("RGB", "RGBA", "L"):
            im = im.convert("RGBA")
        return np.asarray(im)
    except Exception:
        import matplotlib.image as mpimg
        arr = mpimg.imread(path)
        if arr.dtype != np.uint8 and arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        return arr


_UNIT_UM = {"µm": 1.0, "um": 1.0, "micron": 1.0, "microns": 1.0,
            "micrometer": 1.0, "micrometre": 1.0, "nm": 1e-3, "mm": 1e3,
            "cm": 1e4, "m": 1e6, "inch": 25400.0, "in": 25400.0}


def _unit_to_um(u):
    return _UNIT_UM.get(str(u).strip().lower().rstrip("."))


def _read_image_scale_um_per_px(path):
    """Best-effort read of the physical scale (µm per pixel) from image metadata.

    Handles OME-TIFF (PhysicalSizeX), ImageJ TIFF (unit= + XResolution), plain
    TIFF resolution tags (ResolutionUnit inch/cm + XResolution), and PNG/JPEG DPI.
    Returns (um_per_px, source_str) or (None, None) if nothing usable is found.
    """
    import re
    try:
        from PIL import Image
        im = Image.open(path)
    except Exception:
        return None, None

    # description text (ImageJ / OME live here; TIFF tag 270 or PIL info)
    desc = ""
    try:
        tags = getattr(im, "tag_v2", None)
        if tags is not None and 270 in tags:
            desc = tags[270]
    except Exception:
        desc = ""
    if not desc:
        desc = (im.info or {}).get("ImageDescription", "")
    if isinstance(desc, (bytes, bytearray)):
        desc = desc.decode("latin1", "ignore")

    # 1) OME-TIFF PhysicalSizeX="v" PhysicalSizeXUnit="µm"
    m = re.search(r'PhysicalSizeX\s*=\s*"([-\d.eE+]+)"', desc)
    if m:
        val = float(m.group(1))
        mu = re.search(r'PhysicalSizeXUnit\s*=\s*"([^"]+)"', desc)
        # OME PhysicalSizeX is almost always in microns; default to µm if the unit
        # string is missing or unrecognised.
        f = (_unit_to_um(mu.group(1)) if mu else None) or 1.0
        if val > 0:
            return val * f, "OME-TIFF PhysicalSizeX"

    # 2) explicit pixel width in the description (some vendors write this)
    m = re.search(r'(?:pixelWidth|PixelWidth|scale)\s*[=:]\s*([-\d.eE+]+)', desc)
    unit = None
    mu = re.search(r'unit\s*=\s*([^\s\r\n]+)', desc)
    if mu:
        unit = mu.group(1)
    if m:
        val = float(m.group(1))
        f = _unit_to_um(unit) if unit else 1.0
        if val > 0 and f:
            return val * f, "image description"

    # 3) TIFF XResolution (pixels per unit) + unit (ImageJ 'unit=' or ResolutionUnit)
    xres = None
    resunit = None
    try:
        tags = getattr(im, "tag_v2", None)
        if tags is not None:
            if 282 in tags:
                xr = tags[282]
                xres = (float(xr[0]) / float(xr[1])
                        if isinstance(xr, tuple) else float(xr))
            if 296 in tags:
                resunit = int(tags[296])
    except Exception:
        pass
    if xres and xres > 0:
        if unit and _unit_to_um(unit):
            return _unit_to_um(unit) / xres, "ImageJ/TIFF resolution"
        if resunit == 3:                       # centimetre
            return 1e4 / xres, "TIFF cm resolution"
        if resunit == 2:                       # inch
            return 25400.0 / xres, "TIFF inch resolution"

    # 4) PNG/JPEG DPI (pixels per inch) — last resort. Require > 1 to skip the
    #    trivial (1,1) default TIFFs carry when they have no real resolution.
    dpi = (im.info or {}).get("dpi")
    try:
        if dpi and float(dpi[0]) > 1.0:
            return 25400.0 / float(dpi[0]), "image DPI"
    except Exception:
        pass
    return None, None


def main(argv=None):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    mea_theme.apply_qt(app, "solar")         # shared MEA-Suite theme (amber accent)
    _install_excepthook()
    win = MainWindow()
    # optional: open a project passed on the command line (used by Reload/restart)
    for a in (argv or sys.argv[1:]):
        if a.lower().endswith((".mscproj", ".json")) and os.path.exists(a):
            try:
                win.project = proj.Project.load(a)
                tp = win.project.templates_path
                if tp and os.path.exists(tp):
                    try:
                        win.templates = tpl.TemplateSet.load(tp)
                    except Exception:
                        pass
                win._refresh_all()
            except Exception:
                pass
    win.show()
    run = getattr(app, "exec", None) or app.exec_
    return run()


def _run_main():
    """Entry point that logs ANY startup crash to error.log (so a windowless
    launch that dies before the Qt excepthook is installed is still diagnosable)."""
    logpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "error.log"))
    try:
        return main()
    except SystemExit:
        raise
    except BaseException:
        tb = traceback.format_exc()
        try:
            with open(logpath, "a", encoding="utf-8") as fh:
                fh.write("=== startup crash ===\n" + tb + "\n")
        except Exception:
            pass
        try:
            _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
            QtWidgets.QMessageBox.critical(None, "MEA Solar Cell Analyzer failed to start",
                                           tb[-3000:] + "\n\nLogged to:\n" + logpath)
        except Exception:
            pass
        sys.stderr.write(tb)
        return 1


if __name__ == "__main__":
    raise SystemExit(_run_main())
