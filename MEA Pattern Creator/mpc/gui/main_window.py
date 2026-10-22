"""
main_window.py  --  The MEA Pattern Creator main window.

Ties together the live view, the frame timeline, per-frame properties, the
calibration panel, camera control, and .seq import/export.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from .. import camera as cammod
from .. import seqfile
from ..project import Project
from ..patterns import Frame
from .imageview import ImageView
from .calibration_panel import CalibrationPanel
from .alignment_panel import AlignmentPanel
from ..calibration import Calibration, default_store_path

APP_NAME = "MEA Pattern Creator"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 950)

        self.project = Project()
        self.current_index = -1
        self.project_path: Optional[str] = None
        self.camera: Optional[cammod.CameraBackend] = None
        self._cam_settings = self._load_cam_settings()   # remembered binning/exposure

        self.view = ImageView()
        self.view.calibration = self.project.calibration
        self.view.set_canvas_size(self.project.canvas_width, self.project.canvas_height)
        self.setCentralWidget(self.view)
        self.view.shapeAdded.connect(self._sync_shapes)
        self.view.mouseMovedCanvas.connect(self._on_mouse_canvas)

        self._build_actions()
        self._build_toolbar()
        self._build_docks()
        self._build_menu()
        self.statusBar().showMessage("Ready.")

        # live refresh timer
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh_camera)
        self._timer.start(50)                     # ~20 fps

        # playback timer
        self._play_timer = QtCore.QTimer(self)
        self._play_timer.setSingleShot(True)
        self._play_timer.timeout.connect(self._play_step)
        self._play_seq_index = 0

        # start with one frame and the default camera
        self._add_frame()
        self._connect_default_camera()
        self._apply_remembered_calibration(announce=True)
        self._maybe_restore_recovery()            # after a "Reload app" restart

    # ======================================================================
    #  Construction
    # ======================================================================
    def _build_actions(self):
        self.act_new = QtGui.QAction("New project", self, shortcut="Ctrl+N")
        self.act_new.triggered.connect(self._new_project)
        self.act_open = QtGui.QAction("Open project…", self, shortcut="Ctrl+O")
        self.act_open.triggered.connect(self._open_project)
        self.act_save = QtGui.QAction("Save project", self, shortcut="Ctrl+S")
        self.act_save.triggered.connect(self._save_project)
        self.act_save_as = QtGui.QAction("Save project as…", self)
        self.act_save_as.triggered.connect(lambda: self._save_project(as_new=True))
        self.act_import = QtGui.QAction("Import .seq…", self)
        self.act_import.triggered.connect(self._import_seq)
        self.act_export = QtGui.QAction("Export .seq…", self, shortcut="Ctrl+E")
        self.act_export.triggered.connect(self._export_seq)
        self.act_quit = QtGui.QAction("Quit", self, shortcut="Ctrl+Q")
        self.act_quit.triggered.connect(self.close)

        self.act_load_ref = QtGui.QAction("Load reference image…", self)
        self.act_load_ref.triggered.connect(self._load_reference)
        self.act_snap = QtGui.QAction("Freeze current frame as reference", self)
        self.act_snap.triggered.connect(self._snapshot)
        self.act_connect = QtGui.QAction("Connect AmScope camera", self)
        self.act_connect.triggered.connect(lambda: self._connect_default_camera(force_live=True))
        self.act_synth = QtGui.QAction("Use synthetic test source", self)
        self.act_synth.triggered.connect(self._use_synthetic)
        self.act_reload = QtGui.QAction("Reload app (live update)", self,
                                            shortcut="Ctrl+R")
        self.act_reload.setToolTip("Restart the app to load updated scripts; "
                                   "your current patterns are kept.")
        self.act_reload.triggered.connect(self._reload_app)

    def _build_toolbar(self):
        tb = self.addToolBar("Tools")
        tb.setIconSize(QtCore.QSize(18, 18))
        self._tool_group = QtGui.QActionGroup(self)
        tools = [("Select", "select"), ("Pan", "pan"), ("Rect", "rect"),
                 ("Ellipse", "ellipse"), ("Polygon", "polygon"),
                 ("Freehand", "freehand"), ("Point", "point")]
        for label, name in tools:
            a = QtGui.QAction(label, self, checkable=True)
            a.setData(name)
            a.triggered.connect(lambda _, n=name: self.view.set_tool(n))
            self._tool_group.addAction(a)
            tb.addAction(a)
            if name == "pan":
                a.setChecked(True)
        tb.addSeparator()

        self.mode_box = QtWidgets.QComboBox()
        self.mode_box.addItems(["add (illuminate)", "subtract (mask)"])
        self.mode_box.currentIndexChanged.connect(
            lambda i: setattr(self.view, "mode", "add" if i == 0 else "subtract"))
        tb.addWidget(QtWidgets.QLabel(" Mode "))
        tb.addWidget(self.mode_box)

        tb.addWidget(QtWidgets.QLabel(" Brush/point px "))
        self.brush = QtWidgets.QSpinBox(); self.brush.setRange(1, 400); self.brush.setValue(20)
        self.brush.valueChanged.connect(lambda v: setattr(self.view, "brush_width", float(v)))
        tb.addWidget(self.brush)

        tb.addSeparator()
        self.chk_overlay = QtWidgets.QCheckBox("Overlay")
        self.chk_overlay.setChecked(True)
        self.chk_overlay.toggled.connect(self.view.set_overlay_visible)
        tb.addWidget(self.chk_overlay)
        self.opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.opacity.setRange(10, 100); self.opacity.setValue(50); self.opacity.setFixedWidth(90)
        self.opacity.valueChanged.connect(lambda v: self.view.set_overlay_opacity(v / 100.0))
        tb.addWidget(self.opacity)
        self.chk_cam = QtWidgets.QCheckBox("Camera")
        self.chk_cam.setChecked(True)
        self.chk_cam.toggled.connect(self.view.set_show_camera)
        tb.addWidget(self.chk_cam)

        tb.addSeparator()
        b_reload = QtWidgets.QToolButton()
        b_reload.setText("⟳ Reload")
        b_reload.setToolTip("Restart to load updated scripts (keeps your patterns) — Ctrl+R")
        b_reload.clicked.connect(self._reload_app)
        tb.addWidget(b_reload)

    def _build_docks(self):
        # --- frames dock ---
        self.frames_dock = QtWidgets.QDockWidget("Sequence frames", self)
        fw = QtWidgets.QWidget(); fl = QtWidgets.QVBoxLayout(fw)
        self.frame_list = QtWidgets.QListWidget()
        self.frame_list.currentRowChanged.connect(self._select_frame)
        self.frame_list.model().rowsMoved.connect(self._frames_reordered)
        self.frame_list.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        fl.addWidget(self.frame_list, 1)
        btns = QtWidgets.QHBoxLayout()
        for label, slot in [("Add", self._add_frame), ("Duplicate", self._dup_frame),
                            ("Delete", self._del_frame), ("↑", lambda: self._move_frame(-1)),
                            ("↓", lambda: self._move_frame(1))]:
            b = QtWidgets.QPushButton(label); b.clicked.connect(slot); btns.addWidget(b)
        fl.addLayout(btns)

        # playback
        pb = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("▶ Preview")
        self.btn_play.clicked.connect(self._toggle_play)
        pb.addWidget(self.btn_play)
        fl.addLayout(pb)
        self.frames_dock.setWidget(fw)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.frames_dock)

        # --- properties dock ---
        self.props_dock = QtWidgets.QDockWidget("Frame properties", self)
        pw = QtWidgets.QWidget(); form = QtWidgets.QFormLayout(pw)
        self.f_name = QtWidgets.QLineEdit()
        self.f_name.editingFinished.connect(self._apply_props)
        self.f_dur = QtWidgets.QDoubleSpinBox(); self.f_dur.setRange(0.1, 1e6); self.f_dur.setValue(100); self.f_dur.setSuffix(" ms")
        self.f_dur.valueChanged.connect(self._apply_props)
        self.f_int = QtWidgets.QDoubleSpinBox(); self.f_int.setRange(0, 100); self.f_int.setValue(100); self.f_int.setSuffix(" %")
        self.f_int.valueChanged.connect(self._apply_props)
        self.f_inv = QtWidgets.QCheckBox("invert frame")
        self.f_inv.toggled.connect(self._apply_props)
        form.addRow("Name", self.f_name)
        form.addRow("Duration", self.f_dur)
        form.addRow("Intensity", self.f_int)
        form.addRow("", self.f_inv)
        form.addRow(QtWidgets.QLabel("Shapes in frame:"))
        self.shape_list = QtWidgets.QListWidget()
        self.shape_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.shape_list.itemSelectionChanged.connect(self._shape_list_selection_changed)
        self.view.selectionChanged.connect(self._view_selection_changed)
        self._sel_guard = False
        form.addRow(self.shape_list)
        srow = QtWidgets.QHBoxLayout()
        b_del_shape = QtWidgets.QPushButton("Delete shape"); b_del_shape.clicked.connect(self._del_shape)
        b_clear_shapes = QtWidgets.QPushButton("Clear frame"); b_clear_shapes.clicked.connect(self._clear_shapes)
        srow.addWidget(b_del_shape); srow.addWidget(b_clear_shapes)
        sw = QtWidgets.QWidget(); sw.setLayout(srow); form.addRow(sw)
        b_import_mask = QtWidgets.QPushButton("Import pattern image into frame…")
        b_import_mask.clicked.connect(self._import_mask)
        form.addRow(b_import_mask)
        self.props_dock.setWidget(pw)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.props_dock)

        # --- calibration dock ---
        self.calib_dock = QtWidgets.QDockWidget("Camera calibration", self)
        self.calib_panel = CalibrationPanel(self.view, lambda: self.project)
        self.calib_panel.calibrationChanged.connect(self.view.update_overlay)
        self.calib_dock.setWidget(self.calib_panel)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.calib_dock)

        # --- sample alignment dock ---
        self.align_dock = QtWidgets.QDockWidget("Sample alignment", self)
        self.align_panel = AlignmentPanel(self.view, lambda: self.project)
        self.align_dock.setWidget(self.align_panel)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.align_dock)
        self.tabifyDockWidget(self.calib_dock, self.align_dock)
        self.calib_dock.raise_()

        self.tabifyDockWidget(self.frames_dock, self.props_dock)
        self.frames_dock.raise_()

    def _build_menu(self):
        m = self.menuBar()
        fm = m.addMenu("&File")
        for a in [self.act_new, self.act_open, self.act_save, self.act_save_as]:
            fm.addAction(a)
        fm.addSeparator(); fm.addAction(self.act_import); fm.addAction(self.act_export)
        fm.addSeparator(); fm.addAction(self.act_reload)
        fm.addSeparator(); fm.addAction(self.act_quit)

        em = m.addMenu("&Edit")
        a = em.addAction("Copy selected shape(s)\tCtrl+C")
        a.triggered.connect(self.view.copy_selected)
        a = em.addAction("Paste\tCtrl+V")
        a.triggered.connect(self.view.paste_clipboard)
        a = em.addAction("Duplicate\tCtrl+D")
        a.triggered.connect(self.view.duplicate_selected)
        em.addSeparator()
        a = em.addAction("Delete selected shape(s)\tDel")
        a.triggered.connect(self.view.delete_selected)
        em.addSeparator()
        hint = em.addAction("Tip: pick the Select tool, click a shape; "
                            "Shift-click to multi-select")
        hint.setEnabled(False)

        cm = m.addMenu("&Camera")
        cm.addAction(self.act_connect); cm.addAction(self.act_synth)
        cm.addSeparator(); cm.addAction(self.act_load_ref); cm.addAction(self.act_snap)
        self.res_menu = cm.addMenu("Resolution")
        self.exp_menu = cm.addMenu("Exposure / gain")
        self._build_exposure_menu()

        cm.addSeparator()
        a_forget = QtGui.QAction("Forget remembered camera calibration", self)
        a_forget.triggered.connect(self._forget_calibration)
        cm.addAction(a_forget)

        vm = m.addMenu("&View")
        vm.addAction(self.frames_dock.toggleViewAction())
        vm.addAction(self.props_dock.toggleViewAction())
        vm.addAction(self.calib_dock.toggleViewAction())
        vm.addAction(self.align_dock.toggleViewAction())

        sm = m.addMenu("&Sequence")
        a_canvas = QtGui.QAction("Canvas size…", self)
        a_canvas.triggered.connect(self._set_canvas_size)
        sm.addAction(a_canvas)
        sm.addSeparator()
        a_pulse = QtGui.QAction("Build pulse train…", self)
        a_pulse.triggered.connect(self._build_pulse_train)
        sm.addAction(a_pulse)
        a_timed = QtGui.QAction("Export timed .seq (expand durations)…", self)
        a_timed.triggered.connect(self._export_timed_seq)
        sm.addAction(a_timed)

        hm = m.addMenu("&Help")
        a_about = QtGui.QAction("About / quick guide", self)
        a_about.triggered.connect(self._about)
        hm.addAction(a_about)

    def _build_exposure_menu(self):
        self.exp_menu.clear()
        self._act_auto = QtGui.QAction("Auto exposure", self, checkable=True)
        self._act_auto.setChecked(bool(self._cam_settings.get("auto", True)))
        self._act_auto.toggled.connect(self._on_auto_toggle)
        self.exp_menu.addAction(self._act_auto)
        a_exp = QtGui.QAction("Set exposure (ms)…", self)
        a_exp.triggered.connect(self._set_exposure)
        a_gain = QtGui.QAction("Set gain (%)…", self)
        a_gain.triggered.connect(self._set_gain)
        self.exp_menu.addAction(a_exp); self.exp_menu.addAction(a_gain)

    # ======================================================================
    #  Camera
    # ======================================================================
    def _connect_default_camera(self, force_live: bool = False):
        if self.camera:
            try: self.camera.stop(); self.camera.close()
            except Exception: pass
        cam, msg = cammod.open_default_camera(prefer_live=True)
        self.camera = cam
        self.statusBar().showMessage(msg, 8000)
        self._rebuild_res_menu()
        self._apply_cam_settings()               # restore remembered binning/exposure

    def _use_synthetic(self):
        if self.camera:
            try: self.camera.stop(); self.camera.close()
            except Exception: pass
        self.camera = cammod.SyntheticBackend(self.project.canvas_width, self.project.canvas_height)
        self.camera.start()
        self.statusBar().showMessage("Synthetic test source active.", 5000)
        self._rebuild_res_menu()

    def _rebuild_res_menu(self):
        self.res_menu.clear()
        if not self.camera:
            return
        try:
            res = self.camera.list_resolutions()
        except Exception:
            res = [self.camera.frame_size()]
        for i, (w, h) in enumerate(res):
            a = QtGui.QAction(f"{w} x {h}", self)
            a.triggered.connect(lambda _, idx=i: self._set_resolution(idx))
            self.res_menu.addAction(a)

    def _set_resolution(self, idx):
        try:
            self.camera.set_resolution(idx)
            self._cam_settings["resolution_index"] = int(idx)
            self._cam_settings["resolution_wh"] = list(self.camera.frame_size())
            self._save_cam_settings()
        except Exception as e:
            self.statusBar().showMessage(f"Resolution change failed: {e}", 5000)

    def _set_exposure(self):
        default_ms = float(self._cam_settings.get("exposure_us", 10000)) / 1000.0
        ms, ok = QtWidgets.QInputDialog.getDouble(
            self, "Exposure", "Exposure (ms):", default_ms, 0.01, 10000, 3)
        if ok and self.camera:
            self.camera.set_exposure(int(ms * 1000))
            self._cam_settings["exposure_us"] = int(ms * 1000)
            self._cam_settings["auto"] = False
            self._save_cam_settings()
            if hasattr(self, "_act_auto"):
                self._act_auto.blockSignals(True)
                self._act_auto.setChecked(False)
                self._act_auto.blockSignals(False)

    def _set_gain(self):
        default_g = int(self._cam_settings.get("gain", 100))
        g, ok = QtWidgets.QInputDialog.getInt(
            self, "Gain", "Analog gain (%):", default_g, 100, 5000)
        if ok and self.camera:
            self.camera.set_gain(g)
            self._cam_settings["gain"] = int(g)
            self._save_cam_settings()

    def _on_auto_toggle(self, v):
        if self.camera:
            self.camera.set_auto_exposure(bool(v))
            if not v and self._cam_settings.get("exposure_us"):
                self.camera.set_exposure(int(self._cam_settings["exposure_us"]))
        self._cam_settings["auto"] = bool(v)
        self._save_cam_settings()

    # -- camera-settings persistence (binning / exposure / gain) ----------
    def _app_folder(self):
        mpc_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.dirname(mpc_dir)

    def _cam_settings_path(self):
        return os.path.join(self._app_folder(), "camera_settings.json")

    # -- live reload / restart -------------------------------------------
    def _recovery_path(self):
        return os.path.join(self._app_folder(), "_reload_recovery.mpcproj")

    def _reload_flag_path(self):
        return os.path.join(self._app_folder(), "_reload_recovery.flag")

    def _reload_app(self):
        """Restart the app so updated scripts load, preserving current work.

        A clean process restart is used (not in-place module reload), because
        pushed updates often change class structure, which importlib.reload
        cannot safely apply to already-live Qt objects.
        """
        import sys
        # 1) save the in-progress project so nothing is lost across the restart
        try:
            self.project.save(self._recovery_path())
            with open(self._reload_flag_path(), "w", encoding="utf-8") as fh:
                fh.write(self._recovery_path())
        except Exception:
            pass
        # 2) release the camera so the new instance can open it
        try:
            if self.camera:
                self.camera.stop(); self.camera.close()
        except Exception:
            pass
        self.camera = None
        # 3) launch a fresh instance, then quit this one
        prog = sys.executable
        script = os.path.abspath(sys.argv[0])
        extra = list(sys.argv[1:])
        started = False
        try:
            started = QtCore.QProcess.startDetached(prog, [script] + extra)
        except Exception:
            started = False
        if started:
            self.statusBar().showMessage("Reloading…", 3000)
            QtCore.QTimer.singleShot(0, QtWidgets.QApplication.instance().quit)
        else:
            try:                                   # fallback: replace this process
                os.execv(prog, [prog, script] + extra)
            except Exception as e:                 # noqa: BLE001
                QtWidgets.QMessageBox.warning(
                    self, "Reload failed",
                    f"Could not restart automatically:\n{e}\n\n"
                    "Please close and relaunch the app manually.")

    def _maybe_restore_recovery(self):
        """After a Reload-restart, reload the auto-saved project so it feels
        like a live update rather than a fresh start."""
        flag = self._reload_flag_path()
        if not os.path.exists(flag):
            return
        rec = self._recovery_path()
        try:
            if os.path.exists(rec):
                self.project = Project.load(rec)
                self.view.calibration = self.project.calibration
                self.view.set_canvas_size(self.project.canvas_width,
                                          self.project.canvas_height)
                if not self.project.frames:
                    self.project.new_frame()
                self._reload_frame_list()
                self.frame_list.setCurrentRow(0)
                self.calib_panel.refresh()
                self.align_panel.refresh()
                self.statusBar().showMessage(
                    "Reloaded updated scripts — your patterns were restored.", 7000)
        except Exception:
            pass
        finally:
            for p in (flag, rec):
                try:
                    os.remove(p)
                except Exception:
                    pass

    def _load_cam_settings(self) -> dict:
        import json
        try:
            with open(self._cam_settings_path(), encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def _save_cam_settings(self):
        import json
        try:
            with open(self._cam_settings_path(), "w", encoding="utf-8") as fh:
                json.dump(self._cam_settings, fh, indent=2)
        except Exception:
            pass

    def _apply_cam_settings(self):
        """Re-apply remembered binning/exposure/gain to a freshly opened camera."""
        cam = self.camera
        if type(cam).__name__ != "AmcamBackend":
            return
        s = self._cam_settings
        if not s:
            return
        # resolution / binning: match remembered WxH to an available mode,
        # else fall back to the stored index.
        try:
            res = cam.list_resolutions()
        except Exception:
            res = []
        idx = None
        if res and s.get("resolution_wh"):
            target = tuple(s["resolution_wh"])
            for i, (w, h) in enumerate(res):
                if (w, h) == target:
                    idx = i
                    break
        if idx is None and res and isinstance(s.get("resolution_index"), int):
            if 0 <= s["resolution_index"] < len(res):
                idx = s["resolution_index"]
        if idx is not None and (cam.frame_size() != tuple(s.get("resolution_wh", ()))):
            try:
                cam.set_resolution(idx)
                self._rebuild_res_menu()
            except Exception:
                pass
        # exposure / auto
        auto = bool(s.get("auto", True))
        try:
            cam.set_auto_exposure(auto)
        except Exception:
            pass
        if not auto and s.get("exposure_us"):
            try:
                cam.set_exposure(int(s["exposure_us"]))
            except Exception:
                pass
        if s.get("gain"):
            try:
                cam.set_gain(int(s["gain"]))
            except Exception:
                pass
        if hasattr(self, "_act_auto"):
            self._act_auto.blockSignals(True)
            self._act_auto.setChecked(auto)
            self._act_auto.blockSignals(False)
        exp = "auto" if auto else f"{s.get('exposure_us', '?')} µs"
        self.statusBar().showMessage(
            f"Restored camera settings — {cam.frame_size()[0]}×{cam.frame_size()[1]}, "
            f"exposure {exp}.", 6000)

    def _load_reference(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load reference image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
        if path:
            try:
                self.camera = cammod.ReferenceImageBackend.from_file(path)
                self.statusBar().showMessage(f"Loaded reference image: {os.path.basename(path)}", 5000)
                self._rebuild_res_menu()
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Load failed", str(e))

    def _snapshot(self):
        if not self.camera:
            return
        frame = self.camera.get_frame()
        if frame is None:
            self.statusBar().showMessage("No frame available to freeze yet.", 4000)
            return
        self.camera = cammod.ReferenceImageBackend(frame.copy(), source="snapshot")
        self.statusBar().showMessage("Frozen current camera frame as the working reference.", 5000)

    def _apply_remembered_calibration(self, announce: bool = False):
        """Load the app-level camera calibration (same optics every session)."""
        path = default_store_path()
        if not os.path.isfile(path):
            return False
        try:
            cal = Calibration.load(path)
        except Exception:
            return False
        cal.canvas_width = self.project.canvas_width
        cal.canvas_height = self.project.canvas_height
        self.project.calibration = cal
        self.view.calibration = cal
        cal.fit(camera_size=self.view._cam_size)
        self.view._refresh_overlay_transform()
        self.view.update_overlay()
        if hasattr(self, "calib_panel"):
            self.calib_panel.mode.blockSignals(True)
            self.calib_panel.mode.setCurrentText(cal.mode)
            self.calib_panel.mode.blockSignals(False)
            self.calib_panel.refresh()
        if announce and cal.pairs:
            rms = cal.rms_error()
            extra = f" (RMS {rms:.2f} px)" if rms is not None else ""
            self.statusBar().showMessage(
                f"Loaded remembered camera calibration: {len(cal.pairs)} points, "
                f"mode {cal.mode}{extra}.", 8000)
        return True

    def _forget_calibration(self):
        path = default_store_path()
        try:
            if os.path.isfile(path):
                os.remove(path)
        except Exception:
            pass
        self.project.calibration = Calibration(
            canvas_width=self.project.canvas_width,
            canvas_height=self.project.canvas_height)
        self.view.calibration = self.project.calibration
        self.view.clear_calib_markers()
        self.view.update_overlay()
        self.calib_panel.refresh()
        self.statusBar().showMessage("Remembered camera calibration cleared.", 5000)

    def _refresh_camera(self):
        if not self.camera:
            return
        try:
            frame = self.camera.get_frame()
        except Exception:
            frame = None
        if frame is not None:
            self.view.set_camera_image(frame)

    # ======================================================================
    #  Frames
    # ======================================================================
    def _add_frame(self):
        fr = self.project.new_frame()
        self._reload_frame_list()
        self.frame_list.setCurrentRow(len(self.project.frames) - 1)

    def _dup_frame(self):
        if self.current_index < 0:
            return
        import copy
        fr = copy.deepcopy(self.project.frames[self.current_index])
        fr.name = fr.name + " copy"
        self.project.frames.insert(self.current_index + 1, fr)
        self._reload_frame_list()
        self.frame_list.setCurrentRow(self.current_index + 1)

    def _del_frame(self):
        if self.current_index < 0:
            return
        del self.project.frames[self.current_index]
        if not self.project.frames:
            self.project.new_frame()
        self._reload_frame_list()
        self.frame_list.setCurrentRow(min(self.current_index, len(self.project.frames) - 1))

    def _move_frame(self, d):
        i = self.current_index
        j = i + d
        if 0 <= i < len(self.project.frames) and 0 <= j < len(self.project.frames):
            self.project.frames[i], self.project.frames[j] = \
                self.project.frames[j], self.project.frames[i]
            self._reload_frame_list()
            self.frame_list.setCurrentRow(j)

    def _frames_reordered(self, *_):
        # rebuild project.frames order from the list widget's frame refs
        new = []
        for i in range(self.frame_list.count()):
            fr = self.frame_list.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
            new.append(fr)
        if len(new) == len(self.project.frames):
            self.project.frames = new
            self.current_index = self.frame_list.currentRow()

    def _reload_frame_list(self):
        self.frame_list.blockSignals(True)
        self.frame_list.clear()
        for i, fr in enumerate(self.project.frames):
            it = QtWidgets.QListWidgetItem(f"{i+1:03d}  {fr.name}  ·  {fr.duration_ms:g}ms  {fr.intensity:g}%")
            it.setData(QtCore.Qt.ItemDataRole.UserRole, fr)
            self.frame_list.addItem(it)
        self.frame_list.blockSignals(False)

    def _select_frame(self, row):
        self.current_index = row
        if 0 <= row < len(self.project.frames):
            fr = self.project.frames[row]
            self.view.set_frame(fr)
            self._load_props(fr)
            self._sync_shapes()

    def _load_props(self, fr: Frame):
        for w in (self.f_name, self.f_dur, self.f_int, self.f_inv):
            w.blockSignals(True)
        self.f_name.setText(fr.name)
        self.f_dur.setValue(fr.duration_ms)
        self.f_int.setValue(fr.intensity)
        self.f_inv.setChecked(fr.invert)
        for w in (self.f_name, self.f_dur, self.f_int, self.f_inv):
            w.blockSignals(False)

    def _apply_props(self, *_):
        if self.current_index < 0:
            return
        fr = self.project.frames[self.current_index]
        fr.name = self.f_name.text()
        fr.duration_ms = self.f_dur.value()
        fr.intensity = self.f_int.value()
        fr.invert = self.f_inv.isChecked()
        self.view.update_overlay()
        it = self.frame_list.item(self.current_index)
        if it:
            it.setText(f"{self.current_index+1:03d}  {fr.name}  ·  {fr.duration_ms:g}ms  {fr.intensity:g}%")

    def _sync_shapes(self):
        self._sel_guard = True
        try:
            self.shape_list.clear()
            if self.current_index < 0:
                return
            fr = self.project.frames[self.current_index]
            for i, s in enumerate(fr.shapes):
                self.shape_list.addItem(
                    f"{i+1}. {s.kind} [{s.mode}] ({len(s.points)} pts)")
            for i in range(self.shape_list.count()):
                self.shape_list.item(i).setSelected(i in self.view.selected)
        finally:
            self._sel_guard = False
        self._reload_frame_list_label()

    def _view_selection_changed(self):
        """View → shape-list highlight."""
        if getattr(self, "_sel_guard", False):
            return
        self._sel_guard = True
        try:
            for i in range(self.shape_list.count()):
                self.shape_list.item(i).setSelected(i in self.view.selected)
        finally:
            self._sel_guard = False

    def _shape_list_selection_changed(self):
        """Shape-list → view selection."""
        if getattr(self, "_sel_guard", False):
            return
        rows = {ix.row() for ix in self.shape_list.selectedIndexes()}
        self._sel_guard = True
        try:
            self.view.select_indices(rows)
        finally:
            self._sel_guard = False

    def _reload_frame_list_label(self):
        it = self.frame_list.item(self.current_index) if self.current_index >= 0 else None
        if it and self.current_index < len(self.project.frames):
            fr = self.project.frames[self.current_index]
            it.setText(f"{self.current_index+1:03d}  {fr.name}  ·  {fr.duration_ms:g}ms  {fr.intensity:g}%")

    def _del_shape(self):
        if self.current_index < 0:
            return
        if self.view.selected:
            self.view.delete_selected()
            return
        r = self.shape_list.currentRow()
        fr = self.project.frames[self.current_index]
        if 0 <= r < len(fr.shapes):
            del fr.shapes[r]
            self.view.update_overlay(); self._sync_shapes()

    def _clear_shapes(self):
        if self.current_index < 0:
            return
        self.project.frames[self.current_index].shapes.clear()
        self.project.frames[self.current_index].clear_mask()
        self.view.update_overlay(); self._sync_shapes()

    def _import_mask(self):
        if self.current_index < 0:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import pattern image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
        if not path:
            return
        from PIL import Image
        fit = "stretch"
        r = QtWidgets.QMessageBox.question(
            self, "Fit mode", "Stretch image to fill the canvas?\n"
            "Yes = stretch, No = center at native size.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
        if r == QtWidgets.QMessageBox.StandardButton.No:
            fit = "center"
        img = Image.open(path)
        self.project.frames[self.current_index].set_mask_from_image(
            img, self.project.canvas_width, self.project.canvas_height, fit=fit)
        self.view.update_overlay(); self._sync_shapes()

    # ======================================================================
    #  Playback preview
    # ======================================================================
    def _toggle_play(self):
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.btn_play.setText("▶ Preview")
        else:
            if not self.project.frames:
                return
            self._play_seq_index = 0
            self.btn_play.setText("■ Stop")
            self._play_step()

    def _play_step(self):
        if not self.project.frames:
            return
        i = self._play_seq_index % len(self.project.frames)
        self.frame_list.setCurrentRow(i)
        dur = max(20, self.project.frames[i].duration_ms)
        self._play_seq_index += 1
        self._play_timer.start(int(dur))

    # ======================================================================
    #  Mouse readout
    # ======================================================================
    def _on_mouse_canvas(self, cx, cy):
        self.statusBar().showMessage(
            f"canvas ({cx:.0f}, {cy:.0f})  ·  canvas size {self.project.canvas_width}×{self.project.canvas_height}")

    # ======================================================================
    #  Canvas size
    # ======================================================================
    def _set_canvas_size(self):
        w, ok = QtWidgets.QInputDialog.getInt(self, "Canvas width",
            "Width (must be multiple of 8):", self.project.canvas_width, 8, 8192, 8)
        if not ok:
            return
        h, ok = QtWidgets.QInputDialog.getInt(self, "Canvas height",
            "Height:", self.project.canvas_height, 8, 8192, 1)
        if not ok:
            return
        if w % 8 != 0:
            w = (w // 8) * 8
        self.project.canvas_width, self.project.canvas_height = w, h
        self.project.calibration.canvas_width = w
        self.project.calibration.canvas_height = h
        self.view.set_canvas_size(w, h)
        self.statusBar().showMessage(f"Canvas set to {w}×{h}.", 4000)

    # ======================================================================
    #  Project / file IO
    # ======================================================================
    def _new_project(self):
        self.project = Project()
        self.view.calibration = self.project.calibration
        self.view.set_canvas_size(self.project.canvas_width, self.project.canvas_height)
        self.project_path = None
        self.project.new_frame()
        self._reload_frame_list()
        self.frame_list.setCurrentRow(0)
        self.calib_panel.refresh()
        self._apply_remembered_calibration()
        self.align_panel.refresh()

    def _open_project(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open project", "", "MEA Pattern project (*.mpcproj *.json)")
        if not path:
            return
        try:
            self.project = Project.load(path)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Open failed", str(e)); return
        self.project_path = path
        self.view.calibration = self.project.calibration
        self.view.set_canvas_size(self.project.canvas_width, self.project.canvas_height)
        if not self.project.frames:
            self.project.new_frame()
        self._reload_frame_list(); self.frame_list.setCurrentRow(0)
        self.calib_panel.refresh()
        self.align_panel.refresh()
        self.statusBar().showMessage(f"Opened {os.path.basename(path)}", 5000)

    def _save_project(self, as_new: bool = False):
        if as_new or not self.project_path:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save project", "patterns.mpcproj", "MEA Pattern project (*.mpcproj)")
            if not path:
                return
            self.project_path = path
        try:
            self.project.save(self.project_path)
            self.statusBar().showMessage(f"Saved {os.path.basename(self.project_path)}", 5000)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(e))

    def _import_seq(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import .seq", "", "Polygon sequence (*.seq)")
        if not path:
            return
        try:
            self.project = Project.from_seq(path)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Import failed", str(e)); return
        self.view.calibration = self.project.calibration
        self.view.set_canvas_size(self.project.canvas_width, self.project.canvas_height)
        self._reload_frame_list(); self.frame_list.setCurrentRow(0)
        self._apply_remembered_calibration()
        self.calib_panel.refresh()
        self.align_panel.refresh()
        self.statusBar().showMessage(
            f"Imported {len(self.project.frames)} frames from {os.path.basename(path)}", 6000)

    def _export_seq(self):
        if not self.project.frames:
            QtWidgets.QMessageBox.information(self, "Nothing to export", "Add at least one frame.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export .seq", "sequence.seq", "Polygon sequence (*.seq)")
        if not path:
            return
        try:
            self.project.export_seq(path)
            self.statusBar().showMessage(
                f"Exported {len(self.project.frames)} frames → {os.path.basename(path)} "
                f"(+ timing sidecar).", 8000)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export failed", str(e))

    # ======================================================================
    #  Pulse train + timed (duration-aware) export
    # ======================================================================
    def _build_pulse_train(self):
        if self.current_index < 0:
            return
        src = self.project.frames[self.current_index]
        if not src.shapes and not src.mask_b64:
            r = QtWidgets.QMessageBox.question(
                self, "Empty pattern",
                "The current frame has no pattern drawn on it, so the ON pulses "
                "will be blank. Draw your pattern on this frame first.\n\n"
                "Build an empty (blank) pulse train anyway?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
            if r != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Build pulse train")
        form = QtWidgets.QFormLayout(dlg)
        info = QtWidgets.QLabel(
            "Builds: N pulses of (ON, OFF) then one REST, using the current "
            "frame's pattern for each ON. Export with 'Export timed .seq' at the "
            "base period below.")
        info.setWordWrap(True); form.addRow(info)
        n = QtWidgets.QSpinBox(); n.setRange(1, 100000); n.setValue(5)
        on = QtWidgets.QDoubleSpinBox(); on.setRange(0.1, 1e6); on.setValue(10); on.setSuffix(" ms")
        off = QtWidgets.QDoubleSpinBox(); off.setRange(0, 1e6); off.setValue(10); off.setSuffix(" ms")
        rest = QtWidgets.QDoubleSpinBox(); rest.setRange(0, 1e7); rest.setValue(1900); rest.setSuffix(" ms")
        inten = QtWidgets.QDoubleSpinBox(); inten.setRange(0, 100); inten.setValue(src.intensity); inten.setSuffix(" %")
        base = QtWidgets.QDoubleSpinBox(); base.setRange(0.1, 1e5)
        base.setValue(self.project.base_period_ms); base.setSuffix(" ms")
        replace = QtWidgets.QComboBox(); replace.addItems(
            ["Replace all frames", "Append to current frames"])
        form.addRow("Number of pulses", n)
        form.addRow("ON time", on)
        form.addRow("OFF time", off)
        form.addRow("Rest time", rest)
        form.addRow("Intensity", inten)
        form.addRow("Base period (Pattern Period)", base)
        form.addRow("", replace)
        preview = QtWidgets.QLabel(); preview.setWordWrap(True)
        preview.setStyleSheet("color:#8fd0ff;")
        form.addRow(preview)

        def _update(*_):
            npulses = n.value()
            gui_frames = npulses * (2 if off.value() > 0 else 1) + (1 if rest.value() > 0 else 0)
            b = base.value()
            reps_on = max(1, round(on.value() / b))
            reps_off = max(1, round(off.value() / b)) if off.value() > 0 else 0
            reps_rest = max(1, round(rest.value() / b)) if rest.value() > 0 else 0
            total = npulses * (reps_on + reps_off) + reps_rest
            cycle = total * b
            preview.setText(
                f"→ {gui_frames} editable frames; exported timed = {total} frames "
                f"at {b:g} ms = {cycle:g} ms per cycle "
                f"({cycle/1000:.3f} s). Set PolyScan Pattern Period = {b:g} ms.")
        for w in (n, on, off, rest, base):
            w.valueChanged.connect(_update)
        _update()

        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        self.project.base_period_ms = base.value()
        self.project.build_pulse_train(
            on_shapes=src.shapes, n_pulses=n.value(), on_ms=on.value(),
            off_ms=off.value(), rest_ms=rest.value(), intensity=inten.value(),
            replace=(replace.currentIndex() == 0))
        self._reload_frame_list()
        self.frame_list.setCurrentRow(0)
        self.statusBar().showMessage(
            f"Built pulse train: {len(self.project.frames)} frames. "
            f"Use Sequence → Export timed .seq at {base.value():g} ms.", 8000)

    def _export_timed_seq(self):
        if not self.project.frames:
            QtWidgets.QMessageBox.information(self, "Nothing to export", "Add at least one frame.")
            return
        b, ok = QtWidgets.QInputDialog.getDouble(
            self, "Timed export — base period",
            "Base pattern period (ms).\nSet PolyScan's Pattern Period to this value.\n"
            "Each frame is repeated round(duration ÷ base) times:",
            self.project.base_period_ms, 0.1, 100000, 3)
        if not ok:
            return
        self.project.base_period_ms = b
        plan = self.project.expansion_plan(b)
        msg = (f"This will write {plan['total_frames']} frames "
               f"({len(self.project.frames)} editable frames expanded).\n"
               f"Total = {plan['total_ms']:g} ms "
               f"({plan['total_ms']/1000:.3f} s) per run.\n\n"
               f"➜ Set PolyScan Pattern Period = {b:g} ms.\n")
        if plan["warnings"]:
            msg += ("\nRounded (not exact multiples of base):\n  "
                    + "\n  ".join(plan["warnings"][:8]))
            if len(plan["warnings"]) > 8:
                msg += f"\n  …and {len(plan['warnings']) - 8} more"
        if plan["total_frames"] > 4000:
            msg += (f"\n\n⚠ {plan['total_frames']} frames is large "
                    f"(~{plan['total_frames']*0.264:.0f} MB). Consider a larger "
                    "base period if your durations allow.")
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Question)
        box.setWindowTitle("Export timed .seq")
        box.setText(msg)
        box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel)
        if box.exec() != QtWidgets.QMessageBox.StandardButton.Ok:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export timed .seq", "sequence_timed.seq", "Polygon sequence (*.seq)")
        if not path:
            return
        try:
            plan = self.project.export_timed_seq(path, b)
            QtWidgets.QMessageBox.information(
                self, "Timed .seq exported",
                f"Wrote {plan['total_frames']} frames → {os.path.basename(path)}\n\n"
                f"In PolyScan Master Mode set Pattern Period = {b:g} ms.\n"
                f"One full run = {plan['total_ms']:g} ms; use Loop Count for repeats.")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export failed", str(e))

    def _about(self):
        QtWidgets.QMessageBox.information(self, "MEA Pattern Creator — quick guide",
            "1. Connect the AmScope camera (Camera menu) or load a reference snapshot.\n"
            "2. Calibrate (left dock): export the dot .seq, project it, snap the image, "
            "click the dots. Homography corrects perspective.\n"
            "3. Add frames (right dock). For each frame, draw ROIs with the toolbar tools "
            "(rectangle, ellipse, polygon, freehand, point) or import a pattern image.\n"
            "4. Magenta overlay shows exactly what will be projected.\n"
            "5. Set per-frame duration/intensity, preview, then File → Export .seq to load "
            "into the Polygon system.\n\n"
            "Add = illuminate, Subtract = mask out. Right-click / double-click finishes a polygon.")

    def closeEvent(self, ev):
        try:
            if self.camera:
                self.camera.stop(); self.camera.close()
        except Exception:
            pass
        super().closeEvent(ev)
