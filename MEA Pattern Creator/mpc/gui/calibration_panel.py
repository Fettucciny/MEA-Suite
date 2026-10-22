"""
calibration_panel.py  --  Dockable panel that builds the camera<->canvas map.

Workflow (no direct hardware control required):
  1. Set a dot grid (cols x rows) and click "Export calibration .seq".  Load &
     project that .seq with the Polygon software.
  2. Snap / freeze the camera image showing where the dots landed on the sample.
  3. Click "Start picking", then click each projected dot in the image, in the
     listed order.  Each click pairs a known canvas coordinate with the camera
     coordinate you clicked.
  4. Choose Affine (>=3 pts) or Homography (>=4 pts, corrects perspective) and
     the fit updates live; the RMS error (in canvas px) tells you how good it is.

The resulting transform is stored on the project's calibration and used to warp
the pattern overlay and to convert drawings into canvas coordinates.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

from PyQt6 import QtCore, QtWidgets

from ..calibration import Calibration, calibration_dot_canvas_points
from .. import seqfile
from ..patterns import Frame, Shape


class CalibrationPanel(QtWidgets.QWidget):
    calibrationChanged = QtCore.pyqtSignal()

    def __init__(self, view, get_project, parent=None):
        super().__init__(parent)
        self.view = view
        self.get_project = get_project           # callable -> Project
        self._expected: List[Tuple[float, float]] = []
        self._pick_index = 0
        self._picking = False
        self._build()
        self.view.canvasClicked.connect(self._on_image_click)

    # -- ui ---------------------------------------------------------------
    def _build(self):
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)

        form = QtWidgets.QFormLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["identity", "affine", "homography"])
        self.mode.setCurrentText("homography")
        self.mode.currentTextChanged.connect(self._refit)
        form.addRow("Transform", self.mode)

        self.cols = QtWidgets.QSpinBox(); self.cols.setRange(2, 8); self.cols.setValue(3)
        self.rows = QtWidgets.QSpinBox(); self.rows.setRange(2, 8); self.rows.setValue(3)
        self.dot = QtWidgets.QSpinBox(); self.dot.setRange(4, 200); self.dot.setValue(40)
        grid = QtWidgets.QHBoxLayout()
        grid.addWidget(QtWidgets.QLabel("cols")); grid.addWidget(self.cols)
        grid.addWidget(QtWidgets.QLabel("rows")); grid.addWidget(self.rows)
        grid.addWidget(QtWidgets.QLabel("dot px")); grid.addWidget(self.dot)
        gw = QtWidgets.QWidget(); gw.setLayout(grid)
        form.addRow("Dot grid", gw)
        lay.addLayout(form)

        self.btn_export = QtWidgets.QPushButton("Export calibration .seq…")
        self.btn_export.clicked.connect(self._export_calib_seq)
        lay.addWidget(self.btn_export)

        self.btn_pick = QtWidgets.QPushButton("Start picking dots")
        self.btn_pick.setCheckable(True)
        self.btn_pick.toggled.connect(self._toggle_pick)
        lay.addWidget(self.btn_pick)

        self.status = QtWidgets.QLabel("Not calibrated.")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["canvas x", "canvas y", "cam x", "cam y"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self.table, 1)

        row = QtWidgets.QHBoxLayout()
        b_undo = QtWidgets.QPushButton("Undo last")
        b_undo.clicked.connect(self._undo)
        b_clear = QtWidgets.QPushButton("Clear")
        b_clear.clicked.connect(self._clear)
        row.addWidget(b_undo); row.addWidget(b_clear)
        lay.addLayout(row)

        row2 = QtWidgets.QHBoxLayout()
        b_save = QtWidgets.QPushButton("Save…"); b_save.clicked.connect(self._save)
        b_load = QtWidgets.QPushButton("Load…"); b_load.clicked.connect(self._load)
        row2.addWidget(b_save); row2.addWidget(b_load)
        lay.addLayout(row2)

    # -- helpers ----------------------------------------------------------
    def _cal(self) -> Calibration:
        return self.get_project().calibration

    def _expected_points(self):
        proj = self.get_project()
        return calibration_dot_canvas_points(
            proj.canvas_width, proj.canvas_height,
            self.cols.value(), self.rows.value())

    def _export_calib_seq(self):
        proj = self.get_project()
        pts = self._expected_points()
        fr = Frame(name="calibration")
        for (x, y) in pts:
            fr.shapes.append(Shape("point", [(x, y)], width=float(self.dot.value())))
        seq = seqfile.new_sequence(proj.canvas_width, proj.canvas_height)
        seq.add_frame(fr.rasterize(proj.canvas_width, proj.canvas_height))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export calibration .seq", "calibration.seq", "Polygon sequence (*.seq)")
        if path:
            seqfile.write_seq(path, seq)
            QtWidgets.QMessageBox.information(
                self, "Calibration pattern",
                f"Wrote {len(pts)} dots to:\n{path}\n\n"
                "Project this with the Polygon software, snap the camera image, "
                "then click 'Start picking dots' and click each dot in order "
                "(left-to-right, top-to-bottom).")

    def _toggle_pick(self, on: bool):
        self._picking = on
        if on:
            self._expected = self._expected_points()
            self._pick_index = len(self._cal().pairs)
            self.view.set_tool("calibrate")
            self.btn_pick.setText("Picking… (click dots)")
            self._update_status()
        else:
            self.view.set_tool("pan")
            self.btn_pick.setText("Start picking dots")

    def _on_image_click(self, cam_x: float, cam_y: float):
        if not self._picking:
            return
        pts = self._expected_points()
        if self._pick_index >= len(pts):
            self.btn_pick.setChecked(False)
            return
        canvas_xy = pts[self._pick_index]
        self._cal().pairs.append(((cam_x, cam_y), canvas_xy))
        self._pick_index += 1
        self._refit()
        self._sync_table()
        self._draw_markers()
        if self._pick_index >= len(pts):
            self.btn_pick.setChecked(False)

    def _refit(self, *_):
        cal = self._cal()
        cal.mode = self.mode.currentText()
        cal.fit(camera_size=self.view._cam_size)
        self.view._refresh_overlay_transform()
        self.view.update_overlay()
        self._update_status()
        self._remember(cal)
        self.calibrationChanged.emit()

    def _remember(self, cal):
        """Persist the camera calibration so it is reused next session."""
        if len(cal.pairs) >= 3:
            try:
                from ..calibration import default_store_path
                cal.save(default_store_path())
            except Exception:
                pass

    def _update_status(self):
        cal = self._cal()
        rms = cal.rms_error()
        n = len(cal.pairs)
        if self._picking and self._pick_index < len(self._expected_points()):
            nxt = self._pick_index + 1
            self.status.setText(f"Click dot #{nxt} "
                                f"(row {(self._pick_index)//self.cols.value()+1}, "
                                f"col {(self._pick_index)%self.cols.value()+1}).  "
                                f"{n} pairs so far.")
        elif rms is not None:
            self.status.setText(f"{n} pairs · mode={cal.mode} · RMS = {rms:.2f} canvas px")
        else:
            self.status.setText(f"{n} pairs · need ≥3 (affine) / ≥4 (homography).")

    def _sync_table(self):
        cal = self._cal()
        self.table.setRowCount(len(cal.pairs))
        for i, ((cx, cy), (kx, ky)) in enumerate(cal.pairs):
            for col, val in enumerate((kx, ky, cx, cy)):
                self.table.setItem(i, col, QtWidgets.QTableWidgetItem(f"{val:.1f}"))

    def _draw_markers(self):
        self.view.clear_calib_markers()
        for i, ((cx, cy), _) in enumerate(self._cal().pairs):
            self.view.add_calib_marker((cx, cy), str(i + 1))

    def _undo(self):
        if self._cal().pairs:
            self._cal().pairs.pop()
            self._pick_index = max(0, self._pick_index - 1)
            self._refit(); self._sync_table(); self._draw_markers()

    def _clear(self):
        self._cal().pairs.clear()
        self._pick_index = 0
        self._refit(); self._sync_table()
        self.view.clear_calib_markers()

    def _save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save calibration", "calibration.json", "JSON (*.json)")
        if path:
            self._cal().save(path)

    def _load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load calibration", "", "JSON (*.json)")
        if path:
            cal = Calibration.load(path)
            proj = self.get_project()
            proj.calibration = cal
            self.view.calibration = cal
            self.mode.setCurrentText(cal.mode)
            self._refit(); self._sync_table(); self._draw_markers()

    def refresh(self):
        self._sync_table(); self._update_status(); self._draw_markers()
