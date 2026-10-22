"""
alignment_panel.py  --  Align a saved pattern to the current sample.

Two steps:
  A. Mark reference electrodes ON THE PATTERN (design time).  Click identifiable
     electrodes (usually edge/corner ones) on the layout; they save with the
     pattern as canvas-space fiducials.
  B. Align to the current sample (each time you load one).  Click the same
     electrodes, in the same order, on the live feed.  A similarity transform
     (shift + rotation + uniform scale) is fit that maps the pattern's reference
     electrodes onto this sample's real electrode positions.  Preview it in
     green, then export the aligned .seq next to your original.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
from PyQt6 import QtCore, QtWidgets

from .. import alignment as _align


class AlignmentPanel(QtWidgets.QWidget):
    def __init__(self, view, get_project, parent=None):
        super().__init__(parent)
        self.view = view
        self.get_project = get_project
        self._observed: List[Tuple[float, float]] = []
        self._matrix = None
        self._build()
        self.view.referenceClicked.connect(self._on_reference_click)
        self.view.alignPicked.connect(self._on_align_click)

    # -- ui ---------------------------------------------------------------
    def _build(self):
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)

        lay.addWidget(QtWidgets.QLabel("<b>A. Reference electrodes (on pattern)</b>"))
        self.btn_mark = QtWidgets.QPushButton("Mark reference electrodes")
        self.btn_mark.setCheckable(True)
        self.btn_mark.toggled.connect(self._toggle_mark)
        lay.addWidget(self.btn_mark)
        self.ref_list = QtWidgets.QListWidget()
        self.ref_list.setMaximumHeight(110)
        lay.addWidget(self.ref_list)
        rrow = QtWidgets.QHBoxLayout()
        b_undo = QtWidgets.QPushButton("Undo"); b_undo.clicked.connect(self._undo_ref)
        b_clear = QtWidgets.QPushButton("Clear"); b_clear.clicked.connect(self._clear_ref)
        rrow.addWidget(b_undo); rrow.addWidget(b_clear)
        lay.addLayout(rrow)

        lay.addSpacing(8)
        lay.addWidget(QtWidgets.QLabel("<b>B. Align to current sample</b>"))
        self.btn_align = QtWidgets.QPushButton("Pick electrodes on live feed")
        self.btn_align.setCheckable(True)
        self.btn_align.toggled.connect(self._toggle_align)
        lay.addWidget(self.btn_align)

        self.chk_scale = QtWidgets.QCheckBox("allow uniform scale")
        self.chk_scale.setChecked(True)
        self.chk_scale.toggled.connect(self._refit)
        lay.addWidget(self.chk_scale)

        self.status = QtWidgets.QLabel("Mark reference electrodes first.")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        self.result = QtWidgets.QLabel("")
        self.result.setWordWrap(True)
        self.result.setStyleSheet("color:#8fd0ff;")
        lay.addWidget(self.result)

        self.chk_preview = QtWidgets.QCheckBox("show aligned preview (green)")
        self.chk_preview.setChecked(True)
        self.chk_preview.toggled.connect(self._update_preview)
        lay.addWidget(self.chk_preview)

        arow = QtWidgets.QHBoxLayout()
        b_reset = QtWidgets.QPushButton("Reset picks"); b_reset.clicked.connect(self._reset_align)
        arow.addWidget(b_reset)
        lay.addLayout(arow)

        self.btn_export = QtWidgets.QPushButton("Export aligned .seq…")
        self.btn_export.clicked.connect(self._export_aligned)
        self.btn_export.setEnabled(False)
        lay.addWidget(self.btn_export)
        lay.addStretch(1)

    # -- reference electrodes --------------------------------------------
    def _toggle_mark(self, on: bool):
        if on:
            self.btn_align.setChecked(False)
            self.view.set_tool("reference")
            self.btn_mark.setText("Marking… click electrodes")
        else:
            self.view.set_tool("pan")
            self.btn_mark.setText("Mark reference electrodes")

    def _on_reference_click(self, cx, cy):
        self.get_project().reference_fiducials.append((cx, cy))
        self._sync_ref()

    def _sync_ref(self):
        proj = self.get_project()
        self.ref_list.clear()
        self.view.clear_ref_markers()
        for i, (x, y) in enumerate(proj.reference_fiducials):
            self.ref_list.addItem(f"E{i+1}: ({x:.0f}, {y:.0f})")
            self.view.add_ref_marker((x, y), f"E{i+1}")
        n = len(proj.reference_fiducials)
        if n < 2:
            self.status.setText("Mark at least 2 reference electrodes (3+ recommended).")
        else:
            self.status.setText(f"{n} reference electrodes set. "
                                "Load a sample, then pick them on the live feed.")

    def _undo_ref(self):
        proj = self.get_project()
        if proj.reference_fiducials:
            proj.reference_fiducials.pop()
            self._sync_ref()

    def _clear_ref(self):
        self.get_project().reference_fiducials.clear()
        self._sync_ref()

    # -- align to sample --------------------------------------------------
    def _toggle_align(self, on: bool):
        proj = self.get_project()
        if on and len(proj.reference_fiducials) < 2:
            QtWidgets.QMessageBox.information(
                self, "Mark electrodes first",
                "Mark at least 2 reference electrodes on the pattern before aligning.")
            self.btn_align.setChecked(False)
            return
        if on:
            self.btn_mark.setChecked(False)
            self._observed = []
            self.view.clear_align_markers()
            self.view.set_tool("align")
            self._update_align_status()
        else:
            self.view.set_tool("pan")
            self.btn_align.setText("Pick electrodes on live feed")

    def _on_align_click(self, cx, cy):
        proj = self.get_project()
        if len(self._observed) >= len(proj.reference_fiducials):
            return
        self._observed.append((cx, cy))
        self.view.add_align_marker((cx, cy), str(len(self._observed)))
        self._refit()
        self._update_align_status()
        if len(self._observed) >= len(proj.reference_fiducials):
            self.btn_align.setChecked(False)

    def _update_align_status(self):
        proj = self.get_project()
        k = len(self._observed)
        n = len(proj.reference_fiducials)
        if k < n:
            self.status.setText(f"Click electrode E{k+1} on the live feed "
                                f"({k}/{n} picked).")
        else:
            self.status.setText(f"All {n} electrodes picked.")

    def _refit(self, *_):
        proj = self.get_project()
        k = min(len(self._observed), len(proj.reference_fiducials))
        if k < 2:
            self._matrix = None
            self.result.setText("")
            self.btn_export.setEnabled(False)
            self._update_preview()
            return
        src = np.array(proj.reference_fiducials[:k], float)
        dst = np.array(self._observed[:k], float)
        M = _align.fit_similarity(src, dst, allow_scale=self.chk_scale.isChecked())
        self._matrix = M
        info = _align.decompose_similarity(M)
        rms = _align.rms(M, src, dst)
        self.result.setText(
            f"shift = ({info['tx']:.1f}, {info['ty']:.1f}) px\n"
            f"rotation = {info['rotation_deg']:.2f}°   scale = {info['scale']:.4f}\n"
            f"fit RMS = {rms:.2f} canvas px  ({k} electrodes)")
        self.btn_export.setEnabled(True)
        self._update_preview()

    def _update_preview(self, *_):
        if self._matrix is not None and self.chk_preview.isChecked():
            self.view.set_align_preview(self._matrix)
        else:
            self.view.set_align_preview(None)

    def _reset_align(self):
        self._observed = []
        self._matrix = None
        self.result.setText("")
        self.btn_export.setEnabled(False)
        self.view.clear_align_markers()
        self.view.set_align_preview(None)
        self._update_align_status()

    def _export_aligned(self):
        if self._matrix is None:
            return
        proj = self.get_project()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export aligned .seq", "pattern_aligned.seq", "Polygon sequence (*.seq)")
        if not path:
            return
        try:
            proj.export_aligned_seq(path, self._matrix)
            QtWidgets.QMessageBox.information(
                self, "Aligned sequence exported",
                f"Wrote aligned pattern:\n{path}\n\n"
                "Your original (unaligned) pattern is unchanged — export it "
                "separately with File → Export .seq if you also want it saved.")
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Export failed", str(e))

    # -- external refresh -------------------------------------------------
    def refresh(self):
        self._reset_align()
        self._sync_ref()
