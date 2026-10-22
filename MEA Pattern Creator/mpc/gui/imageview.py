"""
imageview.py  --  The interactive live-view / drawing canvas.

A QGraphicsView whose scene is in *camera pixel space*.  It shows:
  * the live (or reference/synthetic) camera image, and
  * a magenta overlay of the current frame's rasterized pattern, warped from
    canvas space into camera space using the calibration transform, so what you
    see over the sample is exactly what will be projected.

Drawing tools operate in camera space; points are converted to canvas space via
the calibration before being stored on the Frame.  This keeps stored geometry
independent of camera resolution.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from ..patterns import Shape, Frame
from ..calibration import Calibration
from .. import alignment as _align


def np_to_qimage(arr: np.ndarray) -> QtGui.QImage:
    """RGB uint8 HxWx3 -> QImage (keeps a ref to the buffer)."""
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    qimg = QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
    qimg._buf = arr  # prevent GC
    return qimg


def _force_add(shape):
    """A copy of `shape` forced to 'add' mode (for the selection highlight)."""
    s = shape.copy()
    s.mode = "add"
    return s


def np3x3_to_qtransform(H: np.ndarray) -> QtGui.QTransform:
    """Convert a column-vector 3x3 (H @ [x,y,1]) to Qt's row-vector QTransform."""
    return QtGui.QTransform(
        H[0, 0], H[1, 0], H[2, 0],
        H[0, 1], H[1, 1], H[2, 1],
        H[0, 2], H[1, 2], H[2, 2],
    )


class ImageView(QtWidgets.QGraphicsView):
    shapeAdded = QtCore.pyqtSignal()
    canvasClicked = QtCore.pyqtSignal(float, float)   # camera/scene coords (calib mode)
    referenceClicked = QtCore.pyqtSignal(float, float)  # canvas coords (fiducial)
    alignPicked = QtCore.pyqtSignal(float, float)       # canvas coords (observed)
    mouseMovedCanvas = QtCore.pyqtSignal(float, float)
    selectionChanged = QtCore.pyqtSignal()              # shape selection changed

    TOOLS = ("select", "pan", "rect", "ellipse", "polygon", "freehand", "point",
             "calibrate", "reference", "align")

    # shape clipboard shared across all views / frames (session-wide)
    _clipboard: List = []

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing |
                            QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
        # Centre the feed in the viewport when it fits; show scrollbars only
        # when the (zoomed-in) content is larger than the viewport.
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QtGui.QColor(20, 20, 24))

        self._pix_item = QtWidgets.QGraphicsPixmapItem()
        self._pix_item.setZValue(0)
        self.scene().addItem(self._pix_item)

        self._overlay_item = QtWidgets.QGraphicsPixmapItem()
        self._overlay_item.setZValue(5)
        self._overlay_item.setOpacity(0.5)
        self.scene().addItem(self._overlay_item)

        # selection highlight (yellow) drawn above the magenta pattern overlay
        self._sel_overlay_item = QtWidgets.QGraphicsPixmapItem()
        self._sel_overlay_item.setZValue(8)
        self._sel_overlay_item.setOpacity(0.9)
        self.scene().addItem(self._sel_overlay_item)

        # transient drawing item
        self._temp_item: Optional[QtWidgets.QGraphicsItem] = None
        self._poly_points: List[QtCore.QPointF] = []
        self._drawing = False
        self._start = QtCore.QPointF()

        self.tool = "pan"
        self.mode = "add"                 # add | subtract
        self.brush_width = 20.0
        self.calibration = Calibration()
        self.frame: Optional[Frame] = None
        self.canvas_size = (self.calibration.canvas_width,
                            self.calibration.canvas_height)
        self._cam_size = (0, 0)
        self._overlay_visible = True
        self._show_camera = True
        self._calib_markers: List[QtWidgets.QGraphicsItem] = []
        self._ref_markers: List[QtWidgets.QGraphicsItem] = []
        self._align_markers: List[QtWidgets.QGraphicsItem] = []
        self._align_M = None                 # 2x3 similarity preview or None
        self._fitted_once = False
        self._auto_fit = True                 # keep feed fit+centred until the
                                              # user manually zooms in
        self._last_fit_size = None            # camera size we last fitted to
        self.selected = set()                 # indices into frame.shapes
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    # -- image / overlay --------------------------------------------------
    def set_camera_image(self, arr: Optional[np.ndarray]):
        if arr is None:
            return
        h, w = arr.shape[:2]
        self._cam_size = (w, h)
        if self._show_camera:
            self._pix_item.setPixmap(QtGui.QPixmap.fromImage(np_to_qimage(arr)))
        else:
            self._pix_item.setPixmap(QtGui.QPixmap())
        # make sure identity calibration knows the camera size
        self.calibration.canvas_width, self.calibration.canvas_height = self.canvas_size
        self.calibration.fit(camera_size=self._cam_size)
        # (Re)fit only when the frame SIZE changes (first frame, or a
        # resolution/binning change) and the user hasn't taken over zoom.
        # Live frames of the same size keep the current zoom/pan.
        if self._last_fit_size != (w, h):
            self.setSceneRect(0, 0, w, h)
            if self._auto_fit:
                self.fitInView(0, 0, w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self._last_fit_size = (w, h)
            self._fitted_once = True
        self._refresh_overlay_transform()

    def fit_to_view(self):
        """Fit the whole feed into the viewport, centred, and resume auto-fit."""
        w, h = self._cam_size
        if w <= 0 or h <= 0:
            return
        self.setSceneRect(0, 0, w, h)
        self.fitInView(0, 0, w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self._auto_fit = True

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        # While the feed is fit-to-window (not manually zoomed in), keep it
        # fit + centred as the window/panels change size.
        if self._auto_fit and self._cam_size[0] > 0:
            w, h = self._cam_size
            self.fitInView(0, 0, w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def _refresh_overlay_transform(self):
        try:
            canvas_to_cam = np.linalg.inv(self.calibration.matrix())
        except np.linalg.LinAlgError:
            canvas_to_cam = np.eye(3)
        self._overlay_item.setTransform(np3x3_to_qtransform(canvas_to_cam))

    def update_overlay(self):
        if self.frame is None or not self._overlay_visible:
            self._overlay_item.setPixmap(QtGui.QPixmap())
            return
        cw, ch = self.canvas_size
        bits = self.frame.rasterize(cw, ch)
        if self._align_M is not None:
            bits = _align.warp_bool(bits, self._align_M, cw, ch)
        rgba = np.zeros((ch, cw, 4), np.uint8)
        if self._align_M is not None:
            rgba[..., 1] = 255        # green tint when showing aligned preview
            rgba[..., 2] = 120
        else:
            rgba[..., 0] = 255        # magenta
            rgba[..., 2] = 255
        rgba[..., 3] = np.where(bits, 200, 0).astype(np.uint8)
        rgba = np.ascontiguousarray(rgba)
        qimg = QtGui.QImage(rgba.data, cw, ch, 4 * cw, QtGui.QImage.Format.Format_RGBA8888)
        qimg._buf = rgba
        self._overlay_item.setPixmap(QtGui.QPixmap.fromImage(qimg))
        self._refresh_overlay_transform()
        self._update_sel_overlay()

    def _update_sel_overlay(self):
        """Highlight the currently-selected shapes in yellow."""
        self._sel_overlay_item.setTransform(self._overlay_item.transform())
        valid = [i for i in self.selected
                 if self.frame is not None and 0 <= i < len(self.frame.shapes)]
        if self.frame is None or not valid or not self._overlay_visible:
            self._sel_overlay_item.setPixmap(QtGui.QPixmap())
            return
        cw, ch = self.canvas_size
        from ..patterns import Frame as _F
        # force every selected shape to 'add' so even subtract shapes highlight
        hs = [_force_add(self.frame.shapes[i]) for i in valid]
        bits = _F(shapes=hs).rasterize(cw, ch)
        rgba = np.zeros((ch, cw, 4), np.uint8)
        rgba[..., 0] = 255            # yellow
        rgba[..., 1] = 230
        rgba[..., 3] = np.where(bits, 235, 0).astype(np.uint8)
        rgba = np.ascontiguousarray(rgba)
        qimg = QtGui.QImage(rgba.data, cw, ch, 4 * cw, QtGui.QImage.Format.Format_RGBA8888)
        qimg._buf = rgba
        self._sel_overlay_item.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def set_overlay_visible(self, v: bool):
        self._overlay_visible = v
        self.update_overlay()

    def set_overlay_opacity(self, v: float):
        self._overlay_item.setOpacity(v)

    def set_show_camera(self, v: bool):
        self._show_camera = v

    def set_frame(self, frame: Optional[Frame]):
        self.frame = frame
        self.selected = set()
        self.update_overlay()
        self.selectionChanged.emit()

    # -- shape selection & editing ---------------------------------------
    def hit_test(self, cx: float, cy: float):
        """Index of the topmost shape at canvas point (cx, cy), or None.
        Tolerance scales with the current zoom so clicking thin traces is easy."""
        if self.frame is None:
            return None
        t = self.transform().m11()
        tol = 4.0 / t if t > 1e-6 else 4.0
        for i in range(len(self.frame.shapes) - 1, -1, -1):   # topmost first
            if self.frame.shapes[i].hit(cx, cy, tol=tol):
                return i
        return None

    def select_indices(self, idxs):
        self.selected = set(i for i in idxs
                            if self.frame is not None and 0 <= i < len(self.frame.shapes))
        self._update_sel_overlay()
        self.selectionChanged.emit()

    def clear_selection(self):
        if self.selected:
            self.selected = set()
            self._update_sel_overlay()
            self.selectionChanged.emit()

    def delete_selected(self):
        if self.frame is None or not self.selected:
            return
        keep = [s for i, s in enumerate(self.frame.shapes) if i not in self.selected]
        self.frame.shapes = keep
        self.selected = set()
        self.update_overlay()
        self.shapeAdded.emit()            # main window re-syncs the shape list
        self.selectionChanged.emit()

    def copy_selected(self):
        if self.frame is None or not self.selected:
            return
        ImageView._clipboard = [self.frame.shapes[i].copy()
                                for i in sorted(self.selected)]

    def paste_clipboard(self):
        if self.frame is None or not ImageView._clipboard:
            return
        off = max(8.0, self.canvas_size[0] * 0.02)
        start = len(self.frame.shapes)
        for s in ImageView._clipboard:
            self.frame.shapes.append(s.translated(off, off))
        self.selected = set(range(start, len(self.frame.shapes)))
        self.update_overlay()
        self.shapeAdded.emit()
        self.selectionChanged.emit()

    def duplicate_selected(self):
        if self.frame is None or not self.selected:
            return
        off = max(8.0, self.canvas_size[0] * 0.02)
        originals = [self.frame.shapes[i] for i in sorted(self.selected)]
        start = len(self.frame.shapes)
        for s in originals:
            self.frame.shapes.append(s.translated(off, off))
        self.selected = set(range(start, len(self.frame.shapes)))
        self.update_overlay()
        self.shapeAdded.emit()
        self.selectionChanged.emit()

    def set_canvas_size(self, w: int, h: int):
        self.canvas_size = (w, h)
        self.calibration.canvas_width = w
        self.calibration.canvas_height = h
        self.calibration.fit(camera_size=self._cam_size)
        self.update_overlay()

    # -- tool state -------------------------------------------------------
    def set_tool(self, tool: str):
        self._cancel_temp()
        self.tool = tool
        if tool == "pan":
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        else:
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        if tool != "select":
            self.clear_selection()

    @staticmethod
    def _constrain_square(start: QtCore.QPointF, p: QtCore.QPointF) -> QtCore.QPointF:
        """Constrain the drag to a square/circle (Shift held): equal side lengths,
        preserving the drag direction from `start`."""
        dx = p.x() - start.x()
        dy = p.y() - start.y()
        s = max(abs(dx), abs(dy))
        sx = 1.0 if dx >= 0 else -1.0
        sy = 1.0 if dy >= 0 else -1.0
        return QtCore.QPointF(start.x() + sx * s, start.y() + sy * s)

    # -- coordinate helpers ----------------------------------------------
    def _scene_to_canvas(self, p: QtCore.QPointF) -> Tuple[float, float]:
        out = self.calibration.camera_to_canvas([[p.x(), p.y()]])[0]
        return float(out[0]), float(out[1])

    # -- calibration markers ---------------------------------------------
    def clear_calib_markers(self):
        for m in self._calib_markers:
            self.scene().removeItem(m)
        self._calib_markers = []

    def add_calib_marker(self, cam_xy: Tuple[float, float], label: str):
        x, y = cam_xy
        r = max(6.0, min(self._cam_size) * 0.006) if min(self._cam_size) else 8.0
        it = self.scene().addEllipse(x - r, y - r, 2 * r, 2 * r,
                                     QtGui.QPen(QtGui.QColor(0, 255, 120), 2))
        it.setZValue(20)
        txt = self.scene().addSimpleText(label)
        txt.setBrush(QtGui.QColor(0, 255, 120))
        txt.setPos(x + r, y - r)
        txt.setZValue(20)
        self._calib_markers += [it, txt]

    # -- reference / align markers (stored in canvas coords) --------------
    def _add_canvas_marker(self, store, canvas_xy, label, color):
        cam = self.calibration.canvas_to_camera([[canvas_xy[0], canvas_xy[1]]])[0]
        x, y = float(cam[0]), float(cam[1])
        r = max(6.0, min(self._cam_size) * 0.006) if min(self._cam_size) else 8.0
        it = self.scene().addEllipse(x - r, y - r, 2 * r, 2 * r, QtGui.QPen(color, 2))
        it.setZValue(22)
        txt = self.scene().addSimpleText(label)
        txt.setBrush(color); txt.setPos(x + r, y - r); txt.setZValue(22)
        store += [it, txt]

    def clear_ref_markers(self):
        for m in self._ref_markers:
            self.scene().removeItem(m)
        self._ref_markers = []

    def add_ref_marker(self, canvas_xy, label):
        self._add_canvas_marker(self._ref_markers, canvas_xy, label,
                                QtGui.QColor(0, 230, 90))

    def clear_align_markers(self):
        for m in self._align_markers:
            self.scene().removeItem(m)
        self._align_markers = []

    def add_align_marker(self, canvas_xy, label):
        self._add_canvas_marker(self._align_markers, canvas_xy, label,
                                QtGui.QColor(0, 200, 255))

    def set_align_preview(self, M):
        """M: 2x3 similarity to preview the aligned pattern, or None to clear."""
        self._align_M = M
        self.update_overlay()

    # -- events -----------------------------------------------------------
    def wheelEvent(self, ev: QtGui.QWheelEvent):
        factor = 1.25 if ev.angleDelta().y() > 0 else 1 / 1.25
        self.scale(factor, factor)
        self._update_autofit_state()

    def _update_autofit_state(self):
        """Auto-fit (centre + refit on resize) while the content fits the
        viewport; switch to zoom/pan (scrollbars) once it exceeds it."""
        w, h = self._cam_size
        if w <= 0:
            return
        t = self.transform()
        content_w = w * abs(t.m11())
        content_h = h * abs(t.m22())
        vp = self.viewport().size()
        self._auto_fit = (content_w <= vp.width() + 1 and
                          content_h <= vp.height() + 1)

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if self.tool == "pan":
            return super().mousePressEvent(ev)
        p = self.mapToScene(ev.position().toPoint())
        if ev.button() == QtCore.Qt.MouseButton.RightButton and self.tool == "polygon":
            self._finish_polygon()
            return
        if self.tool == "calibrate":
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                self.canvasClicked.emit(p.x(), p.y())
            return
        if self.tool == "reference":
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                cx, cy = self._scene_to_canvas(p)
                self.referenceClicked.emit(cx, cy)
            return
        if self.tool == "align":
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                cx, cy = self._scene_to_canvas(p)
                self.alignPicked.emit(cx, cy)
            return
        if self.tool == "select":
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
                cx, cy = self._scene_to_canvas(p)
                idx = self.hit_test(cx, cy)
                additive = bool(ev.modifiers() &
                                (QtCore.Qt.KeyboardModifier.ShiftModifier | QtCore.Qt.KeyboardModifier.ControlModifier))
                if idx is None:
                    if not additive:
                        self.clear_selection()
                elif additive:
                    sel = set(self.selected)
                    sel.symmetric_difference_update({idx})
                    self.select_indices(sel)
                else:
                    self.select_indices({idx})
            return
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return super().mousePressEvent(ev)

        if self.tool in ("rect", "ellipse"):
            self._drawing = True
            self._start = p
            pen = self._pen()
            if self.tool == "rect":
                self._temp_item = self.scene().addRect(QtCore.QRectF(p, p), pen)
            else:
                self._temp_item = self.scene().addEllipse(QtCore.QRectF(p, p), pen)
            self._temp_item.setZValue(15)
        elif self.tool == "freehand":
            self._drawing = True
            self._poly_points = [p]
            path = QtGui.QPainterPath(p)
            self._temp_item = self.scene().addPath(path, self._pen())
            self._temp_item.setZValue(15)
        elif self.tool == "polygon":
            self._poly_points.append(p)
            self._redraw_polygon(preview=p)
        elif self.tool == "point":
            cx, cy = self._scene_to_canvas(p)
            self._commit_shape(Shape("point", [(cx, cy)], width=self.brush_width,
                                     mode=self.mode))

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        p = self.mapToScene(ev.position().toPoint())
        cx, cy = self._scene_to_canvas(p)
        self.mouseMovedCanvas.emit(cx, cy)
        if self.tool in ("rect", "ellipse") and self._drawing and self._temp_item:
            q = (self._constrain_square(self._start, p)
                 if (ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier) else p)
            self._temp_item.setRect(QtCore.QRectF(self._start, q).normalized())
        elif self.tool == "freehand" and self._drawing:
            self._poly_points.append(p)
            path = QtGui.QPainterPath(self._poly_points[0])
            for pt in self._poly_points[1:]:
                path.lineTo(pt)
            self._temp_item.setPath(path)
        elif self.tool == "polygon" and self._poly_points:
            self._redraw_polygon(preview=p)
        else:
            return super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        if self.tool in ("rect", "ellipse") and self._drawing:
            self._drawing = False
            end = self.mapToScene(ev.position().toPoint())
            if ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                end = self._constrain_square(self._start, end)
            rect = QtCore.QRectF(self._start, end).normalized()
            self._cancel_temp()
            if rect.width() > 1 and rect.height() > 1:
                c0 = self._scene_to_canvas(rect.topLeft())
                c1 = self._scene_to_canvas(rect.bottomRight())
                self._commit_shape(Shape(self.tool, [c0, c1], mode=self.mode))
        elif self.tool == "freehand" and self._drawing:
            self._drawing = False
            pts = [self._scene_to_canvas(pt) for pt in self._poly_points]
            self._cancel_temp()
            self._poly_points = []
            if len(pts) >= 2:
                self._commit_shape(Shape("polyline", pts, width=self.brush_width,
                                         mode=self.mode))
        else:
            return super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent):
        if self.tool == "polygon":
            self._finish_polygon()
        elif self.tool == "pan":
            self.fit_to_view()             # double-click to reset fit + centre
        else:
            super().mouseDoubleClickEvent(ev)

    def keyPressEvent(self, ev: QtGui.QKeyEvent):
        key = ev.key()
        ctrl = bool(ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier)
        if key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter) and self.tool == "polygon":
            self._finish_polygon()
        elif key == QtCore.Qt.Key.Key_Escape:
            self._cancel_temp()
            self._poly_points = []
            self.clear_selection()
        elif key in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace) and self.selected:
            self.delete_selected()
        elif ctrl and key == QtCore.Qt.Key.Key_C:
            self.copy_selected()
        elif ctrl and key == QtCore.Qt.Key.Key_V:
            self.paste_clipboard()
        elif ctrl and key == QtCore.Qt.Key.Key_D:
            self.duplicate_selected()
        elif ctrl and key == QtCore.Qt.Key.Key_A and self.tool == "select" and self.frame:
            self.select_indices(range(len(self.frame.shapes)))
        else:
            super().keyPressEvent(ev)

    # -- drawing helpers --------------------------------------------------
    def _pen(self):
        col = QtGui.QColor(255, 60, 255) if self.mode == "add" else QtGui.QColor(80, 200, 255)
        pen = QtGui.QPen(col, 0)
        pen.setCosmetic(True)
        return pen

    def _redraw_polygon(self, preview: Optional[QtCore.QPointF] = None):
        self._cancel_temp()
        pts = list(self._poly_points)
        if preview is not None:
            pts = pts + [preview]
        if len(pts) >= 1:
            path = QtGui.QPainterPath(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)
            self._temp_item = self.scene().addPath(path, self._pen())
            self._temp_item.setZValue(15)

    def _finish_polygon(self):
        pts = [self._scene_to_canvas(pt) for pt in self._poly_points]
        self._cancel_temp()
        self._poly_points = []
        if len(pts) >= 3:
            self._commit_shape(Shape("polygon", pts, mode=self.mode))

    def _cancel_temp(self):
        if self._temp_item is not None:
            self.scene().removeItem(self._temp_item)
            self._temp_item = None

    def _commit_shape(self, shape: Shape):
        if self.frame is not None:
            self.frame.shapes.append(shape)
            self.update_overlay()
            self.shapeAdded.emit()
