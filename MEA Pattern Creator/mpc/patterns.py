"""
patterns.py  --  Region-of-interest (ROI) shapes and rasterization.

Shapes are stored in *canvas coordinates* -- the coordinate system of the
Polygon pattern canvas (default 1824x1216).  The GUI maps between the live
camera view and canvas coordinates using the calibration transform, so the
stored geometry is independent of whatever resolution the camera streams at.

Each shape rasterizes onto a boolean HxW array where True == illuminated.
A frame is the union (OR) of its "add" shapes minus its "subtract" shapes,
optionally combined with an imported bitmap mask.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw

Point = Tuple[float, float]


def _point_in_polygon(x: float, y: float, poly) -> bool:
    """Ray-casting point-in-polygon test (poly = list of (x, y))."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
                (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _dist_to_polyline(x: float, y: float, pts) -> float:
    """Minimum distance from (x, y) to a polyline (list of vertices)."""
    best = float("inf")
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        if seg2 <= 1e-12:
            d = ((x - ax) ** 2 + (y - ay) ** 2) ** 0.5
        else:
            t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / seg2))
            px, py = ax + t * dx, ay + t * dy
            d = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
        best = min(best, d)
    return best


# ---------------------------------------------------------------------------
# Shape model
# ---------------------------------------------------------------------------
@dataclass
class Shape:
    """A single ROI.

    kind: one of 'rect', 'ellipse', 'polygon', 'polyline', 'point'
      - rect / ellipse : points = [(x0,y0), (x1,y1)] bounding box corners
      - polygon        : points = [(x,y), ...] (filled)
      - polyline       : points = [(x,y), ...] (stroked with `width`, for
                          freehand traces / calibration-free line stimuli)
      - point          : points = [(x,y)] centre; `width` = diameter (px)
    mode: 'add' (illuminate) or 'subtract' (mask out)
    """
    kind: str
    points: List[Point] = field(default_factory=list)
    width: float = 3.0                 # stroke width / point diameter (px)
    mode: str = "add"

    # -- serialization ----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "points": [[float(x), float(y)] for x, y in self.points],
            "width": float(self.width),
            "mode": self.mode,
        }

    @staticmethod
    def from_dict(d: dict) -> "Shape":
        return Shape(
            kind=d["kind"],
            points=[(float(x), float(y)) for x, y in d["points"]],
            width=float(d.get("width", 3.0)),
            mode=d.get("mode", "add"),
        )

    # -- editing helpers --------------------------------------------------
    def copy(self) -> "Shape":
        return Shape(kind=self.kind, points=[(float(x), float(y)) for x, y in self.points],
                     width=self.width, mode=self.mode)

    def translated(self, dx: float, dy: float) -> "Shape":
        s = self.copy()
        s.points = [(x + dx, y + dy) for x, y in s.points]
        return s

    def bounds(self):
        """(minx, miny, maxx, maxy) in canvas coords, padded by half-width."""
        if not self.points:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        pad = self.width / 2.0 if self.kind in ("polyline", "point") else 0.0
        return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)

    def hit(self, x: float, y: float, tol: float = 4.0) -> bool:
        """True if the canvas point (x, y) is inside/near this shape."""
        pts = self.points
        if not pts:
            return False
        if self.kind == "rect" and len(pts) >= 2:
            (x0, y0), (x1, y1) = pts[0], pts[1]
            return (min(x0, x1) - tol <= x <= max(x0, x1) + tol and
                    min(y0, y1) - tol <= y <= max(y0, y1) + tol)
        if self.kind == "ellipse" and len(pts) >= 2:
            (x0, y0), (x1, y1) = pts[0], pts[1]
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            rx = abs(x1 - x0) / 2.0 + tol
            ry = abs(y1 - y0) / 2.0 + tol
            if rx <= 0 or ry <= 0:
                return False
            return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0
        if self.kind == "polygon" and len(pts) >= 3:
            return _point_in_polygon(x, y, pts)
        if self.kind == "polyline" and len(pts) >= 2:
            return _dist_to_polyline(x, y, pts) <= self.width / 2.0 + tol
        if self.kind == "point" and len(pts) >= 1:
            px, py = pts[0]
            r = self.width / 2.0 + tol
            return (x - px) ** 2 + (y - py) ** 2 <= r * r
        return False

    # -- rasterization ----------------------------------------------------
    def draw(self, draw_ctx: "ImageDraw.ImageDraw", value: int = 1) -> None:
        pts = self.points
        if not pts:
            return
        if self.kind == "rect" and len(pts) >= 2:
            x0, y0 = pts[0]
            x1, y1 = pts[1]
            draw_ctx.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
                               fill=value)
        elif self.kind == "ellipse" and len(pts) >= 2:
            x0, y0 = pts[0]
            x1, y1 = pts[1]
            draw_ctx.ellipse([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
                             fill=value)
        elif self.kind == "polygon" and len(pts) >= 3:
            draw_ctx.polygon([(x, y) for x, y in pts], fill=value)
        elif self.kind == "polyline" and len(pts) >= 2:
            w = max(1, int(round(self.width)))
            draw_ctx.line([(x, y) for x, y in pts], fill=value, width=w,
                          joint="curve")
            # round the endpoints so thick traces look clean
            r = w / 2.0
            for (x, y) in (pts[0], pts[-1]):
                draw_ctx.ellipse([x - r, y - r, x + r, y + r], fill=value)
        elif self.kind == "point" and len(pts) >= 1:
            x, y = pts[0]
            r = max(0.5, self.width / 2.0)
            draw_ctx.ellipse([x - r, y - r, x + r, y + r], fill=value)


# ---------------------------------------------------------------------------
# Frame model
# ---------------------------------------------------------------------------
@dataclass
class Frame:
    """One entry in a stimulation sequence.

    shapes      : vector ROIs (canvas coordinates)
    mask_b64    : optional imported 1-bit bitmap (PNG bytes, base64) that is
                  OR-ed with the rasterized shapes.  Stored at canvas size.
    duration_ms : intended display time for this frame (metadata; see notes in
                  README about how the Polygon applies timing).
    intensity   : intended LED intensity 0-100 % (metadata).
    name        : label shown in the timeline.
    invert      : if True, the whole frame is inverted after rasterization.
    """
    shapes: List[Shape] = field(default_factory=list)
    mask_b64: Optional[str] = None
    duration_ms: float = 100.0
    intensity: float = 100.0
    name: str = ""
    invert: bool = False

    # -- serialization ----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "shapes": [s.to_dict() for s in self.shapes],
            "mask_b64": self.mask_b64,
            "duration_ms": float(self.duration_ms),
            "intensity": float(self.intensity),
            "name": self.name,
            "invert": bool(self.invert),
        }

    @staticmethod
    def from_dict(d: dict) -> "Frame":
        return Frame(
            shapes=[Shape.from_dict(s) for s in d.get("shapes", [])],
            mask_b64=d.get("mask_b64"),
            duration_ms=float(d.get("duration_ms", 100.0)),
            intensity=float(d.get("intensity", 100.0)),
            name=d.get("name", ""),
            invert=bool(d.get("invert", False)),
        )

    # -- imported bitmap helpers -----------------------------------------
    def set_mask_from_image(self, img: Image.Image, width: int, height: int,
                            threshold: int = 128, fit: str = "stretch") -> None:
        """Store an imported image as a 1-bit mask at canvas resolution.

        fit: 'stretch' (resize to canvas) or 'center' (pad/crop, keep scale).
        """
        g = img.convert("L")
        if fit == "stretch":
            g = g.resize((width, height), Image.NEAREST)
            arr = np.asarray(g) >= threshold
        else:  # center
            canvas = Image.new("L", (width, height), 0)
            gw, gh = g.size
            ox = (width - gw) // 2
            oy = (height - gh) // 2
            canvas.paste(g, (ox, oy))
            arr = np.asarray(canvas) >= threshold
        self.mask_b64 = _bool_to_png_b64(arr)

    def clear_mask(self) -> None:
        self.mask_b64 = None

    def _mask_array(self, width: int, height: int) -> Optional[np.ndarray]:
        if not self.mask_b64:
            return None
        arr = _png_b64_to_bool(self.mask_b64)
        if arr.shape != (height, width):
            im = Image.fromarray(arr.astype(np.uint8) * 255).resize(
                (width, height), Image.NEAREST)
            arr = np.asarray(im) >= 128
        return arr

    # -- rasterization ----------------------------------------------------
    def rasterize(self, width: int, height: int) -> np.ndarray:
        """Return a boolean HxW array for this frame."""
        img = Image.new("L", (width, height), 0)
        d = ImageDraw.Draw(img)
        # additive first
        for s in self.shapes:
            if s.mode == "add":
                s.draw(d, value=255)
        # subtractive
        for s in self.shapes:
            if s.mode == "subtract":
                s.draw(d, value=0)
        arr = np.asarray(img) >= 128

        mask = self._mask_array(width, height)
        if mask is not None:
            arr = arr | mask
            # re-apply subtractive shapes over the imported mask
            if any(s.mode == "subtract" for s in self.shapes):
                sub = Image.new("L", (width, height), 0)
                sd = ImageDraw.Draw(sub)
                for s in self.shapes:
                    if s.mode == "subtract":
                        s.draw(sd, value=255)
                arr = arr & ~(np.asarray(sub) >= 128)

        if self.invert:
            arr = ~arr
        return arr


# ---------------------------------------------------------------------------
# PNG <-> bool helpers (used for masks and previews)
# ---------------------------------------------------------------------------
def _bool_to_png_b64(arr: np.ndarray) -> str:
    im = Image.fromarray((np.asarray(arr).astype(np.uint8) * 255)).convert("1")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _png_b64_to_bool(b64: str) -> np.ndarray:
    data = base64.b64decode(b64)
    im = Image.open(io.BytesIO(data)).convert("L")
    return np.asarray(im) >= 128


def rasterize_frames(frames: List[Frame], width: int, height: int) -> List[np.ndarray]:
    return [f.rasterize(width, height) for f in frames]
