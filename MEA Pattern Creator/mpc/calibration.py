"""
calibration.py  --  Geometric mapping between the live camera image and the
Polygon pattern canvas.

Two coordinate spaces:
  * camera space  : pixels of the live/snapped microscope image the user draws on
  * canvas space  : pixels of the Polygon pattern canvas (default 1824x1216)
                    that get written into the .seq file.

The user establishes correspondences (a point clicked in the camera image and
the canvas coordinate it should map to) and we fit a transform:
  * 'affine'     : >=3 point pairs, least-squares 2x3 affine.
  * 'homography' : >=4 point pairs, DLT 3x3 projective (handles perspective /
                   keystone from the DMD projection).
  * 'identity'   : camera pixels already equal canvas pixels, up to a uniform
                   scale (useful when the reference snapshot is taken at canvas
                   resolution and the Polygon's own calibration is trusted).

Everything is pure NumPy -- no OpenCV dependency.

Typical high-accuracy workflow (no direct hardware control needed):
  1. Export a calibration pattern (dots at known canvas coordinates) as a .seq.
  2. Load & project it with the Polygon software; snap the camera image.
  3. Click each projected dot in this app; the known canvas coordinate is paired
     with the clicked camera coordinate.
  4. Fit a homography -> drawings in camera space now map accurately to canvas.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np

Pair = Tuple[Tuple[float, float], Tuple[float, float]]  # (camera_xy, canvas_xy)


def default_store_path() -> str:
    """App-level file where the (hardware-fixed) camera calibration is remembered."""
    here = os.path.dirname(os.path.abspath(__file__))          # .../mpc
    return os.path.join(os.path.dirname(here), "camera_calibration.json")


# ---------------------------------------------------------------------------
# Transform fitting
# ---------------------------------------------------------------------------
def fit_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares 2x3 affine mapping src -> dst. src/dst: (N,2)."""
    n = src.shape[0]
    A = np.zeros((2 * n, 6))
    b = np.zeros((2 * n,))
    for i in range(n):
        x, y = src[i]
        A[2 * i] = [x, y, 1, 0, 0, 0]
        A[2 * i + 1] = [0, 0, 0, x, y, 1]
        b[2 * i] = dst[i, 0]
        b[2 * i + 1] = dst[i, 1]
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    M = np.array([[sol[0], sol[1], sol[2]],
                  [sol[3], sol[4], sol[5]],
                  [0, 0, 1]])
    return M


def fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """DLT 3x3 homography mapping src -> dst (normalized for stability)."""
    def _normalize(pts):
        c = pts.mean(axis=0)
        d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
        s = np.sqrt(2) / (d + 1e-12)
        T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
        ph = np.hstack([pts, np.ones((pts.shape[0], 1))])
        return (T @ ph.T).T[:, :2], T

    sN, Ts = _normalize(src)
    dN, Td = _normalize(dst)
    n = src.shape[0]
    A = np.zeros((2 * n, 9))
    for i in range(n):
        x, y = sN[i]
        u, v = dN[i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]
    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def apply_transform(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homogeneous transform to (N,2) points."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    ph = np.hstack([pts, np.ones((pts.shape[0], 1))])
    out = (M @ ph.T).T
    out = out[:, :2] / out[:, 2:3]
    return out


# ---------------------------------------------------------------------------
# Calibration container
# ---------------------------------------------------------------------------
@dataclass
class Calibration:
    canvas_width: int = 1824
    canvas_height: int = 1216
    mode: str = "identity"                  # identity | affine | homography
    pairs: List[Pair] = field(default_factory=list)
    # cached camera->canvas transform (3x3); None for identity
    _M: Optional[np.ndarray] = None

    # -- fitting ----------------------------------------------------------
    def fit(self, camera_size: Optional[Tuple[int, int]] = None) -> None:
        """Recompute the transform from the current point pairs.

        camera_size (w,h) is used only for the identity fallback so a reference
        snapshot at a different resolution still maps onto the canvas.
        """
        if self.mode == "identity" or len(self.pairs) < 3:
            if camera_size and camera_size[0] > 0 and camera_size[1] > 0:
                cw, ch = camera_size
                sx = self.canvas_width / cw
                sy = self.canvas_height / ch
                self._M = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
            else:
                self._M = np.eye(3)
            return
        src = np.array([p[0] for p in self.pairs], dtype=float)
        dst = np.array([p[1] for p in self.pairs], dtype=float)
        if self.mode == "homography" and len(self.pairs) >= 4:
            self._M = fit_homography(src, dst)
        else:
            self._M = fit_affine(src, dst)

    def matrix(self) -> np.ndarray:
        if self._M is None:
            self.fit()
        return self._M

    # -- mapping ----------------------------------------------------------
    def camera_to_canvas(self, pts) -> np.ndarray:
        return apply_transform(self.matrix(), pts)

    def canvas_to_camera(self, pts) -> np.ndarray:
        return apply_transform(np.linalg.inv(self.matrix()), pts)

    # -- diagnostics ------------------------------------------------------
    def residuals(self) -> Optional[np.ndarray]:
        """Per-pair mapping error in canvas pixels (RMS diagnostics)."""
        if not self.pairs or self._M is None:
            return None
        src = np.array([p[0] for p in self.pairs], dtype=float)
        dst = np.array([p[1] for p in self.pairs], dtype=float)
        pred = self.camera_to_canvas(src)
        return np.sqrt(((pred - dst) ** 2).sum(axis=1))

    def rms_error(self) -> Optional[float]:
        r = self.residuals()
        return float(np.sqrt((r ** 2).mean())) if r is not None else None

    # -- serialization ----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "mode": self.mode,
            "pairs": [[list(a), list(b)] for a, b in self.pairs],
        }

    @staticmethod
    def from_dict(d: dict) -> "Calibration":
        c = Calibration(
            canvas_width=int(d.get("canvas_width", 1824)),
            canvas_height=int(d.get("canvas_height", 1216)),
            mode=d.get("mode", "identity"),
            pairs=[((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
                   for a, b in d.get("pairs", [])],
        )
        c.fit()
        return c

    def save(self, path) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @staticmethod
    def load(path) -> "Calibration":
        with open(path) as fh:
            return Calibration.from_dict(json.load(fh))


# ---------------------------------------------------------------------------
# Calibration target generation
# ---------------------------------------------------------------------------
def calibration_dot_canvas_points(width: int, height: int,
                                  cols: int = 3, rows: int = 3,
                                  margin: float = 0.15) -> List[Tuple[float, float]]:
    """Grid of canvas coordinates used as calibration dots."""
    xs = np.linspace(width * margin, width * (1 - margin), cols)
    ys = np.linspace(height * margin, height * (1 - margin), rows)
    return [(float(x), float(y)) for y in ys for x in xs]
