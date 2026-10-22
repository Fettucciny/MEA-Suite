"""
alignment.py  --  Register a saved pattern to the actual sample.

The pattern is designed once over an idealized electrode layout, with a handful
of *reference electrodes* (fiducials) marked on identifiable electrodes (canvas
coordinates).  Each sample sits slightly differently in the ROI, so before
projecting we align the pattern to where the electrodes actually are.

Workflow:
  * mark reference electrodes on the pattern    -> ref (canvas coords)
  * click those same electrodes on the live feed -> obs (mapped to canvas coords
    through the remembered camera calibration)
  * fit a similarity transform (rotation + uniform scale + translation) that maps
    ref -> obs, apply it to the whole pattern, and export the aligned .seq.

The fit uses the Umeyama closed-form least-squares similarity estimate.  Applying
the transform warps the rasterized frames (so rectangles / ellipses / imported
masks all rotate correctly), rather than transforming vector control points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Similarity fit (Umeyama)
# ---------------------------------------------------------------------------
def fit_similarity(src: np.ndarray, dst: np.ndarray,
                   allow_scale: bool = True) -> np.ndarray:
    """Least-squares similarity mapping src -> dst.

    src, dst : (N, 2) arrays of corresponding points (N >= 2).
    Returns a 2x3 affine [sR | t] (rotation + uniform scale + translation).
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    n = src.shape[0]
    if n < 2:
        raise ValueError("need at least 2 point pairs to align")

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d

    var_s = (sc ** 2).sum() / n
    cov = (dc.T @ sc) / n                      # 2x2

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1                            # reflection guard
    R = U @ S @ Vt
    if allow_scale:
        scale = (D * np.diag(S)).sum() / (var_s + 1e-12)
    else:
        scale = 1.0
    t = mu_d - scale * (R @ mu_s)

    M = np.zeros((2, 3))
    M[:2, :2] = scale * R
    M[:, 2] = t
    return M


def decompose_similarity(M: np.ndarray) -> dict:
    """Return {'scale','rotation_deg','tx','ty'} from a 2x3 similarity matrix."""
    a, b = M[0, 0], M[0, 1]
    scale = math.hypot(a, b)
    rot = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    return {"scale": scale, "rotation_deg": rot, "tx": M[0, 2], "ty": M[1, 2]}


def apply_affine(M: np.ndarray, pts) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    return (M[:, :2] @ pts.T).T + M[:, 2]


def residuals(M: np.ndarray, src, dst) -> np.ndarray:
    pred = apply_affine(M, src)
    return np.sqrt(((pred - np.asarray(dst, float)) ** 2).sum(axis=1))


def rms(M: np.ndarray, src, dst) -> float:
    r = residuals(M, src, dst)
    return float(np.sqrt((r ** 2).mean()))


# ---------------------------------------------------------------------------
# Raster warp
# ---------------------------------------------------------------------------
def _affine_to_pil(M: np.ndarray) -> Tuple[float, ...]:
    """PIL Image.transform(AFFINE) maps OUTPUT->INPUT, i.e. it needs the inverse
    of a forward (input->output) transform.  Build 3x3, invert, take top rows."""
    H = np.eye(3)
    H[:2, :] = M
    Hinv = np.linalg.inv(H)
    a, b, c = Hinv[0]
    d, e, f = Hinv[1]
    return (a, b, c, d, e, f)


def warp_bool(frame: np.ndarray, M: np.ndarray,
              out_w: Optional[int] = None, out_h: Optional[int] = None) -> np.ndarray:
    """Apply forward similarity M (pattern -> aligned) to a boolean frame."""
    h, w = frame.shape
    out_w = out_w or w
    out_h = out_h or h
    im = Image.fromarray((np.asarray(frame).astype(np.uint8) * 255))
    coeffs = _affine_to_pil(M)
    warped = im.transform((out_w, out_h), Image.AFFINE, coeffs,
                          resample=Image.NEAREST, fillcolor=0)
    return np.asarray(warped) >= 128


@dataclass
class Alignment:
    """A computed pattern->sample alignment."""
    matrix: np.ndarray                 # 2x3 similarity (pattern -> aligned)
    rms_px: float
    n_points: int

    @property
    def info(self) -> dict:
        d = decompose_similarity(self.matrix)
        d["rms_px"] = self.rms_px
        d["n_points"] = self.n_points
        return d
