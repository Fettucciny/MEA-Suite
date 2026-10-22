"""
camera.py  --  Camera backends with graceful fallback.

The GUI talks to a `CameraBackend`.  Three concrete backends are provided:

  * AmcamBackend        : live AmScope / ToupTek camera via the `amcam` SDK
                          (see amcam_backend.py).  Preferred when available.
  * ReferenceImageBackend : a still image loaded from disk, served as if it were
                          a (frozen) live feed.  Always available -- lets you
                          design patterns from a snapshot with no camera.
  * SyntheticBackend    : a generated moving test pattern, for trying the app
                          out with no hardware and no image.

`open_default_camera()` tries the AmScope SDK first and falls back to the
synthetic source, reporting what happened so the GUI can surface it.

All backends return frames as HxWx3 uint8 RGB arrays.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None       # microscope snaps can be very large


class CameraBackend:
    name = "camera"
    is_live = False

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...

    def frame_size(self) -> Tuple[int, int]:
        """(width, height) of the frames this backend yields."""
        raise NotImplementedError

    def get_frame(self) -> Optional[np.ndarray]:
        """Latest frame as HxWx3 uint8 RGB, or None if not ready."""
        raise NotImplementedError

    # optional controls (no-ops unless a backend overrides them) -----------
    def list_resolutions(self) -> List[Tuple[int, int]]:
        return [self.frame_size()]

    def set_resolution(self, index: int) -> None: ...
    def set_exposure(self, microseconds: int) -> None: ...
    def set_gain(self, percent: int) -> None: ...
    def set_auto_exposure(self, enabled: bool) -> None: ...
    def capabilities(self) -> dict:
        return {"exposure": False, "gain": False, "resolutions": False}


# ---------------------------------------------------------------------------
# Reference-image backend
# ---------------------------------------------------------------------------
class ReferenceImageBackend(CameraBackend):
    name = "reference image"
    is_live = False

    def __init__(self, image: np.ndarray, source: str = ""):
        self._img = image
        self.source = source

    @classmethod
    def from_file(cls, path: str) -> "ReferenceImageBackend":
        im = Image.open(path).convert("RGB")
        return cls(np.asarray(im), source=path)

    def frame_size(self) -> Tuple[int, int]:
        h, w = self._img.shape[:2]
        return (w, h)

    def get_frame(self) -> Optional[np.ndarray]:
        return self._img


# ---------------------------------------------------------------------------
# Synthetic backend (no hardware needed)
# ---------------------------------------------------------------------------
class SyntheticBackend(CameraBackend):
    name = "synthetic test source"
    is_live = True

    def __init__(self, width: int = 1824, height: int = 1216):
        self._w, self._h = width, height
        self._t0 = time.time()
        yy, xx = np.mgrid[0:height, 0:width]
        # a faux MEA electrode grid so ROIs have something to align to
        gx = ((xx % (width // 8)) - width // 16)
        gy = ((yy % (height // 6)) - height // 12)
        self._grid = (np.sqrt(gx ** 2 + gy ** 2) < 12).astype(np.float32)

    def frame_size(self) -> Tuple[int, int]:
        return (self._w, self._h)

    def get_frame(self) -> Optional[np.ndarray]:
        t = time.time() - self._t0
        base = 30 + 20 * np.sin(
            0.6 * t + 0.005 * np.arange(self._w))[None, :]
        img = np.broadcast_to(base, (self._h, self._w)).astype(np.float32)
        img = img + 120 * self._grid
        # a slow sweeping bright bar to show "live"
        bar = int((0.5 + 0.5 * np.sin(0.8 * t)) * (self._w - 1))
        img[:, max(0, bar - 3):bar + 3] += 60
        img = np.clip(img, 0, 255).astype(np.uint8)
        rgb = np.stack([img, img, np.clip(img.astype(int) + 15, 0, 255).astype(np.uint8)], axis=-1)
        return rgb


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def open_default_camera(prefer_live: bool = True) -> Tuple[CameraBackend, str]:
    """Return (backend, status_message).

    Tries the AmScope SDK first; on any failure falls back to a synthetic
    source so the application is always usable.
    """
    if prefer_live:
        try:
            from . import amcam_backend
            cam = amcam_backend.AmcamBackend()
            cam.start()
            return cam, f"Connected to AmScope camera via {cam.driver_name}."
        except Exception as exc:            # noqa: BLE001 - want any failure
            msg = ("AmScope camera not connected ({}). "
                   "Falling back to a synthetic test source. "
                   "Use 'Camera > Load reference image…' to design from a snapshot, "
                   "or see the README to enable live capture."
                   ).format(exc)
            cam = SyntheticBackend()
            cam.start()
            return cam, msg
    cam = SyntheticBackend()
    cam.start()
    return cam, "Using synthetic test source."
