"""
amcam_backend.py  --  Live capture from AmScope / ToupTek cameras.

AmScope cameras (including the model driven by "AmScope x64 4.12.x") use the
ToupTek "amcam" SDK.  This backend drives that SDK through the vendor's official
Python wrapper module `amcam.py`, which is the reliable, tested way to talk to
the hardware.

To enable live capture, two vendor files must be reachable:
    amcam.py     (the Python wrapper)
    amcam.dll    (64-bit native library)

The app looks for them in this order:
    1. The application folder  (drop both files next to this .py file), OR
    2. an already-installed `amcam` Python module, AND
    3. amcam.dll on the DLL search path -- we also probe common AmScope install
       directories and add the first match automatically.

Get the SDK free from AmScope/ToupTek ("AmScope SDK" / "ToupCam SDK"); it ships
amcam.py + x64/amcam.dll.  See README for step-by-step instructions.

If any of this is missing, constructing AmcamBackend raises, and camera.py falls
back to the synthetic / reference-image sources so the rest of the app still
works.
"""

from __future__ import annotations

import os
import sys
import glob
import ctypes
import threading
from typing import List, Optional, Tuple

import numpy as np

AMCAM_EVENT_IMAGE = 0x0004
AMCAM_EVENT_STILLIMAGE = 0x0005
AMCAM_EVENT_ERROR = 0x0080
AMCAM_EVENT_DISCONNECTED = 0x0081


def _app_roots():
    here = os.path.dirname(os.path.abspath(__file__))   # .../mpc
    app = os.path.dirname(here)                          # .../MEA Pattern Creator
    return here, app, os.path.dirname(app)               # + parent (root)


def _find_amcam_py() -> Optional[str]:
    """Return the directory containing the vendor amcam.py, or None.

    Handles a plain amcam.py dropped next to the app, and a bundled SDK folder
    such as  amcamsdk.<date>/python/amcam.py.
    """
    here, app, parent = _app_roots()
    for direct in (os.path.join(app, "amcam.py"), os.path.join(here, "amcam.py")):
        if os.path.isfile(direct):
            return os.path.dirname(direct)
    for root in (app, parent):
        for m in glob.glob(os.path.join(root, "**", "amcam.py"), recursive=True):
            if "sample" in m.lower():          # skip the SDK's example scripts
                continue
            return os.path.dirname(m)
    return None


def _rank_dll(path: str) -> int:
    """Lower is better: strongly prefer the 64-bit build."""
    p = path.lower()
    score = 0
    if os.sep + "x86" + os.sep in p or p.endswith("x86" + os.sep + "amcam.dll"):
        score += 100                            # avoid 32-bit
    if "x64" in p:
        score -= 10                             # prefer 64-bit
    if "sample" in p:
        score += 50
    return score


def _find_amcam_dll() -> Optional[str]:
    """Locate the best amcam.dll (bundled SDK win/x64, app folder, or install)."""
    here, app, parent = _app_roots()
    candidates = [
        os.path.join(app, "amcam.dll"),
        os.path.join(app, "x64", "amcam.dll"),
        os.path.join(here, "amcam.dll"),
    ]
    for root in (app, parent):                  # bundled SDK: **/win/x64/amcam.dll
        candidates += glob.glob(os.path.join(root, "**", "amcam.dll"), recursive=True)
    for base in (r"C:\Program Files", r"C:\Program Files (x86)"):
        candidates += glob.glob(os.path.join(base, "**", "amcam.dll"), recursive=True)
    existing = [c for c in candidates if c and os.path.isfile(c)]
    if not existing:
        return None
    existing.sort(key=_rank_dll)
    return existing[0]


def _probe_amcam_dll() -> Optional[str]:
    """Find amcam.dll and put its folder on the DLL search path."""
    dll = _find_amcam_dll()
    if not dll:
        return None
    d = os.path.dirname(dll)
    try:
        if hasattr(os, "add_dll_directory"):    # Python 3.8+ on Windows
            os.add_dll_directory(d)
    except Exception:
        pass
    os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    # Preload by absolute path so the vendor amcam.py binds this exact DLL even
    # though it lives in a different SDK subfolder (win/x64) from amcam.py.
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.WinDLL(dll)
        except Exception:
            pass
    return dll


def _import_amcam():
    """Import the vendor `amcam` module (from a bundled SDK if present)."""
    here, app, parent = _app_roots()
    py_dir = _find_amcam_py()
    for p in (py_dir, here, app):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    _probe_amcam_dll()
    import amcam  # type: ignore
    return amcam


class AmcamBackend:
    name = "AmScope camera"
    is_live = True
    driver_name = "amcam SDK"

    def __init__(self, device_id: Optional[str] = None):
        self._amcam = _import_amcam()
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._w = 0
        self._h = 0
        self._stride = 0
        self._buf = None
        self._bufsize = 0
        self._hcam = None
        self._device_id = device_id
        # live-stream diagnostics
        self._event_count = 0
        self._image_events = 0
        self._frame_count = 0
        self._last_error = None
        self._open()

    # -- lifecycle --------------------------------------------------------
    def _open(self):
        am = self._amcam
        devs = am.Amcam.EnumV2()
        if not devs:
            raise RuntimeError("no AmScope/ToupTek camera detected")
        dev = devs[0]
        for d in devs:
            if self._device_id and d.id == self._device_id:
                dev = d
                break
        self._device = dev
        self._hcam = am.Amcam.Open(dev.id)
        if self._hcam is None:
            raise RuntimeError("failed to open camera (in use by another app?)")
        # ask the SDK to deliver RGB byte order (matches the vendor Qt sample),
        # so we can build the image with no channel swap.
        self._rgb = False
        try:
            self._hcam.put_Option(am.AMCAM_OPTION_BYTEORDER, 0)
            self._rgb = True
        except Exception:
            self._rgb = False
        self._w, self._h = self._hcam.get_Size()
        self._alloc()

    def _alloc(self):
        # 24-bit RGB, 4-byte-aligned stride (TDIBWIDTHBYTES), per the SDK
        try:
            self._stride = self._amcam.TDIBWIDTHBYTES(self._w * 24)
        except Exception:
            self._stride = ((self._w * 24 + 31) // 32) * 4
        self._bufsize = self._stride * self._h
        # A ctypes buffer -- the SDK's PullImage* buffer arg is declared c_char_p,
        # which accepts this (a Python bytearray is REJECTED by ctypes, which was
        # silently killing every frame pull).
        self._buf = ctypes.create_string_buffer(self._bufsize)

    def start(self) -> None:
        self._hcam.StartPullModeWithCallback(self._on_event, self)

    def stop(self) -> None:
        try:
            if self._hcam:
                self._hcam.Stop()
        except Exception:
            pass

    def close(self) -> None:
        try:
            if self._hcam:
                self._hcam.Close()
        finally:
            self._hcam = None

    # -- callback / pull --------------------------------------------------
    @staticmethod
    def _on_event(event, ctx):
        self = ctx
        self._event_count += 1
        if event == AMCAM_EVENT_IMAGE:
            self._image_events += 1
            self._pull()

    def _pull(self):
        try:
            buf = self._buf                      # reused ctypes c_char buffer
            # rowPitch = 0 uses the SDK's default 4-byte-aligned stride.
            # Prefer V4 (as in the vendor samples), fall back to V3 then V2.
            if hasattr(self._hcam, "PullImageV4"):
                self._hcam.PullImageV4(buf, 0, 24, 0, None)
            elif hasattr(self._hcam, "PullImageV3"):
                self._hcam.PullImageV3(buf, 0, 24, 0, None)
            else:
                self._hcam.PullImageV2(buf, 24, None)
            arr = np.frombuffer(buf, dtype=np.uint8, count=self._bufsize)
            arr = arr.reshape(self._h, self._stride)[:, : self._w * 3]
            arr = arr.reshape(self._h, self._w, 3)
            if not self._rgb:            # SDK gave BGR -> swap to RGB
                arr = arr[:, :, ::-1]
            with self._lock:
                self._latest = np.ascontiguousarray(arr)
            self._frame_count += 1
        except Exception as e:               # noqa: BLE001
            self._last_error = f"{type(e).__name__}: {e}"

    def stream_status(self) -> dict:
        return {
            "resolution": f"{self._w}x{self._h}",
            "rgb_byte_order": self._rgb,
            "event_count": self._event_count,
            "image_events": self._image_events,
            "frame_count": self._frame_count,
            "last_error": self._last_error,
            "has_frame": self._latest is not None,
        }

    # -- CameraBackend interface -----------------------------------------
    def frame_size(self) -> Tuple[int, int]:
        return (self._w, self._h)

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest is None else self._latest

    def list_resolutions(self) -> List[Tuple[int, int]]:
        out = []
        try:
            n = self._hcam.ResolutionNumber()
            for i in range(n):
                out.append(tuple(self._hcam.get_Resolution(i)))
        except Exception:
            out = [self.frame_size()]
        return out or [self.frame_size()]

    def set_resolution(self, index: int) -> None:
        self.stop()
        self._hcam.put_eSize(index)
        self._w, self._h = self._hcam.get_Size()
        self._alloc()
        self.start()

    def set_exposure(self, microseconds: int) -> None:
        try:
            self._hcam.put_AutoExpoEnable(0)
            self._hcam.put_ExpoTime(int(microseconds))
        except Exception:
            pass

    def set_gain(self, percent: int) -> None:
        try:
            self._hcam.put_ExpoAGain(int(percent))
        except Exception:
            pass

    def set_auto_exposure(self, enabled: bool) -> None:
        try:
            self._hcam.put_AutoExpoEnable(1 if enabled else 0)
        except Exception:
            pass

    def capabilities(self) -> dict:
        return {"exposure": True, "gain": True, "resolutions": True}


# ---------------------------------------------------------------------------
# Diagnostics -- run each connection step and report exactly what happens.
# ---------------------------------------------------------------------------
def _all_dll_candidates():
    here, app, parent = _app_roots()
    cands = [
        os.path.join(app, "amcam.dll"),
        os.path.join(app, "x64", "amcam.dll"),
        os.path.join(here, "amcam.dll"),
    ]
    for root in (app, parent):
        cands += glob.glob(os.path.join(root, "**", "amcam.dll"), recursive=True)
    seen, out = set(), []
    for c in cands:
        if c and os.path.isfile(c) and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def diagnose() -> dict:
    """Run the connection steps without raising; return a structured report."""
    import traceback
    info = {
        "platform": sys.platform,
        "amcam_py_dir": None,
        "amcam_dll": None,
        "dll_candidates": [],
        "import_ok": False,
        "import_error": None,
        "enum_count": None,
        "enum_error": None,
        "devices": [],
        "open_ok": None,
        "open_error": None,
    }
    try:
        info["amcam_py_dir"] = _find_amcam_py()
    except Exception as e:                       # noqa: BLE001
        info["amcam_py_dir"] = f"<error: {e!r}>"
    try:
        info["dll_candidates"] = _all_dll_candidates()
        info["amcam_dll"] = _find_amcam_dll()
    except Exception as e:                       # noqa: BLE001
        info["amcam_dll"] = f"<error: {e!r}>"
    try:
        am = _import_amcam()
        info["import_ok"] = True
    except Exception:
        info["import_error"] = traceback.format_exc()
        return info
    try:
        devs = am.Amcam.EnumV2()
        info["enum_count"] = len(devs)
        info["devices"] = [getattr(d, "displayname", "?") for d in devs]
    except Exception:
        info["enum_error"] = traceback.format_exc()
        return info
    if info["enum_count"]:
        try:
            h = am.Amcam.Open(devs[0].id)
            info["open_ok"] = h is not None
            if h:
                h.Close()
        except Exception:
            info["open_error"] = traceback.format_exc()
    return info


def diagnose_text() -> str:
    d = diagnose()
    L = []
    L.append(f"platform          : {d['platform']}")
    L.append(f"amcam.py folder    : {d['amcam_py_dir']}")
    L.append(f"amcam.dll (chosen) : {d['amcam_dll']}")
    if d["dll_candidates"]:
        L.append("amcam.dll found    :")
        for c in d["dll_candidates"]:
            L.append(f"    {c}")
    L.append(f"import amcam        : {'OK' if d['import_ok'] else 'FAILED'}")
    if d["import_error"]:
        L.append(d["import_error"].rstrip())
        L.append(">> The vendor amcam.py or amcam.dll could not be loaded. "
                 "Check that the amcamsdk folder is inside 'MEA Pattern Creator' "
                 "and includes python\\amcam.py and win\\x64\\amcam.dll.")
        return "\n".join(L)
    if d["enum_error"]:
        L.append("enumerate cameras  : FAILED")
        L.append(d["enum_error"].rstrip())
        return "\n".join(L)
    L.append(f"cameras found      : {d['enum_count']}  {d['devices']}")
    if d["enum_count"] == 0:
        L.append(">> No camera detected. Is the AmScope camera plugged in, and is "
                 "the AmScope application CLOSED? Only one program can use the "
                 "camera at a time.")
    elif d["open_ok"] is False or d["open_error"]:
        L.append("open first camera  : FAILED")
        if d["open_error"]:
            L.append(d["open_error"].rstrip())
        L.append(">> The camera is detected but could not be opened. Close the "
                 "AmScope application (or any other app using the camera) and try "
                 "again.")
    else:
        L.append("open first camera  : OK")
        L.append(">> Camera is reachable. If live view still fails, use "
                 "Camera > Connect AmScope camera.")
    return "\n".join(L)
