"""
project.py  --  The editable document model.

A Project holds everything the user builds: the canvas size, the camera<->canvas
calibration, and the ordered list of stimulation frames (each a set of ROIs /
imported masks with timing + intensity metadata).

Projects save to a human-readable JSON file (`.mpcproj`).  Exporting produces a
Polygon-compatible `.seq` by rasterizing every frame to the canvas and handing
the boolean stack to seqfile.

A sidecar `<name>.timing.json` is written next to the .seq on export, recording
each frame's intended duration and intensity.  The reference .seq format does
not carry unambiguous per-frame timing (the Polygon applies playback timing when
the sequence runs), so the sidecar preserves the user's intent and can be re-read
into the app.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import seqfile
from . import alignment as _align
from .patterns import Frame
from .calibration import Calibration

PROJECT_VERSION = 2


@dataclass
class Project:
    canvas_width: int = seqfile.DEFAULT_WIDTH
    canvas_height: int = seqfile.DEFAULT_HEIGHT
    frames: List[Frame] = field(default_factory=list)
    calibration: Calibration = field(default_factory=Calibration)
    reference_image_path: Optional[str] = None
    # reference electrodes (fiducials) marked on the pattern, canvas coords.
    # Used to align the pattern to each sample's real electrode positions.
    reference_fiducials: List = field(default_factory=list)   # [(x, y), ...]
    # advanced header params carried into the .seq
    param_a: int = seqfile.DEFAULT_PARAM_A
    param_b: int = seqfile.DEFAULT_PARAM_B
    param_c: int = seqfile.DEFAULT_PARAM_C
    # base pattern period (ms) used to expand per-frame durations into repeated
    # frames on a "timed" export. Set PolyScan's Pattern Period to this value.
    base_period_ms: float = 10.0

    # -- construction -----------------------------------------------------
    def __post_init__(self):
        self.calibration.canvas_width = self.canvas_width
        self.calibration.canvas_height = self.canvas_height

    def new_frame(self, name: str = "") -> Frame:
        f = Frame(name=name or f"Frame {len(self.frames) + 1}")
        self.frames.append(f)
        return f

    # -- pulse-train builder ---------------------------------------------
    def build_pulse_train(self, on_shapes, n_pulses: int, on_ms: float,
                          off_ms: float, rest_ms: float, intensity: float = 100.0,
                          replace: bool = True):
        """Generate frames for a burst: n_pulses of (on_ms ON, off_ms OFF),
        then one rest_ms blank.  `on_shapes` is the pattern shown during each ON
        (a list of patterns.Shape, e.g. the current frame's shapes)."""
        import copy as _copy
        made = []
        for k in range(int(n_pulses)):
            on = Frame(shapes=[s.copy() for s in on_shapes],
                       duration_ms=float(on_ms), intensity=float(intensity),
                       name=f"pulse {k + 1}")
            made.append(on)
            if off_ms > 0:
                made.append(Frame(shapes=[], duration_ms=float(off_ms),
                                  name=f"gap {k + 1}"))
        if rest_ms > 0:
            made.append(Frame(shapes=[], duration_ms=float(rest_ms), name="rest"))
        if replace:
            self.frames = made
        else:
            self.frames.extend(made)
        return made

    # -- export -----------------------------------------------------------
    def to_sequence(self) -> seqfile.Sequence:
        if not self.frames:
            raise ValueError("project has no frames to export")
        seq = seqfile.new_sequence(self.canvas_width, self.canvas_height)
        seq.header.param_a = self.param_a
        seq.header.param_b = self.param_b
        seq.header.param_c = self.param_c
        for fr in self.frames:
            seq.add_frame(fr.rasterize(self.canvas_width, self.canvas_height))
        return seq

    def export_seq(self, path: str, write_timing_sidecar: bool = True) -> None:
        seq = self.to_sequence()
        seqfile.write_seq(path, seq)
        if write_timing_sidecar:
            self._write_sidecars(path)

    # -- duration-aware ("timed") export ---------------------------------
    def expansion_plan(self, base_period_ms: float) -> dict:
        """How each frame expands at a given base period. Returns totals, the
        per-frame repeat counts, and any rounding warnings."""
        base = float(base_period_ms)
        reps, warnings = [], []
        for i, fr in enumerate(self.frames):
            r = max(1, int(round(fr.duration_ms / base))) if base > 0 else 1
            reps.append(r)
            actual = r * base
            if abs(actual - fr.duration_ms) > 1e-6:
                warnings.append(
                    f"'{fr.name}': {fr.duration_ms:g} ms → {r}×{base:g} = "
                    f"{actual:g} ms")
        total_frames = sum(reps)
        total_ms = total_frames * base
        return {"reps": reps, "total_frames": total_frames,
                "total_ms": total_ms, "base_period_ms": base,
                "warnings": warnings}

    def timed_sequence(self, base_period_ms: float) -> seqfile.Sequence:
        """Rasterize every frame and repeat it round(duration/base_period) times
        so the sequence plays with the right timing when PolyScan's Pattern
        Period equals base_period_ms."""
        if not self.frames:
            raise ValueError("project has no frames to export")
        base = float(base_period_ms)
        seq = seqfile.new_sequence(self.canvas_width, self.canvas_height)
        seq.header.param_a = self.param_a
        seq.header.param_b = self.param_b
        seq.header.param_c = self.param_c
        for fr in self.frames:
            bits = fr.rasterize(self.canvas_width, self.canvas_height)
            reps = max(1, int(round(fr.duration_ms / base))) if base > 0 else 1
            for _ in range(reps):
                seq.add_frame(bits)
        return seq

    def export_timed_seq(self, path: str, base_period_ms: float,
                         write_timing_sidecar: bool = True) -> dict:
        plan = self.expansion_plan(base_period_ms)
        seq = self.timed_sequence(base_period_ms)
        seqfile.write_seq(path, seq)
        if write_timing_sidecar:
            base = os.path.splitext(path)[0]
            with open(base + ".timing.json", "w") as fh:
                json.dump({
                    "canvas": [self.canvas_width, self.canvas_height],
                    "timed": True,
                    "base_period_ms": float(base_period_ms),
                    "polyscan_pattern_period_ms": float(base_period_ms),
                    "total_frames": plan["total_frames"],
                    "total_ms": plan["total_ms"],
                    "frames": [
                        {"index": i, "name": fr.name,
                         "duration_ms": fr.duration_ms,
                         "intensity_pct": fr.intensity,
                         "repeats": plan["reps"][i]}
                        for i, fr in enumerate(self.frames)
                    ],
                }, fh, indent=2)
        return plan

    def _write_sidecars(self, path: str) -> None:
        base = os.path.splitext(path)[0]
        with open(base + ".timing.json", "w") as fh:
            json.dump({
                "canvas": [self.canvas_width, self.canvas_height],
                "frames": [
                    {"index": i, "name": fr.name,
                     "duration_ms": fr.duration_ms,
                     "intensity_pct": fr.intensity}
                    for i, fr in enumerate(self.frames)
                ],
            }, fh, indent=2)
        # reference electrodes so a previously exported .seq can be re-aligned
        if self.reference_fiducials:
            with open(base + ".align.json", "w") as fh:
                json.dump({
                    "canvas": [self.canvas_width, self.canvas_height],
                    "reference_fiducials": [[float(x), float(y)]
                                            for x, y in self.reference_fiducials],
                }, fh, indent=2)

    # -- alignment / aligned export --------------------------------------
    def aligned_sequence(self, matrix) -> seqfile.Sequence:
        """Rasterize every frame and warp it by a 2x3 similarity matrix."""
        if not self.frames:
            raise ValueError("project has no frames to export")
        seq = seqfile.new_sequence(self.canvas_width, self.canvas_height)
        seq.header.param_a = self.param_a
        seq.header.param_b = self.param_b
        seq.header.param_c = self.param_c
        M = np.asarray(matrix, dtype=float)
        for fr in self.frames:
            bits = fr.rasterize(self.canvas_width, self.canvas_height)
            warped = _align.warp_bool(bits, M, self.canvas_width, self.canvas_height)
            seq.add_frame(warped)
        return seq

    def export_aligned_seq(self, path: str, matrix,
                           write_timing_sidecar: bool = True) -> None:
        seq = self.aligned_sequence(matrix)
        seqfile.write_seq(path, seq)
        if write_timing_sidecar:
            base = os.path.splitext(path)[0]
            with open(base + ".timing.json", "w") as fh:
                json.dump({
                    "canvas": [self.canvas_width, self.canvas_height],
                    "aligned": True,
                    "transform_2x3": [list(map(float, row)) for row in np.asarray(matrix)],
                    "frames": [
                        {"index": i, "name": fr.name,
                         "duration_ms": fr.duration_ms,
                         "intensity_pct": fr.intensity}
                        for i, fr in enumerate(self.frames)
                    ],
                }, fh, indent=2)

    # -- import an existing .seq into editable frames --------------------
    @staticmethod
    def from_seq(path: str) -> "Project":
        seq = seqfile.read_seq(path)
        proj = Project(canvas_width=seq.width, canvas_height=seq.height)
        proj.param_a = seq.header.param_a
        proj.param_b = seq.header.param_b
        proj.param_c = seq.header.param_c
        from .patterns import _bool_to_png_b64
        for i, bits in enumerate(seq.frames):
            fr = Frame(name=f"Frame {i + 1}")
            fr.mask_b64 = _bool_to_png_b64(bits)
            proj.frames.append(fr)
        # read timing sidecar if present
        base = os.path.splitext(path)[0]
        side = base + ".timing.json"
        if os.path.isfile(side):
            try:
                meta = json.load(open(side))
                for m in meta.get("frames", []):
                    idx = m.get("index")
                    if idx is not None and 0 <= idx < len(proj.frames):
                        proj.frames[idx].duration_ms = float(m.get("duration_ms", 100))
                        proj.frames[idx].intensity = float(m.get("intensity_pct", 100))
                        if m.get("name"):
                            proj.frames[idx].name = m["name"]
            except Exception:
                pass
        # read reference-electrode sidecar if present
        aside = base + ".align.json"
        if os.path.isfile(aside):
            try:
                a = json.load(open(aside))
                proj.reference_fiducials = [(float(x), float(y))
                                            for x, y in a.get("reference_fiducials", [])]
            except Exception:
                pass
        return proj

    # -- serialization ----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": PROJECT_VERSION,
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "reference_image_path": self.reference_image_path,
            "param_a": self.param_a,
            "param_b": self.param_b,
            "param_c": self.param_c,
            "base_period_ms": self.base_period_ms,
            "reference_fiducials": [[float(x), float(y)]
                                    for x, y in self.reference_fiducials],
            "calibration": self.calibration.to_dict(),
            "frames": [fr.to_dict() for fr in self.frames],
        }

    @staticmethod
    def from_dict(d: dict) -> "Project":
        proj = Project(
            canvas_width=int(d.get("canvas_width", seqfile.DEFAULT_WIDTH)),
            canvas_height=int(d.get("canvas_height", seqfile.DEFAULT_HEIGHT)),
            reference_image_path=d.get("reference_image_path"),
            param_a=int(d.get("param_a", seqfile.DEFAULT_PARAM_A)),
            param_b=int(d.get("param_b", seqfile.DEFAULT_PARAM_B)),
            param_c=int(d.get("param_c", seqfile.DEFAULT_PARAM_C)),
        )
        proj.base_period_ms = float(d.get("base_period_ms", 10.0))
        if "calibration" in d:
            proj.calibration = Calibration.from_dict(d["calibration"])
        proj.reference_fiducials = [(float(x), float(y))
                                    for x, y in d.get("reference_fiducials", [])]
        proj.frames = [Frame.from_dict(f) for f in d.get("frames", [])]
        return proj

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @staticmethod
    def load(path: str) -> "Project":
        with open(path) as fh:
            return Project.from_dict(json.load(fh))
