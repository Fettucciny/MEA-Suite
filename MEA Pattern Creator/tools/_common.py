"""Shared helpers for the command-line tools in this folder.

Run any tool from the "MEA Pattern Creator" folder, e.g.
    python tools\\inspect_seq.py "..\\config\\Sept2021_SweepBRtoTL.seq"
Every tool imports the app's own mpc.seqfile, so there is exactly one .seq
reader/writer in the project.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(HERE)                  # ...\MEA Pattern Creator
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import numpy as np                               # noqa: E402
from mpc import seqfile                          # noqa: E402

W, H = seqfile.DEFAULT_WIDTH, seqfile.DEFAULT_HEIGHT
CX, CY = (W - 1) / 2.0, (H - 1) / 2.0           # canvas centre, pixel-centre convention

# Optical scale at 4X, AmScope at 1824x1216 binning, identity calibration.
# Measured 2026-10 from the 200 um electrode pitch in config\1 Solar Cell.png
# (tools\measure_mea_pitch.py).  PolyScan's 120123_Calibration_4X.map stores
# 2.70004 instead (4.9 % higher) -- unresolved, see docs\HANDOFF.md.
UM_PER_CANVAS_PX = 2.5749

# 60-electrode MEA (8x8, corners absent), 200 um pitch, 30 um diameter.
MEA_PITCH_UM = 200.0
MEA_ELECTRODE_DIAM_UM = 30.0


def electrode_centres(cx=CX, cy=CY, um_per_px=UM_PER_CANVAS_PX, rotation_deg=0.0):
    """Canvas-px centres of the 60 electrodes, keyed by MCS label 'ij'
    (i = column 1..8 left->right, j = row 1..8 top->bottom), array centred at
    (cx, cy).  Assumes the array is centred in the field -- real samples sit a
    little differently every time."""
    p = MEA_PITCH_UM / um_per_px
    a = np.deg2rad(rotation_deg)
    ca, sa = np.cos(a), np.sin(a)
    out = {}
    for i in range(8):
        for j in range(8):
            if (i, j) in ((0, 0), (0, 7), (7, 0), (7, 7)):
                continue
            dx, dy = (i - 3.5) * p, (j - 3.5) * p
            out[f"{i+1}{j+1}"] = (cx + ca * dx - sa * dy, cy + sa * dx + ca * dy)
    return out


def electrode_mask(cx=CX, cy=CY, um_per_px=UM_PER_CANVAS_PX):
    r = MEA_ELECTRODE_DIAM_UM / 2.0 / um_per_px
    yy, xx = np.mgrid[0:H, 0:W]
    m = np.zeros((H, W), bool)
    for (ex, ey) in electrode_centres(cx, cy, um_per_px).values():
        m |= np.hypot(xx - ex, yy - ey) <= r
    return m


def electrode_span(frames, mask):
    """(first, last) frame index whose lit pixels touch any electrode, or None."""
    hits = [k for k, f in enumerate(frames) if (f & mask).any()]
    return (hits[0], hits[-1]) if hits else None


def write_frames(path, frames):
    seq = seqfile.new_sequence(W, H)
    for f in frames:
        seq.add_frame(f)
    seqfile.write_seq(path, seq)
    back = seqfile.read_seq(path)
    assert back.n_frames == len(frames)
    assert all(np.array_equal(a, b) for a, b in zip(back.frames, frames))
    return os.path.getsize(path)
