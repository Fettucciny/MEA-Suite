#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 mea_nap_polygon_seq.py  --  read Mightex Polygon (.seq) DMD pattern sequences
================================================================================
The MCS Digital Event Recorder only stores the TRIGGER pulses that clock the
Mightex Polygon 1000 to its next frame. The *spatial* pattern projected onto the
sample at each trigger lives in the Polygon ".seq" file. This module decodes that
file so the pipeline knows exactly WHAT was projected WHERE, at each trigger.

File format (magic "1QSM" = MSQ1, Mightex sequence v1), reverse-engineered and
verified on Sept2021_SweepBRtoTL.seq:
  - header holds, at fixed offsets, nframes (@4228), width (@4232), height (@4236)
    as little-endian uint32;
  - pixel data starts at  filesize - nframes * framebytes ;
  - each frame is a 1-bit-packed bitmap, row-major, MSB-first, rowbytes =
    ceil(width/8), framebytes = rowbytes * height  (1 = mirror ON / laser on).

For Sept2021_SweepBRtoTL.seq this decodes to 160 frames of 1824x1216 px: a set of
parallel diagonal bars (~135 deg) that sweeps bottom-right -> top-left across the
field (in SAMPLE coordinates, after the projection optics' flip), i.e. a moving
wavefront along the 135 deg axis.

Import-safe (numpy only; matplotlib only for the optional overview figure).
Run directly to decode + render an overview:
    python mea_nap_polygon_seq.py  path\\to\\pattern.seq
================================================================================
"""

import os
import struct
import numpy as np

SEQ_MAGIC = b"1QSM"
_OFF_NFRAMES, _OFF_WIDTH, _OFF_HEIGHT = 4228, 4232, 4236


# ==============================================================================
# header + frame access
# ==============================================================================
def read_header(path):
    """Parse the .seq header -> dict with geometry and the data offset."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(max(_OFF_HEIGHT + 4, 64))
    if head[:4] != SEQ_MAGIC:
        raise ValueError(f"{os.path.basename(path)}: not a Mightex .seq "
                         f"(magic {head[:4]!r}, expected {SEQ_MAGIC!r})")
    nframes = struct.unpack_from("<I", head, _OFF_NFRAMES)[0]
    width = struct.unpack_from("<I", head, _OFF_WIDTH)[0]
    height = struct.unpack_from("<I", head, _OFF_HEIGHT)[0]
    if not (0 < nframes < 1_000_000 and 0 < width < 20000 and 0 < height < 20000):
        raise ValueError(f"{os.path.basename(path)}: implausible header "
                         f"(n={nframes}, w={width}, h={height})")
    rowbytes = (width + 7) // 8
    framebytes = rowbytes * height
    data_offset = size - nframes * framebytes
    if data_offset < 0:
        raise ValueError(f"{os.path.basename(path)}: file too small for "
                         f"{nframes}x{framebytes} bytes of frame data")
    return {"path": path, "nframes": nframes, "width": width, "height": height,
            "rowbytes": rowbytes, "framebytes": framebytes,
            "data_offset": data_offset, "filesize": size}


def frame_mask(path, k, hdr=None):
    """Boolean (height, width) mask of frame k (True = laser on)."""
    h = hdr or read_header(path)
    if not (0 <= k < h["nframes"]):
        raise IndexError(f"frame {k} out of range 0..{h['nframes']-1}")
    with open(path, "rb") as f:
        f.seek(h["data_offset"] + k * h["framebytes"])
        raw = np.frombuffer(f.read(h["framebytes"]), dtype=np.uint8)
    bits = np.unpackbits(raw.reshape(h["height"], h["rowbytes"]), axis=1)
    return bits[:, :h["width"]].astype(bool)


def roi_centroids(path, hdr=None):
    """Per-frame illuminated-area centroid (x, y) in px and pixel area.
    Returns (centroids[n,2] with NaN where empty, areas[n])."""
    h = hdr or read_header(path)
    cents = np.full((h["nframes"], 2), np.nan)
    areas = np.zeros(h["nframes"], int)
    for k in range(h["nframes"]):
        m = frame_mask(path, k, h)
        ys, xs = np.nonzero(m)
        areas[k] = xs.size
        if xs.size:
            cents[k] = (xs.mean(), ys.mean())
    return cents, areas


# ==============================================================================
# sweep geometry
# ==============================================================================
def sweep_geometry(path, hdr=None):
    """Summarise the sweep: net direction, per-frame phase (offset of the pattern
    centre-of-mass projected onto the sweep axis), and the coverage extent.

    Returns dict:
      sweep_dir_deg  : net travel direction of the pattern (DMD px coords, y-DOWN;
                       0=+x/right, 90=+y/down), from a linear fit of COM vs frame.
      bar_angle_deg  : orientation of the projected bars (perpendicular to sweep).
      phase          : per-frame scalar position of the pattern along the sweep axis
                       (px), 0 at frame 0; monotone if the sweep is one-directional.
      dmd            : (width, height).
    NOTE: DMD px coordinates. Sample/electrode coordinates require the DMD->MEA
    calibration (see dmd_to_electrode); the projection optics typically flip x/y,
    which is why a DMD bottom-left->top-right sweep is BR->TL on the sample.
    """
    h = hdr or read_header(path)
    cents, areas = roi_centroids(path, h)
    ok = np.isfinite(cents[:, 0]) & (areas > 0)
    ks = np.nonzero(ok)[0]
    # --- bar orientation from the 2-D FFT of the busiest frame (robust for a
    #     grating, where the centroid is unreliable) ---
    kbest = int(np.argmax(areas))
    m = frame_mask(path, kbest, h).astype(float)
    step = max(1, min(m.shape) // 600)
    m = m[::step, ::step]
    F = np.abs(np.fft.fftshift(np.fft.fft2(m - m.mean())))
    cy, cx = np.array(F.shape) // 2
    F[cy - 3:cy + 4, cx - 3:cx + 4] = 0.0                # suppress DC
    py, px = np.unravel_index(int(np.argmax(F)), F.shape)
    wx, wy = px - cx, py - cy                             # grating wavevector
    wave_ang = np.degrees(np.arctan2(wy, wx))
    bar_angle = (wave_ang + 90.0) % 180.0
    period_px = float(m.shape[1] / max(abs(wx), 1) * step)
    # --- sweep sign: does the pattern advance +wave or -wave over frames? ---
    u = np.array([np.cos(np.radians(wave_ang)), np.sin(np.radians(wave_ang))])
    proj_all = cents @ u
    trend = np.polyfit(ks, proj_all[ks], 1)[0] if ks.size > 1 else 1.0
    sweep_dir = (wave_ang if trend >= 0 else wave_ang + 180.0) % 360.0
    us = np.array([np.cos(np.radians(sweep_dir)), np.sin(np.radians(sweep_dir))])
    phase = np.full(h["nframes"], np.nan)
    ppp = cents @ us
    phase[ks] = ppp[ks] - ppp[ks[0]]
    return {"sweep_dir_deg": float(sweep_dir), "bar_angle_deg": float(bar_angle),
            "grating_period_px": period_px, "phase": phase,
            "centroids": cents, "areas": areas,
            "dmd": (h["width"], h["height"]), "nframes": h["nframes"]}


# ==============================================================================
# trigger -> frame -> position
# ==============================================================================
def trigger_frames(n_triggers, nframes, start_frame=0, loop=True):
    """Map each external trigger to the frame it advances the Polygon to.
    Default: one trigger advances one frame, looping the sequence (the standard
    Polygon 'trigger-advance' mode). Returns an int array of length n_triggers."""
    idx = start_frame + np.arange(n_triggers)
    return (idx % nframes) if loop else np.clip(idx, 0, nframes - 1)


def dmd_to_electrode(xy_px, calib):
    """Map DMD pixel coords -> MEA electrode/grid coords via an affine calibration.
    calib = {"scale": s | (sx,sy), "theta_deg": r, "flip_x": bool, "flip_y": bool,
             "offset": (ox, oy)} applied as: flip -> scale -> rotate -> offset.
    Returns an (...,2) array in the same units as 'offset' (e.g. µm or grid units).
    Provide the calibration from your Polygon<->MEA registration; without it the
    sweep DIRECTION (sweep_dir_deg) and speed-per-trigger are still usable in
    relative terms."""
    p = np.asarray(xy_px, float).reshape(-1, 2)
    s = calib.get("scale", 1.0)
    sx, sy = (s if isinstance(s, (tuple, list)) else (s, s))
    if calib.get("flip_x"):
        sx = -sx
    if calib.get("flip_y"):
        sy = -sy
    q = p * np.array([sx, sy])
    th = np.radians(calib.get("theta_deg", 0.0))
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    q = q @ R.T
    q = q + np.asarray(calib.get("offset", (0.0, 0.0)), float)
    return q.reshape(np.asarray(xy_px).shape)


# ==============================================================================
# overview figure
# ==============================================================================
def render_overview(path, out_png=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = read_header(path)
    g = sweep_geometry(path, h)
    W, H = g["dmd"]
    cents, areas = g["centroids"], g["areas"]
    nf = h["nframes"]
    ks = [0, nf // 4, nf // 2, 3 * nf // 4, nf - 1]
    cov = np.zeros((H // 8, W // 8))
    for k in range(nf):
        b = frame_mask(path, k, h)[::8, ::8]
        cov[:b.shape[0], :b.shape[1]] += b
    fig = plt.figure(figsize=(15, 4.6))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(cov, cmap="inferno", origin="upper", extent=[0, W, H, 0])
    ax.set_title("Coverage (all frames summed)"); ax.set_xlabel("DMD x"); ax.set_ylabel("DMD y")
    ax2 = fig.add_subplot(1, 3, 2)
    sc = ax2.scatter(cents[:, 0], cents[:, 1], c=np.arange(nf), cmap="viridis", s=16)
    ax2.scatter(cents[0, 0], cents[0, 1], c="lime", s=110, edgecolors="k",
                marker="o", zorder=5, label="frame 0")
    ax2.scatter(cents[-1, 0], cents[-1, 1], c="red", s=140, edgecolors="k",
                marker="*", zorder=5, label=f"frame {nf-1}")
    ax2.set_xlim(0, W); ax2.set_ylim(H, 0); ax2.set_aspect("equal"); ax2.legend(fontsize=8)
    ax2.set_title(f"Pattern centroid path\nsweep dir {g['sweep_dir_deg']:.0f}° (DMD), "
                  f"bars ~{g['bar_angle_deg']:.0f}°")
    plt.colorbar(sc, ax=ax2, fraction=0.046, label="frame #")
    ax3 = fig.add_subplot(1, 3, 3)
    cols = plt.cm.viridis(np.linspace(0, 1, len(ks)))
    for k, c in zip(ks, cols):
        ys, xs = np.nonzero(frame_mask(path, k, h)[::4, ::4])
        ax3.scatter(xs * 4, ys * 4, s=1, color=c, label=f"frame {k}")
    ax3.set_xlim(0, W); ax3.set_ylim(H, 0); ax3.set_aspect("equal")
    ax3.legend(fontsize=8, markerscale=6); ax3.set_title("Example projected frames")
    fig.suptitle(f"{os.path.basename(path)} — {nf} frames, {W}x{H} px DMD "
                 f"(DMD coords, white/points = laser ON)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_png = out_png or (os.path.splitext(path)[0] + "_decoded.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png


# ==============================================================================
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        h5 = [f for f in os.listdir(".") if f.lower().endswith(".seq")]
        path = h5[0] if h5 else None
    if not path or not os.path.exists(path):
        print("usage: python mea_nap_polygon_seq.py <pattern.seq>")
        sys.exit(1)
    h = read_header(path)
    print(f"== {os.path.basename(path)} ==")
    print(f"frames={h['nframes']}  DMD={h['width']}x{h['height']}px  "
          f"rowbytes={h['rowbytes']}  framebytes={h['framebytes']}  "
          f"header={h['data_offset']} bytes")
    g = sweep_geometry(path, h)
    print(f"sweep direction (DMD, y-down): {g['sweep_dir_deg']:.1f}°  |  "
          f"bar orientation: {g['bar_angle_deg']:.1f}°")
    ph = g["phase"][np.isfinite(g["phase"])]
    print(f"sweep phase span: {ph.min():.0f} .. {ph.max():.0f} px "
          f"over {h['nframes']} frames  ({np.median(np.diff(ph)):.1f} px/frame)")
    out = render_overview(path, h["path"].rsplit(".", 1)[0] + "_decoded.png")
    print("overview figure ->", out)
