"""Generate a centre-out expanding-ring .seq for TTL-stepped playback.

    python tools\\make_radial_seq.py OUT.seq [--step 13.435] [--width 40.305]
        [--max-radius R] [--pad N | --match-gap FRAMES] [--cx X --cy Y]

Defaults reproduce config\\Oct2026_RadialCenterOut.seq exactly (133 frames):
the ring copies the BR->TL sweep's bar (57 diagonal px wide = 40.3 px, steps
19 diagonal px = 13.4 px per frame, so every pixel is lit for 3 frames), grows
from the canvas centre until it leaves the field corners (82 frames), then
--match-gap 108 appends blank frames so the dark interval over the electrode
array equals the sweep's (108 frames = 2.16 s at 50 Hz).

Ring k lights pixels whose centre distance d satisfies k*step <= d < k*step+width.
"""
import argparse
import math

import _common as C
from _common import np

SWEEP_STEP = 19 / math.sqrt(2)      # px per frame, perpendicular to the sweep bar
SWEEP_WIDTH = 57 / math.sqrt(2)     # px, bar thickness


def rings(step, width, cx, cy, max_radius=None):
    # float32 on purpose: the deployed Oct2026_RadialCenterOut.seq was built in
    # float32, and float64 flips ~200 exact-boundary pixels.  Keep it so the
    # default output stays byte-identical (md5 bee2579fe9ae06170f57e9c166b373b7).
    yy, xx = np.mgrid[0:C.H, 0:C.W].astype(np.float32)
    D = np.hypot(xx - np.float32(cx), yy - np.float32(cy))
    out, k = [], 0
    while True:
        r0 = np.float64(k * step)        # float32 distances vs float64 radii
        if max_radius is not None and r0 >= max_radius:
            break
        f = (D >= r0) & (D < r0 + np.float64(width))
        if not f.any():
            break
        out.append(f)
        k += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--step", type=float, default=SWEEP_STEP)
    ap.add_argument("--width", type=float, default=SWEEP_WIDTH)
    ap.add_argument("--cx", type=float, default=C.CX)
    ap.add_argument("--cy", type=float, default=C.CY)
    ap.add_argument("--max-radius", type=float, default=None,
                    help="stop once the ring's inner edge reaches this radius (px)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--pad", type=int, default=None, help="append N blank frames")
    g.add_argument("--match-gap", type=int, default=108,
                   help="pad so the dark gap over the centred array is this many "
                        "frames (default 108 = BR->TL sweep)")
    a = ap.parse_args()

    F = rings(a.step, a.width, a.cx, a.cy, a.max_radius)
    first, last = C.electrode_span(F, C.electrode_mask())
    gap0 = (len(F) - 1 - last) + first
    pad = a.pad if a.pad is not None else max(0, a.match_gap - gap0)
    F += [np.zeros((C.H, C.W), bool)] * pad
    size = C.write_frames(a.out, F)
    gap = gap0 + pad
    print(f"wrote {a.out}: {len(F)} frames ({len(F)-pad} ring + {pad} blank), {size:,} B")
    print(f"  over electrodes frames {first}-{last}; dark gap {gap} frames "
          f"= {gap/50:.2f} s at 50 Hz; cycle {len(F)/50:.2f} s")
    print(f"  speed at 50 Hz: {a.step*C.UM_PER_CANVAS_PX*50/1000:.2f} mm/s, "
          f"ring width {a.width*C.UM_PER_CANVAS_PX:.0f} um")
    if size > 30 * 2**20:
        print("  NOTE: >30 MiB -- Cowork could not transfer this; generate it locally.")


if __name__ == "__main__":
    main()
