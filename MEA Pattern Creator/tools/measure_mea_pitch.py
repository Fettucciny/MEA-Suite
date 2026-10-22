"""Measure the optical scale from a microscope photo of the MEA electrode grid.

    python tools\\measure_mea_pitch.py PHOTO.png [--pitch-um 200] [--bin 3]
        [--roi x0,y0,x1,y1] [--overlay out.png]

Detects the dark electrode discs, fits an integer lattice (origin + a*i + b*j)
by iterated least squares, and reports pitch in camera px, um per native camera
px, the array rotation, and um per canvas px assuming the AmScope runs at
1/--bin of native resolution with identity camera<->DMD calibration.

Reference result (config\\1 Solar Cell.png, 5440x3648, 4X): pitch 233.03 px,
0.8583 um/native px, rotation -0.71 deg, 2.5748 um/canvas px (bin 3), 30
lattice inliers, median residual 1.2 px.  (The original 2026-10 measurement,
with scikit-image peak finding, gave 233.015 px / 2.5749 -- same to 0.01 %.)
--roi defaults to that photo's electrode region.
"""
import argparse

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

Image.MAX_IMAGE_PIXELS = None


def detect(A, roi, r, min_dist, rel_thresh):
    x0, y0, x1, y1 = roi
    D = np.clip(ndi.uniform_filter(A, 201) - A, 0, None)[y0:y1, x0:x1]   # dark -> +
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = ((xx ** 2 + yy ** 2) <= r * r).astype(np.float32)
    R = ndi.correlate(D, disk / disk.sum(), mode="nearest")
    peaks = (R == ndi.maximum_filter(R, size=min_dist)) & (R > rel_thresh * R.max())
    pts = []
    for y, x in zip(*np.nonzero(peaks)):
        a, b = max(0, y - r), min(D.shape[0], y + r + 1)
        c, d = max(0, x - r), min(D.shape[1], x + r + 1)
        w = D[a:b, c:d]
        if w.sum() > 0:
            gy, gx = np.mgrid[a:b, c:d]
            pts.append(((gx * w).sum() / w.sum() + x0, (gy * w).sum() / w.sum() + y0))
    return np.array(pts)


def fit_lattice(P, p0, iters=60):
    a, b, o = np.array([p0, 0.0]), np.array([0.0, p0]), P.mean(0)
    for _ in range(iters):
        ij = np.rint(np.linalg.solve(np.stack([a, b]).T, (P - o).T)).T
        X = np.hstack([ij, np.ones((len(P), 1))])
        a, b, o = np.linalg.lstsq(X, P, rcond=None)[0]
    pred = np.hstack([ij, np.ones((len(P), 1))]) @ np.stack([a, b, o])
    return a, b, o, np.linalg.norm(P - pred, axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo")
    ap.add_argument("--pitch-um", type=float, default=200.0)
    ap.add_argument("--guess-px", type=float, default=231.0, help="rough pitch in px")
    ap.add_argument("--bin", type=float, default=3.0,
                    help="native/live resolution ratio (5440x3648 -> 1824x1216 = 3)")
    ap.add_argument("--roi", default="1780,900,3560,2680")
    ap.add_argument("--overlay")
    a = ap.parse_args()

    A = np.asarray(Image.open(a.photo).convert("L")).astype(np.float32)
    roi = [int(v) for v in a.roi.split(",")]
    P = detect(A, roi, r=19, min_dist=int(a.guess_px * 0.65), rel_thresh=0.25)
    va, vb, o, res = fit_lattice(P, a.guess_px)
    keep = res < 0.11 * a.guess_px
    va, vb, o, res = fit_lattice(P[keep], a.guess_px)
    pitch = (np.linalg.norm(va) + np.linalg.norm(vb)) / 2
    um = a.pitch_um / pitch
    print(f"{len(P)} candidates, {keep.sum()} lattice inliers, median residual "
          f"{np.median(res):.2f} px (max {res.max():.2f})")
    print(f"a = ({va[0]:.3f}, {va[1]:.3f}) |a| = {np.linalg.norm(va):.3f} px; "
          f"b = ({vb[0]:.3f}, {vb[1]:.3f}) |b| = {np.linalg.norm(vb):.3f} px")
    print(f"rotation {np.degrees(np.arctan2(va[1], va[0])):.3f} deg, included angle "
          f"{np.degrees(np.arctan2(vb[1], vb[0]) - np.arctan2(va[1], va[0])):.3f} deg")
    print(f"PITCH {pitch:.3f} px -> {um:.5f} um/native px -> "
          f"{um * a.bin:.4f} um/canvas px (bin {a.bin:g})")
    if a.overlay:
        from PIL import ImageDraw
        im = Image.open(a.photo).convert("RGB")
        d = ImageDraw.Draw(im)
        for i in range(-4, 5):
            for j in range(-4, 5):
                X, Y = o + va * i + vb * j
                d.ellipse([X - 28, Y - 28, X + 28, Y + 28], outline=(219, 127, 255), width=5)
        im.resize((im.width // 4, im.height // 4)).save(a.overlay)
        print(f"overlay -> {a.overlay}")


if __name__ == "__main__":
    main()
