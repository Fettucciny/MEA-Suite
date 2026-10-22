"""Describe a Polygon .seq file frame by frame, optionally compare it to another.

    python tools\\inspect_seq.py FILE.seq [--compare OTHER.seq] [--png out.png]
                                 [--frames 0,10,20]

Prints the header ints, frame count, per-frame lit-pixel count / bounding box /
centroid, which frames touch the (assumed centred) 60-electrode array, and the
dark gap over the array when the sequence loops.  --png writes a contact sheet.
"""
import argparse

import _common as C
from _common import np, seqfile


def describe(path, png=None, pick=None):
    raw = open(path, "rb").read()
    seq = seqfile.read_seq(path)
    F = np.array(seq.frames)
    print(f"{path}\n  size {len(raw):,} B   header ints "
          f"{np.frombuffer(raw[:24], '<u4')[1:].tolist()}   footer {raw[-8:].hex()}")
    print(f"  {seq.n_frames} frames of {seq.width}x{seq.height}")
    yy, xx = np.mgrid[0:seq.height, 0:seq.width]
    blank = 0
    for k, f in enumerate(F):
        n = int(f.sum())
        if n == 0:
            blank += 1
            continue
        X, Y = xx[f], yy[f]
        print(f"  frame {k:4d}: {n:8d} px  x {X.min():4d}-{X.max():4d}  "
              f"y {Y.min():4d}-{Y.max():4d}  centroid ({X.mean():7.1f},{Y.mean():7.1f})")
    print(f"  blank frames: {blank}")
    span = C.electrode_span(F, C.electrode_mask())
    if span:
        a, b = span
        gap = len(F) - (b - a + 1)
        print(f"  over electrodes (centred array): frames {a}-{b}; "
              f"dark gap when looped {gap} frames = {gap/50:.2f} s at 50 Hz")
    if png:
        from PIL import Image
        ks = pick or list(np.linspace(0, len(F) - 1, min(8, len(F))).astype(int))
        tw, th = 304, 203
        sheet = Image.new("L", (len(ks) * (tw + 6), th), 60)
        for i, k in enumerate(ks):
            sheet.paste(Image.fromarray((F[k] * 255).astype(np.uint8)).resize((tw, th)),
                        (i * (tw + 6), 0))
        sheet.save(png)
        print(f"  contact sheet of frames {list(map(int, ks))} -> {png}")
    return F


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seq")
    ap.add_argument("--compare")
    ap.add_argument("--png")
    ap.add_argument("--frames", help="comma-separated frame indices for --png")
    a = ap.parse_args()
    pick = [int(x) for x in a.frames.split(",")] if a.frames else None
    F = describe(a.seq, a.png, pick)
    if a.compare:
        G = describe(a.compare)
        if F.shape == G.shape and np.array_equal(F, G):
            print("\nIDENTICAL frame bitmaps")
        else:
            n = min(len(F), len(G))
            diff = [k for k in range(n) if not np.array_equal(F[k], G[k])]
            print(f"\nDIFFERENT: {len(F)} vs {len(G)} frames; "
                  f"{len(diff)} of the first {n} frames differ")


if __name__ == "__main__":
    main()
