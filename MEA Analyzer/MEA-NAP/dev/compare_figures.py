r"""Byte-compare every PNG/CSV under two output trees (e.g. the self-test folder
before and after a change), and for PNGs that differ, say how much.

    python dev\compare_figures.py BEFORE_DIR AFTER_DIR

Identical bytes = the change provably did not touch that figure. A handful of
pixels at +-1/255 is anti-aliasing on arrows whose endpoints moved by ~1e-17
(floating-point order of summation) - report it as such, not as a change.
"""
import os, sys, hashlib

def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()

def walk(root):
    out = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith((".png", ".csv")):
                p = os.path.join(d, f)
                out[os.path.relpath(p, root)] = p
    return out

def pixel_report(a, b):
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return "(install pillow + numpy for a pixel report)"
    A = np.asarray(Image.open(a).convert("RGB"), int)
    B = np.asarray(Image.open(b).convert("RGB"), int)
    if A.shape != B.shape:
        return f"size differs {A.shape[:2]} vs {B.shape[:2]}"
    d = np.abs(A - B).max(axis=2)
    n = int((d > 0).sum())
    return f"{n} of {d.size:,} pixels differ ({100 * n / d.size:.4f}%), max |delta| {int(d.max())}/255"

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    before, after = walk(sys.argv[1]), walk(sys.argv[2])
    same = diff = 0
    for rel in sorted(set(before) | set(after)):
        if rel not in before or rel not in after:
            print(f"  ONLY IN {'AFTER' if rel not in before else 'BEFORE'}  {rel}")
            diff += 1
            continue
        if md5(before[rel]) == md5(after[rel]):
            print(f"  IDENTICAL  {rel}"); same += 1
        else:
            extra = pixel_report(before[rel], after[rel]) if rel.lower().endswith(".png") else ""
            print(f"  DIFFERS    {rel}   {extra}"); diff += 1
    print(f"\n{same} identical, {diff} differ")
    sys.exit(1 if diff else 0)
