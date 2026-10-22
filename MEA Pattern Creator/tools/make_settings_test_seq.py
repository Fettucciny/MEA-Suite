"""Generate the numbered diagnostic .seq used to probe PolyScan / TTL behaviour.

    python tools\\make_settings_test_seq.py OUT.seq [--frames 20]

Every frame shows its number (large, upright in canvas coordinates) and a row
of boxes with box k filled; every frame also has a centre crosshair and a 1 mm
scale bar (at UM_PER_CANVAS_PX).  Frame 1 adds a full-field border so loop
restarts are visible.  No frame is blank, so a dark field always means PolyScan
itself is showing nothing.

What it tests: trigger edge (one number per TTL pulse?), batch boundaries,
off-by-one at start/stop, image orientation, field centre, optical scale (the
bar should span exactly 5 electrode pitches if 2.575 um/px is right; ~5.24 if
PolyScan's 2.70 um/px is right).
"""
import argparse

import _common as C
from _common import np

FONT = ["DejaVuSans-Bold.ttf", "arialbd.ttf", "segoeuib.ttf"]


def _font(size):
    from PIL import ImageFont
    for name in FONT:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def frames(n):
    from PIL import Image, ImageDraw
    big, small = _font(300), _font(60)
    L = 1000 / C.UM_PER_CANVAS_PX
    out = []
    for k in range(1, n + 1):
        im = Image.new("1", (C.W, C.H), 0)
        d = ImageDraw.Draw(im)
        t = str(k)
        b = d.textbbox((0, 0), t, font=big)
        d.text((C.CX - (b[0] + b[2]) / 2, 330 - (b[1] + b[3]) / 2), t, font=big, fill=1)
        sw, gap = 60, 20
        x0, y0 = (C.W - (n * sw + (n - 1) * gap)) / 2, 760
        if x0 < 10:                                   # many frames: shrink the boxes
            sw = gap = max(4, int((C.W - 40) / (2 * n)))
            x0 = (C.W - (n * sw + (n - 1) * gap)) / 2
        for i in range(n):
            x = x0 + i * (sw + gap)
            if i == k - 1:
                d.rectangle([x, y0, x + sw - 1, y0 + sw - 1], fill=1)
            else:
                d.rectangle([x, y0, x + sw - 1, y0 + sw - 1], outline=1, width=4)
        d.rectangle([C.CX - 120, C.CY - 2, C.CX + 120, C.CY + 1], fill=1)
        d.rectangle([C.CX - 2, C.CY - 120, C.CX + 1, C.CY + 120], fill=1)
        by = 1000
        d.rectangle([C.CX - L / 2, by, C.CX + L / 2, by + 9], fill=1)
        d.rectangle([C.CX - L / 2, by - 30, C.CX - L / 2 + 5, by + 39], fill=1)
        d.rectangle([C.CX + L / 2 - 5, by - 30, C.CX + L / 2, by + 39], fill=1)
        b = d.textbbox((0, 0), "1 mm", font=small)
        d.text((C.CX - (b[0] + b[2]) / 2, by + 60), "1 mm", font=small, fill=1)
        if k == 1:
            d.rectangle([0, 0, C.W - 1, C.H - 1], outline=1, width=16)
        out.append(np.array(im, dtype=bool))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=20)
    a = ap.parse_args()
    F = frames(a.frames)
    size = C.write_frames(a.out, F)
    print(f"wrote {a.out}: {len(F)} frames, {size:,} B")


if __name__ == "__main__":
    main()
