"""
Core (non-GUI) tests for MEA Pattern Creator.

Run from the "MEA Pattern Creator" folder:
    python -m pytest tests            # if pytest is available
    python tests/test_core.py         # plain-python fallback (no pytest needed)

The byte-exact round-trip test runs only if the reference sample file is found;
set MPC_SAMPLE_SEQ to its path, or drop it beside this repo.
"""
import os
import sys
import hashlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpc import seqfile, calibration
from mpc.patterns import Shape, Frame
from mpc.project import Project


def _find_sample():
    for p in [
        os.environ.get("MPC_SAMPLE_SEQ", ""),
        os.path.join(os.path.dirname(__file__), "..", "..", "config", "Sept2021_SweepBRtoTL.seq"),
        os.path.join(os.path.dirname(__file__), "..", "..", "Sample Sequence File.seq"),
        os.path.join(os.path.dirname(__file__), "Sample Sequence File.seq"),
    ]:
        if p and os.path.isfile(p):
            return p
    return None


def test_new_sequence_roundtrip():
    seq = seqfile.new_sequence()
    f = np.zeros((seq.height, seq.width), bool)
    f[100:300, 200:900] = True
    seq.add_frame(f)
    blob = seqfile.serialize_seq(seq)
    assert blob[:4] == b"1QSM"
    re = seqfile.parse_seq(blob)
    assert re.n_frames == 1
    assert np.array_equal(re.frames[0], f)


def test_project_export_fidelity():
    proj = Project()
    fr = proj.new_frame("t")
    fr.shapes = [Shape("rect", [(100, 100), (500, 400)]),
                 Shape("ellipse", [(600, 200), (1000, 700)]),
                 Shape("ellipse", [(700, 300), (900, 600)], mode="subtract")]
    seq = proj.to_sequence()
    re = seqfile.parse_seq(seqfile.serialize_seq(seq))
    assert np.array_equal(re.frames[0], fr.rasterize(proj.canvas_width, proj.canvas_height))


def test_calibration_homography_exact():
    H = np.array([[0.9, 0.02, 30.0], [0.01, 0.95, -20.0], [1e-5, 2e-5, 1.0]])
    rng = np.random.default_rng(1)
    cam = rng.uniform(0, 2000, (12, 2))
    ph = np.hstack([cam, np.ones((12, 1))])
    proj_pts = (H @ ph.T).T
    canvas = proj_pts[:, :2] / proj_pts[:, 2:3]
    cal = calibration.Calibration(mode="homography")
    cal.pairs = [((c[0], c[1]), (k[0], k[1])) for c, k in zip(cam, canvas)]
    cal.fit()
    assert cal.rms_error() < 1e-6


def test_sample_byte_exact_roundtrip():
    sample = _find_sample()
    if not sample:
        print("  (skipped: sample .seq not found)")
        return
    orig = open(sample, "rb").read()
    seq = seqfile.parse_seq(orig)
    rebuilt = seqfile.serialize_seq(seq)
    assert hashlib.md5(rebuilt).hexdigest() == hashlib.md5(orig).hexdigest()


def test_bitmap_offset_in_block():
    """Pixel (0,0) must land in the first byte after the 20-byte frame header."""
    seq = seqfile.new_sequence()
    f = np.zeros((seq.height, seq.width), bool)
    f[0, 0] = True
    seq.add_frame(f)
    blob = seqfile.serialize_seq(seq)
    frame0 = 128 + 31 * 132 + 4
    assert blob[frame0 + 20] == 0x80
    assert blob[frame0 + 16:frame0 + 20] == (1).to_bytes(4, "little")


def test_sample_sweep_geometry():
    """The reference sweep must decode as one band from corner to corner."""
    sample = _find_sample()
    if not sample:
        print("  (skipped: sample .seq not found)")
        return
    seq = seqfile.read_seq(sample)
    h, w = seq.height, seq.width
    assert seq.frames[0][h - 1, w - 1]           # starts on the bottom-right pixel
    assert seq.frames[-1][0, 0]                  # ends on the top-left pixel
    for fr in seq.frames:
        ys, xs = np.nonzero(fr)
        s = np.unique(xs + ys)
        assert s.size <= 57 and s[-1] - s[0] == s.size - 1   # one contiguous band


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
