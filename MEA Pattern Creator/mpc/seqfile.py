"""
seqfile.py  --  Reader / writer for Mightex Polygon 1000 ".seq" sequence files.

The format was reverse-engineered from a reference file created by the Polygon
sequence editor.  A byte-exact round-trip against that reference is verified in
tests/test_seqfile.py, so files written here are structurally identical to ones
the Polygon software produces.

------------------------------------------------------------------------------
Container layout ("MSQ1")
------------------------------------------------------------------------------
    offset 0      : 128-byte file header
                        u32  magic          = b"1QSM"   (ASCII '1','Q','S','M')
                        u32  version        = 1
                        u32  param_a        = 1000      (see SeqHeader)
                        u32  param_b        = 10000
                        u32  reserved0      = 0
                        u32  param_c        = 15000
                        ... zero padding to 128 bytes
    offset 128    : NREC (=31) records of 132 bytes each.
                        Each record is all zero except a u32 == 1 at record
                        offset 8.  These appear to be fixed slots the editor
                        always emits; we reproduce them verbatim.
    then          : 4 zero bytes of alignment padding.
    then          : NFRAMES frame blocks, each:
                        u32  flag0          = 0
                        u32  count          = NFRAMES on frame 0, else 0
                        u32  width          = 1824
                        u32  height         = 1216
                        u32  one            = 1
                        <width*height/8 bytes>  1-bpp packed bitmap,
                                                MSB-first, row-major,
                                                (width/8) bytes per row.
                        u32  trailer        = 0
    end           : 8 zero bytes of footer.

NOTE (fixed Oct 2026): the bitmap starts 20 bytes into each frame block and is
followed by a 4-byte zero trailer.  Earlier versions assumed a 24-byte header
with the bitmap at +24.  That still round-tripped byte-exactly, but it read and
wrote every image 32 px (4 bytes) out of place: PolyScan saw patterns shifted
32 DMD px to the right, with the rightmost 32 columns wrapped onto the next row.
The proof is in the reference sweep: read at +20, every frame is one clean
45-degree band that starts exactly at the bottom-right corner pixel and ends
exactly at the top-left one, and the 4 bytes after each bitmap are always zero.
Read at +24, a 32-px sliver of the band appears at the right edge and pixel data
leaks into the "header".

A pixel bit == 1 means "mirror ON" (that pixel is illuminated / white).

Everything except the bitmaps themselves is small and fixed, so we keep the
exact header/record bytes we read (or sensible defaults for new files) and only
swap the frame bitmaps in and out.  That guarantees compatibility even for any
header fields whose meaning we have not fully decoded.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Format constants (defaults taken from the reference file)
# ---------------------------------------------------------------------------
MAGIC = b"1QSM"                 # bytes as they appear on disk
HEADER_SIZE = 128
NUM_RECORDS = 31                # fixed 132-byte records after the header
RECORD_SIZE = 132
GAP_AFTER_RECORDS = 4           # zero padding before the first frame
FRAME_HEADER_SIZE = 20           # bitmap begins at block offset +20
FRAME_TRAILER_SIZE = 4          # zero u32 after each bitmap
FOOTER_SIZE = 8

DEFAULT_WIDTH = 1824            # Polygon reference/DMD canvas width
DEFAULT_HEIGHT = 1216          # Polygon reference/DMD canvas height

# File-header integer fields as found in the reference file.  Their exact
# meaning is not fully documented by Mightex; reproducing them keeps written
# files identical to editor output.  Exposed so the UI can tweak them later.
DEFAULT_PARAM_A = 1000
DEFAULT_PARAM_B = 10000
DEFAULT_PARAM_C = 15000


@dataclass
class SeqHeader:
    """The decoded 128-byte file header plus the fixed record block."""
    version: int = 1
    param_a: int = DEFAULT_PARAM_A
    param_b: int = DEFAULT_PARAM_B
    param_c: int = DEFAULT_PARAM_C
    # Raw bytes preserved verbatim so we can rewrite unknown fields exactly.
    raw_header: Optional[bytes] = None          # full 128 bytes
    raw_records: Optional[bytes] = None         # NUM_RECORDS * RECORD_SIZE
    raw_gap: Optional[bytes] = None             # GAP_AFTER_RECORDS bytes


@dataclass
class Sequence:
    """A full decoded .seq: metadata + a stack of boolean pattern frames."""
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    frames: List[np.ndarray] = field(default_factory=list)   # bool HxW arrays
    header: SeqHeader = field(default_factory=SeqHeader)
    # Per-frame 20-byte headers preserved when round-tripping (optional).
    _frame_headers: Optional[List[bytes]] = None

    # -- convenience -------------------------------------------------------
    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def add_frame(self, frame: np.ndarray) -> None:
        self.frames.append(_as_bool_frame(frame, self.width, self.height))

    def frame_row_bytes(self) -> int:
        return self.width // 8

    def frame_bitmap_bytes(self) -> int:
        return (self.width // 8) * self.height


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _u32(buf: bytes, off: int) -> int:
    return int(np.frombuffer(buf[off:off + 4], dtype="<u4")[0])


def _as_bool_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    a = np.asarray(frame)
    if a.shape != (height, width):
        raise ValueError(
            f"frame shape {a.shape} does not match canvas ({height}, {width})"
        )
    return a.astype(bool)


def _default_records() -> bytes:
    """Reproduce the 31 fixed records (each: u32==1 at offset 8, rest zero)."""
    rec = bytearray(RECORD_SIZE)
    rec[8:12] = (1).to_bytes(4, "little")
    return bytes(rec) * NUM_RECORDS


def _default_header_bytes() -> bytes:
    h = bytearray(HEADER_SIZE)
    h[0:4] = MAGIC
    h[4:8] = (1).to_bytes(4, "little")                 # version
    h[8:12] = (DEFAULT_PARAM_A).to_bytes(4, "little")
    h[12:16] = (DEFAULT_PARAM_B).to_bytes(4, "little")
    h[16:20] = (0).to_bytes(4, "little")
    h[20:24] = (DEFAULT_PARAM_C).to_bytes(4, "little")
    return bytes(h)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def read_seq(path) -> Sequence:
    with open(path, "rb") as fh:
        data = fh.read()
    return parse_seq(data)


def parse_seq(data: bytes) -> Sequence:
    if data[0:4] != MAGIC:
        raise ValueError(
            f"not a Polygon .seq file (magic={data[0:4]!r}, expected {MAGIC!r})"
        )

    raw_header = data[0:HEADER_SIZE]
    version = _u32(raw_header, 4)
    param_a = _u32(raw_header, 8)
    param_b = _u32(raw_header, 12)
    param_c = _u32(raw_header, 20)

    rec_start = HEADER_SIZE
    rec_end = rec_start + NUM_RECORDS * RECORD_SIZE
    raw_records = data[rec_start:rec_end]
    raw_gap = data[rec_end:rec_end + GAP_AFTER_RECORDS]

    frame0 = rec_end + GAP_AFTER_RECORDS
    # width / height live in the first frame header (offsets +8, +12)
    width = _u32(data, frame0 + 8)
    height = _u32(data, frame0 + 12)
    if width <= 0 or height <= 0 or width % 8 != 0:
        raise ValueError(f"implausible frame size {width}x{height}")
    n_frames = _u32(data, frame0 + 4)

    fb = (width // 8) * height
    period = FRAME_HEADER_SIZE + fb + FRAME_TRAILER_SIZE

    frames: List[np.ndarray] = []
    frame_headers: List[bytes] = []
    for i in range(n_frames):
        off = frame0 + i * period
        fh_bytes = data[off:off + FRAME_HEADER_SIZE]
        raw = np.frombuffer(
            data, dtype=np.uint8, count=fb, offset=off + FRAME_HEADER_SIZE
        )
        bits = np.unpackbits(raw, bitorder="big").reshape(height, width).astype(bool)
        frames.append(bits)
        frame_headers.append(bytes(fh_bytes))

    header = SeqHeader(
        version=version, param_a=param_a, param_b=param_b, param_c=param_c,
        raw_header=raw_header, raw_records=raw_records, raw_gap=raw_gap,
    )
    seq = Sequence(width=width, height=height, frames=frames, header=header)
    seq._frame_headers = frame_headers
    return seq


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def _make_frame_header(index: int, n_frames: int, width: int, height: int,
                       preserved: Optional[bytes]) -> bytes:
    if preserved is not None and len(preserved) == FRAME_HEADER_SIZE:
        # Keep an exact header we previously read, but refresh width/height and
        # the frame count field so edits stay consistent.
        b = bytearray(preserved)
    else:
        b = bytearray(FRAME_HEADER_SIZE)
        b[16:20] = (1).to_bytes(4, "little")           # the constant '1' field
    b[4:8] = (n_frames if index == 0 else 0).to_bytes(4, "little")
    b[8:12] = int(width).to_bytes(4, "little")
    b[12:16] = int(height).to_bytes(4, "little")
    return bytes(b)


def write_seq(path, seq: Sequence) -> None:
    with open(path, "wb") as fh:
        fh.write(serialize_seq(seq))


def serialize_seq(seq: Sequence) -> bytes:
    if seq.n_frames == 0:
        raise ValueError("cannot write a sequence with zero frames")
    width, height = seq.width, seq.height
    if width % 8 != 0:
        raise ValueError("canvas width must be a multiple of 8")

    out = io.BytesIO()

    # ---- 128-byte header ------------------------------------------------
    if seq.header.raw_header is not None and len(seq.header.raw_header) == HEADER_SIZE:
        header_bytes = bytearray(seq.header.raw_header)
    else:
        header_bytes = bytearray(_default_header_bytes())
    # keep magic + refresh known param fields
    header_bytes[0:4] = MAGIC
    header_bytes[4:8] = int(seq.header.version).to_bytes(4, "little")
    header_bytes[8:12] = int(seq.header.param_a).to_bytes(4, "little")
    header_bytes[12:16] = int(seq.header.param_b).to_bytes(4, "little")
    header_bytes[20:24] = int(seq.header.param_c).to_bytes(4, "little")
    out.write(bytes(header_bytes))

    # ---- 31 fixed records ----------------------------------------------
    records = seq.header.raw_records
    if records is None or len(records) != NUM_RECORDS * RECORD_SIZE:
        records = _default_records()
    out.write(records)

    # ---- 4-byte gap -----------------------------------------------------
    gap = seq.header.raw_gap
    if gap is None or len(gap) != GAP_AFTER_RECORDS:
        gap = b"\x00" * GAP_AFTER_RECORDS
    out.write(gap)

    # ---- frames ---------------------------------------------------------
    n = seq.n_frames
    preserved = seq._frame_headers
    for i, frame in enumerate(seq.frames):
        ph = preserved[i] if (preserved and i < len(preserved)) else None
        out.write(_make_frame_header(i, n, width, height, ph))
        bits = _as_bool_frame(frame, width, height).astype(np.uint8)
        out.write(np.packbits(bits, bitorder="big").tobytes())
        out.write(b"\x00" * FRAME_TRAILER_SIZE)

    # ---- footer ---------------------------------------------------------
    out.write(b"\x00" * FOOTER_SIZE)
    return out.getvalue()


# ---------------------------------------------------------------------------
# Small utilities used by the GUI / exporters
# ---------------------------------------------------------------------------
def new_sequence(width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT) -> Sequence:
    """A fresh, empty sequence with editor-default header values."""
    return Sequence(width=width, height=height, frames=[])


def frame_to_png_bytes(frame: np.ndarray) -> bytes:
    """Encode a boolean frame as a 1-bit PNG (for previews / export)."""
    from PIL import Image
    img = Image.fromarray((np.asarray(frame).astype(np.uint8) * 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
