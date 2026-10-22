"""Best-effort reader for Mightex Polygon (PolyScan2) '.seq' flood-pattern files.

A .seq starts with the magic '1QSM' followed by a small header of little-endian
uint32 fields. On the files this was written for, the layout is:

    @0  magic '1QSM'
    @4  frame_count           (number of pulses in the pattern)
    @8  field_a_us            (a short pre/sync time, µs)
    @12 on_us                 (illumination time per pulse, µs)
    @20 off_us                (dark gap per pulse, µs)

so one pulse period = on_us + off_us and the whole train = frame_count * period.
Example: 10 frames, 15000 µs on + 10000 µs off -> 25 ms/pulse -> 250 ms train.

`read_seq(path)` returns a dict of the decoded values (all times in ms) plus
`total_ms`, or None if the file isn't a recognisable .seq. The decode is
heuristic; treat `total_ms` as a good default the user can override.
"""
import struct


def read_seq(path):
    try:
        with open(path, "rb") as fh:
            d = fh.read(64)
    except Exception:
        return None
    if len(d) < 24 or d[:4] != b"1QSM":
        return None
    u32 = lambda o: struct.unpack_from("<I", d, o)[0]
    try:
        frames = int(u32(4))
        field_a_us = int(u32(8))
        on_us = int(u32(12))
        off_us = int(u32(20))
    except Exception:
        return None
    period_us = on_us + off_us
    total_us = frames * period_us if (frames > 0 and period_us > 0) else 0
    return {
        "frames": frames,
        "pre_us": field_a_us,
        "on_us": on_us,
        "off_us": off_us,
        "on_ms": on_us / 1000.0,
        "off_ms": off_us / 1000.0,
        "period_ms": period_us / 1000.0,
        "total_ms": total_us / 1000.0,
    }


def describe(info):
    if not info:
        return "not a recognisable .seq file"
    return ("%d pulses x (%.0f ms on + %.0f ms off = %.0f ms) = %.0f ms train"
            % (info["frames"], info["on_ms"], info["off_ms"],
               info["period_ms"], info["total_ms"]))
