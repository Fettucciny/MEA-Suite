"""Level-of-detail policy: decide overview (binned image) vs detail (ticks),
with hysteresis so the mode doesn't flip-flop at the boundary.
"""

# Enter DETAIL when fewer than this many spikes are visible; return to OVERVIEW
# only above the (higher) exit threshold. The gap is the hysteresis band.
DETAIL_ENTER = 30_000
DETAIL_EXIT = 80_000

# Safety cap on ticks actually drawn in detail mode (protects the GUI if the
# thresholds are ever mis-tuned for an unusually dense window).
MAX_TICKS = 250_000

# Target overview bins ~= one per horizontal pixel, clamped to a sane range.
MIN_BINS = 200
MAX_BINS = 4000


def bin_count(width_px):
    """Number of time bins for the overview image given the plot's pixel width."""
    try:
        w = int(width_px)
    except Exception:
        w = 1000
    return max(MIN_BINS, min(MAX_BINS, w))


def choose_mode(prev_mode, visible_count):
    """Return 'detail' or 'overview' with hysteresis.

    prev_mode: the mode currently displayed ('detail' | 'overview' | None).
    """
    if prev_mode == "detail":
        return "overview" if visible_count > DETAIL_EXIT else "detail"
    # currently overview or unset
    return "detail" if visible_count < DETAIL_ENTER else "overview"
