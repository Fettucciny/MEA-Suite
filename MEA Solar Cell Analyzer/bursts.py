"""Single-electrode burst detection (pure numpy).

Max-ISI method (Chiappalone-style, the transparent MEA standard): a burst is a
maximal run of spikes whose consecutive inter-spike intervals are all <= max_isi,
containing at least min_spikes spikes. Burst RATE is bursts per minute per
electrode — the metric the Filter-tab burst heatmap colours by.
"""
import numpy as np


def detect_bursts(spike_times, max_isi_s=0.1, min_spikes=5):
    """List of (start_s, end_s) bursts in a single electrode's spike train."""
    t = np.sort(np.asarray(spike_times, float))
    if t.size < min_spikes:
        return []
    gaps = np.where(np.diff(t) > max_isi_s)[0]        # indices before an ISI gap
    starts = np.concatenate([[0], gaps + 1])
    ends = np.concatenate([gaps, [t.size - 1]])
    return [(float(t[s]), float(t[e])) for s, e in zip(starts, ends)
            if (e - s + 1) >= min_spikes]


def burst_count(spike_times, max_isi_s=0.1, min_spikes=5):
    return len(detect_bursts(spike_times, max_isi_s, min_spikes))


def burst_rate_per_min(spike_times, duration_s, max_isi_s=0.1, min_spikes=5):
    """Bursts per minute over `duration_s`."""
    if duration_s <= 0:
        return 0.0
    return burst_count(spike_times, max_isi_s, min_spikes) / (duration_s / 60.0)
