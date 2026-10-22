#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 mea_nap_digital_events.py  --  read the laser / digital-event timing from MCS h5
================================================================================
The MCS "Digital Event Recorder" writes when the laser power is ON as a square
wave. In an MCS HDF5 file this shows up in ONE of two forms, and this module
auto-detects and reads either:

  (A) a *Digital* AnalogStream  — a stream (DataSubType='Digital', usually 1
      channel) whose sample values are the digital word; the laser bit goes HIGH
      while the power is on. We threshold it and take the rising edges as laser
      ON times (and falling edges as OFF).

  (B) an *EventStream*          — MCS stores each digital transition as an event
      with a timestamp (µs). We read the timestamps as ON/OFF times.

`read_digital_events(filepath, ...)` returns:
    {"on_s": array, "off_s": array, "intervals": [(on,off),...],
     "source": "analog:Stream_1" | "event:Stream_0" | None,
     "fs": float|None, "detail": str}
All times are in SECONDS from the start of the recording (same clock as the
electrode AnalogStream this pipeline already uses).

Import-safe (numpy + h5py). Run directly on a file to print a diagnosis:
    python mea_nap_digital_events.py  path\\to\\recording.h5
================================================================================
"""

import os
import sys
import numpy as np
import h5py

# ------------------------------------------------------------------------------
DIGITAL_PARAMS = {
    # digital_bit : which digital-input bit is the laser. None = ANY non-zero bit
    #   counts as "on". Set 0..15 to select one line if several are wired. Range 0-15.
    "digital_bit":          None,
    # min_pulse_ms : ignore on/off transitions shorter than this (debounce). 0-50.
    "min_pulse_ms":         0.5,
    # analog_label_hint : substring that marks the digital analog stream label /
    #   DataSubType (case-insensitive). MCS usually tags it 'Digital'.
    "analog_label_hint":    "digital",
    # invert : treat LOW as "on" instead of HIGH (rare). True | False.
    "invert":               False,
    # use_in_stage2 : if True, Stage 2 reads the laser events and adds the
    #   stimulus-locked propagation figure + digital CV-field option. Costs one
    #   extra single-channel read per file; turn off if you have no digital data.
    "use_in_stage2":        True,
}

# ------------------------------------------------------------------------------
# Sweep-speed calibration. All speed groups use the SAME Polygon .seq pattern and
# differ ONLY in trigger rate, so the sweep speed is set entirely by the measured
# trigger rate (median inter-trigger interval, µs-accurate from the MEA clock).
# The trigger rate (Hz) is the exact RELATIVE speed with no calibration. To convert
# to an absolute mm/s, set frames_per_crossing from your DMD->MEA registration
# (how many frames the bar takes to cross the sample); until then mm/s is left NaN.
SWEEP_CALIB = {
    "sample_diameter_mm":   4.0,     # sample the bar sweeps across (mm)
    "frames_per_crossing":  None,    # frames for the bar to cross the sample; None -> mm/s not computed
    "seq_path":             None,    # optional path to the .seq (logged for confirmation)
}


def sweep_metrics(ev):
    """From a read_digital_events() result -> trigger-rate = sweep-speed metrics.
    Returns {n_triggers, median_isi_s, trigger_hz}. trigger_hz is the precise
    relative sweep speed (same pattern across groups)."""
    on = np.sort(np.asarray(ev.get("on_s", []), float))
    if on.size < 2:
        return {"n_triggers": int(on.size), "median_isi_s": float("nan"),
                "trigger_hz": float("nan")}
    isi = np.diff(on)
    isi = isi[isi > 0]
    med = float(np.median(isi)) if isi.size else float("nan")
    return {"n_triggers": int(on.size), "median_isi_s": med,
            "trigger_hz": (1.0 / med if med and med > 0 else float("nan"))}


def sweep_speed_mm_s(trigger_hz, calib=SWEEP_CALIB):
    """Absolute sweep speed (mm/s) if frames_per_crossing is calibrated, else NaN.
    speed = sample_diameter_mm * trigger_hz / frames_per_crossing."""
    fpc = calib.get("frames_per_crossing")
    if not fpc or not np.isfinite(trigger_hz):
        return float("nan")
    return calib["sample_diameter_mm"] * trigger_hz / float(fpc)


# ==============================================================================
# low-level structure walk
# ==============================================================================
def _first_recording(f):
    recs = sorted(k for k in f.get("Data", {}) if k.startswith("Recording_"))
    if not recs:
        raise ValueError("no Data/Recording_* group in file")
    return f["Data"][recs[0]]


def _stream_attr(g, name, default=""):
    v = g.attrs.get(name, default)
    if isinstance(v, (bytes, bytearray)):
        return v.decode(errors="replace")
    if isinstance(v, np.ndarray) and v.size == 1:
        v = v.reshape(-1)[0]
        return v.decode(errors="replace") if isinstance(v, (bytes, bytearray)) else str(v)
    return str(v)


def survey_streams(filepath, stream_index=None):
    """Return a human-readable list of every stream, tagging likely digital ones."""
    out = []
    with h5py.File(filepath, "r") as f:
        rec = _first_recording(f)
        for kind in ("AnalogStream", "EventStream"):
            if kind not in rec:
                continue
            for sk in sorted(rec[kind]):
                g = rec[kind][sk]
                entry = {"kind": kind, "stream": sk,
                         "label": _stream_attr(g, "Label"),
                         "subtype": _stream_attr(g, "DataSubType"),
                         "children": list(g.keys())}
                if kind == "AnalogStream" and "InfoChannel" in g:
                    info = g["InfoChannel"][:]
                    labs = [l.decode() if isinstance(l, (bytes, bytearray)) else str(l)
                            for l in info["Label"]]
                    entry["n_channels"] = len(labs)
                    entry["ch_labels"] = labs[:12]
                    if "ChannelData" in g:
                        entry["data_shape"] = tuple(g["ChannelData"].shape)
                if kind == "EventStream":
                    ents = [k for k in g if k.startswith("EventEntity")]
                    entry["event_entities"] = {k: tuple(g[k].shape) for k in ents}
                    if "InfoEvent" in g:
                        ie = g["InfoEvent"][:]
                        if "Label" in ie.dtype.names:
                            entry["event_labels"] = [
                                l.decode() if isinstance(l, (bytes, bytearray)) else str(l)
                                for l in ie["Label"]]
                out.append(entry)
    return out


# ==============================================================================
# edge extraction
# ==============================================================================
def _edges_from_square(sig, fs, p):
    """Rising/falling edge times (s) from a digital square-wave sample vector."""
    sig = np.asarray(sig)
    if p.get("digital_bit") is not None:
        hi = ((sig.astype(np.int64) >> int(p["digital_bit"])) & 1).astype(bool)
    else:
        lo, hh = float(np.min(sig)), float(np.max(sig))
        thr = (lo + hh) / 2.0 if hh > lo else 0.5
        hi = sig > thr
    if p.get("invert"):
        hi = ~hi
    d = np.diff(hi.astype(np.int8))
    on = (np.where(d == 1)[0] + 1) / fs
    off = (np.where(d == -1)[0] + 1) / fs
    if hi.size and hi[0]:                          # already high at t=0
        on = np.insert(on, 0, 0.0)
    return _debounce(on, off, p)


def _debounce(on, off, p):
    """Drop ON pulses shorter than min_pulse_ms. Vectorised (O(n log n)); each ON is
    paired with the next OFF after it, dangling ONs (no later OFF) are kept."""
    minp = p.get("min_pulse_ms", 0.0) / 1000.0
    on = np.sort(np.asarray(on, float))
    off = np.sort(np.asarray(off, float))
    if minp <= 0 or on.size == 0 or off.size == 0:
        return on, off
    idx = np.searchsorted(off, on, side="right")          # first OFF strictly after each ON
    ends = np.where(idx < off.size, off[np.clip(idx, 0, off.size - 1)], np.inf)
    keep = (ends - on) >= minp                             # inf for dangling ONs -> kept
    return on[keep], off


def _read_digital_analog(g, p):
    """From a Digital AnalogStream group -> (on_s, off_s, fs). Returns empty if the
    stream isn't actually a digital square wave (guards against mistaking a neural
    voltage channel for a digital line, which would otherwise produce millions of
    spurious edges)."""
    info = g["InfoChannel"][:]
    tick = float(np.asarray(info["Tick"])[0])
    fs = 1.0e6 / tick
    data = g["ChannelData"]
    raw = data[:] if data.shape[0] <= 4 else data[:1]
    sig = raw[0] if raw.shape[0] == 1 else np.bitwise_or.reduce(raw.astype(np.int64), axis=0)
    # a genuine digital line takes very few distinct values; neural voltage does not
    samp = sig[:: max(1, sig.size // 200000)]
    if p.get("digital_bit") is None and np.unique(samp).size > 64:
        return np.array([]), np.array([]), fs
    on, off = _edges_from_square(sig, fs, p)
    if on.size > 200000:                                  # implausible pulse count -> noise
        return np.array([]), np.array([]), fs
    return on, off, fs


def _read_event_stream(g, p):
    """From an EventStream group -> (on_s, off_s, fs=None). MCS EventEntity is a
    (rows, count) array: row 0 = timestamp (µs); row 1 = duration (µs) for the MCS
    Digital Event Detector. Two entities = separate ON / OFF rising/falling edges."""
    ents = sorted(k for k in g if k.startswith("EventEntity"))
    if not ents:
        return np.array([]), np.array([]), None
    # two entities: rising (ON) and falling (OFF) edges in separate datasets
    if len(ents) >= 2:
        on = np.sort(np.asarray(g[ents[0]][:])[0].astype(np.float64) / 1e6)
        off = np.sort(np.asarray(g[ents[1]][:])[0].astype(np.float64) / 1e6)
        return _debounce(on, off, p) + (None,)
    # single entity: row 0 = onset (µs); row 1 = duration (µs) if it looks like one
    arr = np.asarray(g[ents[0]][:])
    if arr.ndim == 2 and arr.shape[0] >= 1:
        on = arr[0].astype(np.float64) / 1e6
        off = np.array([])
        if arr.shape[0] >= 2:
            dur = arr[1].astype(np.float64) / 1e6
            if dur.size and np.all(dur >= 0) and np.any(dur > 0) \
                    and np.median(dur) < 10.0:            # plausible pulse widths
                off = on + dur
    else:
        on = arr.astype(np.float64) / 1e6
        off = np.array([])
    order = np.argsort(on)
    on = on[order]
    off = off[order] if off.size == on.size else off
    return on, off, None


# ==============================================================================
# main entry
# ==============================================================================
def read_digital_events(filepath, stream_index=None, params=DIGITAL_PARAMS):
    """Auto-detect and read laser ON/OFF timing. Returns a dict (see module docs)."""
    p = dict(DIGITAL_PARAMS); p.update(params or {})
    hint = str(p.get("analog_label_hint", "digital")).lower()
    with h5py.File(filepath, "r") as f:
        rec = _first_recording(f)
        # 1) prefer an explicit Digital AnalogStream
        cand = []
        if "AnalogStream" in rec:
            for sk in sorted(rec["AnalogStream"]):
                g = rec["AnalogStream"][sk]
                if "InfoChannel" not in g or "ChannelData" not in g:
                    continue
                sub = (_stream_attr(g, "DataSubType") + " " + _stream_attr(g, "Label")).lower()
                nch = g["ChannelData"].shape[0]
                if nch > 8:                    # electrode arrays (e.g. 60 ch) are never digital
                    continue
                score = 0
                if hint in sub or "digital" in sub:
                    score += 10
                if nch <= 4:
                    score += 2
                cand.append((score, sk, g))
            cand.sort(key=lambda c: c[0], reverse=True)
            if cand and cand[0][0] >= 10:
                _score, sk, g = cand[0]
                on, off, fs = _read_digital_analog(g, p)
                return _result(on, off, f"analog:{sk}", fs,
                               f"Digital AnalogStream {sk}")
        # 2) fall back to an EventStream — prefer a Digital Event Detector /
        #    DigitalPort over the Stimulator (STG) sideband stream.
        if "EventStream" in rec:
            estreams = []
            for sk in sorted(rec["EventStream"]):
                g = rec["EventStream"][sk]
                lab = (_stream_attr(g, "DataSubType") + " "
                       + _stream_attr(g, "Label")).lower()
                score = 0
                if "digital" in lab or hint in lab:
                    score += 10
                if "stg" in lab or "stimulat" in lab or "sideband" in lab:
                    score -= 5
                estreams.append((score, sk, g, lab))
            estreams.sort(key=lambda c: c[0], reverse=True)
            for _s, sk, g, lab in estreams:
                on, off, fs = _read_event_stream(g, p)
                if np.size(on):
                    return _result(on, off, f"event:{sk}", fs,
                                   f"EventStream {sk} ({lab.strip()[:40]})")
        # 3) last resort: a low-channel-count analog stream that isn't the electrodes
        if cand:
            cand.sort(key=lambda c: c[2]["ChannelData"].shape[0])
            _s, sk, g = cand[0]
            on, off, fs = _read_digital_analog(g, p)
            if np.size(on):
                return _result(on, off, f"analog:{sk}", fs,
                               f"low-channel AnalogStream {sk} (heuristic)")
    return {"on_s": np.array([]), "off_s": np.array([]), "intervals": [],
            "source": None, "fs": None,
            "detail": "no digital/event stream found"}


def _result(on, off, source, fs, detail):
    on = np.sort(np.asarray(on, float)); off = np.sort(np.asarray(off, float))
    if off.size:                                          # pair each ON with the next OFF
        idx = np.searchsorted(off, on, side="right")
        ends = np.where(idx < off.size, off[np.clip(idx, 0, off.size - 1)], np.nan)
        intervals = [(float(t), (float(e) if np.isfinite(e) else None))
                     for t, e in zip(on, ends)]
    else:
        intervals = [(float(t), None) for t in on]
    return {"on_s": on, "off_s": off, "intervals": intervals,
            "source": source, "fs": fs, "detail": detail}


def events_in_window(ev, t0, t1):
    """Laser ON times (s, relative to t0) that fall inside [t0, t1)."""
    on = ev.get("on_s", np.array([]))
    w = on[(on >= t0) & (on < t1)]
    return w - t0


# ==============================================================================
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        cwd = os.getcwd()
        h5s = [f for f in os.listdir(cwd) if f.lower().endswith(".h5")]
        path = os.path.join(cwd, h5s[0]) if h5s else None
    if not path or not os.path.exists(path):
        print("usage: python mea_nap_digital_events.py <recording.h5>")
        sys.exit(1)
    print(f"== streams in {os.path.basename(path)} ==")
    for s in survey_streams(path):
        print(s)
    print("\n== detected laser events ==")
    ev = read_digital_events(path)
    print("source:", ev["source"], "| fs:", ev["fs"], "|", ev["detail"])
    on = ev["on_s"]
    print(f"n ON = {on.size}")
    if on.size:
        print("first ON times (s):", np.round(on[:12], 4))
        if ev["off_s"].size:
            durs = np.array([b - a for a, b in ev["intervals"] if b is not None])
            if durs.size:
                print(f"pulse durations (ms): med {np.median(durs)*1000:.1f}, "
                      f"min {durs.min()*1000:.1f}, max {durs.max()*1000:.1f}")
        if on.size > 1:
            isi = np.diff(on)
            print(f"inter-ON interval (s): med {np.median(isi):.3f}, "
                  f"min {isi.min():.3f}, max {isi.max():.3f}")
