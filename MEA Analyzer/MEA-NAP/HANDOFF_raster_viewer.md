# Handoff — Interactive Raster Viewer (reuse spec)

**Goal of the new tool:** an interactive viewer for MCS MEA recordings — scroll
through the recording in time, zoom in/out, and export or inspect a selected
section in detail. Must stay responsive on hour-long recordings with millions of
spikes.

**This is a separate project.** It lives in the same folder
(`Scripts\MEA-NAP\`) so it can `import` the tested loaders below. Do **not**
re-derive the HDF5 byte layout or the spike-file format — import these functions.

---

## Golden rule

The raster itself is built from **detected spike times**, which are small and
cheap to load — you do **not** need the raw voltage to draw a raster. Load raw
voltage only when the user zooms in to see a channel's waveform (optional).

Two data sources, two costs:

| Source | Function | Size | Use for |
|---|---|---|---|
| Spike times | `io.load_spike_file(stem)` | small (times only) | the raster (primary) |
| Raw voltage | `io.MCSFile(path)` | **large** (whole file in RAM) | optional waveform on zoom |

---

## Reusable API (exact signatures + returns)

All in `Scripts\MEA-NAP\`. Import as:

```python
import sys; sys.path.insert(0, r"C:\Users\Yikan\Nextcloud2\MEA Data\New MEA 06172026\Scripts\MEA-NAP")
import mea_nap_io as io
import mea_nap_original_plots as orig
import mea_nap_digital_events as de
```

### 1. Paths / config — `io.CONFIG`
- `io.CONFIG["DATA_DIR"]`   folder with the `.h5` files (and `Batch.csv`)
- `io.CONFIG["SPIKE_DIR"]`  folder with detected spike files (`<stem>_spikes.npz`)
- `io.CONFIG["STREAM_INDEX"]`  `None` = auto-pick the largest analog stream
- `io.parse_batch(csv_path)` -> list of row dicts (`stem`, `age`, `group`, `win`,
  `region`, `ground`) if you want to drive the viewer off a Batch.csv.

### 2. Spike times (the raster data) — `io.load_spike_file(stem)`
`stem` = the `.h5` filename without extension. Returns `None` if not yet detected
(run Stage-1 detection first), else a dict:
```python
{
  "spike_times": [np.ndarray, ...],   # one array PER CHANNEL, aligned to "labels"
                                       # times are SECONDS over the WHOLE recording
  "labels":      ["21","22",...],      # MCS labels, str; index i <-> spike_times[i]
  "fs":          25000.0,              # sampling rate (Hz)
  "auto_ground": ["47", ...],          # channels excluded by quality control
  "noise_uv":    np.ndarray | None,    # per-channel noise floor (µV)
}
```
Notes:
- Times are **absolute seconds** (0 .. duration), NOT window-relative.
- To draw a raster for a time window `[t0, t1]`, just filter each channel:
  `s[(s >= t0) & (s < t1)]`.
- Drop the reference (`io.CONFIG["REFERENCE_ELECTRODE"]`, e.g. `Ref`/`15`) and,
  if you want QC applied, the `auto_ground` labels.

### 3. Raw voltage (optional, for waveform-on-zoom) — `io.MCSFile(path, stream_index=None)`
Loads the whole recording into RAM on construction (int16), so it is **heavy**
(≈ n_channels × n_samples × 2 bytes; a 60-ch 1-hour @25 kHz file ≈ 10 GB). For a
viewer, construct it lazily (only when the user asks for a waveform) and read
windows — never `get_matrix()` on a long file.
- attrs: `.fs`, `.labels` (list[str]), `.n_channels`, `.n_samples`, `.duration`
- `.get_channel_window(i, s0, s1)` -> µV float array for channel index `i`
  (index into `.labels`), samples `[s0, s1)`. Convert seconds to samples with
  `int(t * mcs.fs)`. **This is the call to use in a viewer.**
- `.get_channel(i)` -> whole µV trace for channel `i` (heavy)
- `.get_matrix()` -> full `(n_ch, n_samples)` µV float32 (avoid on long files)
- `.free()` -> release the in-RAM raw buffer when done
Align spikes to voltage by matching label strings (`spike["labels"][i]` <->
`mcs.labels[j]`).

### 4. Electrode geometry — `orig.mcs_coord(label)`
`"47"` (col 4, row 7) -> `(4.0, 2.0)` = `(col, 9-row)`, y flipped so row 1 is on
top; returns `None` for non-grid labels (e.g. `Ref`). Use it to order electrodes
spatially, or to offer an array-layout view. `orig._mcs60_labels()` lists the 60
grid labels.

### 5. Laser / digital triggers (nice overlay) — `de.read_digital_events(path, stream_index=None)`
Returns `{"on_s": np.ndarray, "off_s": ..., "intervals": ..., "source": str,
"fs": ..., "detail": ...}`. `on_s` = trigger ON times in **seconds** — draw them
as vertical lines on the raster to see the stimulus against the activity.

---

## Data conventions (do not re-guess these)

- **Time:** seconds everywhere in spike data; `on_s` triggers in seconds. Convert
  to samples for voltage with `round(t * fs)`.
- **Voltage:** µV (already scaled by `MCSFile`: `(raw - ADZero) * ConversionFactor
  * 10^Exponent * 1e6`).
- **Labels:** MCS grid labels as 2-digit strings, `col*10 + row` (e.g. `"47"`).
  Corners `11/18/81/88` are absent; `Ref`/`15` is the hardware reference.
- **Coordinate frame:** `mcs_coord` returns `(col, 9-row)` — row 1 at the top.
- **fs:** `1e6 / Tick_us`; typically 25 kHz.

---

## Suggested architecture

**Two-resolution raster** (this is the key to staying responsive):
1. **Overview (zoomed out):** don't plot millions of ticks. Bin spikes into a
   `(n_channels × n_time_bins)` count/rate image and show it (like the existing
   `orig.fig_raster`, which is a per-second firing-rate heatmap). Cheap, instant
   pan. Recompute bin width from the current zoom.
2. **Detail (zoomed in):** once the visible window is short enough (say < a few
   thousand spikes on screen), switch to actual spike ticks per electrode from
   `load_spike_file`. Optionally overlay the raw waveform for a clicked channel
   via `MCSFile.get_channel_window`.

**Library:** `pyqtgraph` (Qt) is the standard for fast, scrollable electro-
physiology views — hardware-accelerated, built-in ROI selection + PNG/SVG export,
and it handles pan/zoom over long series far better than matplotlib. The loaders
above are GUI-agnostic (pure numpy), so they drop straight in. Add `pyqtgraph`
and `PyQt5`/`PySide6` to the `.venv`. (matplotlib is fine for a first prototype
but will stutter on the full recording.)

**Export a section:** capture the current x-range `[t0, t1]` and the visible
channels, then either (a) save the view as PNG/SVG (pyqtgraph `ImageExporter`/
`SVGExporter`), or (b) export the spike subset (`s[(s>=t0)&(s<t1)]` per channel)
to CSV/npz, and/or the raw window via `get_channel_window` for offline analysis.

---

## Gotchas

- **Spikes must exist first.** `load_spike_file` returns `None` if Stage-1
  detection hasn't been run for that recording. Either require detection first,
  or (advanced) call the detector on a window on the fly via
  `mea_nap_wavelet_detect` — but reusing the cached spike files is far simpler.
- **Never `get_matrix()` a long file** — it's tens of GB. Use windowed reads.
- **MCSFile loads the whole file into RAM on construction** — so only build it
  when the user actually wants a waveform, and `.free()` it when switching files.
- **Reference / grounded channels:** exclude `io.CONFIG["REFERENCE_ELECTRODE"]`
  and (optionally) the spike file's `auto_ground` list from the raster.
- **Stream index:** leave `STREAM_INDEX=None` to auto-pick the largest analog
  stream, matching the rest of the pipeline.

---

## Minimal starter (load + window a raster)

```python
import os, numpy as np
import mea_nap_io as io
import mea_nap_digital_events as de

io.CONFIG["SPIKE_DIR"] = r"...\Output\SpikeData"     # where detection wrote spikes
io.CONFIG["DATA_DIR"]  = r"...\H5"                    # where the .h5 files are

stem = "4000um_2_D3"
sp = io.load_spike_file(stem)                         # None if not detected yet
assert sp is not None, "run Stage-1 detection for this recording first"

ref = set(map(str, io.CONFIG.get("REFERENCE_ELECTRODE", [])))
drop = ref | set(sp.get("auto_ground") or [])
chans = [(lab, st) for lab, st in zip(sp["labels"], sp["spike_times"])
         if lab not in drop]                          # (label, spike_seconds)

# raster for a visible window [t0, t1]
t0, t1 = 120.0, 130.0
window = [(lab, s[(s >= t0) & (s < t1)]) for lab, s in chans]

# laser triggers to overlay (if present)
ev = de.read_digital_events(os.path.join(io.CONFIG["DATA_DIR"], stem + ".h5"))
triggers = ev["on_s"][(ev["on_s"] >= t0) & (ev["on_s"] < t1)]

# optional raw waveform for a channel in this window:
# mcs = io.MCSFile(os.path.join(io.CONFIG["DATA_DIR"], stem + ".h5"))
# i = mcs.labels.index("47")
# uv = mcs.get_channel_window(i, int(t0*mcs.fs), int(t1*mcs.fs)); mcs.free()
```

---

## Kick off the new session with

> Read `HANDOFF_raster_viewer.md` and the loaders it references in this folder,
> then propose an architecture + plotting-library choice for an interactive,
> scrollable, zoomable raster viewer with section export. Reuse the loaders;
> do not reimplement HDF5/spike reading.
