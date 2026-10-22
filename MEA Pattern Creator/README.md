# MEA Pattern Creator

A GUI for building **Polygon 1000** stimulation sequences (`.seq`) directly from
your **AmScope** microscope camera feed — a simpler alternative to the Polygon
system's built-in calibration and sequence editor.

You draw regions of interest straight onto the live image, see a magenta overlay
of exactly what will be projected, set per-frame timing and intensity, and export
a `.seq` that loads into the Polygon software.

Launch it from the **MEA Suite** selector (`MEA Suite.exe` in the MEA Analyzer
root) or `MEA Pattern Creator.exe`. Use `run_debug.bat` in this folder for a
visible console when troubleshooting. After a code update, press **⟳ Reload**
(Ctrl+R) instead of relaunching: your open project is restored.

---

## What it does

- **Live camera view** from the AmScope camera (via the ToupTek/AmScope `amcam`
  SDK), or from a snapshot image, or a built-in synthetic test source.
- **Draw ROIs** on the image: rectangle, ellipse, polygon, freehand trace, and
  point/brush. Each ROI is *add* (illuminate) or *subtract* (mask out).
- **Import pattern images** (any PNG/TIFF/BMP) as a 1-bit mask into a frame.
- **Camera ↔ pattern calibration.** The Polygon is already calibrated to the
  camera, so the default is *identity* (camera px = DMD px at 1824×1216). The
  optional dot-grid calibration (affine or homography, with an RMS readout) is
  still available and is remembered across sessions.
- **Select / edit shapes:** Select tool, Ctrl+C / Ctrl+V / Ctrl+D, Del. Hold
  **Shift** while drawing for a perfect square or circle.
- **Camera settings remembered:** resolution (binning), exposure and gain persist
  between launches. Camera → diagnostics explains connection problems.
- **Sample alignment** — mark reference electrodes on a pattern once, then for
  each new sample click those electrodes on the live feed and the pattern is
  shifted/rotated/scaled onto the real electrode positions, exported as an
  aligned `.seq` next to your original. Reuse saved patterns instead of redrawing.
- **Frame timeline** with per-frame duration, intensity, name, invert, reorder,
  duplicate, and a playback preview.
- **Export `.seq`** in the Polygon "MSQ1" format (round-trip and geometry tests
  against a PolyScan-made file), plus a `.timing.json` sidecar recording each
  frame's duration/intensity.
- **Timed export** (Sequence → Export timed .seq) expands per-frame durations into
  repeated frames at one base Pattern Period. **Pulse-train builder** (Sequence →
  Build pulse train) makes N on/off pulses followed by a rest.
- **Import an existing `.seq`** back into editable frames.
- **Save/Load projects** as `.mpcproj` (JSON).

---

## The `.seq` format (for reference)

Reverse-engineered from your reference file and verified with a byte-exact
round-trip:

```
128-byte header  : magic "1QSM", version, and three parameter ints
31 × 132-byte records (fixed slots the editor always writes)
4-byte alignment gap
N frame blocks, each:
    20-byte frame header (width, height, frame-count on frame 0)
    width*height/8 bytes : 1-bit-per-pixel bitmap, MSB-first, row-major
    4-byte zero trailer
8-byte footer
```

The reference file used a **1824 × 1216** canvas with 160 frames; a bit set to 1
means that pixel is illuminated. The app defaults to 1824 × 1216 but the canvas
size is configurable (Sequence → Canvas size).

---

## Enabling the live AmScope camera

The app talks to the camera through the vendor's `amcam` SDK. Out of the box the
app runs fine **without** it (synthetic source + snapshot import), but for live
capture it needs the vendor `amcam.py` wrapper and the 64-bit `amcam.dll`.

**Easiest (what's set up here):** drop the whole unzipped **AmScope SDK folder**
anywhere inside the `MEA Pattern Creator` folder — e.g.
`MEA Pattern Creator\amcamsdk.20250415\`. The app auto-detects it: it finds the
wrapper at `…\python\amcam.py` and automatically prefers the 64-bit DLL at
`…\win\x64\amcam.dll` (over the 32-bit `win\x86` build), adds it to the load path,
and preloads it so the wrapper binds the right library. Nothing to copy or rename.

Alternatively you can copy `amcam.py` and the 64-bit `amcam.dll` directly next to
`app.py`, or rely on an installed AmScope/ToupTek SDK under
`C:\Program Files\…` — the app searches those too.

Then: **close the AmScope application** (only one program can hold the camera at a
time) and choose **Camera → Connect AmScope camera**. The status bar reports
success, or explains the fallback if the camera can't be opened.

If the camera can't be opened, the status bar explains why and the app falls back
to the synthetic source so you can keep working.

> Note: AmScope cameras are not standard USB webcams, so a generic webcam capture
> won't see them — the `amcam` SDK is the supported path. If you'd rather not use
> the SDK at all, snap a still in the AmScope app and use **Camera → Load
> reference image…** to design patterns against it.

---

## Suggested workflow

1. **Connect** the camera (or load a snapshot).
2. **Calibrate** (left dock), *optional*: identity is the default, because the
   Polygon is already calibrated to the camera. Only do this to re-derive the map:
   - Set a dot grid (e.g. 3×3) and click **Export calibration .seq…**.
   - Load & project that `.seq` with the Polygon software.
   - Snap the camera image (Camera → Freeze current frame as reference).
   - Click **Start picking dots** and click each projected dot in order.
   - Choose **homography** (≥4 dots) to correct perspective/keystone. Watch the
     RMS error — a few canvas pixels or less is good.
3. **Build frames** (right dock): Add a frame, pick a tool, and draw. The magenta
   overlay shows what will be projected. Use *subtract* to cut holes.
4. Set each frame's **duration** and **intensity**, reorder as needed, and hit
   **▶ Preview**.
5. **File → Export .seq** and load it into the Polygon system.

---

## Remembered camera calibration

The dot-grid calibration (left dock, **Camera calibration** tab) ties camera
pixels to DMD/canvas pixels for your specific microscope + camera. Because that
optical setup doesn't change between samples, the app **saves it automatically**
(to `camera_calibration.json` in this folder) whenever you fit it, and reloads it
on startup and whenever you start a new pattern or import a `.seq`. You only
calibrate the optics once. Re-do it any time by picking dots again, or wipe it
with **Camera → Forget remembered camera calibration**.

## Sample alignment — reuse a pattern on a new sample

Each sample sits slightly differently in the ROI, so a saved pattern needs to be
nudged onto *this* sample's electrodes. Use the **Sample alignment** tab (left
dock):

1. **Mark reference electrodes (once, on the pattern).** Click
   **Mark reference electrodes** and click a few identifiable electrodes — edge
   or corner ones work best (3+ recommended). They save with the pattern
   (`.mpcproj`, and an `.align.json` sidecar next to any exported `.seq`).
2. **Load your sample** and make sure the camera calibration is active
   (it's remembered from before).
3. **Pick electrodes on live feed.** Click **Pick electrodes on live feed** and
   click those same electrodes, in the same order, on the live image.
4. The app fits a **similarity transform** (shift + rotation + uniform scale)
   mapping the pattern's reference electrodes onto the sample's real ones, and
   shows the shift/rotation/scale and fit RMS. A **green preview** shows the
   aligned pattern over your sample.
5. **Export aligned .seq…** writes the aligned pattern (your original is left
   untouched — export it separately if you want both). Upload the aligned `.seq`
   to the Polygon for this sample.

So the routine per sample becomes: load pattern → click a few electrodes →
export aligned `.seq`. No redrawing.

---

## Timing & intensity — how they're stored

The Polygon `.seq` container stores **only the pattern bitmaps**. PolyScan gives
every frame the same dwell: Pattern Period in Master mode, or one frame per TTL
pulse in Slave mode. So per-frame durations are made by repeating frames (timed
export), and under external TTL the run length is set by the pulse count. See
`docs/POLYGON_AND_STIMULATION.md`. To preserve your
intent, the app writes a **`<name>.timing.json`** sidecar next to the exported
`.seq` and reads it back on import. The three decoded header parameter ints from
the sample are reproduced on export; if you determine what they control, they're
exposed as `param_a/b/c` on the project and easy to wire to the UI.

---

## Files

```
MEA Pattern Creator/
    app.py                     ← entry point (PyQt6 + shared mea_theme)
    MEA Pattern Creator.exe    ← launcher
    run_debug.bat              ← launch with a visible console for troubleshooting
    README.md                  ← this guide
    CLAUDE.md                  ← instructions for Claude Code
    docs/HANDOFF.md            ← history, decisions, open items
    docs/POLYGON_AND_STIMULATION.md ← rig, PolyScan/TTL behaviour, sequence catalogue
    tools/                     ← inspect/compare .seq, make radial & test sequences,
                                 measure optical scale from a photo
    mpc/
        seqfile.py             ← .seq read/write (MSQ1)
        patterns.py            ← ROI shapes + rasterizer
        calibration.py         ← camera↔canvas affine/homography (+ remembered)
        alignment.py           ← pattern→sample similarity fit + raster warp
        camera.py              ← camera backends + fallback
        amcam_backend.py       ← AmScope/ToupTek SDK integration
        project.py             ← editable document model + export
        gui/
            main_window.py
            imageview.py       ← interactive drawing canvas
            calibration_panel.py
            alignment_panel.py
    tests/
        test_core.py           ← run: python tests/test_core.py
```

Requires the per-machine suite environment at
`%USERPROFILE%\Documents\MEA Analyzer\.venv` (PyQt6, numpy, Pillow), built once
per computer by `config\setup.bat`. If imports fail, run `setup.bat`.
```
