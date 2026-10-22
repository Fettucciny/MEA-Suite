# MEA Pattern Creator — handoff from Claude Cowork (2026-10-22)

This program was built and maintained in Claude Cowork sessions from July to
October 2026. This file carries that context forward: what exists, why it is the
way it is, what went wrong, and what is still open. Sequence, optics and TTL
facts are in `POLYGON_AND_STIMULATION.md`.

## 1. Origin and goal

Jerry stimulates MEA cultures with a Polygon 1000 DMD and found the bundled
calibration and sequence editor hard to use. He asked for:

- a GUI that shows the live AmScope camera feed (AmScope x64 4.12.26598.20240928,
  USB);
- easy creation of Polygon `.seq` files;
- reuse of saved patterns across samples by aligning them to each sample's
  electrodes.

Only `.seq` generation is in scope. The app never talks to the Polygon itself;
Jerry loads the files into PolyScan2.

## 2. Timeline (dates from file timestamps, UTC)

| When | What |
|---|---|
| 2026-07-21 | `.seq` format reverse-engineered from a PolyScan-made 160-frame file (now `..\config\Sept2021_SweepBRtoTL.seq`). The byte-exact round-trip passed, but the frame-header size was wrong (see 2026-10-15). Shape model, rasterizer, project and export written. GUI: drawing tools, frame list, overlay. |
| 2026-07-21 | Camera↔canvas calibration (affine/homography, saved to `camera_calibration.json`). Pattern→sample alignment: reference electrodes are marked on the pattern and clicked on the live feed, then an Umeyama similarity fit (shift + rotation + uniform scale) exports `*_aligned.seq` plus an `.align.json` sidecar. |
| 2026-07-21 | AmScope SDK integration: auto-discovers `amcamsdk.20250415\python\amcam.py` and the x64 DLL. Fixed "connected but no video" (ctypes buffer, rule 7 in CLAUDE.md). Added `Camera → diagnostics` and `camera_connect.log`. |
| 2026-07-21/22 | Camera view auto-centres with scrollbars only when zoomed past the border. Shift-constrained square/circle. Select tool with copy/paste/duplicate/delete (Ctrl+C/V/D, Del). Binning and exposure persist (`camera_settings.json`). **⟳ Reload** live-restart button. |
| 2026-07-22 | Duration-aware **timed export** (expands per-frame durations into repeated frames at a base Pattern Period, with a `.timing.json` sidecar) and a **pulse-train builder** (N pulses × on/off + rest). |
| 2026-07-23 | Suite moved to a per-machine venv under Documents plus `setup.bat`, and native `.exe` launchers (done outside this app's code, by another session). |
| 2026-07-23 | **PyQt5 → PyQt6 port** and shared MEA-Suite theme (`mea_theme.apply_qt(app, "pattern")`). Back-end untouched. Jerry approved from a screenshot. |
| 2026-10-15 | Analysed the BR→TL sweep and measured the optical scale from a photo (2.575 µm/canvas px). **Found and fixed the `.seq` offset bug**: bitmap at +20, not +24. Added regression tests. Generated the radial centre-out sequence and the 20-frame settings-test sequence. Probed PolyScan TTL behaviour with Jerry. Worked out Agilent 33250A burst counts for a 55-min run. |
| 2026-10-15 | Compared a previous lab member's `01212025_RingOut_20ms_50HzTTL_pause.seq`: a different protocol (see POLYGON doc §6). |
| 2026-10-19 | A MEA Explorer session added `..\config\electrode_order_RadialCenterOut.json` (raster electrode order for the radial sequence). |
| 2026-10-22 | This handoff: `CLAUDE.md`, these docs, `tools/`, `.gitignore`. |

## 3. Key decisions

- **Identity calibration is the default.** Jerry confirmed the Polygon is already
  calibrated to the camera, so canvas px = camera px at 1824×1216. The dot-grid
  calibration panel is legacy/optional. It needs the Polygon to project dots and
  the user to click them, and the app has no projector control.
- **Alignment is a similarity transform**, not an affine one (Jerry's choice:
  shift + rotation + uniform scale). Reference electrodes are marked on the
  pattern.
- **Reload restarts the process** (`QProcess.startDetached(sys.executable, app.py)`)
  instead of using `importlib.reload`, because updates change class layouts. The
  open project is saved to `_reload_recovery.mpcproj` and restored.
- **One `.seq` implementation:** `tools/` imports `mpc.seqfile`. Keep it that way.
- **Default header = PolyScan editor output:** `new_sequence()` writes the same
  4224 header/record bytes as the sweep file. Files from another PolyScan version
  (RingOut) carry param_c = 10000 and a non-zero footer; both load in PolyScan.

## 4. Bugs and lessons worth remembering

1. **`.seq` 32-px shift (fixed 2026-10-15).** The bitmap was assumed to start at
   frame-block +24; it starts at +20, with a 4-byte zero trailer after it. The
   round-trip was byte-exact either way. Evidence that +20 is correct:
   - every sweep frame decodes as a single clean band, from exactly the
     bottom-right corner pixel to exactly the top-left one;
   - the trailers are always zero;
   - the RingOut file decodes as rings centred on the field.

   **Any `.seq` exported by the app before 2026-10-15 is shifted 32 px right**
   and should be re-exported. Also, a byte-exact round-trip only proves
   self-consistency: always test geometry against a known image.
2. **"Camera connected, no video."** `PullImageV4` declares its buffer as
   `c_char_p`. Passing a `bytearray` raised `ArgumentError`, which was swallowed
   by a bare `except`. Fix: a reused `ctypes.create_string_buffer`. Pull errors
   are now recorded, and the app checks for frames 2.5 s after connecting.
3. **Silent fallback.** A camera failure used to fall back to the synthetic
   source silently. Failures now show a dialog and are written to the log.
4. **Validate SDK calls against the vendor `amcam.py`** in the SDK folder,
   not memory. Three mismatches were caught that way: byte order option,
   PullImage variant, and the resolution method name.
5. **PyQt6 port:** `QHeaderView.ResizeMode.Stretch` was missed in the first enum
   sweep. Tests that trigger static `QMessageBox` calls hang unless the call is
   patched.
6. **`Calibration.fit`** divided by zero when the camera size was unknown at
   startup; it is now guarded.

## 5. Open items (roughly by priority)

1. **Optical scale disagreement.** The photo gives 2.575 µm/canvas px, while
   PolyScan's `120123_Calibration_4X.map` stores 2.70004 (4.9 % more). Resolve by
   projecting the settings-test sequence: does its 1 mm bar span exactly 5
   electrode pitches (2.575), or about 5.24 (2.70)? Every µm and mm/s figure
   depends on this.
2. **PolyScan off-by-one.** Does frame 1 show at run start, or after the first
   TTL? Test: 20-frame settings test plus a 20-pulse burst. If it ends on "20",
   counts are as documented; if it ends on "1", subtract 1. Not yet reported.
3. **Stale docs.** `LAUNCHER_AND_PACKAGING.md` still says PyQt5, "theme not
   wired", and that the `.exe` lives in the suite root. The suite root now has
   `MEA Suite.exe` (the selector), and `MEA Pattern Creator.exe` (4.9 MB, dated
   2026-07-24) sits inside this folder. **Ask Jerry how launching works now**
   before editing launcher docs. The README was refreshed on 2026-10-22.
4. **Move per-machine state** (`camera_settings.json`, `camera_calibration.json`,
   `camera_connect.log`, `_reload_recovery.*`) out of the synced folder, e.g. to
   `%USERPROFILE%\Documents\MEA Analyzer\pattern_creator\`. Migrate existing
   files once.
5. **Hide or simplify the dot-calibration panel** (see §3). Keep identity as the
   default.
6. **Theme polish:** mark one primary button with `setProperty("primary", True)`,
   raise toolbar label contrast, and replace the hard-coded `#8fd0ff` in
   `alignment_panel.py` and `main_window.py` (pulse-train preview) with a theme token.
7. **Possible features Jerry has shown interest in:**
   - put the `tools/` generators (radial, sweep, settings test) in the
     Sequence menu;
   - a waveform/timeline viewer for frames (he said "skip for now" in July);
   - a "one trigger per burst" option for the pulse train;
   - a radial variant that stops at the illuminated-field edge (~800 px /
     2.06 mm, like RingOut) while keeping the 2.16 s gap. Offered, not yet
     requested.
8. **Housekeeping:** delete `sync_probe.txt`, a leftover from a sync test.
   Cowork could not delete files on the device.

## 6. Where things live outside this folder

- `..\config\`: suite-shared. Holds `setup.bat`, `requirements.txt`, the master
  `mea_theme.py`, the stimulation `.seq` files, `120123_Calibration_4X.map`,
  `1 Solar Cell.png` (the 4X reference photo), and the `electrode_order_*.json`
  files. The electrode-order files are owned by **MEA Explorer**: if a sequence
  changes, its electrode order must be updated there too.
- Claude project "MEA Suite" (claude.ai): `claude/polygon-seq-notes.md`, a
  shorter copy of the POLYGON doc.
- Machines: **Fett** (desktop, camera + Polygon rig) and **FettLaptop**.
