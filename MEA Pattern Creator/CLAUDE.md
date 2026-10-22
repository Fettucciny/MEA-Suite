# MEA Pattern Creator — guide for Claude Code

PyQt6 desktop app (Windows) for designing **Mightex Polygon 1000** DMD stimulation
sequences (`.seq`) on top of a live **AmScope** microscope feed, used to light-
stimulate 60-electrode MEA cultures (optogenetics and photovoltaic/solar-cell
stimulation). Owner: Jerry (Yikan), a neuroscience/bioengineering researcher, not a
software developer. One program in the "MEA Suite" (siblings: MEA Analyzer/MEA-NAP,
MEA Explorer, MEA Solar Cell Analyzer); this folder is self-contained.

**Before non-trivial work, read `docs/HANDOFF.md`** (history, decisions, open
items). For anything about sequences, timing, TTL, optics or scale, also read
`docs/POLYGON_AND_STIMULATION.md`. `README.md` is the user-facing guide.

## Run and test (Windows, cmd)

```
set PY=%USERPROFILE%\Documents\MEA Analyzer\.venv\Scripts\python.exe
"%PY%" tests\test_core.py          # 6 tests; must stay 6/6
"%PY%" app.py                      # or run_debug.bat for a visible console
"%PY%" tools\inspect_seq.py "..\config\Sept2021_SweepBRtoTL.seq"
```

`test_core.py` finds the reference sweep at `..\config\Sept2021_SweepBRtoTL.seq`
(or `MPC_SAMPLE_SEQ`). Without it, two tests print "skipped", so check that they ran.

Jerry launches the app from the suite selector (`MEA Suite.exe`, suite root) or
`MEA Pattern Creator.exe`. After you change code, tell him to press
**⟳ Reload** (toolbar, Ctrl+R): it restarts the process and restores his open
project, so he does not need to relaunch.

## Layout

```
app.py                  entry: QApplication + mea_theme.apply_qt(app, "pattern")
mpc/seqfile.py          .seq (MSQ1) reader/writer, the format reference
mpc/patterns.py         Shape/Frame model, rasterizer, hit-testing
mpc/project.py          document model, export, timed export, pulse train, sidecars
mpc/calibration.py      camera<->canvas affine/homography (identity is default)
mpc/alignment.py        Umeyama similarity fit + PIL raster warp
mpc/camera.py           camera backends + synthetic fallback
mpc/amcam_backend.py    AmScope/ToupTek SDK (ctypes) discovery + streaming
mpc/gui/                main_window, imageview (canvas), calibration/alignment panels
tools/                  CLI scripts: inspect/compare .seq, generate radial and
                        settings-test sequences, measure optical scale from a photo
tests/test_core.py      plain-python tests (pytest also works)
amcamsdk.20250415/      VENDOR SDK, 300 MB, read-only. Never edit or move it.
mea_theme.py            copy of the suite master ..\config\mea_theme.py
```

## Hard rules

1. **`.seq` layout:** each frame block is a 20-byte header, then the
   width*height/8 bitmap, then a 4-byte zero trailer. Until 2026-10-15 the code
   assumed a 24-byte header. The round-trip still passed, but every pattern
   landed 32 DMD px (~82 µm) to the right. A byte-exact round-trip test cannot
   catch offset bugs, which is why `test_sample_sweep_geometry` and
   `test_bitmap_offset_in_block` exist. Never weaken them.
2. **A `.seq` holds bitmaps only, no timing.** PolyScan gives every frame the same
   dwell (Master: Pattern Period; Slave: one frame per TTL pulse). Durations are
   made by repeating frames (`Project.timed_sequence`) or by TTL timing.
3. **Python env:** a per-machine venv at
   `%USERPROFILE%\Documents\MEA Analyzer\.venv`, built by `..\config\setup.bat`
   from `..\config\requirements.txt` (shared by the whole suite). Never create a
   venv inside this Nextcloud-synced folder and never auto-install packages from
   app code. If a dependency is needed, add it to the suite `requirements.txt`
   and tell Jerry to re-run `setup.bat` on **both** machines.
4. **Launchers:** if you touch any `.exe` source, build wide paths with
   `wcscpy`/`wcscat`, never `swprintf("%s")` (mingw truncates). Save every `.bat`
   with CRLF line endings (LF breaks `goto`). See `LAUNCHER_AND_PACKAGING.md`.
5. **Theme:** colours come from `mea_theme` (accent `pattern` = #DB7FFF). Do not
   hard-code hex in app code. Change the master `..\config\mea_theme.py`, then
   re-copy it here. Rules are in `GUI_STYLING.md`. (Known violations: `#8fd0ff`
   in `alignment_panel.py` line 74 and `main_window.py` line 952.)
6. **PyQt6 only:** use scoped enums (`Qt.AlignmentFlag.AlignCenter`), `QAction`
   from `QtGui`, `exec()`, and `ev.position().toPoint()`. Do not reintroduce PyQt5.
7. **AmScope SDK:** buffers passed to `PullImage*` must be
   `ctypes.create_string_buffer`. A `bytearray` raises inside a bare except, which
   shows up as "connected but no video". Only one program can hold the camera, so
   the AmScope app must be closed.
8. **Two machines, one synced folder:** Fett (desktop, has the camera and
   Polygon) and FettLaptop share this folder via Nextcloud. Do not edit a file
   Jerry has open on the other machine. Per-machine state
   (`camera_settings.json`, `camera_calibration.json`, `camera_connect.log`,
   `_reload_recovery.*`) currently lives here, which is a known open item.
9. **Big files:** full-canvas `.seq` files are ~0.28 MB per frame (a 160-frame
   sweep is 44 MB). Do not commit them to git (see `.gitignore`); regenerate
   them with `tools/`.
10. **Back-end vs GUI:** keep back-end modules UI-free (no Qt imports outside
   `mpc/gui` and `app.py`).

## Working with Jerry

- Keep replies short and plain-language. Explain physics or instrument behaviour
  with numbers, and cite **peer-reviewed or data-backed sources** when available.
  This is his standing preference.
- For visible GUI changes, show a screenshot or describe the change and get his
  OK before calling it done. He asked for this during the PyQt6 port.
- He works by trying things on the rig and reporting back. When his observations
  contradict documentation (for example, PolyScan settings with no effect),
  trust the observation and record it in `docs/POLYGON_AND_STIMULATION.md`.
- Ask before deleting files or changing anything outside this folder,
  especially `..\config\` (suite-shared) and the other programs.
