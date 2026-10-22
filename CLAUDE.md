# MEA Suite — instructions for Claude Code

Four Windows GUI programs for laser-sweep STDP experiments on 60-electrode MCS
arrays, plus a launcher. Owner: Jerry (Yikan), machine "fett". This file is the
suite-wide contract; `MEA Analyzer\CLAUDE.md` adds the Analyzer specifics.

## Layout

```
MEA Analyzer\                      <- suite root (this file)
  MEA Suite.exe                    launcher
  config\                          shared theme kit (mea_theme.py = MASTER), icons,
                                   electrode orders, Polygon .seq files (36-44 MB)
  MEA Analyzer\                    the Analyzer (PyQt6)  -> see its CLAUDE.md
    MEA-NAP\                       all analysis code lives here
  MEA Analyzer (Legacy)\           Tk version, frozen; do not develop here
  MEA Explorer\                    PyQt5 (migration to Qt6 pending), app.py
  MEA Pattern Creator\             PyQt5
  MEA Solar Cell Analyzer\         PyQt5
```

Data and outputs live OUTSIDE the code tree, e.g.
`C:\Users\Yikan\Nextcloud2\MEA Data\<experiment>\H5\` (recordings + Batch.csv)
and `...\Output\{SpikeData,Plots}`. The whole tree is inside Nextcloud: the
sync client holds file handles, so any atomic `os.replace` must tolerate
`PermissionError` (see `mea_nap_io.save_progress` for the pattern).

## Conventions that are not optional

- **Python env:** `%USERPROFILE%\Documents\MEA Analyzer\.venv` (built by
  `config\setup.bat`). The launchers run `pythonw.exe`, so a startup exception
  shows NOTHING — the window simply never appears. After any change to a module
  the GUI imports, prove it constructs: see "Verification ritual" below.
- **Theme:** every program applies `mea_theme.apply_qt(app, "<accent>")`
  (analyzer blue, explorer emerald, pattern magenta, solar amber). Never hard-code
  colours in a GUI; pull them from `mea_theme`. Edit the MASTER in `config\` and
  re-copy to each program (they each carry a local copy they import).
- **Figures** (matplotlib, light background): the house palette is PARULA for
  scalar data, matplotlib "Blues" for day/epoch progression, crimson dashed for
  the nominal stimulus reference, greys `#555/#777/#888` for annotations,
  `#FAFAFA` panel faces. Do not introduce new palettes.
- **Never change numerical routines casually.** When a change must leave results
  identical, prove it with `MEA-NAP\dev\astdiff.py` (which top-level functions
  changed) and `MEA-NAP\dev\compare_figures.py` (byte-compare self-test PNGs
  before/after). Floating-point-only differences (≤1e-12) are acceptable and
  must be reported as such.
- **GUI <-> module parameter coupling:** `mea_nap_gui.build_schema()` binds
  `_P(PLK, "PHASE_LOCKIN", "<key>", ...)` etc. and `_cur()` does a bare
  `obj[key]` lookup. A key in the schema that is missing from the module's dict
  raises `KeyError` while the window is being built and, under pythonw, kills
  the app silently. Add the module key FIRST, then the schema entry, then run
  `MEA-NAP\dev\check_gui_keys.py`.
- **Batch.csv** is the experiment manifest (see `mea_nap_io.py` BLOCK B):
  `stem, age, group, "start-end"(s), region, grounded, [frames]`. Header optional.
  Group names starting with `radial` select the radial (centre-out) model in the
  phase lock-in; everything else is the linear sweep.
- Windows line endings are not required; write files with `newline=""` so you
  do not re-flow them. Don't touch `__pycache__`, `*.exe`, `*.seq`.

## Verification ritual (do this before saying something works)

1. `python -m py_compile <file>` for every file you touched.
2. Headless GUI construction:
   `set QT_QPA_PLATFORM=offscreen && python -c "import sys; sys.argv=['x']; import mea_nap_gui as g; from PyQt6 import QtWidgets; a=QtWidgets.QApplication(sys.argv); w=g.App(); w.close()"`
   from inside `MEA-NAP\`.
3. `python MEA-NAP\dev\check_gui_keys.py` — every schema key resolves.
4. For the phase lock-in: `python mea_nap_phase_lockin.py` runs the self-test
   (synthetic sweep + radial data) and writes PNGs under `_phaselock_selftest\`;
   compare against a pre-change run with `dev\compare_figures.py`.
5. Say exactly what changed numerically, if anything.

## Git

The suite root is a git repository (initialised 2026-10-22, branch `main`).
`.gitignore` excludes `__pycache__`, `*.exe`, `*.seq`, `*.h5`, `*.npz`, `Output\`,
`.venv`, the camera SDK, per-machine settings and `MEA Analyzer (Legacy)\`;
`.gitattributes` pins LF for text and CRLF for `.bat`/`.ps1`. Commit after every
verified change. The dated legacy copies (e.g.
`mea_nap_phase_lockin_legacy_10_19_26.py` + its `Rollback ... .bat`) remain as a
belt-and-braces rollback for the phase lock-in. The tree is Nextcloud-synced:
add `.git` to the Nextcloud client's ignored-files list so the two do not fight
over repository internals.

## Where the history is

- `MEA Analyzer\MEA-NAP\HANDOFF_2026-10-22.md` — full state of the analysis
  code, what changed in the last Cowork sessions and why, open items. READ IT
  FIRST when touching `mea_nap_phase_lockin.py`.
- `MEA Analyzer\MEA-NAP\mea-analysis-pipeline.md` — the MEA-NAP pipeline port
  (Stage 1 detection, Stage 2 activity/network).
- `MEA Analyzer\MEA-NAP\PYQT6_PORT.md`, `MEA Explorer\GUI_STYLING.md`,
  `*\THEME.md` — the Qt6 migration pattern and theming.
- `MEA Analyzer\MEA-NAP\HANDOFF_raster_viewer.md` — the raster viewer.
