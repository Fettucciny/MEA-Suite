# MEA Suite

Four Windows desktop programs for laser-sweep STDP experiments on 60-electrode
Multi Channel Systems (MCS) arrays driven by a Mightex **Polygon 1000** DMD and
recorded with an **MCS MEA2100**. Everything is Python (PyQt6 / PyQt5) and shares
one virtual environment, one theme kit and one set of electrode-order and
Polygon-sequence files in `config\`.

| Program | Folder | What it does |
|---|---|---|
| **MEA Analyzer** | `MEA Analyzer\` | GUI for the MEA-NAP analysis pipeline port in `MEA-NAP\`: wavelet spike detection, activity and network metrics, SpKit EGM, phase lock-in to the laser sweep, spectral propagation, digital (laser) events. |
| **MEA Explorer** | `MEA Explorer\` | Interactive raster viewer: level-of-detail raster, electrode map and highlight groups, trigger overlay, single-electrode explorer with burst detection, image and spike export. |
| **MEA Pattern Creator** | `MEA Pattern Creator\` | Draws stimulation patterns on the live AmScope camera feed, calibrates camera to DMD, aligns patterns to each sample, and reads and writes Polygon `.seq` files. |
| **MEA Solar Cell Analyzer** | `MEA Solar Cell Analyzer\` | Classifies and removes solar-cell and piezo artifacts from recordings of spheroids co-cultured with optically driven solar cells; exports cleaned MEA-NAP-format spikes. |

`MEA Suite.exe` at the root is a launcher that picks a program; each program
folder has its own `.exe` launcher that builds the environment on first run.

## Getting started

1. Clone the repository anywhere on a Windows PC.
2. Run `config\setup.bat`. It installs Python 3.14 if needed, creates
   `%USERPROFILE%\Documents\MEA Analyzer\.venv`, installs PyTorch (CUDA 12.8 by
   default, see the comments at the top of the script for CPU-only) and
   `config\requirements.txt`. The environment deliberately lives outside the
   code tree so that synced copies on several machines do not share it.
3. Launch a program with its `.exe`, its `.bat` (`MEA Analyzer\MEA Analyzer.bat`,
   `MEA Explorer\run.bat`) or `python app.py` from inside the program folder with
   the venv activated. The `run_debug.bat` scripts keep a console open so a
   startup exception is visible.

Experiment data stays outside the code tree, for example
`...\MEA Data\<experiment>\H5\` for recordings plus `Batch.csv` and
`...\MEA Data\<experiment>\Output\{SpikeData,Plots}` for results. `Batch.csv` is
the experiment manifest; its format is described in `MEA Analyzer\MEA-NAP\mea_nap_io.py`.

## What is not in the repository

These files are needed on a working machine but are excluded from git by
`.gitignore` because of size or licensing. Fetch them from the lab share or
regenerate them:

- `MEA Pattern Creator\amcamsdk.<date>\` — the AmScope/ToupTek camera SDK
  (`python\amcam.py` and `win\x64\amcam.dll` are what the app looks for).
  Download it from AmScope; the Pattern Creator also runs without it using a
  snapshot image or the synthetic test source.
- `config\*.seq` — the Polygon sequences (`Sept2021_SweepBRtoTL.seq`,
  `Oct2026_RadialCenterOut.seq`). Regenerate with the scripts in
  `MEA Pattern Creator\tools\`. The Pattern Creator round-trip test in
  `tests\test_core.py` needs the Sept2021 file.
- `config\1 Solar Cell.png` — the 18 MB 4X reference photo used to measure the
  electrode pitch.
- `*.exe` launchers — built from a small C program with mingw-w64 as described
  in each `LAUNCHER_AND_PACKAGING.md`. Attach them to a GitHub Release.
- Per-machine settings (`mea_explorer_settings.json`, `settings.json`,
  `camera_settings.json`, `camera_calibration.json`, `mea_nap_gui_settings.json`),
  logs, diagnostic `.npz` dumps and the frozen Tk-era `MEA Analyzer (Legacy)\`.

## Verifying a change

Follow the ritual in `CLAUDE.md`: byte-compile every touched file, construct
each GUI headless (`QT_QPA_PLATFORM=offscreen`), run
`MEA Analyzer\MEA-NAP\dev\check_gui_keys.py`, and for numerical modules prove
results are unchanged with `dev\astdiff.py` and `dev\compare_figures.py`.
Program-level tests: `python tests\run_tests.py` in `MEA Explorer\`,
`python tests\test_core.py` in `MEA Pattern Creator\`, `python selftest.py` in
`MEA Solar Cell Analyzer\`. The GitHub Actions workflow only byte-compiles; the
GUIs need the full CUDA/Qt environment and are verified on the lab machines.

## Documentation

- `docs\HANDOFF.md` — state of every program, open items, change log. Start here.
- `docs\polygon-seq-notes.md` — Polygon `.seq` byte layout, 4X scale, PolyScan TTL behaviour.
- `MEA Analyzer\MEA-NAP\mea-analysis-pipeline.md` — the MEA-NAP pipeline port.
- `MEA Analyzer\MEA-NAP\HANDOFF_2026-10-22.md` — phase lock-in history; read before touching `mea_nap_phase_lockin.py`.
- Per program: `README.md`, `THEME.md`, `GUI_STYLING.md`, `LAUNCHER_AND_PACKAGING.md`.
- `CLAUDE.md` files — working conventions for Claude Code sessions (also useful to humans).

## Conventions

- One theme kit: `config\mea_theme.py` is the master; each program carries a
  copy it imports. Edit the master and re-copy.
- Figures use PARULA for scalar data, matplotlib "Blues" for progression,
  crimson dashed for the nominal stimulus, greys for annotations.
- Numerical routines are never changed casually; a refactor must leave results
  byte-identical or report the floating-point difference.
- Batch scripts keep CRLF line endings (`.gitattributes` enforces this).

## Acknowledgements and licence

The analysis pipeline in `MEA Analyzer\MEA-NAP\` is a Python port of
[MEA-NAP](https://github.com/SAND-Lab/MEA-NAP) by the SAND Lab (University of
Cambridge), which is released under the GNU GPL-3.0. This repository is
licensed under the GNU General Public License v3.0; see `LICENSE`.
