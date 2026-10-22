# MEA Explorer — launcher & packaging notes

_Last updated 2026-07-22. Written by a Claude Cowork session so future sessions can catch up on how this program is launched and packaged._

## What this program is

The raster / spike-data viewer (a pyqtgraph application). Opens a spike-data folder and lets you scroll large multi-electrode rasters interactively.

Entry script (relative to this folder): `app.py`.

## How to launch it

Double-click **`MEA Explorer.exe`** in the `MEA Analyzer` root folder. It opens the program with no console window, using the shared environment below. If the environment has not been built yet, it shows a dialog asking you to run `setup.bat` first.

The `.exe` must stay in the root folder: it locates this program's `app.py`/GUI as `<root>\MEA Explorer\...`, i.e. relative to its own location.

### Icon

A spike raster (rows of ticks) with a bright vertical scan line, emerald accent.

### Dependencies

numpy, scipy, h5py, matplotlib, pyqtgraph / PyQt5 (see this folder's requirements.txt). All of these are installed into the shared environment by `setup.bat`.

## The shared Python environment

All four MEA programs (MEA Analyzer, MEA Explorer, MEA Pattern Creator, MEA Solar
Cell Analyzer) share **one** Python environment:

    %USERPROFILE%\Documents\MEA Analyzer\.venv

It lives under **Documents on purpose**. The `MEA Analyzer` program folder is
synced between machines with Nextcloud, but a Python virtual environment is
machine-specific (its `pyvenv.cfg` hard-codes a path to the base Python it was
built from). If the `.venv` were kept inside the synced folder, the two
machines would repeatedly overwrite and rebuild each other's environment. Keeping
it under Documents (which is local to each PC) avoids that entirely.

The environment is built by **`setup.bat`** in the `MEA Analyzer` root folder.
`setup.bat` is standalone and must be run **manually, once per machine**:

  1. finds or installs Python 3.14 (via the `py` launcher, common install paths,
     or `winget install Python.Python.3.14`);
  2. creates the `.venv` under Documents (rebuilding it if a broken/copied one
     is found);
  3. installs PyTorch (CUDA 12.8 build, CPU fallback) then everything in the
     root `requirements.txt`;
  4. verifies the core packages import.

None of the launchers install anything — if the environment is missing they
just tell you to run `setup.bat`.

## How the `.exe` launcher was packaged

The `.exe` is **not** a bundled Python or a converted script. It is a tiny
native Windows program (compiled from ~60 lines of C) whose only job is to
launch this program the same way the old `.bat`/`.vbs` did, but with a custom
icon and no console window. At run time it:

  1. finds its own folder (`GetModuleFileNameW`);
  2. builds the path to this program's entry script, relative to itself;
  3. locates `pythonw.exe` (fallback `python.exe`) in the Documents `.venv`;
  4. launches the app windowless with `CreateProcessW`, or shows a small
     message box if the entry script or the environment is missing.

Key build facts (so a future session can reproduce or modify it):

  - **Compiler:** mingw-w64 cross-toolchain — `x86_64-w64-mingw32-gcc` plus
    `x86_64-w64-mingw32-windres`. Built on Linux, targets 64-bit Windows.
  - **Subsystem:** GUI (`-mwindows -municode`), so double-clicking shows no
    console window. Entry point is `wWinMain`.
  - **Icon:** a multi-resolution `.ico` (16, 24, 32, 48, 64, 128, 256 px)
    embedded as resource id 1 via a one-line `.rc` file (`1 ICON "name.ico"`).
    Windows uses the lowest-id icon as the application icon.
  - **Path handling gotcha (important):** Unicode paths are assembled with
    `wcscpy`/`wcscat`, **never** `swprintf("%s", ...)`. Under this toolchain a
    wide `swprintf` treats `%s` as a *narrow* string, which truncated the folder
    path to just its first character ("C") — an early build failed with
    "could not find the GUI" for exactly this reason. Use `%ls` or plain
    `wcscat` for wide strings.
  - **Batch/VBS line endings:** `.bat` files must be saved with Windows CRLF
    line endings. With Unix LF endings `goto` fails at run time with
    "The system cannot find the batch label specified". (`setup.bat` is full of
    `goto`; the simple launchers are not, which is why they tolerated LF but
    `setup.bat` did not.)

### Rebuilding the icon and the exe

On a Linux box with `mingw-w64` and Python+Pillow:

```
# 1. (re)generate the .ico with Pillow, then:
x86_64-w64-mingw32-windres launcher.rc -O coff -o launcher_res.o
x86_64-w64-mingw32-gcc launcher.c launcher_res.o -o "MEA Explorer.exe" \
    -mwindows -municode -O2 -s
```

The launcher C source is generic: it derives the app path from the program's
folder name, so the same source builds all four exes — only the embedded icon
and the folder name differ.

## Files

- Root folder: `MEA Explorer.exe` — the icon-bearing launcher.
- Replaced launcher: MEA Explorer.vbs (root) — safe to delete once the .exe is in place.
- This folder also has `run.bat` (windowless) and `run_debug.bat` (visible console for troubleshooting). Both were repointed to the Documents `.venv` and no longer use the old root `.venv`.

## Shared UI theme (mea_theme.py)

_Added 2026-07-23._ This program now ships with the shared MEA Suite theme kit
`mea_theme.py in this folder` and a `THEME.md` in this folder. The kit carries the same indigo->teal
palette, this program's accent (#34D399 (emerald)), Segoe UI fonts, rounded "card" panels, and
the parula plot colormap used by the icons, splash screens, and the MEA Suite
selector. `THEME.md` has the one-call integration for this program's toolkit; the
suite master + a visual reference are in `config\mea_theme.py` /
`config\theme_preview.png`. The theme is not yet wired into the app code — adopt it
from this session when convenient.
