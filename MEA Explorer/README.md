# MEA Explorer

Interactive, scrollable, zoomable raster viewer for MCS MEA recordings, with
section export. It **reuses the tested MEA-NAP loaders** for all HDF5 and
spike-file reading — nothing about those formats is reimplemented here.

## What it does

- **Two-resolution raster.** Zoomed out, it shows a binned spikes/bin rate image
  (~one bin per pixel) so panning an hour-long file stays instant. Zoom in and
  it automatically switches to real spike ticks per electrode. A **View**
  selector (Auto / Heatmap / Ticks) forces a mode when comparing recordings.
- **Trigger overlay.** Laser/digital ON times (from `read_digital_events`) are
  drawn as vertical lines.
- **Electrode map.** The square MCS 8x8 grid on the right mirrors the array
  layout (electrode numbers on each site; scales with the window). **Left-click**
  an electrode to cycle it through 3 highlight-color groups (none → 1 → 2 → 3 →
  none) — each group is a translucent band on its raster rows. **Right-click** or
  **Ctrl-click** shows/hides an electrode (hidden ones drop out and the raster
  re-stacks). The y-axis shows the real electrode labels. Buttons: Show all,
  Hide all, Clear highlight, Show only highlighted. A **legend** above the raster
  names the groups; edit the names in the **Legend labels** fields under the map.
- **Section export.** Toggle **Selection**, drag the shaded region over a span,
  then **Export spikes…** (`.npz`/`.csv`) or **Export image…** (`.png`/`.svg`).
- **Waveform on demand.** Double-click a channel to load its raw voltage for the
  visible window (constructs the heavy `MCSFile` on first use; **Clear waveform**
  frees it).

## Run

Double-click **`MEA Explorer.vbs`** in the parent folder (next to the `.venv`)
for a launch with **no console window**. It opens a folder picker — choose the
spike-data folder to browse, then pick a recording.

Alternatively, inside this folder: **`run.bat`** (also windowless, uses
`pythonw`), or **`run_debug.bat`** which keeps a console open and prints any
error. Crashes are also written to `error.log` here and shown in a dialog.

From a terminal you can skip the pickers:

```
run.bat                              # folder picker, then recording picker
run.bat 400um_2_D4                   # open a specific stem directly
run.bat --spike-dir="D:\path\to\SpikeData"   # start in a folder, then pick
run.bat 400um_2_D4 --no-qc           # keep auto-grounded channels in the raster
```

A folder's "recordings" = every `<stem>_spikes.npz` in it. If a folder has none,
the picker says so and lets you choose another — the raster is built from
detected spikes, so run Stage-1 detection first if a recording hasn't been
processed yet.

## Environment

This runs on the **shared MEA `.venv`** in the parent folder (the same one
MEA Analyzer uses). That venv already has everything the loaders need plus
**PyQt5** and **pyqtgraph**, so there's nothing to install. If the venv is ever
rebuilt from scratch and pyqtgraph is missing, add it with:

```
..\.venv\Scripts\python.exe -m pip install pyqtgraph
```

(`requirements.txt` lists that one addition; `requirements-full.txt` lists the
whole stack for building a brand-new venv.)

## Where it reads from

By default it uses the loaders' own `io.CONFIG` paths (`DATA_DIR`, `SPIKE_DIR`).
Override with environment variables if needed:

- `MEANAP_DIR` — folder containing `mea_nap_io.py` (default set in `config.py`
  and commented in `run.bat`).
- `MEANAP_DATA_DIR` — folder with the `.h5` files.
- `MEANAP_SPIKE_DIR` — folder with the `<stem>_spikes.npz` files.

## Layout

```
MEA Analyzer\                (shared root: .venv, Scripts\MEA-NAP, MEA Analyzer)
  MEA Explorer.vbs           double-click to launch (no console)
  MEA Explorer\              this folder
    app.py                   the pyqtgraph window + entry point (python app.py)
    config.py                puts the loaders on sys.path
    datasource.py            the only module that imports the loaders
    lod.py                   overview/detail decision (with hysteresis)
    export.py                image + spike/raw export
    run.bat                  windowless launcher
    run_debug.bat            launcher with a console (for errors)
    README.md, requirements.txt, requirements-full.txt
```

The modules run as plain scripts (the folder name has a space, so `python -m`
isn't used); `app.py` adds its own folder and the loaders folder to `sys.path`.
Only `datasource.py` touches the loaders, so the UI never couples to their
internals.
