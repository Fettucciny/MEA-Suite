# MEA Suite — handoff (written 2026-10-22, migrating from Claude Cowork to Claude Code)

This is the running memory of the project. Earlier work happened in Claude Cowork
sessions attached to a claude.ai project called "MEA Suite"; those sessions can't be
read from here, so everything a future session needs is in this folder. Update this
file at the end of every task.

## Who / what

- User: Jerry (Yikan). Lab work: motor-neuron spheroids and primary muscle on MCS
  60-electrode MEAs (200 µm pitch, 60MEA200), optogenetic / laser stimulation projected
  with a Mightex Polygon 1000 DMD through a 4X objective, AmScope camera for alignment.
- Machines: Windows PCs (one is named `fett`), folder synced by Nextcloud at
  `C:\Users\Yikan\Nextcloud2\MEA\MEA Analyzer\`. Data: `C:\Users\Yikan\Nextcloud2\MEA Data\…`.
- Python 3.14 venv at `%USERPROFILE%\Documents\MEA Analyzer\.venv` (built by
  `config\setup.bat`; PyTorch CUDA 12.8, PyQt6, PyQt5, pyqtgraph 0.14, scipy, h5py, …).
- User preference: cite peer-reviewed or data-backed sources when making scientific
  claims; keep answers concise.

## Program status

### MEA Analyzer (MEA-NAP pipeline GUI) — `MEA Analyzer\`
- Reference implementation of the SAND-Lab MEA-NAP pipeline for MCS HDF5: spike
  detection (bior1.5 CWT, `mea_nap_wavelet_detect` + `sand_spike_detection_reference`),
  activity / network metrics (`meanap\pipeline\`), SpKit EGM, phase lock-in, spectral
  propagation, digital (laser) events.
- GUI ported Tkinter → PyQt6 on 2026-07-23 (`MEA-NAP\PYQT6_PORT.md`); back end unchanged.
  Legacy Tkinter copy frozen in `MEA Analyzer (Legacy)\`.
- Treated as read-only by the other tools' sessions. Known quirk: `mea_nap_io.CONFIG`
  has hard-coded `DATA_DIR` / `SPIKE_DIR` defaults that go stale when a new dataset
  is opened; consumers must set them (MEA Explorer resolves the `.h5` by convention
  instead, see below).
- Open item noted once by the user (not investigated): with "SpKit EGM" and "Phase
  Lock-In" both checked, the run stayed at "0% starting…".

### MEA Explorer — `MEA Explorer\` (most recent work, Oct 2026)
Interactive raster viewer. `app.py` (main window), `explorer.py` (single-electrode
window), `datasource.py` (only module touching the loaders), `export.py`, `lod.py`,
`config.py` (auto-discovers `MEA-NAP`), `mea_theme.py` (copy).
- Features: LOD raster (heatmap when dense / ticks when sparse, with hysteresis;
  Auto/Heatmap/Ticks selector), electrode map with show/hide and 3 highlight groups +
  editable legend, electrode reorder dialog with save/load JSON, trigger lines toggle,
  time axis that follows zoom (prefers ms), heatmap scale slider, image export
  (QPainter-based PNG/SVG with font sizes, width/DPI, hide-tick option), spike export
  (MEA-NAP npz layout — keep as is), waveform on double-click, QC toggle (drop
  auto-ground channels).
- Settings persisted in `mea_explorer_settings.json` next to app.py: view mode, time
  unit, triggers, heat gain, export options, last electrode order (+source file),
  spike_dir, explorer options, data_dir (only when the raw file was located by hand).
  This file is machine-specific but lives in the synced folder — open item.
- **Electrode explorer (added 2026-10-06):** toolbar "Explore electrode…" or
  Shift-click an electrode on the map. Top: filtered trace, **3rd-order Butterworth
  200–3000 Hz, zero-phase (scipy sosfiltfilt)** — a display-only filter set by
  `TRACE_FILTER_ORDER` / `TRACE_FILTER_BAND` in datasource.py, independent of the
  detector's own bandpass. Spike markers on the trace (green = in burst). Bottom:
  spike ticks + burst bars. Bursts via `meanap.pipeline.burst_detection`
  (Bakkum ISI_N; auto threshold = histogram valley capped at 100 ms, or manual);
  summary card, burst table (click → zoom), CSV export. Trace drawn only for views
  ≤ 30 s; raw `.h5` loads fully into RAM on first use (asks above 4 GB); "Locate raw
  .h5…" if not found. Verified identical to `single_channel_burst_detection`.
- Raw-file discovery: `datasource.resolve_h5` checks config/io DATA_DIR, then
  `<root>\H5` next to `<root>\Output\SpikeData`, then nearby folders. This also fixed
  duration, trigger lines and waveforms for datasets other than the hard-coded one.
- Tests: `python tests\run_tests.py` — `tests\stubs\` fake `mea_nap_io` /
  `digital_events` / `original_plots` over a synthetic 120 s fixture
  (`tests\fixtures\MEA 071826\…`, 33 labels, 4 triggers); the real
  `mea_nap_wavelet_detect` + `meanap` are imported from the real MEA-NAP folder.
  Covers h5 discovery, filter correctness, burst identity with the pipeline, and the
  GUI end to end offscreen (writes `tests\out\*.png`).
- Electrode-order files in `config\`: `electrode_order.json`, `electrode_order_BR_to_TL.json`
  (58 entries; omits 15 and 31), `electrode_order_RadialCenterOut.json` (all 60;
  distance from the array centre between 44/45/54/55, ties clockwise from 12 o'clock,
  centre at the top of the raster; `rings` key holds distances and approximate arrival
  frames; preview PNG beside it).
- README.md still describes the PyQt5 / `.vbs` era — update it when convenient.

### MEA Pattern Creator — `MEA Pattern Creator\`
PyQt6 editor for Polygon `.seq` sequences: live AmScope feed (`amcam` SDK auto-found
under `amcamsdk.20250415\`), ROI drawing, camera↔canvas calibration (remembered in
`camera_calibration.json`), per-sample alignment, timeline, `.seq` import/export
(`mpc\seqfile.py`), `.mpcproj` projects. See README.md.
- **Frame layout bug fixed 2026-10-15:** bitmap starts right after the 20-byte frame
  header (earlier code used +24, shifting every exported pattern 32 DMD px ≈ 82 µm to
  the right). `.seq` files exported before the fix carry the shift — re-export them.
  Regression test in `tests\test_core.py` (byte-exact round trip of
  `config\Sept2021_SweepBRtoTL.seq`).
- Patterns in `config\`: `Sept2021_SweepBRtoTL.seq` (160 frames, 45° band, BR→TL),
  `Oct2026_RadialCenterOut.seq` (133 frames: 82 growing rings + 51 blank),
  `Oct2026_SettingsTest_20frames.seq` (diagnostic), `radial_vs_sweep_preview.png`.
  Format and timing details: `docs\polygon-seq-notes.md`.
- README still says PyQt5 in its last line; the code is PyQt6.

### MEA Solar Cell Analyzer — `MEA Solar Cell Analyzer\`
Artifact classification and removal for recordings with solar cells + spheroids
(peaks → solar / piezo / biological; STA subtraction; exports MEA-NAP-format cleaned
spikes; field-radius fits; footprint maps). Large README.md; `selftest.py` runs
headless synthetic tests; `verify_coupling.py` confirmed the headstage is DC-coupled.
Built in separate sessions — the README is the authoritative description; no open
items were recorded here.

## Cross-cutting facts worth remembering
- Scale at 4X: 0.8583 µm per native camera px (5440×3648), 2.575 µm per DMD canvas
  px (1824×1216). PolyScan's own `.map` says 2.70004 (4.9 % higher) — unreconciled.
- PolyScan2 under external TTL: each pulse advances one frame, sequence wraps forever,
  Batch Size must be ≥ frame count, pause/loop-count settings have no effect; dark gaps
  must be blank frames or gaps in the TTL train.
- Electrode geometry helper: `mea_nap_original_plots.mcs_coord(label) -> (col, 9-row)`.

## Open items / ideas (none in progress)
1. Move `mea_explorer_settings.json` out of the synced folder (e.g. `%APPDATA%`).
2. Refresh the MEA Explorer and Pattern Creator READMEs (PyQt6, `.exe` launchers,
   electrode explorer, tests).
3. ISI_N histogram diagnostic panel in the electrode explorer (offered, not requested).
4. Reconcile the 2.575 vs 2.70004 µm/px calibration discrepancy.
5. The MEA Analyzer "0% starting…" report with SpKit EGM + Phase Lock-In.
6. Consider `git init` here with the provided `.gitignore` (Nextcloud + git can
   conflict on `.git\` internals; a bare repo outside the synced folder, or syncing
   only the working tree, is safer).

## Recent change log
- 2026-10-19 `config\electrode_order_RadialCenterOut.json` + preview.
- 2026-10-06 MEA Explorer: electrode explorer (explorer.py), trace filter 3rd-order
  200–3000 Hz, h5 discovery, `export_bursts_csv`, settings for explorer/data_dir.
- 2026-10-15 Pattern Creator: `.seq` frame-offset fix; radial and settings-test
  sequences; polygon notes (see docs).
- 2026-07/08 (earlier Cowork sessions): PyQt6 migrations, shared theme, `.exe`
  launchers, per-machine venv under Documents, Explorer feature set above.
