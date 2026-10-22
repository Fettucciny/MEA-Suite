# CLAUDE.md — MEA Solar Cell Analyzer

Context for Claude Code working in **this folder** (`MEA Solar Cell Analyzer/`).
This program was built and iterated across several Claude Cowork sessions; this file
carries that accumulated context so you can continue without re-deriving it. Read
`README.md` for end-user functionality — this file is the engineering/agent view:
conventions, architecture, domain knowledge, decisions, and gotchas.

---

## 1. What this is

A PyQt6 + pyqtgraph desktop app that analyzes **MCS (Multi Channel Systems) HDF5 MEA
recordings** of **motor-neuron spheroids + optically-driven solar cells** on the
same 60-electrode array (8×8, 4 corners absent). It does solar-cell field mapping,
spheroid footprinting, peak detection/classification (solar vs piezo-laser-artifact
vs biological), and artifact-subtraction to recover clean biological spikes exported
in **MEA-NAP format** for the rest of the suite.

It is one of four programs in the **MEA Suite** (MEA Analyzer/MEA-NAP, MEA Explorer,
MEA Pattern Creator, MEA Solar Cell Analyzer). Only this program is in scope for this
repo.

### Golden rules (do not violate without asking)

1. **Reuse the MEA-NAP loaders; never reimplement HDF5 / spike / digital-event
   formats.** All of that lives in `loaders.py`, which is the ONLY module allowed to
   import the `mea_nap_*` modules. Everything else uses plain numpy + the small
   dataclasses `loaders.py` exposes.
2. **Never modify the MEA-NAP folder** (`…/MEA Analyzer/MEA Analyzer/MEA-NAP`) or the
   MEA Analyzer `.bat`/`setup.bat`. It is a vendored, tested dependency shared by the
   whole suite and treated read-only.
3. **Never hard-code theme colors in app code.** Pull them from `mea_theme.py`
   (`mea_theme.PANEL`, `mea_theme.accent("solar")`, …). This program's accent is
   `solar` = `#FBBF24` (amber). See §8.
4. **Respect the MCS coordinate frame** (§5). Getting label→grid wrong is the single
   most common bug here — it has bitten every session.
5. **Verify scientific claims against real data, not intuition.** The user is a
   domain expert who repeatedly (and correctly) catches hand-wavy reasoning. The
   working pattern is to write a small `verify_*.py` diagnostic script, run it on the
   real recording, and look at the numbers/plots before concluding. See §6, §10.

---

## 2. Where everything is (paths on the build machine, Windows)

This folder is synced between machines via **Nextcloud**, so paths look like:

```
C:\Users\Yikan\Nextcloud2\MEA\MEA Analyzer\      <- suite root (.bat launchers, config\, MEA Suite.exe)
  config\                                         <- SHARED: master mea_theme.py, theme_preview.png, .seq configs
  MEA Analyzer\MEA Analyzer\MEA-NAP\              <- the loaders (READ-ONLY vendor dep): mea_nap_io.py, meanap\, ...
  MEA Solar Cell Analyzer\                        <- THIS repo
  MEA Explorer\  MEA Pattern Creator\  ...        <- sibling programs (out of scope)
```

- **Loaders discovery:** `config.py` finds the MEA-NAP folder via `MEANAP_DIR` env
  override → known absolute paths → a search up a couple of parent levels. If
  `import mea_nap_io` fails, set `MEANAP_DIR` to the folder that contains
  `mea_nap_io.py`.
- **Shared Python env (all four programs):** `%USERPROFILE%\Documents\MEA Analyzer\.venv`
  — deliberately under Documents (local per machine), NOT in the Nextcloud-synced
  folder, because a venv's `pyvenv.cfg` hard-codes an absolute base-Python path and
  would thrash between machines. Built once per machine by `setup.bat` in the suite
  root (installs PyTorch + the root `requirements.txt`). Nothing auto-installs at
  launch; if imports fail, the user runs `setup.bat`. Python is 3.14.
- **Data (MCS .h5 recordings):** `C:\Users\Yikan\Nextcloud2\MEA Data\MEA 071826\H5\`
  (the pipeline layout is `<root>\H5\` beside `<root>\Output\SpikeData`). The main
  solar-cell recording stem used for diagnostics:
  `2026-07-21T14-22-461 Solar Cell Laser`.
- **Launcher:** `MEA Solar Cell Analyzer.exe` in the suite root (a tiny native C
  launcher, not a bundled Python). Entry script is this folder's `app.py`. See
  `LAUNCHER_AND_PACKAGING.md`.

---

## 3. How to run, test, and verify

Use the shared venv's interpreter:

```bat
"%USERPROFILE%\Documents\MEA Analyzer\.venv\Scripts\python.exe" <script>
```

- **Launch the GUI:** run `app.py` (or double-click the suite-root `.exe`).
- **Self-tests (fast, loader-free):** `python selftest.py` — pure numpy/geometry/DSP
  checks that do NOT need the MEA-NAP loaders, torch, or h5py, so they run anywhere.
  It must print `ALL SELFTESTS PASSED`. **Add a test here for any new pure-logic
  primitive.** Current suites include geometry, peaks, templates/classify, filtering,
  field map, per-cell max, sustained plateau, footprint, and more.
- **Compile check:** `python -m py_compile <module>.py` on every module you touch.
- **Offscreen GUI smoke test** (construct the window with no display):
  ```
  QT_QPA_PLATFORM=offscreen MEANAP_DIR="<path to MEA-NAP>" python -c "..."
  ```
  Construct `app.MainWindow()` and exercise the methods you changed. (On the cloud
  Cowork sandbox, PyQt6 + pyqtgraph had to be `pip install`ed; on the real machine
  they're in the shared venv.)
- **Real-data diagnostics — the `verify_*.py` pattern:** the raw culture H5 is ~700 MB
  and too big to move around, so the established workflow is: write a small
  `verify_<thing>.py` that opens the recording with `loaders`, computes something, and
  **saves a SMALL `.npz` (and often a `.png`)** diagnostic; run it; then read that
  small artifact back to analyze. Existing examples: `verify_solar.py`,
  `verify_coupling.py` (DC-vs-AC headstage test), `verify_filter.py`, `verify_span.py`,
  `verify_stim.py`, `verify_bio_shape.py`. Follow their structure (module-level
  `DATA_DIR`/`STEM`, `main()` that writes `*_diag.npz`).

---

## 4. Architecture & module map

Data flow: **loaders.py** (raw µV traces + laser events + validated detector) →
**peaks.py** (triggered averages, peak detection, features) → **filtering.py** (STA
bank, artifact subtraction, biological re-detection) / **fieldmap.py** +
**footprint.py** (spatial analysis) → **pipeline.py** (orchestration: ties project +
loaders + analysis together) → **app.py** / **electrode_map.py** (GUI) +
**heatmap_export.py** / **calibration.py** (figure/PNG output).

| Module | Role |
|---|---|
| `loaders.py` | **Only** importer of `mea_nap_*`. Exposes `RawRecording` (calibrated µV traces, `fs`, labels), `laser_events()` (flood ON/OFF), `electrode_xy`/`grid_labels`, the validated `bandpass`/`detect_spikes` bior1.5 detector, MEA-NAP spike-file read/write. |
| `config.py` | Locates the MEA-NAP loaders and puts them on `sys.path` (`ensure_loaders_on_path`). Env overrides: `MEANAP_DIR`, `MEANAP_DATA_DIR`, `MEANAP_SPIKE_DIR`. |
| `geometry.py` | Electrode array + `SolarCell` + `Spheroid` geometry in grid/µm. Nearest-cell assignment (`nearest_cell_distance_um`, `max_by_cell`), distances. Pure numpy; heavily unit-tested. |
| `peaks.py` | `flood_triggered_average` (STA + p2p), `flood_plateau` (sustained level), `detect_peaks`, `train_starts`, event windows, robust noise. Pure numpy/scipy. |
| `filtering.py` | `build_sta_bank` (per-electrode STA + p2p + noise + plateau), `subtract_flood_artifact`, `filter_recording` → `FilterResult`. The artifact-removal core. |
| `fieldmap.py` | `compute_field_map` → `FieldMapResult` (decay fit, effective radius, anisotropy, per-cell peaks). |
| `footprint.py` | Spheroid electrophysiological footprint (centroid, spread, reach). |
| `templates.py` / `classify.py` | Learned peak-shape wavelets per class; solar/piezo/bio classification. |
| `pipeline.py` | Orchestration: `_sta_p2p`, `field_map_from_solar`, `filter_sample`/batch, `solar_cell_peaks`, `solar_cell_plateaus`, `footprint_from_filter`, event/region windows. The layer the GUI calls. |
| `project.py` | Per-project state: solar cells per recording, spheroid outline, image alignment, analysis params, data/spike dirs. Persisted to `settings.json`. |
| `app.py` | The tabbed main window + entry point. Large. Tabs: Setup/placement, Peaks, Filter, Field map, Footprint, Calibration. |
| `electrode_map.py` | Click-to-place pyqtgraph canvas (cells, spheroid, microscope-image overlay). |
| `heatmap_export.py` | Dark-theme electrode heatmap PNGs (pre/stim/post triptych) for the Filter tab. |
| `calibration.py` | Calibration report: learned wavelets + detection-wavelet figures + summary. |
| `bursts.py`, `seqfile.py` | Burst-rate detection; .seq helper. |
| `selftest.py` | Loader-free regression suite (run this). |
| `mea_theme.py` | Shared suite theme kit (see §8). A synced copy of the suite master in `config\`. |

---

## 5. Domain conventions (get these right)

- **MCS electrode labels are "CR" = Column, Row** (e.g. `"47"` = col 4, row 7). The
  array is 8×8 with the four corners removed → 60 electrodes. Labels like `Ref` are
  non-grid.
- **Label → grid mapping flips the row so row 1 is at the TOP:** label `"CR"` → grid
  `(C, 9 − R)`. So `"47"` → grid `(4, 2)`. **This flip is the #1 source of bugs** —
  when planting synthetic values in tests, convert through this mapping (e.g. a cell
  at grid `(2,2)` is fed by label `"27"`, NOT `"22"`). `mcs_coord`/`electrode_xy`
  already apply the flip; keep heatmaps un-flipped in y to match the on-screen map.
- **Pitch:** default **200 µm**; µm = grid × pitch.
- **Array center grid** = `(4.5, 4.5)`; **FOV** = 3000 µm diameter (cells may be
  placed off-grid / outside the electrode square, so crops must widen to include
  cell corners).
- **Laser floods:** ~one exposure every ~5 s; may be a train of pulses. `train_starts`
  collapses pulse onsets into one start per train (~0.25 s grouping). `flood_on_off`
  returns individual pulse onsets + offsets + single-pulse duration.

### The two per-electrode voltage metrics (important distinction)

- **STA peak-to-peak** (`sta_bank[lab]["p2p"]`, µV): `max − min` of the flood-
  triggered average. **Dominated by the fast capacitive edge transient at laser onset
  (~8.5 mV),** NOT the sustained photovoltage. This is what the field map, the
  effective-radius fit, and the per-cell "max voltage detected" annotation use.
- **Sustained plateau** (`sta_bank[lab]["plateau"]`, µV, signed): `flood_plateau` —
  the median held level during illumination on a single-pulse average, edges
  excluded. This is the real sustained photovoltage (~0.1–0.4 mV), far smaller than
  the edge spike. Surfaced per cell via `pipeline.solar_cell_plateaus` and shown as a
  "plat …" line on the cell annotations and the field-map info line. Requires the
  recording's single-pulse ON/OFF duration (else NaN).

---

## 6. Key decisions & findings (accumulated across sessions)

- **The edge transient vs the sustained photovoltage.** The large signal at a solar-
  cell electrode is the electrode–electrolyte double-layer *capacitively charging* at
  the light edge (~0.3 ms, ~8.5 mV), not the sustained cell output (~0.36–0.39 mV).
  The p2p metric therefore reports the edge spike; the plateau metric was added to
  report the sustained level alongside it.
- **The headstage is DC-coupled (verified 2026-07).** `verify_coupling.py` checks this
  two ways: (A) the MCS file's `InfoChannel` HighPass filter field reads 0/DC, and
  (B) the raw (un-filtered) STA holds a plateau across the ~20 ms illumination instead
  of decaying in ~0.5 ms. Both agree: DC-coupled. **Consequence:** the sustained
  photovoltage IS present in the raw recording; a ~0.5 ms "differentiated" transient
  in analysis would come from a *software* 300 Hz high-pass, not the hardware. (This
  ruled out the "AC-coupled 300 Hz → τ=0.53 ms differentiation" hypothesis.)
- **Per-cell annotation feature (built this series of sessions).** Every placed solar
  cell is labeled, on all three surfaces (Field-map tab image + info line, in-GUI
  Filter map, exported heatmap PNG triptych), with the **max voltage detected on its
  nearest electrodes** (each electrode assigned to the cell whose *edge* it is
  nearest), generalizing to N cells; plus the **sustained plateau** as a second line.
  Core primitive: `geometry.max_by_cell(..., by_magnitude=?)`.
- **Spike detection reuses the validated MEA-NAP bior1.5 wavelet detector**
  (600–8000 Hz band). The optional "Tighten" gate is a 2nd-order Butterworth
  200–3000 Hz + 5× robust-noise amplitude threshold ANDed with wavelet hits.
  Open scientific question discussed but NOT acted on: whether filtering before
  wavelet detection distorts the true peak — relevant now that we know coupling is DC.
- **Theme adoption.** The GUI was migrated PyQt5 → PyQt6 and wired to the shared
  `mea_theme.py` (accent `solar`). See `GUI_STYLING.md` for the migration specifics.

---

## 7. Gotchas (things that have broken before)

- **`numpy.trapz` was removed in NumPy 2.x** → use `numpy.trapezoid`; guard with
  `_trapz = getattr(np, "trapezoid", None) or np.trapz` for cross-version support.
- **PyQt6 scoped enums:** `Qt.Horizontal` → `Qt.Orientation.Horizontal`,
  `Qt.DashLine` → `Qt.PenStyle.DashLine`, `QMessageBox.Yes` →
  `QMessageBox.StandardButton.Yes`, `dialog.exec_()` → `dialog.exec()`. The GUI is
  binding-agnostic via `from pyqtgraph.Qt import ...`, so prefer that import.
- **The row-flip** (§5) — re-stated because it keeps recurring.
- **`.bat` files need Windows CRLF** line endings (`goto` fails on LF). Not relevant
  to the Python but relevant if you touch launchers.
- **Nextcloud + git:** this folder is Nextcloud-synced. A `.git` folder inside a
  synced folder can cause sync churn/conflicts across machines. The user accepted this
  trade-off (per-program repo). If sync conflicts appear, consider a Nextcloud ignore
  rule for `.git/` or moving the repo. Don't commit from two machines simultaneously.
- **Big files:** the culture H5 is ~700 MB; don't try to copy/stage it. Use the
  `verify_*.py` → small `.npz` pattern. The `*_diag.npz`, `*.png`, `*.exe`, and
  `__pycache__` are gitignored (see `.gitignore`).

---

## 8. Theme / styling

Shared across the suite; only the accent differs per program. This program = `solar`
(`#FBBF24` amber). Apply once after `QApplication`:

```python
import mea_theme
app.setStyle("Fusion")
mea_theme.apply_qt(app, "solar")       # palette + fonts + full stylesheet
mea_theme.style_pyqtgraph("solar")     # dark plot bg + antialias (once)
lut = mea_theme.parula_lut(256)        # shared heatmap colormap
```

Palette tokens and the full design summary are in `THEME.md` and `GUI_STYLING.md`.
`mea_theme.py` here is a synced copy of the suite master `config\mea_theme.py` — if
the palette changes, edit the master and re-copy; don't fork colors. Exported
scientific figures (heatmaps, calibration) deliberately keep their own dark styling
(viridis/magma) and are NOT forced to the chrome theme.

---

## 9. Dependencies

Declared in `requirements.txt` (installed into the shared venv by `setup.bat`, not
per-project): `numpy`, `scipy`, `h5py`, `PyWavelets`, `pyqtgraph>=0.13`, `PyQt6`
(PyQt5 also works), `matplotlib`, `pillow`; and via the MEA-NAP loaders: `torch`,
`tqdm`. Python 3.14.

---

## 10. How the user works (observed working agreement)

- **Domain expert; data-backed only.** Standing preference: *"Only use peer-reviewed
  or data-backed sources when available."* When a scientific question comes up, don't
  assert — measure. Write a `verify_*.py`, run it on the real recording, read the
  numbers/plot, then conclude. The user has repeatedly caught incorrect reasoning, so
  show the evidence.
- **Deliver files, verify first.** Always `py_compile` + run `selftest.py` + an
  offscreen GUI smoke test before claiming done. Add selftests for new pure logic.
- **Small, surgical changes** to the existing architecture; keep the loader-reuse
  contract intact. Match existing patterns (e.g. new metrics flow through the STA bank
  and `pipeline.*`, parallel to `p2p`).
- Under Cowork the delivery was SendUserFile→device commit; under Claude Code you edit
  the files in place (this folder) directly.

---

## 11. Open offers / possible next steps (not started — ask before doing)

- Make the **sustained plateau** (τ-corrected with the fitted ~18.4 ms relaxation) a
  first-class calibration-report quantity, separate from the edge-spike p2p.
- Revisit **filter-before-detect**: now that coupling is confirmed DC, quantify
  whether the spike-band filter distorts the recovered peak amplitude.
- Characterize the **laser pulse-train structure** (single sustained pulse vs train)
  per recording — `verify_coupling.py` output hinted at extra pulse edges.
```
