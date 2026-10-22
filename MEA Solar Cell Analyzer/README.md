# MEA Solar Cell Analyzer

Analyze MCS MEA recordings of **motor-neuron spheroids + optically-driven solar
cells** on the same 60-electrode array. It **reuses the tested MEA-NAP loaders**
(`mea_nap_io`, `mea_nap_digital_events`, `mea_nap_wavelet_detect`,
`mea_nap_original_plots`) for all HDF5 / laser-event / spike-detection work —
nothing about those formats is reimplemented here — and adds the solar-cell /
spheroid-specific analysis on top.

## What it does

1. **Electrode map you click on.** Place **solar cells per recording** — 0 (laser
   only), 1, or 2+ (to test field amplification between two devices near each
   other) — each a square block with its own **rotation and electrode-contact
   axis**, by click *or* µm coordinates (from the array centre), including
   **off-grid / between electrodes**. Pick the recording under *Solar cells → For
   recording*, then *Add cell* / click the map; each cell's electrode direction is
   set independently. Trace the
   **spheroid** by clicking points around its perimeter — the closed outline is
   fitted through your clicks (with undo / clear), so irregular, non-circular
   spheroids are captured faithfully. A **3 mm field** is drawn around the array
   so cells off to the side stay placeable, and the placement map's electrode
   rings are translucent (no numbers) so a loaded **microscope image** of the MEA
   surface — which you can scale / rotate (0.1° steps) / move (1 µm steps) / fade —
   shows through for exact alignment. The **µm/pixel scale is auto-read from the
   image metadata** (ImageJ/OME-TIFF/TIFF-resolution/DPI) when present, else set
   manually. Positions, the traced outline, and the image alignment are stored in
   the project and drive every distance metric.
2. **Peak detection + classification.** Detects all peaks (either polarity), then
   labels each as **solar-cell voltage**, **piezoelectric laser**, or
   **biological** using the laser-flood timing (from the digital-event data — one
   flood ≈ every 5 s) plus learned peak-shape templates. Because your neurons fire
   *spontaneously*, any peak that is time-locked to a flood is an artifact; the
   solar-vs-piezo split is by template shape + proximity to the cell.
3. **Filter to pure biological.** Subtracts each electrode's flood-triggered-
   average artifact and re-runs the validated bior1.5 detector on the residual, so
   biological spikes are recovered even during the stim window. An optional
   **Tighten** gate (2nd-order Butterworth **200–3000 Hz** band-pass + **5× robust-
   noise** amplitude threshold, MEA-NAP style) is ANDed with the wavelet detections
   to reject low-amplitude non-biological events. On **artifact-contaminated
   electrodes**, any re-detected spike that still falls inside a flood-train window is
   also dropped — on a multi-mV solar/piezo electrode even the STA-subtraction residual
   dwarfs the biological threshold, and a spontaneous neuron is never time-locked to
   the laser, so these are residual artifacts (`reject_flood_locked`). Export the cleaned spike train **in
   the exact MEA-NAP format** (`<stem>_spikes.npz` + `.mat`) so it loads and plots
   directly in **MEA Analyzer (MEA-NAP)** and the raster viewer — or **batch-filter
   the whole data folder** in one click. Every filter run writes a **dark-theme
   heatmap plot** to a `Filtered Signal` subfolder of the Output folder, **split into
   pre / stim / post panels** (per-region spike rate on a shared colour scale); the
   **solar-cell field overlay is drawn on the STIM panel only** (the laser is off in
   pre/post). On every run it also writes a **burst-rate**
   version of the same plot (`<stem>_filtered_burst_heatmap.png`, bursts/min via a
   max-ISI detector). A **Metric** dropdown (spike rate / burst rate) picks what the
   live map shows; both plots are always saved. **Plot-option checkboxes** control the
   output and the live map: artifact rings on/off, **log vs linear-95th-percentile**
   colour, solar-field overlay (modeled from the cell location + measured amplitude,
   or the measured response interpolated), and the spheroid outline. Every placed
   **solar cell is outlined and labelled with the MAX voltage detected on its own
   electrodes** — each electrode is assigned to whichever cell its *edge* is
   nearest, so three cells in the field of view get three independent numbers (the
   value is the flood-STA peak-to-peak, the same quantity the field map fits, with
   the electrode it came from shown underneath). Across every map, a **gold ring marks
   artifact-subtracted electrodes** (consistent convention), electrode dots are large
   with numbers legible on any colour, and the linear colour scale is capped at the
   95th percentile so a couple of hyperactive electrodes can't wash out the footprint.
4. **Solar-cell field radius + shape (1 or 2 cells).** From a no-culture recording
   with its own placed cell(s), maps each electrode's flood-evoked response
   (distance measured to the *nearest* cell), fits the spatial decay, and reports
   the **effective field radius**, length constant, **anisotropy** (the field
   reaches farther along the electrode axis than across it), and field
   **peak / mean / total** — so you can compare a **1-cell vs 2-cell** recording to
   see whether two nearby devices amplify the field. Optionally **subtract a
   laser-only recording** to remove the piezo baseline and isolate the pure solar
   field. Works for any square size; shows a 2-D field map with all cells overlaid,
   each labelled with the **max voltage detected on its nearest electrodes** and,
   beneath it, the **sustained plateau** — the held photovoltage measured during
   illumination on a single-pulse average (edges excluded), reported alongside the
   fast edge-spike max. The edge spike is the large capacitive transient at laser
   onset; the plateau is the DC-coupled sustained level the headstage genuinely
   holds (confirmed DC-coupled via `verify_coupling.py`), typically far smaller. The
   field-map info line lists both per cell (`Max per cell …` and `Sustained
   (plateau) per cell …`), and both the Filter-tab map and the exported heatmap PNGs
   show the plateau as a third line (`plat …`) on each cell when it is available (it
   needs the recording's single-pulse ON/OFF duration).
5. **Spheroid footprint.** Maps biological spike rate onto the array → the
   electrophysiological **footprint**, its centroid/spread, and the **spiking
   reach** (farthest active electrode from the traced soma outline) so you can
   compare against imaging soma location / neurite density.

## Learn templates → classify → filter (typical flow)

1. **Setup tab:** pick the data folder (`.h5` + `Batch.csv`), optionally a
   **cleaned-spikes output folder** (where the Filter tab / batch filtering write
   the MEA-NAP `<stem>_spikes.npz` files that MEA Analyzer then reads), set the
   pitch (default **200 µm**, MCS 60MEA200), and tag each recording's
   **condition**: *Laser only* (bare MEA → piezo template + field baseline),
   *Solar cell x1* (→ solar template + single-cell field), *Solar cells x2* (→
   two-cell field / amplification), *Culture only* (→ biological template), *Solar
   cell + culture* (the sample to clean). Place each recording's solar cell(s) and
   the spheroid by clicking or typing µm coordinates, and (optionally) load + align
   a microscope image.
2. **Peaks tab:** choose the **source recording for each template** (Solar from
   your **one-cell** file, Piezo from laser-only, Biological from culture/sample;
   `(auto)` picks the sensible default), *Learn templates*, then *Detect &
   classify* any recording. Inspect a single electrode's trace with flood markers
   and colour-coded peaks. **Classification** matches each peak's *shape* against
   an extremum-aligned solar/piezo/biological peak template AND checks flood
   timing. The flood is a **pulsed train** — the `flood.seq` fires e.g. **10 pulses
   of 15 ms on / 10 ms off = a 250 ms train** — so each solar event is a *burst* of
   ~10 transients. A peak is an artifact if it falls anywhere in that whole-train
   window (the master rule) **or**, for piezo, matches its peak *shape* better than
   the biological one (a shape backstop just outside the window). By default the
   **artifact-window length is auto-detected per recording from the data** (the
   high-passed flood-triggered average — how long the sharp pulse transients last),
   so it adapts to any pulse count / width, single pulses, and different or mixed
   stimulation durations with no manual setting. Untick *Auto-detect stim duration*
   on the Setup tab to set a fixed train length instead (or **Load flood .seq** to
   read it from the Polygon sequence file).

   *The solar template IS the whole 250 ms train.* Because a solar cell has no single
   characteristic peak — it fires the whole pulsed burst of ~10 transients (the first
   onset largest) — the **solar template is the entire flood-onset-aligned 250 ms
   train**, not a single-peak average. Because the cell is a **dipole** (electrodes
   on two opposite edges), that train is **sign-aligned by correlation** across
   electrodes so the up/down lobes reinforce instead of cancelling. Each electrode's
   waveform is **amplitude-normalised before averaging** — an electrode sitting right
   on the cell contact reads ~19× larger than the rest, and without this it alone
   would define the template — and the mean is **oriented down-first** so it shows the
   physical repeating unit (a deep DOWN peak at each pulse onset, then the relaxation /
   OFF up-peak), verified against the real solar recording. Solar vs piezo is
   therefore decided by **flood-train locking + proximity to the cell** (solar is
   localised at the cell; the piezo transient is array-wide), and the Filter
   subtraction spans the whole train so every pulse's transient is removed. Piezo
   still carries a short peak template (its per-transient shape) plus a train event
   for display.

   *The biological template is built only from detector-confirmed spikes.* To keep
   the bio wavelet and its ±SD error band clean, bio peaks are collected only where
   the **validated bior1.5 detector** confirms a spike (non-flood-locked,
   negative-going — matching the detector's own polarity gate), then averaged with
   shape-based outlier rejection. Noise threshold-crossings the broadband detector
   picks up but bior1.5 rejects never enter the template, so the error shadow no
   longer shows spurious positive peaks.

   *The three biological maps show different things:* the **Peaks & Classify** map
   is the count of peaks the *broadband* detector classified biological — a quick
   classification check, polarity-agnostic. The **Filter** and **Footprint** maps
   are the biological spike **rate** from the *validated bior1.5 wavelet* detector
   run on the artifact-removed trace (the actual biological output); those two use
   the same method and match when run on the same recording+region. Each map also
   shows only its analyzed recording's own placed solar cells, plus the spheroid.
3. **Filter tab:** *Subtract artifacts & re-detect* on a sample, then either
   **Export for MEA Analyzer…** (writes `<stem>_spikes.npz` + `.mat` in the exact
   MEA-NAP format — reusing the lab's own `io.save_spike_file`, so MEA Analyzer
   Stage 2 and the raster viewer read/plot the cleaned biological spikes with no
   conversion), or **Batch filter folder…** to filter *every* `.h5` in the data
   folder at once and write one `<stem>_spikes.npz` per recording into a chosen
   output folder (batched GPU detection on the artifact-removed matrix; per-file
   errors are collected and reported, not fatal). The exported spike times are
   absolute (whole-recording) seconds, exactly as MEA-NAP expects.
4. **Field map / Footprint tabs:** compute from the solar-only / culture files.
   *Clean class wavelets.* The peak wavelets (piezo / bio) and their ±SD error bands
   are built with two outlier rejections before averaging: **amplitude-outlier
   rejection** (a robust MAD ceiling on peak amplitude — a rare electrode artifact can
   be 100–1000× a real spike and, though it barely moves the mean, one such event
   dominates the ±SD band; verified on real data, a single 21 mV artifact drove the
   trough SD from ~9 to 331 µV) and **shape-outlier rejection** (peaks inconsistent
   with the emerging class mean). The **biological** wavelet additionally takes only
   **detector-confirmed, negative-going** peaks (see above), matching the validated
   **bior1.5 detector**'s own polarity gate rather than adding an assumption. The
   **solar** template is the whole 250 ms train (above). All tunable in the project
   analysis (`bio_polarity`, `template_amp_mad_k`, `template_min_corr`,
   `template_keep_frac`).

5. **Templates tab → Calibration report…:** writes, into a `Calibration Report`
   subfolder of your cleaned-spikes output dir, a PNG of each condition's learned
   wavelet (**mean ± 1 SD error band**), a normalised overlay, the **bior1.5
   detection wavelet with its per-scale shapes**, and a `calibration_summary.md`
   listing the detection parameters and per-condition stats.

Save everything (folders, conditions, solar & spheroid placement, pitch) to a
`.mscproj` project file; templates save to their own `.npz`. The app also
remembers, across sessions (in `settings.json`), the **last-used folder** (every
file dialog opens there), the **image µm/px scale**, and the **electrode pitch**,
so you don't re-navigate or re-enter them each launch.

## Run

Double-click **`MEA Solar Cell Analyzer.exe`** in the parent `MEA Analyzer`
folder — a small native Windows launcher (windowless, custom icon) that runs
`MEA Solar Cell Analyzer\app.py` on the shared environment below. No console
window opens; startup problems are written to `startup_stderr.log` and tracebacks
to `error.log` in this folder, and fatal errors also pop a dialog. From a
terminal you can also run
`"%USERPROFILE%\Documents\MEA Analyzer\.venv\Scripts\python.exe" app.py`.
See `LAUNCHER_AND_PACKAGING.md` for how the `.exe` and environment are built.

**Live update:** after a new version is pushed into the folder, click
**↻ Reload scripts (restart)** on the Setup tab — it compile-checks every script
(aborting safely if one has an error), preserves your current project + learned
templates, and relaunches on the new code, so you don't have to close and reopen
manually.

## Environment

Runs on the **shared MEA virtual environment** used by all four MEA programs,
built once per machine by **`setup.bat`** in the `MEA Analyzer` root folder:

    %USERPROFILE%\Documents\MEA Analyzer\.venv

It lives under Documents on purpose — a venv is machine-specific, so keeping it
out of the Nextcloud-synced program folder stops two machines from overwriting
each other's environment. **Nothing auto-installs:** if imports fail (or on a new
machine), run `setup.bat` once. It installs Python 3.14, PyTorch, and everything
in the root `requirements.txt` (numpy, scipy, h5py, PyWavelets, matplotlib,
pillow, pyqtgraph, PyQt6, tqdm, …) — covering all of this app's needs (calibration
plots use matplotlib; the microscope-image overlay uses pillow; spike detection
reuses the MEA-NAP torch/PyWavelets stack).

## Where it finds the loaders

`config.py` auto-discovers the `MEA-NAP` folder (env `MEANAP_DIR` override →
known paths → search near this folder). Set `MEANAP_DIR` to force a location.

## Layout

```
MEA Solar Cell Analyzer\
  app.py            the tabbed GUI + entry point
  electrode_map.py  the clickable electrode-map widget
  pipeline.py       orchestration: loaders -> analysis (the only heavy path)
  loaders.py        the ONLY module importing mea_nap_* (format reuse)
  config.py         puts the MEA-NAP loaders on sys.path
  geometry.py       electrode grid (µm), square solar-cell model, spheroid
  peaks.py          multi-polarity peak detection + snippet features
  templates.py      learned solar/piezo/bio peak-shape templates
  classify.py       flood-locking + template classification
  filtering.py      flood-artifact (STA) subtraction -> pure biological spikes
  fieldmap.py       solar-cell effective field radius / shape (+ modeled field grid)
  heatmap_export.py electrode-heatmap PNG writer (Filter tab -> Filtered Signal)
  bursts.py         single-electrode burst detection (max-ISI) + burst rate
  footprint.py      spheroid footprint + spiking-reach metrics
  project.py        project state + JSON persistence
  selftest.py       headless synthetic-data tests (python selftest.py)
```

`peaks/templates/classify/filtering/fieldmap/footprint/geometry` are loader-free
and validated by `selftest.py`; only `loaders`/`pipeline`/`app` touch real data.
