# MEA-NAP Python pipeline — project memory

**Authoritative working memory for this project.** Reconstructed 2026-07-16 after
the original `/areas/mea-analysis-pipeline.md` (Cowork app-managed local memory)
did not reach a cloud session. This copy lives beside the scripts so it travels
with the project; keep it updated.

Owner: Jerry (Yikan). Machine: Windows desktop "fett", RTX 4090. Scripts +
vendored `meanap/` port live in
`C:\Users\Yikan\Nextcloud2\MEA Data\New MEA 06172026\Scripts\MEA-NAP`.

---

## Phase Lock-In stage (from sweep_lockin.py) — 2026-07-18

NEW module `mea_nap_phase_lockin.py` — folds the user's standalone `sweep_lockin.py`
into the pipeline as a 4th optional stage (do4), GUI tab "Phase Lock-In",
`PHASE_LOCKIN["enable"]`. This is the method purpose-built for the sweep-MEMORY
question: it measures whether the network locks to the SLOW periodic laser sweep
(200-3200 um/s), unlike the fast intra-burst LAT/spkit-EGM CV (10^4-10^5 um/s).

Algorithm (spikes, not voltage): per electrode, phase-locking resultant
`z_e = mean(exp(i*2pi*frac((t-t0)/T)))`; |z|=lock strength, arg z=preferred phase.
Propagation = spatial gradient of preferred phase via wrap-immune plane-wave
matched filter over wavevector k: dir=atan2(ky,kx), speed=2pi/(|k|*T)=wavelength/
period, coherence in [0,1]. refine_period searches around a seed; peak_ratio +
coherence_floor flag low confidence; optional phase-rotation permutation p-value.

Integration specifics:
- spikes in SECONDS from io.iter_batch_units (window-relative); geometry via
  orig.mcs_coord (x=col,y=9-row)*pitch — same frame as spectral_propagation so
  135deg=BR->TL alignment is consistent.
- period SEED: frames_per_sweep / trigger_hz from de.sweep_metrics(read_digital_events)
  (160 frames from Sept2021_SweepBRtoTL.seq; e.g. 160/80Hz=2.0s). Fallbacks:
  period_s override, else population-rate autocorrelation. STIM region computed
  first so its refined period seeds pre/post (carry-over prior).
- driver run_phase_lockin(): analyze-group filtered (io.filter_rows_by_group),
  per stem one row of pre/stim/post phase maps -> `Phase Lock/<stem>/phase_lockin.png`
  + `Phase Lock/phase_lockin_summary.csv` (dir/speed/coh/period/peak_ratio/alignment/
  p_value/low_confidence). Output root = phase_lock_root(PLOT_DIR)=PLOT_DIR/"Phase Lock".
- GUI: import phase_lockin; PLK tab in build_schema; PARAM_HELP entries; do4 in
  _worker (after SpKit) + _on_run nothing-to-run/analysis-group checks. Added a new
  param type "floatn" (None-or-float) to _fmt/_parse for period_s override.
Self-test (synthetic BR->TL, T=2s): stim dir=+135 align=+1.00 PR=4.0 T=2.00; pre
(background only) coh~0.02 PR~1.1 correctly flagged no-lock. Verified GUI tab
renders (17th tab) + do4 enable applies.

## Phase Lock-In — how it works (plain language) — 2026-07-20

Plain-language companion to the technical note above; useful for methods sections
and explaining the stage to collaborators.

The setup: the laser bar sweeps across the array in the same direction over and
over at a steady rhythm — like a lighthouse beam retracing the same arc every
couple of seconds. Because it repeats on a fixed period, every sweep can be
treated as one repetition of the same "lap."

1. Find the rhythm (the period). The period comes straight from the laser trigger
   pulses (frames per sweep ÷ trigger rate), then gets fine-tuned by trying nearby
   values and keeping whichever makes the wave below cleanest.
2. Fold every spike onto one cycle. For each spike, ask "where in the sweep cycle
   did this happen?" — start, middle, end. This is like a long-exposure photo whose
   shutter resets once per lap, so all laps pile onto one. Random firing smears
   evenly around the cycle; reliable firing at the same point piles up there.
3. Turn each electrode into a clock hand. Picture the sweep cycle as a clock face
   and each spike as a mark on it. Average the marks for one electrode → one arrow
   (standard circular-statistics trick). Its LENGTH = how tightly the electrode
   locks to the sweep (shown as colour = "lock strength" in the plots); its
   DIRECTION = when in the sweep that electrode prefers to fire.
4. Look for a wave marching across the array. If the network follows the sweep, the
   electrodes hit first should prefer an early moment and those hit last a later
   moment — so the preferred time changes smoothly across space, like a stadium wave
   or falling dominoes. Fitting a tilted plane ("plane wave") to all the clock-hand
   directions gives the wave's DIRECTION and SPEED.
5. Confidence + significance. COHERENCE (0–1) = how well one clean wave explains
   everything (high = trustworthy). ALIGNMENT = cos(direction − sweep axis): +1 =
   travels exactly along BR→TL, 0 = sideways, −1 = backwards. A shuffle
   (permutation) test scrambles each electrode's timing many times and reports how
   often chance beats the real wave → the p-value.

Why it fits this experiment: folding on the period is what lets a SLOW, repeating
drive show up clearly, and the space-varying preferred time is what carries the
direction. The payoff is pre/stim/post: a strong, sweep-aligned wave in stim = the
network locked onto the laser; a leftover wave at the same rhythm in post (laser
off) = the "memory" signature being tested.

Not a single published method — a purpose-built combination of standard pieces:
- Per-electrode phase-locking resultant / phase-locking value: Lachaux, Rodriguez,
  Martinerie & Varela (1999), *Measuring phase synchrony in brain signals*, Human
  Brain Mapping 8(4):194–208. https://pubmed.ncbi.nlm.nih.gov/10619414/
- Circular-statistics conventions (resultant vector, mean phase, resultant length):
  Berens (2009), *CircStat: A MATLAB Toolbox for Circular Statistics*, J. Stat.
  Software 31(10). https://www.jstatsoft.org/v31/i10
- Plane-wave / phase-gradient recovery of a travelling wave from array phases:
  Rubino, Robbins & Hatsopoulos (2006), *Propagating waves mediate information
  transfer in the motor cortex*, Nat. Neurosci. 9:1549–1557.
  https://www.nature.com/articles/nn1802 ; Muller et al. (2014), *The
  stimulus-evoked population response ... is a propagating wave*, Nat. Commun.
  5:3675, https://www.nature.com/articles/ncomms4675 ; Muller, Chavane, Reynolds &
  Sejnowski (2018), *Cortical travelling waves*, Nat. Rev. Neurosci. 19:255–268,
  https://www.nature.com/articles/nrn.2018.20
- Permutation/surrogate significance: Maris & Oostenveld (2007), *Nonparametric
  statistical testing of EEG- and MEG-data*, J. Neurosci. Methods 164(1):177–190.
  https://pubmed.ncbi.nlm.nih.gov/17517438/

## Output layout: one clean tree under PLOT_DIR — 2026-07-18

BUG FIXED: `orig.output_root` and `spkit_egm.spkit_root` used
`dirname(os.path.normpath(plot_dir))` = the PARENT of the GUI "Plot output
folder", so MEA-NAP-Original and SpKit output landed OUTSIDE the selected folder
(appeared "missing"). Both now return a subfolder UNDER plot_dir.

Everything the pipeline writes now lives under the single GUI-selected PLOT_DIR:
  Recordings/<stem>/            per-recording Stage-2 plots (activity, spectral,
                               propagation, cv_field) + lag<X>ms/  — was PLOT_DIR/<stem>/
  Comparisons/                 Stage-2c group/age comparisons — COMPARE_PARAMS
                               ["out_subdir"] renamed "comparisons"->"Comparisons"
  MEA-NAP Original Output/     MEA-NAP original-style tree (orig.output_root)
  SpKit/<stem>/<region>/       SpKit continuous-EGM CV (spkit_root)
  Tables/                      activity_summary / node_metrics / network_metrics /
                               sweep_speed_summary CSVs (were at PLOT_DIR root; `P`
                               in run_stage2 now = PLOT_DIR/Tables)
  <QC dir>/                    spike-detection QC images (io QC_PARAMS save_dir_name;
                               already under PLOT_DIR, unchanged)
Edits: activity base_dir (line ~1942) -> join(PLOT_DIR,"Recordings",stem);
out_subdir -> "Comparisons"; CSV `P` -> join(PLOT_DIR,"Tables")+makedirs.
NOTE: SpKit still only RUNS when the GUI "SpKit EGM" tab enable is on (do3);
MEA-NAP original runs whenever MEANAP_ORIGINAL_PLOTS=True (default).

## What this is

A personal Python reimplementation of the SAND-Lab **MEA-NAP** pipeline for MCS
HDF5 microelectrode-array recordings from a laser-sweep stimulation experiment.
Goal is parity with MEA-NAP's own outputs, achieved by calling the lab's own
validated Python port (the vendored `meanap/` package) directly for the network
math, with the heavy CWT/STTC moved onto the GPU for speed.

## Files (all must sit in one flat folder, beside `meanap/`)

- **`mea_nap_io.py`** — Stage 1. MCS HDF5 reader (whole-file load to RAM),
  6-column `Batch.csv` parser, resumable per-H5 spike detection (atomic `.npz` +
  MEA-NAP `.mat`, `_progress.json` manifest, retry-once-then-skip); also the raw
  voltage reader `MCSFile` (get_channel_window / get_matrix) reused by the SpKit
  pipeline; auto-grounding
  of dead/noisy electrodes (QUALITY_PARAMS, cheap noise backfill via
  `ensure_quality`), per-row window slicing with clamp-and-warn, and spike-detection
  QC snippet plots. Detection is ONCE per whole recording; each Batch row slices
  its window from the cached spikes so pre/stim/post share one detection basis.
- **`mea_nap_wavelet_detect.py`** — MEA-NAP-exact `bior1.5` wavelet detection with
  GPU-batched CWT (torch). Corrected detector: Nenadic–Burdick Bayesian threshold
  (`DTh = Mj/2 + Sigmaj^2/Mj*(L*Lmax + log(PN/PS))`, Lmax=36.7368), negated CWT so
  only negative-going APs count, Sigma from stride-subsampled coefficients.
  `_selftest()` validates torch path == lab CPU reference (identical spike indices).
- **`mea_nap_activity.py`** — Stage 2. Activity metrics + plots, STTC functional
  connectivity, multi-lag network + node metrics via the port's
  `compute_network_metrics` (= `ExtractNetMet.m`), node cartography, and Stage-2c
  group/age/region comparisons.
- **`mea_nap_progress.py`** — threaded Tk progress window, one horizontal bar per
  sample. Tk lives in its own daemon thread with its own mainloop; the pipeline
  thread only drops messages on a queue (fixes the "(Not Responding)" freeze).
  Fails soft to a no-op if Tk is unavailable.
- **`mea_nap_spectral_propagation.py`** — Stage-2 add-on (added 2026-07-17). PSD of
  firing/bursting per region + LAT-plane-fit signal-propagation (direction/CV) over
  network bursts and sliding windows. Import-safe (numpy/scipy/matplotlib, imports
  `mea_nap_original_plots`). Wired into `run_stage2` after `make_activity_plots_multi`.
  Now also emits a local conduction-velocity flow-field (`cv_field.png`, CVFIELD_PARAMS).
- **`mea_nap_spkit_egm.py`** — OPTIONAL spkit continuous-EGM conduction-velocity
  pipeline (added 2026-07-17). Runs the real `spkit.mea` functions on RAW MCS voltage,
  output to `SpKit/<stem>/<region>/`. spkit imported lazily. Enabled from the GUI
  "SpKit EGM" tab; runs as a 3rd phase after Stage 2. Needs `pip install spkit`.
- **`mea_nap_digital_events.py`** — reads the laser ON/OFF timing from the MCS Digital
  Event Recorder (added 2026-07-17; numpy+h5py only). Auto-detects Digital AnalogStream
  or EventStream. Runnable as an inspector: `python mea_nap_digital_events.py <h5>`.
- **`meanap/`** — the SAND lab's own validated Python port (github.com/SAND-Lab/
  MEA-NAP). Called directly for parity; MUST sit beside the scripts.
- **`sand_spike_detection_reference.py`** — reference copy of the port's spike
  detection (WATERS), for comparison.

## Config that matters (paths on "fett")

- `CONFIG["DATA_DIR"]` = `...\New MEA 06172026\H5` (flat: .h5 files + Batch.csv)
- `CONFIG["SPIKE_DIR"]` = `...\Output\SpikeData`
- `CONFIG["PLOT_DIR"]` = `...\Output\Plots`
- `CONFIG["MIN_WINDOW_S"]` = 1.0 — batch rows whose clamped window is shorter than
  this are SKIPPED (guards the 420 s-window-on-300 s-Control problem).
- `CONFIG["REFERENCE_ELECTRODE"]` = ["Ref", "15"] (always dropped).

## Detector params hash (verify on every run)

`mea_nap_io._params_hash()` = md5 of `det.DETECT_PARAMS`. Terminal should show:
- **`3046fc7b6b` = correct/new detector** (current code — CONFIRMED 2026-07-16).
- `4e736f3638` = old broken detector (must NOT appear).

## Stage-2 performance — RESOLVED 2026-07-16

**Symptom:** Stage 2 much slower after adopting the MEA-NAP port from GitHub.

**Root cause:** `analyze_network` routes through the port's
`compute_network_metrics` (ExtractNetMet), which runs heavy null-model
normalizations the old custom `compute_graph` path never did — small-worldness
(`latmio_und_v2` 10000 + `randmio_und_v2` 5000 rewires), normalized participation
coefficient (`participation_coef_norm`, n_iter=100 randomizations), consensus
modularity (rep_num=50 Louvain reps) — ONCE per (region × lag), i.e. ~9 calls/file.
The port's swap loops use **numba** when importable and silently fall back to a
pure-NumPy path ~10–30× slower otherwise. numba was NOT installed → the slowdown.

**Benchmarked** `compute_network_metrics` on 40–60-node nets:
no-numba 16–48 s/call vs with-numba 1.3–2.5 s/call (**12–20×**). Enabling numba
changes ONLY the normalized PC (different RNG realization, within its stochastic
range); SW/SWw/Q/Ci/Z/CC/BC/Eloc/degree/strength are identical.

**Fixes applied (2026-07-16), verified parallel==serial bit-identical:**
1. `pip install numba` on "fett" (DONE by Jerry). Port auto-detects it.
2. Added `PERF_PARAMS` to `mea_nap_activity.py`:
   - `require_numba`: "warn" (default, loud banner if fallback would run — analogue
     of `strict_gpu`) | "raise" | "off". Reads port's real `null_models._HAVE_NUMBA`.
   - `parallel_network_metrics`: True. (region × lag) `compute_network_metrics`
     calls fan out over a `ProcessPoolExecutor` (created once per batch; numba JIT
     paid once, disk-cached). STTC/surrogates stay serial on the single GPU. Auto
     falls back to serial if pool can't start or `sttc_backend != "gpu"`; per-file
     rows are staged locally and only committed on success (no double-count).
   - `max_workers`: None → CPU count − 1.
   - `analyze_network` split into `_network_connectivity` (GPU/STTC, serial),
     top-level `_metrics_worker` (picklable, reseeds default_rng(seed) so results
     match serial and are order-independent), `_finalize_network` (active-node
     subset stitch). `analyze_network` itself unchanged for the serial path.
   File is now 95,966 bytes / 1987 lines (was 86,327).

**CAVEAT for Jerry:** now that numba is active, the `PC` column in
`node_metrics.csv` (and borderline node-cartography roles) shifts slightly vs old
NumPy-fallback runs — stochastic, not a regression.

## STTC / surrogate thresholding (Stage 2, unchanged this session)

- `PARITY_PARAMS["sttc_backend"]` = "gpu" (binned-matmul STTC + circular-shift
  surrogates on GPU). `PROB_PARAMS`: n_surrogates=200, tail=0.05, sig=0.95,
  bins_per_dt=25, `use_gpu`=True, `strict_gpu`=True.
- `strict_gpu`=True → a GPU STTC failure RAISES (names the cause) instead of the
  silent ~1000× exact-CPU fallback. Remaining silent hole: if `use_gpu` were ever
  set False, `probabilistic_threshold` goes straight to exact CPU with no guard —
  keep both True.
- VRAM math (verified): on this dataset (≤~420 s windows, 60-ch MEA) the GPU STTC
  path peaks at ~150–205 MB; a full 1-hour window ~1.7 GB. On a 24 GB 4090 a true
  CUDA OOM is essentially impossible, so if `strict_gpu` raises it is almost
  certainly NOT memory — read the exception; dropping `bins_per_dt` 25→10 would
  not be the fix.

## Open items (Jerry's side)

1. Stage-2 slowdown — RESOLVED (numba + parallel above).
2. Detector hash — CONFIRMED `3046fc7b6b`.
3. Batch.csv Control rows specifying 420 s windows on ~300 s recordings: currently
   SKIPPED safely by MIN_WINDOW_S. Real fix is a DOMAIN decision — what window did
   the Controls actually intend? (A "post" window starting at 420 s on a 300 s file
   has nothing to slice.) NOT yet decided. Batch.csv lives in DATA_DIR (the `H5`
   folder), which is NOT the currently connected folder (only `Scripts\MEA-NAP` is).

## MEA-NAP Original Output (faithful MATLAB-figure recreation) — added 2026-07-16

- **`mea_nap_original_plots.py`** (new, beside the other scripts) reproduces the
  SAND-Lab MEA-NAP MATLAB figure set (the vendored `meanap/` port has NO plotting)
  into a parallel tree `<Output>/MEA-NAP Original Output/`, alongside `Plots/`
  (which is untouched). Import-safe (matplotlib + scipy only; no torch/pywt).
- Faithful to the MATLAB sources in `MEA-NAP-main/Functions/`: embedded MATLAB
  **parula** colormap, MCS60 electrode coords from labels, `aesthetics.m`
  (white bg, ticks out, Arial), 300-dpi PNG (`pipelineSaveFig.m`).
- Figure set built + visually verified vs the .m sources:
  - **2A** (`2_NeuronalActivity/2A_IndividualNeuronalAnalysis/<stem>_<region>/`):
    `1_FiringRateByElectrode` (HalfViolinPlot.m), `2_Heatmap` (electrodeHeatMaps.m,
    grounded electrodes drawn grey [0.8,0.8,0.8] — full MCS60 grid, non-analysed
    positions greyed), `3_Raster` (rasterPlot.m, per-second FR matrix, parula@85%,
    dual recording/batch scaling).
  - **4A** (`4_NetworkActivity/4A_IndividualNetworkAnalysis/<group>/<stem>_<region>/<lag>mslag/`):
    `1_adjM<lag>msConnectivityStats` (plotConnectivityProperties.m), `2_MEA_NetworkPlot`
    (plain, node size=degree, cyan) + `3..8_MEA_NetworkPlot_<metric>` colour-mapped
    (ND/NS/BC/PC/Z/Eloc, StandardisedNetworkPlot.m), `9_adjM<lag>msNodeCartography`
    (NodeCartography.m scatter + role-region diagram, 6 Guimera-Amaral colours).
  - **4B** group comparisons: half-violin by group/age (plotHalfViolinByX.m) for
    network + node metrics per lag, `5_GraphMetricsByLag`, `6_NodeCartographyByLag`,
    `7_DensityLandscape` (PC-vs-z hexbin). **2B**: activity half-violins by group/age.
- Wire-in: `mea_nap_activity.py` top-level flag `MEANAP_ORIGINAL_PLOTS = True`;
  `run_stage2` computes the dataset-wide max FR (batch-scaled panels), renders per
  recording in the activity + network loops, and the comparisons after the CSVs.
  Every render call is wrapped in try/except (`[MEA-NAP 2A/4A/4B warn]`) so it can
  never break the main pipeline. Volume is large (per recording x region x lag:
  connectivity-stats + plain + 6 metric network plots + cartography) — full count
  kept at Jerry's request. Toggle the flag off to skip.
- **Region-aware group comparisons (updated 2026-07-16).** Batch.csv has two
  columns MEA-NAP lacks (window + region). A "sample" = one filename stem; the
  `region` column (pre/stim/post) marks windows WITHIN that same sample. So the
  2B/4B comparisons treat `region` as the compared factor (`REGION_ORDER`,
  `REGION_COLORS` in the module; `fig_region_by_x`):
    * NodeByAge / RecordingsByAge : x=age(day), one half-violin per region, one
      subplot per group -> compare pre/stim/post across days, per group.
    * NodeByGroup / RecordingsByGroup : x=group, one half-violin per region, pooled
      across samples/days -> compare pre/stim/post across samples by group.
    * GraphMetricsByLag : one line per region across lags.
  (The pipeline's own Plots/2c still keep regions separate their own way.)
- Half-violin spine fix: KDE evaluated ~3 bandwidths past the data + fill_betweenx
  against a vertical spine at x=pos+0.1 (a grid clipped to [min,max] left a tilted
  closing edge on sparse panels).
- All adjustable params carry an adjustment range in their annotation
  (MEANAP_PLOT_PARAMS, REGION_*, _CART, and PERF_PARAMS in mea_nap_activity).
- Deps for this module on "fett": scipy (KDE for half-violins) + matplotlib.

## Overnight launcher GUI — added 2026-07-16

- **`mea_nap_gui.py`** (new, beside the other scripts): one Tk window that runs
  Stage 1 (`io.run_stage1`) then Stage 2 (`activity.run_stage2`) SEQUENTIALLY in a
  background thread, so Stage 2 auto-starts when Stage 1 finishes — start it and
  leave it overnight. Run with the venv python:
  `& "...\.venv\Scripts\python.exe" "...\Scripts\MEA-NAP\mea_nap_gui.py"`.
- Inputs: the 3 paths (Data/Spike/Plot) + Browse buttons and Batch CSV name, plus
  a **tabbed parameter panel** (`build_schema()`) exposing EVERY tunable across all
  modules — I/O, Detection, Quality (auto-ground), QC, Firing, Bursts, Net-bursts,
  STTC, Surrogate, Network/parity, Cartography, Perf, Plots — each row showing its
  adjustment range. Specs edit the live module dicts IN PLACE (a couple of scalars,
  e.g. MEANAP_ORIGINAL_PLOTS / MAX_ATTEMPTS, via getter/setter). QC mode defaults
  to "save" (no popups, right for unattended). Stage 1 / Stage 2 toggles. Settings
  (paths + every param) persist to `mea_nap_gui_settings.json`.
- A large overall-progress bar (percentage = mean of per-sample fracs) sits above
  the per-sample bars. DPI-aware on Windows (`SetProcessDpiAwareness` before Tk()
  + tk scaling) for crisp rendering.
- Progress bars are EMBEDDED in the window (no popup): the GUI monkeypatches
  `mea_nap_progress.make` to an `EmbeddedProgress` object that enqueues
  start/set/done/skipped/failed onto a queue; the Tk main thread drains it and
  draws one bar per sample (same colours as BarProgress). The pipeline runs in a
  worker thread and touches only the queue (Tk stays on the main thread). A Log
  panel captures the pipeline's stdout (tqdm '\r' lines filtered); stderr is
  swallowed so tqdm/pythonw can't crash the worker.
- Verified headless (Xvfb + python3.12) driving a stubbed Stage1→Stage2: bars
  fill, phase relabels between stages, log shows the handoff + completion.
- Needs tkinter (bundled with the Windows Python, no install). Sets matplotlib
  Agg before importing the pipeline so worker-thread figures never touch Tk.
- ProcessPool note: GUI is __main__; on Windows-spawn the children import it as
  __mp_main__ and the `if __name__=="__main__"` guard stops them launching Tk.

## Plot + GUI refinements — added 2026-07-16 (round 2)

- Heatmap/raster second panel now scales to "entire group" (not dataset):
  `group_max_firing_rate()` -> per-group maxFR; wiring passes the unit's group max.
- Half-violin point/mean-dot sizes are params (`violin_point_size`,
  `violin_mean_size`) and scale with violin width (grouped region panels get
  smaller dots) — fixes the oversized points in 4B/2B comparisons.
- Colour schemes (MEANAP_PLOT_PARAMS + CMAP_OPTIONS/PALETTE_OPTIONS): `raster_cmap`,
  `heatmap_cmap`, `network_node_cmap`, `violin_palette`; `_cmap()`/`_cmap_trunc()`
  resolvers ('parula' = embedded map). Exposed in the GUI "Colours" tab.
- Empty-folder fix: `3_EdgeThresholdingCheck` was never written -> added
  `fig_edge_threshold_check` (real-vs-surrogate STTC, significance, retention) into
  `3_EdgeThresholdingCheck/<stem>_<region>/`. 2B `1_NodeByGroup`/`2_NodeByAge` were
  empty -> now filled from a per-electrode activity dataframe (`electrode_activity_row`
  in the wiring -> `electrode_df` -> `ELECTRODE_METRIC_LABELS`). `render_group_comparisons`
  now isolates each sub-section (`_safe`) so one failure can't blank the rest.
- Network nodes smaller + configurable (`node_min_size`/`node_max_size`); the plot
  now draws an EDGE-WEIGHT legend (3 line widths) beside the node-degree legend,
  and edge line width AND shade scale with STTC weight (`edge_max_width`).
- `PLOT_ENABLE` registry (per-figure on/off) gated in every render driver; GUI
  "Plot outputs" tab = a checkbox per figure + a "Check all" master.
- GUI "Preview" tab: pick a figure, "Update preview" renders it from synthetic data
  with the current colour/size params and shows it inline (Tk PhotoImage).
- Every parameter row and plot-output row has an "ⓘ" hover tooltip; explanations
  (PARAM_HELP / PLOT_HELP in mea_nap_gui.py) are sourced from the MEA-NAP paper
  (Sit et al. 2024, Cell Reports Methods) + docs.

## PSD + signal-propagation — added 2026-07-17

Jerry's request: (1) a PSD of firing/bursting **separately per region** to see if
light stimulation shifted the spontaneous firing/bursting frequency; (2) signal-
propagation detection — the stim is a laser **sweep bottom-right → top-left**, so he
wants to see an STDP-like effect where the network "memorises" the sweep direction
or speed. Ref: CinC 2025 paper (CinC2025-300.pdf) + spkit `bsp_mea`, "Figure 3".

- **New module `mea_nap_spectral_propagation.py`** (self-test passes). Config dicts:
  `SPECTRAL_PARAMS` (psd_bin_s 0.05, psd_fmax_hz 10, psd_welch_nperseg 1024, psd_logy),
  `PROP_PARAMS` (electrode_pitch_um 200, min_electrodes 6, min_fit_r2 0.30, window_ms
  100, window_step_ms 50, window_min_active_frac 0.30, sweep_angle_deg 135, max_maps 1),
  `SP_ENABLE` (psd_firing_bursting / propagation_bursts / propagation_windows).
- **PSD** (`fig_psd` -> `psd_firing_bursting.png`): 2 panels. Left = Welch PSD of the
  population firing rate (all spikes binned at psd_bin_s); right = Welch PSD of the
  network-burst envelope (1 during a net burst). pre/stim/post overlaid; peak-freq
  per region in the title. `_welch` caps nperseg to signal length, skips flat/short.
- **Propagation** method = **spike-based Local Activation Time plane fit** (native, NOT
  spkit — spkit's `bsp_mea` expects continuous cardiac EGM, not spikes). Per event
  take each electrode's first-spike LAT, least-squares fit a plane LAT≈ax+by+c;
  θ = atan2(b,a) (propagation direction), CV = 1/sqrt(a²+b²) mm/s. Reject if
  <min_electrodes or R²<min_fit_r2. Events = network bursts (`propagation_bursts.png`)
  AND fixed sliding windows (`propagation_windows.png`) — Jerry chose **both**.
- Figure = 3 rows × nregions: (top) example activation-time map w/ mean-direction
  arrow, (mid) direction rose + resultant vector, (bottom) CV histogram. **Sweep axis
  135° BR→TL drawn as red dashed line; per-region alignment score = mean cos(θ−sweep)**
  (+1 = along sweep, −1 = against). This is the STDP-direction readout.
- Output → `Plots/<stem>/` (root Plots tree, NOT the MEA-NAP Original folder), guarded
  by try/except in `run_stage2`.
- **GUI**: new "Spectral / Propagation" params tab (all SPECTRAL_PARAMS + PROP_PARAMS
  with ranges + ⓘ tooltips); the 3 SP figures added to the "Plot outputs" checkbox
  tab (under a separator, included in "Check all" and SP_ENABLE write-back).
- Self-test (`python mea_nap_spectral_propagation.py`): synthetic pre=40°/CV150,
  stim=135°/CV180, post=135°/CV260 recovered as pre≈40°(align −0.08), stim≈136°,
  post≈138° (align +0.99) — direction/velocity/alignment all correct.

## GUI env box + aesthetics — added 2026-07-17

- **Environment status box** (top-right of the window, beside Paths): shows whether
  the GPU is enabled and whether every required package is present IN THE RUNNING
  INTERPRETER (so it reflects the venv the GUI was launched with). `_probe_environment()`
  imports each of `_REQUIRED_PKGS` (numpy, scipy, matplotlib, PyWavelets, h5py, pandas,
  networkx, torch, numba[optional], meanap) + checks `torch.cuda.is_available()` →
  GPU line shows device name + CUDA version when enabled. Green/amber/red dots
  (ok/optional-missing/required-missing); versions on hover. Banner tag summarises
  "✓ environment ready" or "packages N/M".
- Probe runs on a **daemon thread** (torch import can be slow) that ONLY sets
  `self._env_result`; a main-thread `_env_poll` (root.after) renders it. IMPORTANT:
  never call Tk widget methods from the worker thread (root.after from a non-main
  thread silently fails — that was the first-cut bug).
- **Theme** (`_apply_theme`, PALETTE dict): clam base restyled to a cohesive light
  indigo/blue scheme — deep-indigo banner header, card-style panels, padded notebook
  tabs, accent "Run" button, colored progress bar. Tab padding kept small ((6,5)/font 9)
  so all 17 tabs fit ~1160px width.
- **Animation**: window fade-in via `-alpha` 0→1 (`_fade`), a pulsing dot while the
  probe runs (`_pulse`), and a staggered reveal of the package chips (`_reveal_chip`).
  All best-effort/try-except so they degrade to no-ops if unsupported.
- Verified live under Xvfb + python3.12 (screenshot): env box renders GPU + 2-col
  package grid, red dot on a missing package, tabs readable.

## Local CV field + spkit continuous-EGM pipeline — added 2026-07-17

Jerry asked whether the native LAT-plane-fit propagation does "as good a job" as
spkit. Answer given: for spike data ours is the appropriate method and same core
gradient math, but spkit's `compute_cv` yields a LOCAL velocity FIELD (heterogeneity)
that our single global fit averages away. He then asked for BOTH: (1) a spkit-style
local CV field in Plots/, and (2) spkit's ACTUAL continuous-EGM pipeline as an
optional GUI tab writing to a new `SpKit/` folder.

### Part 1 — local CV flow-field (native, no spkit dep) — in mea_nap_spectral_propagation.py
- New `CVFIELD_PARAMS` (cv_field_mode bursts|windows, cv_neighbor_factor 1.6,
  cv_min_neighbors 4, cv_local_min_r2 0.20) + `SP_ENABLE["cv_field"]`.
- `_region_activation_map` averages each electrode's relative first-spike LAT across
  events -> one activation surface per region. `_local_cv_field` fits a plane in each
  electrode's neighbourhood (radius = factor×pitch) -> per-electrode velocity vector
  (θ=atan2(b,a), CV=1/|∇T| mm/s), keeping only local R²≥threshold. `fig_cv_field`
  draws a quiver flow-map per region coloured by local CV, sweep axis marked, median
  CV + mean dir + alignment annotated -> `Plots/<stem>/cv_field.png`.
- Kept spkit-FREE on purpose so Stage 2 never depends on spkit being installed.
- Self-test recovers pre 36°/align−0.15, stim 135°/+0.99, post 140°/+0.97 (faster).

### Part 2 — spkit continuous-EGM CV pipeline — NEW mea_nap_spkit_egm.py
- Runs the REAL spkit `bsp_mea` functions on RAW MCS voltage (not our spikes), as an
  independent method-diverse cross-check. Optional; OFF by default.
- `SPKIT_EGM` params (enable, event-detection band/threshold/ISI/min-max events,
  egm_pre/post_ms, at_method min_dvdt, at_gradient fdiff, eD_um 200, cv_thr_cm_s 100,
  sweep_angle_deg 135). spkit imported LAZILY inside functions so importing the module
  (e.g. from the GUI) never fails if spkit is absent.
- Per Batch row: `io.MCSFile` reads the window's raw voltage -> `_detect_events`
  (population envelope peaks) -> `_representative_egm` (average per-channel segments
  aligned on event onsets) -> `spkit.mea.activation_time_loc` -> `arrange_mea_grid` ->
  `spkit.fill_nans_2d` -> `spkit.mea.compute_cv` (returns CV_df, CV_thetas, CV0, CV;
  CV is in cm/s — ×10 for mm/s). Verified against the real spkit 0.0.9.7 API here.
- Output: `SpKit/<stem>/<region>/spkit_cv.png` (activation-time map | CV magnitude map
  | CV vector field) + `CV_df.csv`; `SpKit/spkit_cv_summary.csv` + per-stem
  pre/stim/post CV+alignment bar figure. `spkit_root(plot_dir)` = parent(Plots)/SpKit.
- Self-test on synthetic continuous EGM waves recovers pre 40°/CV150, stim 134°/CV181,
  post 135°/CV261 mm/s, align −0.08 / +1.00 / +1.00 — using the genuine spkit chain.

### GUI wiring (mea_nap_gui.py)
- New "SpKit EGM" tab (enable checkbox + all SPKIT_EGM params w/ ranges + ⓘ tooltips).
- CVFIELD_PARAMS added to the "Spectral / Propagation" tab; `cv_field` toggle auto-
  appears in "Plot outputs" (iterates SP_ENABLE). Env box lists `spkit (EGM)` as an
  OPTIONAL package (amber if missing). Worker runs a 3rd phase after Stage 2 when
  `SPKIT_EGM["enable"]`; `_worker(do1,do2,do3)`; a SpKit-only run is allowed (the
  "nothing to run" guard now also checks the enable flag). Verified live under Xvfb.
- NOTE: needs `spkit` installed in the venv (`python -m pip install spkit`); it's the
  only genuinely new dependency and only for this optional tab.

## Laser timing from the MCS Digital Event Recorder — added 2026-07-17

Jerry: the recordings contain a "Digital Event Recorder" that marks laser-power ON as
square waves — the REAL sweep timing. Use it instead of heuristic event detection.

### NEW mea_nap_digital_events.py  (import-safe: numpy + h5py only)
- `read_digital_events(filepath, stream_index, params)` auto-detects and reads laser
  ON/OFF times, handling BOTH MCS storage forms:
    (A) a Digital *AnalogStream* (DataSubType='Digital', usually 1 ch) — threshold the
        square wave, rising edges = ON. `digital_bit` (None=any nonzero, else 0-15).
    (B) an *EventStream* — EventEntity_0/1 timestamp datasets (µs) = ON/OFF.
  Returns {on_s, off_s, intervals, source, fs, detail}. `events_in_window(ev,t0,t1)`
  gives window-relative ON times. `survey_streams()` dumps the HDF5 stream layout.
- `DIGITAL_PARAMS`: digital_bit, min_pulse_ms (debounce), analog_label_hint ('digital'),
  invert, use_in_stage2. **RUN AS INSPECTOR:** `python mea_nap_digital_events.py <h5>`
  prints every stream + the detected ON times — standalone (needs only numpy+h5py), so
  Jerry can run it from anywhere to confirm the reader finds his laser channel.
- Validated here on synthetic MCS files for BOTH the analog-digital and event-stream
  paths (recovers exact ON/OFF times), and end-to-end through the spkit pipeline
  (digital-locked run recovers CV 200 mm/s, dir 135°, align 1.00).
- ASSUMPTION TO CONFIRM on real files: which stream/bit is the laser. If the inspector
  shows a different label/bit, set analog_label_hint / digital_bit accordingly.

### Integration (digital events become the preferred, stimulus-locked event source)
- **spkit EGM** (mea_nap_spkit_egm.py): `SPKIT_EGM["use_digital_events"]` (+digital_lag_ms).
  Driver reads laser events per file; `run_region(..., onsets_s=)` uses laser ON times
  when the window has ≥evt_min_events, else falls back to population detection. Summary
  gains `event_source` ('laser/digital' | 'population').
- **Stage-2 spike propagation** (mea_nap_spectral_propagation.py): new event mode
  'digital' -> `propagation_digital.png` (windows anchored on laser ON; only regions WITH
  laser, i.e. stim, are drawn — pre/post show "no coherent events"). `cv_field_mode` now
  also accepts 'digital'. `render_spectral_propagation(..., laser_by_region=)`.
  New SP_ENABLE key `propagation_digital` (auto-skips if no laser events).
- **Wiring**: `io.window_unit` now returns `"win":(t0,t1)` so Stage 2 can convert absolute
  laser times to window-relative. `activity.run_stage2` reads digital events per stem
  (gated by DIGITAL_PARAMS.use_in_stage2) and passes `laser_by_region` to the renderer.
- **GUI**: digital params on the Spectral/Propagation tab (use_in_stage2, digital_bit,
  min_pulse_ms, invert, hint) + SpKit tab (use_digital_events, digital_lag_ms);
  `propagation_digital` toggle in Plot outputs; cv_field_mode enum gains 'digital'.
- NOTE: laser events only exist during the stim window, so the digital figures
  characterise the EVOKED wave; pre/post remain spontaneous (bursts/windows). The STDP
  test = compare stim's laser-locked direction/speed vs post's spontaneous bias.

## Polygon .seq decode + sweep-speed-from-triggers — added 2026-07-17

Jerry: the MCS Digital Event Recorder only logs the TRIGGER to the Mightex Polygon
1000; the projected pattern is in a .seq file (Sept2021_SweepBRtoTL.seq). Also: all
speed groups use the SAME .seq, differing only in TRIGGER RATE; groups labelled by
(hand-estimated, inaccurate) speed; Control = no digital events. Keeping labels as-is.

### NEW mea_nap_polygon_seq.py — Mightex .seq reader (numpy only)
- Format reverse-engineered + verified: magic "1QSM" (MSQ1); header stores nframes@4228,
  width@4232, height@4236 (uint32 LE); 1-bit-packed frames (rowbytes=ceil(w/8),
  MSB-first), data_offset = filesize - nframes*framebytes.
- Sept2021_SweepBRtoTL.seq = 160 frames, 1824x1216 DMD. Pattern = a GRATING of parallel
  diagonal bars (~124° in DMD, ~27px wide, ~600px pitch) that SWEEPS perpendicular
  across the field. After the projection-optics flip -> BR->TL wavefront on the sample
  = the 135° axis (CONFIRMS PROP_PARAMS sweep_angle_deg=135, no longer an assumption).
- API: read_header, frame_mask(k), roi_centroids, sweep_geometry (FFT-based bar
  orientation, robust for a grating), trigger_frames(n,nframes) (1 trigger=1 frame,
  loops), dmd_to_electrode(xy, calib) affine hook, render_overview. Standalone inspector.
- STILL NEEDS: DMD->MEA registration (affine) for absolute mm/s + per-electrode
  targeting. NOT wired into run_stage2 (direction is fixed 135°); it's a standalone tool.

### Sweep speed from trigger rate (mea_nap_digital_events.py + activity)
- Same pattern across groups => sweep speed is set ENTIRELY by trigger rate, read µs-
  accurately from the digital events (far better than the hand stopwatch). trigger_hz =
  1/median(ISI) is the exact RELATIVE speed with NO calibration needed.
- `sweep_metrics(ev)` -> {n_triggers, median_isi_s, trigger_hz}. `SWEEP_CALIB`
  {sample_diameter_mm=4.0, frames_per_crossing=None, seq_path} + `sweep_speed_mm_s()`
  (mm/s only once frames_per_crossing is calibrated; else NaN).
- run_stage2 collects per-file `sweep_rows`, writes Plots/comparisons? no -> writes
  `<PLOT_DIR>/sweep_speed_summary.csv`, logs per-group median trigger Hz, and FLAGS any
  file whose rate differs >15% from its group median (catches mislabels).
- Group ordering: by-group figures now read Control-first then slow->fast. Mechanism:
  run_stage2 makes the `group` column an ORDERED Categorical (order by measured speed)
  on activity/node/net/electrode dfs; NEW `orig._cat_sorted(series)` returns categories
  in order for ordered-categoricals, else plain sorted() (so lag_ms etc. unaffected).
  Replaced the group/x/hue `sorted(...unique())` sites in activity + orig with it.
- GUI: SWEEP_CALIB (sample_diameter_mm, frames_per_crossing) added to the Spectral/
  Propagation tab. Verified: _cat_sorted gives speed order; comparison figs render with
  the Categorical group with no breakage; no regression in existing self-tests.
- OPEN: give frames_per_crossing (from DMD->MEA calib) to get absolute mm/s and to label
  each trigger with the bar's electrode position; then compare CV vs sweep speed per group.

## BUGFIX: Stage-2 freeze on no-digital (Control) recordings — 2026-07-17

Symptom: "PSD only" run froze at the first long recording at 25% (the un-instrumented
stretch after the activity phase, before the network phase increments the bar).
Root cause: `read_digital_events` had a "last resort" fallback that, on a recording
with NO digital/event stream (i.e. Control files), fell through to reading an
ELECTRODE channel, thresholded the neural voltage as a square wave -> millions of
edges -> `_debounce` (and `_result`) ran an O(n²) Python loop over them -> effectively
frozen. Ran during Stage 2 regardless of plot toggles, so disabling plots didn't help.
Fixes (mea_nap_digital_events.py):
- `_debounce` and `_result` rewritten vectorised (searchsorted, O(n log n)).
- `_read_digital_analog` now rejects non-binary signals (>64 distinct values when no
  digital_bit set) and caps edges at 200k -> a neural voltage channel is refused.
- Digital analog CANDIDATES now require nch<=8, so the 60-ch electrode stream is never
  even considered. Control files now return source=None instantly.
Verified: Control (electrode-only) -> None in 0.00s (was a freeze); real single-entity
Digital Event Detector EventStream -> 1252 ON/OFF, 10 Hz, fine; analog binary digital
stream -> fine; STG sideband still de-prioritised.
Also: `_lat_in_window` (spectral_propagation) switched to searchsorted (spikes are
sorted by io.window_unit) -> prop_windows 16.7s -> 2.0s on an 80 Hz/420 s recording.
Added progress-note updates ("activity plots…", "PSD + propagation…") in run_stage2 so
the bar shows what it's doing during that between-phases stretch instead of looking hung.

## Compute-level checkboxes + speed pass — 2026-07-17

Jerry: make the plot checkboxes compute-level (skip the WORK, not just the PNG) + find
more speedups.
- **Network stage gating (the big win).** `_need_network()` = any of `_NETWORK_FIG_KEYS`
  (the 4A/4B network+node figure keys in orig.PLOT_ENABLE) enabled. When False, run_stage2
  skips the ENTIRE STTC + surrogate + graph-metrics stage: pool not created, serial
  fallback guarded (`if not ok_par and need_net`), n_steps excludes the network steps,
  logged loudly ("network stage SKIPPED …"). So activity/PSD-only runs no longer pay for
  STTC (the dominant cost). node/network CSVs are not written in that mode (expected).
- **Activity plots gated.** `make_activity_plots_multi` now checks orig.PLOT_ENABLE for
  firing_rate_by_electrode / electrode_heatmap / raster (the pipeline's OWN Plots/<stem>/
  figures, gated by the same 2A checkboxes) — unchecking raster skips the ~8 s render.
- **raster_dpi (300 default).** Rasters carry ~1.5M ticks; render cost is the segment
  geometry (DPI-independent-ish) so 300 vs 600 is ~identical visually but smaller files.
  Separate `PLOT_PARAMS["raster_dpi"]` knob (GUI Plots tab, range 150-600); raise to 600
  to match. (Confirmed the 8 s raster is inherent to drawing 1.5M line segments — not the
  layout solver; only skipping or reducing tick count changes it materially.)
- Earlier same-day fixes recap: digital `_debounce`/`_result` O(n²)->vectorised (the
  Control-file freeze), `_lat_in_window` searchsorted (prop_windows 17->2 s).
- NOT changed: global dpi stays 600 (Jerry's explicit preference); STTC internals; the
  tick-raster style (an image/2D-histogram raster mode would be the next big raster win
  but changes appearance — offered, not built).

## BUGFIX: freeze on 1-hour / high-trigger-rate sample (275k laser events) — 2026-07-17

Symptom: Controls (50 laser events) finished; first real sample `4000um_2_D3` (1-hour,
275,705 laser ON @ 80 Hz) stuck at 100% / "PSD + propagation". Cause: the digital-locked
propagation created ONE analysis window PER laser trigger -> 275k plane fits (+ the CV
field the same), and 1-hour sliding windows = 72k windows. Not infinite, but minutes ->
looked hung. Fix: `PROP_PARAMS["max_events_per_region"]` (default 4000) — `_window_events`
and `_digital_events` now uniformly subsample above it via `_subsample_events` (logs
"N events -> subsampled to 4000"). spkit `run_region` caps digital onsets to evt_max_events
too (was stacking 275k EGM segments in _representative_egm). Verified on a 1-hour /
288k-event synthetic: propagation_digital 1.3 s, cv_field digital 0.7 s, windows 1.1 s,
PSD 1.7 s (all were a freeze / ~17 s). Exposed in GUI Spectral/Propagation tab.
NOTE (science): 275k triggers = ~1723 repeats of the 160-frame sweep; uniform subsample
of 4000 still spans all bar positions/cycles -> representative direction/CV. The "correct"
per-bar-position averaging (group triggers by .seq frame index) is the future enhancement
tied to the DMD->MEA calibration, not built yet.

## Straddling-window artifact + missing electrodes — clarified/fixed 2026-07-17

Jerry asked (1) why some electrodes are missing from propagation plots, (2) whether a
window catching the tail of one wave + head of the next fits a bogus direction.
(1) Electrode absent iff: no spike in that window (`_lat_in_window` only includes
electrodes that fired in [t0,t1) — the example LAT map shows only that event's active
electrodes); OR dropped upstream (Ref/15, Batch grounded, auto-grounded) in
io.window_unit; OR (CV field only) fewer than cv_min_neighbors firing neighbours.
(2) CONFIRMED by sim: a straddling window fits a REVERSED direction (315° vs true 135°)
with R²≈0.5–0.6 — which PASSES the default min_fit_r2=0.30, so spurious against-sweep
events leak in. Worst for the digital mode at high trigger rate (100 ms window spanned
~8 waves at 80 Hz). FIX: `_digital_events` now ends each laser window at the NEXT trigger
(min(window_ms, next_onset, dur)) so a stimulus-locked window contains exactly one evoked
wave. Verified: 80 Hz / window_ms=100 -> 80 fits all 135°, R=1.00, 0 reversed (was garbage).
Bursts mode is immune (one detected burst per event). Sliding-window mode on spontaneous
data can still straddle — recommend bursts mode or raising min_fit_r2 (~0.5–0.6) there.

## GUI progress: bars -> reflowing circular rings — 2026-07-17

Jerry: change per-sample progress from a horizontal bar back to a circle, and make
samples wrap into multiple rows by window width. Done in mea_nap_gui.py:
- Per-sample cell = a Canvas ring (`_draw_ring`): grey track oval + coloured arc
  (create_arc style="arc", start=90, extent=-359.9*frac clockwise), centre shows %
  when running or a glyph (✓ done / » skipped / ✕ failed). Colours from _STATE_COLOR.
- Cells laid out with grid() and REFLOWED: `_on_pcanvas_configure` (bound to the
  progress canvas <Configure>) sets the inner window width and calls `_reflow_cells`,
  which grids cells into `cols = max(1, width // CELL_W)` (CELL_W=148). Verified under
  Xvfb: 6 rings/row at 1160px, 4/row at 720px; states/percentages render correctly.
- Constants RING=62, RING_TH=7, CELL_W=148. `_build_rows`/`_update_row` rewritten;
  old BAR_W/BAR_H kept but unused. Name (truncated) + note under each ring.

## Known gaps — deliberately NOT built

NMF, effective rank, rich club (RC is computed in the port but not surfaced),
combined/scaled/circular-module network plots, and Step 5 ML classifiers.

## Environment notes

- Cloud Cowork sessions do NOT see the app-managed `/areas/` memory or the Windows
  data drive by default. To work on real files, Jerry connects a folder via
  "Add folder"; then use device_list_dir / device_stage_files / device_commit_files.
  Writing outputs back to disk via device_commit_files avoids the "unable to open
  file" download issue seen 2026-07-16.
- Deps confirmed on "fett": numba (just installed), plus the usual numpy/scipy/
  pywt/h5py/torch(+CUDA)/networkx/pandas stack. `.cupy` present in home dir.

## Propagation robustness + pitch, crisp rings, SpKit features — 2026-07-17

Diagnosed from real Output figures (4000um_2_D3, 200um_1_D2):
- Propagation "example wave" map showed one SPARSE event (~20 of ~40 active elec) and
  per-event rose was noisy/reversed while the averaged CV field was cleaner.
  NOTE: the "<N>um" filename prefix is the LASER SWEEP SPEED (µm/s), NOT electrode
  spacing — electrodes are fixed 30 µm dia / 200 µm pitch for the whole batch. A
  pitch-from-filename change was tried then REVERTED; electrode_pitch_um stays 200.
  BUT the filename speed IS now surfaced as a per-recording LABEL:
  `sweep_speed_from_stem(stem)` (in mea_nap_spectral_propagation.py; a private copy
  `_sweep_suffix` in mea_nap_spkit_egm.py) regex-parses the leading "<N>um" -> µm/s and
  the propagation, cv_field, spkit_cv, spkit_features, and spkit_cv_summary suptitles now
  append "· nominal laser sweep <N> µm/s" (rendered as mm/s when >=1000). Control names
  (no "<N>um" prefix) -> no label. This is the stimulus's absolute ground-truth sweep
  speed shown ALONGSIDE the measured neural CV (mm/s); the two are different quantities.
Fixes (mea_nap_spectral_propagation.py):
- fig_propagation row 0 now draws the AVERAGED activation map (`_region_activation_map`,
  all active electrodes) not one event; the rose shows a bold BLACK aggregate arrow
  (global fit of the mean map, robust) + grey per-event resultant; alignment/CV headline
  from the aggregate. Fixes both sparse-map and reversed-direction complaints.
- NOTE: some samples are genuinely incoherent (aggregate align~0) — not a bug.
GUI rings crisp (mea_nap_gui.py): PIL supersampled (`_draw_ring_pil`, SS=4, LANCZOS) with
Tk-arc fallback (`_draw_ring_tk`); DejaVuSans-Bold via matplotlib font_manager.
SpKit features (mea_nap_spkit_egm.py): run_region now also runs `extract_egm`+`egm_features`
-> per-electrode peak_to_peak / egm_duration(ms) / f_index grids. Two figures now:
  spkit_features.png = activation | peak-to-peak | EGM duration | fractionation index;
  spkit_cv.png = CV magnitude | CV vector field | STREAMLINE flow map | VELOCITY COMPASS.
DESYNC WARNING: cloud workspace had reverted to a pre-straddle-fix snapshot; machine
io.py had lost the "win" field (breaks digital wiring). Re-synced from machine + re-applied
edits. Likely Nextcloud sync conflicts reverting files between runs — flag to Jerry.
