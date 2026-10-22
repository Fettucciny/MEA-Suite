# MEA Analyzer — instructions for Claude Code

Read the suite-level `..\CLAUDE.md` first. This folder is the PyQt6 Analyzer;
all code is in `MEA-NAP\`. Launch: `MEA Analyzer.bat` (or the .exe, which just
runs the .bat with a splash). The GUI is `MEA-NAP\mea_nap_gui.py`; it imports
every analysis module at start-up, so a broken module = no window (pythonw).

## Module map (MEA-NAP\)

| file | role |
|---|---|
| `mea_nap_gui.py` | the window: paths, section toggles, parameter tabs (`build_schema`), groups panel, run/replot workers, log, embedded progress rows |
| `mea_nap_io.py` | CONFIG, Batch.csv parser (BLOCK B), windowing of spike files into units (`window_unit`), Stage 1 driver, resume manifest (`save_progress`, Nextcloud-tolerant) |
| `mea_nap_wavelet_detect.py` | spike detection (torch/GPU) |
| `mea_nap_activity.py` | Stage 2: activity, bursts, STTC network, cartography, group comparisons (`run_stage2`, `run_stage2c`) |
| `mea_nap_original_plots.py` | MEA-NAP MATLAB figure set; also `mcs_coord()` / `_mcs60_labels()` (electrode geometry: x=col, y=9-row) |
| `mea_nap_phase_lockin.py` | **sweep / radial phase lock-in** — see HANDOFF_2026-10-22.md |
| `mea_nap_spkit_egm.py` | SpKit continuous-EGM conduction velocity + slow-sweep latency |
| `mea_nap_spectral_propagation.py` | burst/laser-locked propagation maps |
| `mea_nap_digital_events.py` | laser trigger reader (`read_digital_events`, `sweep_metrics`) |
| `mea_nap_progress.py` | progress-window abstraction (the GUI replaces `make()` with its embedded rows) |
| `mea_nap_polygon_seq.py` | Polygon `.seq` decoder |
| `mea_theme.py` | local copy of the suite theme kit |
| `meanap\` | the lab's port of MEA-NAP pipeline internals |
| `dev\` | verification tools (astdiff, compare_figures, check_gui_keys) |

Stages run in a worker thread (`App._worker`): Stage 1 -> Stage 2 -> SpKit ->
Phase Lock-In, each gated by its section toggle. `App._replot_worker` re-renders
the phase lock-in figures from the results cache without computing.

## Phase lock-in essentials (details in the handoff)

- Per-electrode phase: spikes folded onto the stimulus period, `z = mean(exp(iφ))`
  (`electrode_locking`). Spatial model: plane-wave matched filter
  (`fit_plane_wave`, vectorised — one matmul over the k grid). Period seed =
  frames / trigger_hz, refined ±30 %.
- Radial groups (`group` starts with `radial`): `quadrant_alignment()` fits a
  plane wave per quadrant about the ring centre (`_RADIAL_CENTRE_CELLS` =
  (4.5, 4.5), i.e. between 44/45/54/55) and scores each against its outward
  direction. The sweep axis parameter is NOT used for radial outputs.
- Outputs per group under `<PLOT_DIR>\Phase Lock\`: lock map, flow map,
  compass, convergence-over-time (per sample) and `..._mean_<condition>`
  (averaged over replicates), `phase_lockin_summary.csv`, and the results cache
  `phase_lockin_results.pkl` (+ `.json` manifest) that the Replot button uses.
- Per-sample work is `_analyse_stem()` (pure, no GUI, no global config) so it
  can run in a spawn-mode process pool (`n_workers`, 0 = auto). Serial and
  parallel are bit-identical; keep it that way (seeds are fixed per window).
- `post_map_window_s` (150 s): the map/flow/compass POST column is a separate
  fit over the first N s of post; the convergence curves use the full post.
  Changing it needs a run; plot-only keys (`_PLOT_ONLY_KEYS`) only need Replot.
- Rollback: `Rollback Phase Lock-In to 10_19_26.bat` swaps in the pre-
  vectorisation legacy build; run it again to swap back.
- `null_model` ("rotation" default | "position"): the rotation null is driven
  to the p floor by synchrony alone (every pre/post/control window on the Oct
  2026 batch); "position" shuffles electrode positions and is the propagation
  test. See HANDOFF section 8.2 before interpreting `p_value`.

## Things not to do

- Don't rename the `Phase Lock` / `MEA-NAP Original Output` output folders or
  the summary CSV columns — downstream notebooks read them.
- Don't change `PHASE_LOCKIN` defaults, `min_cycles`, `max_trigger_period_s`,
  `min_phase_span_rad`, `coherence_floor` without saying so: they decide which
  windows count as locked.
- Don't bump `RESULTS_VERSION` unless the cache layout really changed; it
  invalidates every user's cached results.
