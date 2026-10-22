"""Orchestration: wire the MEA-NAP loaders to the analysis modules.

Everything here touches real recordings (heavy) and runs on the machine that has
the loaders + data. The GUI calls these; the pure-numpy analysis modules stay
loader-free and unit-tested. Each long call takes a `progress(done, total)`
callback so the GUI can show a bar and stay responsive.
"""
import os
import numpy as np

import loaders
import geometry as geom
import peaks as pk
import templates as tpl
import classify as clf
import filtering as flt
import fieldmap as fm
import footprint as fp
import project as proj


# ---------------------------------------------------------------------------
def load_raw(data_dir, stem):
    return loaders.RawRecording(os.path.join(data_dir, stem + ".h5"))


def flood_times(data_dir, stem, window=None):
    """Laser flood ON times (s). If window=(t0,t1) restrict to that span."""
    ev = loaders.laser_events(os.path.join(data_dir, stem + ".h5"))
    on = np.asarray(ev.get("on_s", []), float)
    if window is not None:
        t0, t1 = window
        on = on[(on >= t0) & (on <= t1)]
    return on, ev


def flood_on_off(data_dir, stem, window=None):
    """(on, off, median_duration_s) for the laser floods. `off` is None if the
    digital events carry only onsets; duration is the median ON→OFF gap (0 if
    unknown)."""
    ev = loaders.laser_events(os.path.join(data_dir, stem + ".h5"))
    on = np.asarray(ev.get("on_s", []), float)
    off = ev.get("off_s")
    off = np.asarray(off, float) if off is not None and np.size(off) else None
    if window is not None:
        t0, t1 = window
        keep = (on >= t0) & (on <= t1)
        on = on[keep]
        if off is not None and off.size == np.size(keep):
            off = off[keep]
    dur = 0.0
    if off is not None and off.size == on.size and on.size:
        d = off - on
        d = d[np.isfinite(d) & (d > 0)]
        dur = float(np.median(d)) if d.size else 0.0
    elif off is not None and off.size and on.size:
        # unpaired: estimate duration from nearest following offset
        offs = np.sort(off)
        ds = []
        for a in on:
            nxt = offs[np.searchsorted(offs, a, "right"):]
            if nxt.size:
                ds.append(nxt[0] - a)
        ds = np.asarray([x for x in ds if 0 < x])
        dur = float(np.median(ds)) if ds.size else 0.0
    return on, off, dur


def auto_stim_span(raw, starts, analysis, n_probe=8, k=6.0, probe_cap_s=2.0,
                   margin_s=0.010):
    """DATA-DRIVEN artifact duration (s): how long after each stimulation onset the
    flood-evoked transients last. Measured on the HIGH-PASSED flood-triggered average
    of the strongest-responding electrodes — high-pass so the slow photovoltaic
    relaxation tail (which the spike detector band-passes out and never re-detects) is
    ignored, leaving only the sharp pulse transients that actually leak as spikes.

    The baseline is taken from the CLEAN POST-ARTIFACT TAIL of the window, NOT the
    pre-onset samples: the huge onset transient's (non-causal) filter ringing bleeds
    backwards into the pre-window and would otherwise inflate the noise estimate and
    make the detector report ~0. To avoid ever under-covering (which lets residual
    artifacts leak), the span is the 90th-percentile across the probed electrodes plus
    a small margin. Adapts to any pulse count / width, single pulses, and different
    durations. Returns None if there are no events (caller keeps stim_train_ms).
    """
    starts = np.asarray(starts, float)
    if starts.size == 0:
        return None
    fs = raw.fs
    if starts.size >= 2:                       # don't probe into the next train
        probe = min(probe_cap_s, 0.8 * float(np.min(np.diff(starts))))
    else:
        probe = min(probe_cap_s, 0.5 * raw.duration)
    probe = max(probe, 0.05)
    hp = analysis.get("hp_hz", 100.0)
    cand = []
    for lab in raw.labels:
        avg, t_ms, n, p2p = pk.flood_triggered_average(
            raw.channel(lab), fs, starts, pre_ms=5.0, post_ms=probe * 1000.0)
        cand.append((float(p2p), avg, t_ms))
    cand.sort(key=lambda r: -r[0])
    spans = []
    for p2p, avg, t_ms in cand[:n_probe]:
        if p2p <= 0:
            continue
        raw_env = np.abs(pk.highpass(np.asarray(avg, float), fs, hp))
        sw = max(1, int(0.005 * fs))               # 5 ms moving average: kills lone
        env = np.convolve(raw_env, np.ones(sw) / sw, mode="same")  # noise samples,
        #                                            keeps the (repeating) transients
        tail = t_ms > 0.7 * float(t_ms.max())      # clean post-artifact baseline
        if int(tail.sum()) < 10:
            tail = t_ms > 0.5 * float(t_ms.max())
        base = float(np.median(env[tail]))
        spread = float(pk.robust_noise(env[tail]))
        thr = base + k * max(spread, 1e-9)
        if float(env.max()) < 3.0 * thr:           # no real artifact here
            continue
        above = np.where((t_ms >= 0) & (env > thr))[0]
        if above.size:
            spans.append(float(t_ms[above[-1]]) * 1e-3)
    if not spans:
        return None
    span = float(np.percentile(spans, 90)) + margin_s
    return float(np.clip(span, 0.005, 2.0))


def resolve_stim_span(project, raw, starts):
    """Return (analysis_copy, span_s). When analysis['stim_span_mode'] == 'auto',
    override stim_train_ms with the DATA-DRIVEN artifact duration so the STA
    subtraction, flood-locked rejection, classifier window and solar template all use
    the real per-recording stimulation length. Falls back to the fixed stim_train_ms
    if detection fails."""
    A = dict(project.analysis)
    fixed = A.get("stim_train_ms", 250.0) * 1e-3
    if A.get("stim_span_mode", "auto") != "auto":
        return A, fixed
    span = auto_stim_span(raw, starts, A)
    if span is None or not np.isfinite(span) or span <= 0:
        return A, fixed
    span = float(np.clip(span, 0.005, 2.0))
    A["stim_train_ms"] = span * 1000.0
    return A, span


def grounded_electrodes(data_dir, stem):
    """Set of grounded electrode labels for a stem (Batch.csv column 6). These are
    turned OFF: excluded from detection and shown inactive on the maps."""
    csv = os.path.join(data_dir, "Batch.csv")
    if not os.path.exists(csv):
        return set()
    g = set()
    try:
        for r in loaders.parse_batch(csv):
            if r["stem"] == stem:
                g.update(str(x) for x in r.get("ground", []))
    except Exception:
        return set()
    return g


def region_windows(data_dir, stem):
    """{'pre':(t0,t1), 'stim':..., 'post':...} from Batch.csv (only those present)."""
    out = {}
    for r in ("pre", "stim", "post"):
        w = region_window(data_dir, stem, r)
        if w:
            out[r] = w
    return out


def event_regions(data_dir, stem, duration, analysis=None):
    """pre / stim / post windows derived from the LASER DIGITAL-EVENT trace, not the
    Batch.csv region column: `stim` spans the actual flood activity (first onset to
    the last train's end), `pre` is everything before it, `post` everything after.
    Returns {} if there are no digital events (caller falls back to the region
    column). Windows shorter than ~1 s are dropped."""
    on, off, _dur = flood_on_off(data_dir, stem)
    if on is None or np.size(on) == 0:
        return {}
    t_first = float(np.min(on))
    stim_train_s = (analysis or {}).get("stim_train_ms", 250.0) * 1e-3
    if off is not None and np.size(off):
        t_last = float(np.max(off))
    else:
        t_last = float(np.max(on)) + stim_train_s
    t_last = min(t_last, float(duration))
    out = {}
    if t_first > 1.0:
        out["pre"] = (0.0, t_first)
    out["stim"] = (t_first, max(t_last, t_first + stim_train_s))
    if duration - t_last > 1.0:
        out["post"] = (t_last, float(duration))
    return out


def region_rate_maps(filter_result, windows):
    """{region -> {label -> rate Hz}} by binning the (absolute-time) biological
    spikes into each region window. No re-detection needed — the spike times already
    span the whole recording. Grounded electrodes are NaN (rendered off)."""
    grounded = getattr(filter_result, "grounded", set())
    maps = {}
    for name, (t0, t1) in windows.items():
        dur = max(1e-9, float(t1) - float(t0))
        maps[name] = {lab: (np.nan if lab in grounded
                            else int(np.sum((t >= t0) & (t < t1))) / dur)
                      for lab, t in filter_result.bio_spikes.items()}
    return maps


def region_burst_rate_maps(filter_result, windows, analysis):
    """{region -> {label -> burst rate (bursts/min)}} over each region window."""
    import bursts
    grounded = getattr(filter_result, "grounded", set())
    max_isi = analysis.get("burst_max_isi_ms", 100.0) * 1e-3
    min_sp = int(analysis.get("burst_min_spikes", 5))
    maps = {}
    for name, (t0, t1) in windows.items():
        dur_s = max(1e-9, float(t1) - float(t0))
        m = {}
        for lab, t in filter_result.bio_spikes.items():
            if lab in grounded:
                m[lab] = np.nan
                continue
            tt = t[(t >= t0) & (t < t1)]
            m[lab] = bursts.burst_rate_per_min(tt, dur_s, max_isi, min_sp)
        maps[name] = m
    return maps


def burst_rate_map(filter_result, duration_s, analysis):
    """{label -> burst rate (bursts/min)} over the whole recording (for the live
    in-app map). Grounded electrodes are NaN."""
    import bursts
    grounded = getattr(filter_result, "grounded", set())
    max_isi = analysis.get("burst_max_isi_ms", 100.0) * 1e-3
    min_sp = int(analysis.get("burst_min_spikes", 5))
    out = {}
    for lab, t in filter_result.bio_spikes.items():
        out[lab] = (np.nan if lab in grounded
                    else bursts.burst_rate_per_min(t, duration_s, max_isi, min_sp))
    return out


def region_window(data_dir, stem, region):
    """(t0,t1) for a Batch.csv region ('pre'/'stim'/'post') of a stem, or None."""
    csv = os.path.join(data_dir, "Batch.csv")
    if not os.path.exists(csv):
        return None
    for r in loaders.parse_batch(csv):
        if r["stem"] == stem and str(r["region"]).lower() == str(region).lower():
            return r["win"]
    return None


# ---------------------------------------------------------------------------
def detect_all_peaks(raw, analysis, labels=None, window=None, progress=None):
    """dict label -> peaks.Peaks over the whole trace (or a window)."""
    labels = labels or raw.labels
    out = {}
    for n, lab in enumerate(labels):
        if window is None:
            trace, t0 = raw.channel(lab), 0.0
        else:
            trace, t0 = raw.channel_window(lab, window[0], window[1])
        out[lab] = pk.detect_peaks(
            trace, raw.fs, hp_hz=analysis["hp_hz"], k=analysis["peak_k"],
            refractory_ms=analysis["refractory_ms"],
            pre_ms=analysis["snippet_pre_ms"], post_ms=analysis["snippet_post_ms"],
            t_offset=t0, label=lab)
        if progress:
            progress(n + 1, len(labels))
    return out


# ---------------------------------------------------------------------------
def default_template_sources(project):
    """The recording each template class is learned from by default. SOLAR uses a
    SINGLE-cell recording (never the two-cell file); PIEZO a laser-only; BIO a
    culture-only, else a solar+culture sample. Returns {class: stem or ''}."""
    def first(cond):
        s = project.stems_with_condition(cond)
        return s[0] if s else ""
    bio = first(proj.CULTURE_ONLY) or first(proj.SOLAR_CULTURE)
    return {tpl.SOLAR: first(proj.SOLAR_ONLY),
            tpl.PIEZO: first(proj.LASER_ONLY),
            tpl.BIO: bio}


def learn_templates(project, data_dir=None, sources=None, progress=None,
                    max_channels=None):
    """Learn solar/piezo/bio templates from EXPLICIT source recordings.

    sources : {SOLAR: stem, PIEZO: stem, BIO: stem}. Defaults to
              default_template_sources(project) — SOLAR from the ONE-cell recording.

    solar/piezo are learned as flood-ONSET-aligned EVENT AVERAGES (each spanning the
    whole exposure) so a two-transient (ON+OFF) event is preserved; bio is learned
    as extremum-aligned spike snippets. Returns (TemplateSet, info_dict).
    """
    data_dir = data_dir or project.data_dir
    A = project.analysis
    ts = tpl.TemplateSet()
    info = {}
    if sources is None:
        sources = default_template_sources(project)

    plan = [(tpl.SOLAR, sources.get(tpl.SOLAR, ""), "artifact"),
            (tpl.PIEZO, sources.get(tpl.PIEZO, ""), "artifact"),
            (tpl.BIO, sources.get(tpl.BIO, ""), "bio")]
    steps = sum(1 for _, stem, _ in plan if stem)
    done = [0]

    def sub(name):
        def _p(i, t):
            if progress:
                progress(done[0], max(1, steps), f"{name}: channel {i}/{t}")
        return _p

    for name, stem, kind in plan:
        if not stem:
            info[name] = "no source recording"
            continue
        raw = load_raw(data_dir, stem)
        on, off, dur = flood_on_off(data_dir, stem)
        if kind == "artifact" and name == tpl.SOLAR:
            # SOLAR is a pulsed TRAIN of transients — there is no single "solar peak",
            # so THE solar template IS the whole onset-aligned train (dipole sign-
            # aligned so opposite-edge electrodes reinforce). The train length is
            # data-driven (auto), so it fits any pulse count / width. Solar
            # classification is by flood-train locking + proximity to the cell.
            A_s, _span = resolve_stim_span(project, raw, flt.train_starts(on, A))
            waves, ev_t, ev_amps, _w = tpl.learn_event_class_from_recording(
                raw, on, dur, A_s, max_channels=max_channels, progress=sub(name))
            t = tpl.build_template(name, waves, ev_t, ev_amps,
                                   np.full(len(ev_amps), np.nan), raw.fs,
                                   align_polarity=False, aligned="onset",
                                   sign_align=True, amp_normalize=True,
                                   orient_down=True)
            raw.free()
            ts.templates[name] = t
            info[name] = ("%s: %.0f ms train (whole flood), %d electrode(s)"
                          % (stem, A_s.get("stim_train_ms", 250.0), t.n))
        elif kind == "artifact":
            # PIEZO: (a) extremum-aligned PEAK template from flood-locked peaks -> the
            #     per-peak SHAPE backstop; (b) the onset EVENT (both transients) for
            #     display.
            snips, snip_t, amps, widths = tpl.learn_class_from_recording(
                raw, on, A, locked=True, max_channels=max_channels,
                progress=sub(name))
            t = tpl.build_template(name, snips, snip_t, amps, widths, raw.fs,
                                   align_polarity=True, aligned="peak",
                                   robust=True,
                                   min_corr=A.get("template_min_corr", 0.55),
                                   keep_frac=A.get("template_keep_frac", 0.8),
                                   amp_mad_k=A.get("template_amp_mad_k", 6.0))
            waves, ev_t, ev_amps, _w = tpl.learn_event_class_from_recording(
                raw, on, dur, A, max_channels=max_channels)
            ev = tpl.build_template(name, waves, ev_t, ev_amps,
                                    np.full(len(ev_amps), np.nan), raw.fs,
                                    align_polarity=False, aligned="onset",
                                    sign_align=True)
            t.event_t_ms, t.event_uv = ev.t_ms, ev.mean_uv
            t.event_std, t.event_n = ev.std_uv, ev.n
            raw.free()
            ts.templates[name] = t
            info[name] = ("%s: %d peaks (shape) + %d event electrode(s), dur≈%.0f ms"
                          % (stem, t.n, ev.n, dur * 1000.0))
        else:
            # BIO: learn ONLY from peaks the validated bior1.5 detector confirms as
            # spikes (non-flood-locked, negative-going), so noise crossings the
            # broadband detector picks up can't inflate the wavelet or its ±SD band.
            snips, snip_t, amps, widths = tpl.learn_class_from_recording(
                raw, on, A, locked=False, max_channels=max_channels,
                progress=sub(name), keep_polarity=A.get("bio_polarity", 0),
                detector=loaders.detect_spikes)
            fs = raw.fs
            raw.free()
            ts.templates[name] = tpl.build_template(
                name, snips, snip_t, amps, widths, fs, robust=True,
                min_corr=A.get("template_min_corr", 0.55),
                keep_frac=A.get("template_keep_frac", 0.8),
                amp_mad_k=A.get("template_amp_mad_k", 6.0))
            info[name] = f"{stem}: {ts.templates[name].n} detector-confirmed spikes"
        done[0] += 1
        if progress:
            progress(done[0], max(1, steps), f"{name} done")
    return ts, info


# ---------------------------------------------------------------------------
def _near_solar_labels(project, stem, threshold_um=None):
    """Electrode labels inside/near ANY solar cell placed on this recording."""
    cells = project.solar_cells(stem)
    if not cells:
        return set()
    arr = project.array()
    if threshold_um is None:
        threshold_um = 0.75 * project.pitch_um
    d, _ = geom.nearest_cell_distance_um(cells, arr.grid)
    return {arr.labels[i] for i in range(len(arr.labels)) if d[i] <= threshold_um}


def classify_recording(raw, on, off, dur_s, templateset, project, stem,
                       labels=None, window=None, progress=None):
    """dict label -> ClassifyResult, plus aggregate counts. `on`/`off`/`dur_s`
    define the full-exposure artifact window (both solar transients)."""
    labels = labels or raw.labels
    near = _near_solar_labels(project, stem)
    A, _span = resolve_stim_span(project, raw, flt.train_starts(on, project.analysis))
    peaks_by = detect_all_peaks(raw, A, labels, window, progress)
    results, counts = {}, {}
    for lab, P in peaks_by.items():
        ns = np.full(len(P), lab in near, bool)
        res = clf.classify_peaks(P, on, off, dur_s, templateset, A, ns)
        results[lab] = res
        for c, n in res.counts().items():
            counts[c] = counts.get(c, 0) + n
    return results, counts


# ---------------------------------------------------------------------------
def _sta_p2p(project, data_dir, stem, progress=None):
    """Per-electrode flood-triggered-average peak-to-peak (µV) + noise for a
    recording. Returns (resp_by_label, noise_by_label, bank)."""
    raw = load_raw(data_dir, stem)
    on, off, dur = flood_on_off(data_dir, stem)
    A, _span = resolve_stim_span(project, raw, flt.train_starts(on, project.analysis))
    bank = flt.build_sta_bank(raw, on, A, dur_s=dur, progress=progress)
    raw.free()
    resp = {lab: bank[lab]["p2p"] for lab in bank}
    noise = {lab: bank[lab]["noise"] for lab in bank}
    return resp, noise, bank


def field_map_from_solar(project, data_dir=None, stem=None, baseline_stem=None,
                         progress=None):
    """Build the field map for a recording using ITS placed solar cell(s).

    baseline_stem : optional laser-only recording whose flood response (piezo) is
                    subtracted per electrode, isolating the pure solar field for
                    the 1-cell vs 2-cell amplification comparison.
    """
    data_dir = data_dir or project.data_dir
    if stem is None:
        stems = (project.stems_with_condition(proj.SOLAR_ONLY)
                 or project.stems_with_condition(proj.SOLAR_TWO))
        if not stems:
            raise ValueError("Tag a recording as 'Solar cell x1/x2' first.")
        stem = stems[0]
    cells = project.solar_cells(stem)
    if not cells:
        raise ValueError("Place at least one solar cell on '%s' first "
                         "(Setup tab -> Solar cells for recording)." % stem)
    resp, noise, bank = _sta_p2p(project, data_dir, stem, progress)
    baseline = None
    if baseline_stem:
        baseline, _bn, _bb = _sta_p2p(project, data_dir, baseline_stem)
    arr = project.array()
    fmap = fm.compute_field_map(arr, resp, cells, noise_by_label=noise,
                                baseline_by_label=baseline)
    return fmap, stem, bank


# ---------------------------------------------------------------------------
def filter_sample(project, data_dir=None, stem=None, window=None, progress=None):
    """Artifact-subtract a solar_culture sample and re-detect biological spikes."""
    data_dir = data_dir or project.data_dir
    if stem is None:
        stems = project.stems_with_condition(proj.SOLAR_CULTURE)
        if not stems:
            raise ValueError("Tag a sample as 'Solar cell + culture' first.")
        stem = stems[0]
    raw = load_raw(data_dir, stem)
    on, off, flood_dur = flood_on_off(data_dir, stem, window)
    starts = flt.train_starts(on, project.analysis)
    A, span = resolve_stim_span(project, raw, starts)
    res = flt.filter_recording(raw, on, A,
                               detector=loaders.detect_spikes, dur_s=flood_dur,
                               progress=progress,
                               exclude=grounded_electrodes(data_dir, stem))
    res.stim_span_ms = span * 1000.0
    dur = raw.duration if window is None else (window[1] - window[0])
    raw.free()
    return res, stem, dur


def filter_and_save(project, out_dir, data_dir=None, stem=None, window=None,
                    progress=None, use_matrix=True):
    """Filter ONE recording (subtract flood artifacts, re-detect biological spikes)
    and write the cleaned train in the EXACT MEA-NAP format (`<stem>_spikes.npz` +
    `.mat`) into out_dir, so MEA Analyzer (MEA-NAP) Stage 2 / the raster viewer read
    and plot it directly. Returns (npz_path, FilterResult, duration_s)."""
    data_dir = data_dir or project.data_dir
    if stem is None:
        stems = project.stems_with_condition(proj.SOLAR_CULTURE)
        if not stems:
            raise ValueError("Tag a sample as 'Solar cell + culture' first.")
        stem = stems[0]
    raw = load_raw(data_dir, stem)
    on, off, flood_dur = flood_on_off(data_dir, stem, window)
    excl = grounded_electrodes(data_dir, stem)
    starts = flt.train_starts(on, project.analysis)
    A, span = resolve_stim_span(project, raw, starts)
    if use_matrix:
        res = flt.filter_recording_matrix(
            raw, on, A, detector_matrix=loaders.detect_spikes_matrix,
            progress=progress, exclude=excl)
    else:
        res = flt.filter_recording(raw, on, A, detector=loaders.detect_spikes,
                                   dur_s=flood_dur, progress=progress, exclude=excl)
    res.stim_span_ms = span * 1000.0
    dur = raw.duration if window is None else (window[1] - window[0])
    fs = raw.fs
    raw.free()
    noise_by_label = {lab: res.sta_bank[lab]["noise"] for lab in res.bio_spikes
                      if lab in res.sta_bank}
    npz_path = loaders.save_meanap_spike_file(
        out_dir, stem, res.bio_spikes, fs, noise_by_label=noise_by_label,
        extra_params={"artifact_labels": sorted(res.artifact_labels),
                      "duration_s": float(dur)})
    return npz_path, res, dur


def batch_filter(project, out_dir, data_dir=None, stems=None, progress=None,
                 use_matrix=True):
    """Filter EVERY recording in the data folder and write one `<stem>_spikes.npz`
    (+ `.mat`) per recording into out_dir, all MEA-NAP-readable. Skips nothing —
    every .h5 stem is processed (or the explicit `stems` list). Returns a list of
    per-recording result dicts (stem, path, n_spikes, n_artifact_ch, error)."""
    data_dir = data_dir or project.data_dir
    if stems is None:
        stems = loaders.list_h5_stems(data_dir)
    results = []
    total = max(1, len(stems))
    for k, stem in enumerate(stems):
        def sub(done, tot, _stem=stem, _k=k):
            if progress:
                progress(_k, total,
                         f"[{_k + 1}/{total}] {_stem}: channel {done}/{tot}")
        try:
            npz_path, res, dur = filter_and_save(
                project, out_dir, data_dir, stem, window=None, progress=sub,
                use_matrix=use_matrix)
            results.append({"stem": stem, "path": npz_path,
                            "n_spikes": res.total_spikes(),
                            "n_artifact_ch": len(res.artifact_labels),
                            "error": None})
        except Exception as exc:
            results.append({"stem": stem, "path": None, "n_spikes": 0,
                            "n_artifact_ch": 0, "error": repr(exc)})
        if progress:
            progress(k + 1, total, f"[{k + 1}/{total}] {stem} done")
    return results


def field_overlay(project, filter_result, stem, mode="modeled", n=140):
    """(Z, extent_grid) solar-cell electric-field heatmap for the electrode-map
    BACKGROUND, or None if this recording has no placed solar cell.

    mode='modeled'  : smooth exp-decay field from each placed cell's location,
                      scaled by the measured near-cell solar amplitude (works on any
                      sample). mode='measured' : the actual per-electrode flood
                      response (STA p2p) interpolated across the plane.
    """
    cells = project.solar_cells(stem)
    if not cells or filter_result is None:
        return None
    bank = getattr(filter_result, "sta_bank", None)
    if not bank:
        return None
    arr = project.array()
    resp = {lab: bank[lab]["p2p"] for lab in bank}
    if not resp:
        return None
    if mode == "measured":
        noise = {lab: bank[lab]["noise"] for lab in bank}
        try:
            fmap = fm.compute_field_map(arr, resp, cells, noise_by_label=noise)
            return fmap.interpolate(arr, n=n)
        except Exception:
            return None
    # modeled: amplitude from electrodes near the cell(s)
    d, _ = geom.nearest_cell_distance_um(cells, arr.grid)
    thr = 0.75 * project.pitch_um
    near = [resp[arr.labels[i]] for i in range(len(arr.labels))
            if d[i] <= thr and arr.labels[i] in resp]
    amp0 = float(np.median(near)) if near else float(np.median(list(resp.values())))
    lam = float(project.analysis.get("field_lambda_um", 300.0))
    return fm.model_field_grid(arr, cells, amp0, lam, n=n)


def solar_cell_peaks(project, filter_result, stem, max_dist_um=None):
    """Per-solar-cell MAX detected voltage for the map annotations.

    Every electrode is assigned to the solar cell whose edge it is closest to, and
    each cell takes the largest flood-triggered-average peak-to-peak amplitude
    (µV) among the electrodes it claimed. With three cells in the field of view
    that gives three independent numbers, one per cell.

    Returns (cells, values_uv, peak_labels) — three parallel lists. `values_uv`
    holds NaN for a cell with no usable electrode.

    Note on what the number means: the STA p2p is dominated by the fast capacitive
    edge transient at laser onset, which is much larger than the sustained
    photovoltage during illumination. It is the same quantity the field map and the
    effective-radius fit already use, so the annotation is consistent with them.
    """
    cells = list(project.solar_cells(stem) or [])
    if not cells:
        return [], [], []
    bank = getattr(filter_result, "sta_bank", None) or {}
    resp = {lab: bank[lab]["p2p"] for lab in bank}
    if not resp:
        return cells, [float("nan")] * len(cells), [None] * len(cells)
    recs = geom.max_by_cell(cells, resp, project.array(),
                            max_dist_um=max_dist_um)
    return cells, [r["value"] for r in recs], [r["label"] for r in recs]


def solar_cell_plateaus(project, filter_result, stem, max_dist_um=None):
    """Per-solar-cell SUSTAINED photovoltage (held plateau) for the annotations,
    reported ALONGSIDE the edge-spike max from `solar_cell_peaks`.

    Same nearest-cell assignment, but the per-cell number is the strongest sustained
    plateau (by magnitude, sign kept) among the electrodes the cell claimed — the DC
    level held DURING illumination, edges excluded, which the coupling test showed
    survives in the DC-coupled raw trace. This is the real sustained cell response,
    typically far smaller than the fast capacitive edge transient that dominates the
    p2p. Returns (cells, plateau_uv, plateau_labels) — three parallel lists; the
    plateau list is all-NaN when the recording's single-pulse duration is unknown.
    """
    cells = list(project.solar_cells(stem) or [])
    if not cells:
        return [], [], []
    bank = getattr(filter_result, "sta_bank", None) or {}
    plat = {lab: bank[lab]["plateau"] for lab in bank
            if "plateau" in bank[lab] and np.isfinite(bank[lab]["plateau"])}
    if not plat:
        return cells, [float("nan")] * len(cells), [None] * len(cells)
    recs = geom.max_by_cell(cells, plat, project.array(),
                            max_dist_um=max_dist_um, by_magnitude=True)
    return cells, [r["value"] for r in recs], [r["label"] for r in recs]


def footprint_from_filter(project, filter_result, duration):
    """Footprint from a FilterResult's biological spikes."""
    arr = project.array()
    rate = filter_result.rate_map(duration)
    # mean spike amplitude per electrode from the STA-free residual is not tracked;
    # use spike count as amplitude proxy only if needed. Footprint uses rate.
    return fp.compute_footprint(arr, rate, spheroid=project.spheroid_obj())
