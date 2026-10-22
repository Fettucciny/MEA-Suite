"""Headless self-tests for the analysis logic (no GUI, no loaders, no real data).

Synthesises traces/maps with known ground truth and checks that peak detection,
template learning, classification, artifact subtraction, the field-map fit and
the footprint recover it. Run:  python selftest.py
"""
import numpy as np

import geometry as geo
import peaks as pk
import templates as tpl
import classify as clf
import filtering as flt
import fieldmap as fm
import footprint as fp

FS = 20000.0
_fail = []


def check(name, cond, extra=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  {extra}" if extra else ""))
    if not cond:
        _fail.append(name)


def _ap(width_samples=12, amp=-40.0):
    s = np.sin(np.linspace(0, np.pi, width_samples))
    shape = -np.concatenate([s[:width_samples // 2], -0.6 * s[width_samples // 2:]])
    return shape / np.abs(shape).max() * abs(amp) * (1 if amp > 0 else -1)


def test_geometry():
    print("geometry")
    arr = geo.ElectrodeArray()
    check("60 electrodes", len(arr.labels) == 60, f"n={len(arr.labels)}")
    check("pitch µm scaling", np.allclose(arr.position_um("47"),
          arr.position_grid("47") * 200.0))
    sc = geo.SolarCell(center_grid=(4.5, 4.5), side_um=400.0,
                       orientation_deg=0.0, pitch_um=200.0)
    # centre is inside -> distance 0
    check("inside cell -> dist 0", sc.distance_um([(4.5, 4.5)])[0] == 0.0)
    # a point 2 grid units (400 µm) to the right: edge is at +200 µm, so 200 µm out
    d = sc.distance_um([(6.5, 4.5)])[0]
    check("edge distance", abs(d - 200.0) < 1e-6, f"d={d:.1f}µm")
    sph = geo.Spheroid(center_grid=(4.0, 4.0), radius_um=200.0)
    # edge distance: a point 400 µm from the centre of a 200 µm-radius disc is
    # 200 µm outside the edge; a point inside the disc is 0.
    check("spheroid edge distance",
          abs(sph.distance_um([(6.0, 4.0)])[0] - 200.0) < 1e-6)
    check("spheroid inside -> 0", sph.distance_um([(4.5, 4.0)])[0] == 0.0)
    # hand-traced irregular outline: a square traced clockwise
    out = geo.SpheroidOutline([(3, 3), (5, 3), (5, 5), (3, 5)], pitch_um=200.0)
    check("outline centroid ~ (4,4)",
          np.allclose(out.center_grid, [4.0, 4.0], atol=0.15),
          str(np.round(out.center_grid, 2)))
    check("outline inside -> 0", out.distance_um([(4.0, 4.0)])[0] == 0.0)
    check("outline outside > 0", out.distance_um([(8.0, 4.0)])[0] > 0.0)


def test_peaks():
    print("peaks")
    rng = np.random.default_rng(0)
    dur = 6.0
    N = int(FS * dur)
    x = rng.normal(0, 4.0, N)
    ap = _ap()
    times = np.array([0.5, 1.3, 2.7, 3.9, 5.1])
    for t in times:
        i = int(t * FS)
        x[i:i + ap.size] += ap
    # add a POSITIVE artifact too
    pos = -_ap(20, -60.0)
    x[int(4.5 * FS):int(4.5 * FS) + pos.size] += pos
    p = pk.detect_peaks(x, FS, hp_hz=100.0, k=5.0)
    check("detects >= 6 peaks", len(p) >= 6, f"n={len(p)}")
    # every planted AP has a detected peak within 1 ms
    hit = all(np.min(np.abs(p.t - t)) < 1e-3 for t in times)
    check("all planted APs found", hit)
    check("both polarities present", set(np.unique(p.polarity)) == {-1, 1})

    # flood-triggered average recovers a fixed artifact
    flood = np.arange(0.5, dur, 0.5)
    y = rng.normal(0, 3.0, N)
    art = 50.0 * np.exp(-np.arange(0, int(0.01 * FS)) / (0.002 * FS))
    for f in flood:
        i = int(f * FS)
        y[i:i + art.size] += art
    sta, t_ms, nused, p2p = pk.flood_triggered_average(y, FS, flood,
                                                       pre_ms=5, post_ms=30)
    check("STA n used", nused == flood.size, f"n={nused}")
    check("STA recovers artifact p2p", 40 < p2p < 60, f"p2p={p2p:.1f}µV")


def test_templates_and_classify():
    print("templates + classify")
    # build a solar-like and bio-like template from synthetic snippets
    L = 61
    t_ms = (np.arange(-30, 31) / FS) * 1000.0
    bio_shape = _ap(24, -30.0)
    bio_shape = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, bio_shape.size),
                          bio_shape)
    solar_shape = 40.0 * np.sin(np.linspace(0, np.pi, L))     # broad positive bump
    rng = np.random.default_rng(1)
    bio_snips = np.stack([bio_shape + rng.normal(0, 2, L) for _ in range(50)])
    solar_snips = np.stack([solar_shape + rng.normal(0, 2, L) for _ in range(50)])
    bio_t = tpl.build_template(tpl.BIO, bio_snips, t_ms,
                               np.full(50, -30.0), np.full(50, 0.5), FS)
    solar_t = tpl.build_template(tpl.SOLAR, solar_snips, t_ms,
                                 np.full(50, 40.0), np.full(50, 2.0), FS)
    piezo_t = tpl.build_template(tpl.PIEZO,
                                 np.stack([-solar_shape + rng.normal(0, 2, L)
                                           for _ in range(50)]), t_ms,
                                 np.full(50, -40.0), np.full(50, 1.5), FS)
    ts = tpl.TemplateSet({tpl.BIO: bio_t, tpl.SOLAR: solar_t, tpl.PIEZO: piezo_t})

    # round-trip save/load
    import tempfile, os
    tmp = os.path.join(tempfile.gettempdir(), "_tpl_selftest.npz")
    ts.save(tmp)
    ts2 = tpl.TemplateSet.load(tmp)
    check("template save/load", np.allclose(ts2.get(tpl.BIO).mean_norm,
          bio_t.mean_norm))

    # self-correlation high, cross-correlation lower
    csolar = float(np.atleast_1d(solar_t.correlate(solar_t.mean_norm[None, :]))[0])
    ccross = float(np.atleast_1d(bio_t.correlate(solar_t.mean_norm[None, :]))[0])
    check("solar self-corr ~1", csolar > 0.98, f"c={csolar:.3f}")
    check("bio vs solar low", ccross < 0.6, f"c={ccross:.3f}")

    # two curves per event: laser ON at flood, OFF 50 ms later. A peak at the ON
    # transient AND a peak at the OFF transient must both count as flood-locked.
    on = np.array([1.0, 2.0, 3.0])
    off = on + 0.05                          # 50 ms exposure
    dur = 0.05
    idx = np.arange(7)
    snips = np.stack([solar_shape] * 4 + [bio_shape] * 3)
    amps = np.array([40., 40., 40., 40., -30., -30., -30.])
    widths = np.array([2., 2., 2., 2., 0.5, 0.5, 0.5])
    #   ON peaks at flood+2 ms, one OFF peak at flood+52 ms, 3 bio between floods
    times = np.array([1.002, 2.002, 3.002, 1.052, 1.5, 2.5, 4.7])
    near = np.array([True, True, True, True, False, False, False])
    P = pk.Peaks(idx, times, amps, np.sign(amps).astype(int), widths,
                 snips, t_ms, FS, 3.0)
    analysis = {"lock_pre_ms": 2.0, "lock_post_ms": 25.0, "event_tail_ms": 15.0,
                "corr_min": 0.6, "artifact_corr_min": 0.7, "hp_hz": 100.0}
    res = clf.classify_peaks(P, on, off, dur, ts, analysis, near_solar=near)
    labs = list(res.labels)
    check("ON-transient peaks -> solar (locked, solar shape)",
          labs[:3] == [clf.SOLAR] * 3, str(labs[:3]))
    check("OFF-transient peak (t=+52ms) also locked -> solar",
          labs[3] == clf.SOLAR, str(labs[3]))
    check("3 bio (unlocked) labelled bio",
          labs[4:] == [clf.BIO] * 3, str(labs[4:]))
    # SHAPE backstop: a NON-flood-locked sharp peak matching the solar shape must
    # be rejected as an artifact (not left in bio) — this is the regression fix.
    Pshape = pk.Peaks([0, 1], np.array([4.6, 4.7]),
                      np.array([40., -30.]), np.array([1, -1]),
                      np.array([2.0, 0.5]),
                      np.stack([solar_shape, bio_shape]), t_ms, FS, 3.0)
    rs = clf.classify_peaks(Pshape, on, off, dur, ts, analysis,
                            near_solar=np.array([False, False]))
    check("unlocked solar-shaped peak -> artifact (not bio)",
          rs.labels[0] in (clf.SOLAR, clf.PIEZO), str(rs.labels[0]))
    check("unlocked bio-shaped peak -> bio", rs.labels[1] == clf.BIO,
          str(rs.labels[1]))
    # a locked PIEZO-shaped peak (not near a cell) -> piezo
    Ppz = pk.Peaks([0], np.array([1.002]), np.array([-40.]), np.array([-1]),
                   np.array([2.0]), np.stack([-solar_shape]), t_ms, FS, 3.0)
    rpz = clf.classify_peaks(Ppz, on, off, dur, ts, analysis,
                             near_solar=np.array([False]))
    check("locked piezo-shaped peak -> piezo", rpz.labels[0] == clf.PIEZO,
          str(rpz.labels[0]))

    # event template (onset-aligned, both curves) preserved without sign-flip
    L2 = 121
    biphasic = np.concatenate([np.hanning(20) * 40,        # ON transient (+)
                               np.zeros(60),
                               -np.hanning(20) * 40,       # OFF transient (-)
                               np.zeros(21)])[:L2]
    ev_t = (np.arange(L2) / FS) * 1000.0
    waves = np.stack([biphasic + np.random.randn(L2) * 2 for _ in range(8)])
    ev_tpl = tpl.build_template(tpl.SOLAR, waves, ev_t, np.full(8, 40.0),
                               np.full(8, np.nan), FS, align_polarity=False,
                               aligned="onset")
    check("event template keeps both transients",
          ev_tpl.mean_uv.max() > 20 and ev_tpl.mean_uv.min() < -20,
          f"max={ev_tpl.mean_uv.max():.0f} min={ev_tpl.mean_uv.min():.0f}")
    check("event template flagged onset-aligned", ev_tpl.aligned == "onset")

    # dipolar solar cell: half the electrodes see the biphasic event INVERTED.
    # sign_align must keep both up+down lobes; plain averaging cancels them.
    dip = np.stack([biphasic + np.random.randn(L2) * 2 for _ in range(6)] +
                   [-biphasic + np.random.randn(L2) * 2 for _ in range(6)])
    aligned_tpl = tpl.build_template(tpl.SOLAR, dip, ev_t, np.full(12, 40.0),
                                     np.full(12, np.nan), FS, sign_align=True,
                                     aligned="onset")
    plain_tpl = tpl.build_template(tpl.SOLAR, dip, ev_t, np.full(12, 40.0),
                                   np.full(12, np.nan), FS, align_polarity=False,
                                   aligned="onset")
    check("sign-align keeps both dipole lobes",
          aligned_tpl.mean_uv.max() > 20 and aligned_tpl.mean_uv.min() < -20,
          f"max={aligned_tpl.mean_uv.max():.0f} min={aligned_tpl.mean_uv.min():.0f}")
    check("plain average cancels the dipole",
          abs(plain_tpl.mean_uv).max() < 15,
          f"|max|={abs(plain_tpl.mean_uv).max():.0f}")

    # solar EVENT template: one contact electrode ~19x larger must NOT dominate, and
    # the mean must be oriented down-first (deep down peak leads).
    unit = np.concatenate([-40 * np.hanning(20),       # deep DOWN at onset
                           np.zeros(40),
                           20 * np.hanning(20),        # UP later (OFF transient)
                           np.zeros(L2 - 80)])[:L2]
    typical = np.stack([unit + np.random.randn(L2) * 1.0 for _ in range(8)])
    giant = (unit * 19.0 * -1.0)[None, :]              # 19x AND opposite polarity
    ev_rows = np.vstack([typical, giant])
    ev_amps2 = np.concatenate([np.full(8, 60.0), [1140.0]])
    plain = tpl.build_template(tpl.SOLAR, ev_rows, ev_t, ev_amps2,
                               np.full(9, np.nan), FS, align_polarity=False,
                               aligned="onset", sign_align=True)
    fixed = tpl.build_template(tpl.SOLAR, ev_rows, ev_t, ev_amps2,
                               np.full(9, np.nan), FS, align_polarity=False,
                               aligned="onset", sign_align=True,
                               amp_normalize=True, orient_down=True)
    # plain mean is dominated by the giant (|mean| ~ its scale); normalized is not
    check("amp-normalize stops the 19x contact electrode dominating",
          abs(fixed.mean_uv).max() < 0.5 * abs(plain.mean_uv).max()
          and abs(fixed.mean_uv).max() < 60,
          f"fixed={abs(fixed.mean_uv).max():.0f} plain={abs(plain.mean_uv).max():.0f}")
    # oriented down-first: the largest excursion is negative and early
    ext_i = int(np.argmax(np.abs(fixed.mean_uv)))
    check("solar template oriented down-first (deep down leads)",
          fixed.mean_uv[ext_i] < 0 and fixed.t_ms[ext_i] < 30,
          f"ext t={fixed.t_ms[ext_i]:.1f}ms val={fixed.mean_uv[ext_i]:.1f}")


def test_bio_polarity_and_robust():
    print("bio template polarity + robust cleaning")
    rng = np.random.default_rng(7)
    dur = 8.0
    N = int(FS * dur)
    x = rng.normal(0, 3.0, N)
    _w = 24
    _b = np.exp(-(np.linspace(-1, 1, _w) * 3.0) ** 2)   # narrow bump (~1.2 ms)
    neg = -45.0 * _b                            # negative-going extracellular AP
    pos = +70.0 * _b                            # sharp POSITIVE artifact (up-going)
    for t in np.linspace(0.3, dur - 0.3, 30):   # 30 real (negative) APs
        i = int(t * FS)
        x[i:i + neg.size] += neg
    for t in np.linspace(0.35, dur - 0.35, 10):  # 10 positive contaminants
        i = int(t * FS)
        x[i:i + pos.size] += pos
    raw = _FakeRaw(x[None, :], ["e1"], FS)
    far = np.array([100.0])                      # no floods in range -> all "bio"
    A = dict(hp_hz=100.0, peak_k=5.0, refractory_ms=1.0,
             snippet_pre_ms=1.5, snippet_post_ms=1.5,
             lock_pre_ms=2.0, lock_post_ms=25.0)
    _sB, _tB, aB, _wB = tpl.learn_class_from_recording(
        raw, far, A, locked=False, keep_polarity=0)
    sN, tN, aN, wN = tpl.learn_class_from_recording(
        raw, far, A, locked=False, keep_polarity=-1)
    check("keep both polarities collects the positive contaminants",
          (aB > 0).any(), f"n_pos={(aB > 0).sum()}")
    check("keep_polarity=-1 drops every positive peak",
          aN.size > 0 and not (aN > 0).any(),
          f"max_amp={aN.max() if aN.size else 0:.0f}")
    check("negative-only keeps the real APs", aN.size >= 25, f"n={aN.size}")
    tN_tpl = tpl.build_template(tpl.BIO, sN, tN, aN, wN, FS, robust=True)
    check("clean bio wavelet: trough dominates, no big positive lobe",
          tN_tpl.mean_uv.min() < 0
          and tN_tpl.mean_uv.max() < 0.6 * abs(tN_tpl.mean_uv.min()),
          f"max={tN_tpl.mean_uv.max():.0f} min={tN_tpl.mean_uv.min():.0f}")

    # detector-confirmed path: only peaks the (fake) validated detector confirms
    # enter the bio template — broadband crossings it rejects are dropped
    det_times = np.linspace(0.3, dur - 0.3, 30)[::2]      # confirm half the APs
    sD, tD, aD, wD = tpl.learn_class_from_recording(
        raw, far, A, locked=False, keep_polarity=-1,
        detector=lambda tr, fs: det_times)
    check("detector-confirmed keeps only the confirmed spikes",
          10 <= aD.size <= 20, f"n={aD.size} (30 detected, 15 confirmed)")
    mm = tpl._match_mask(np.array([0.1, 0.2, 0.5]), np.array([0.2]), 0.01)
    check("_match_mask flags only the matched time", list(mm) == [False, True, False],
          str(list(mm)))

    # robust rejection: a few odd-shaped peaks must not blow out the ±SD band
    L = 61
    t_ms = (np.arange(-30, 31) / FS) * 1000.0
    bio = _ap(24, -30.0)
    bio = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, bio.size), bio)
    clean = np.stack([bio + rng.normal(0, 2, L) for _ in range(40)])
    weird = np.stack([  # broad double-lobe artifacts, same (negative) polarity
        (-30.0 * np.sin(np.linspace(0, 2 * np.pi, L)) + rng.normal(0, 2, L))
        for _ in range(8)])
    snips = np.vstack([clean, weird])
    amps = np.concatenate([np.full(40, -30.0), np.full(8, -30.0)])
    widths = np.full(48, 0.5)
    plain = tpl.build_template(tpl.BIO, snips, t_ms, amps, widths, FS, robust=False)
    rob = tpl.build_template(tpl.BIO, snips, t_ms, amps, widths, FS, robust=True,
                             min_corr=0.55, keep_frac=0.6)
    check("robust drops the odd-shaped contaminants",
          40 <= rob.n < 48, f"n={rob.n}")
    check("robust error band tighter than plain average",
          rob.std_uv.max() < plain.std_uv.max(),
          f"rob={rob.std_uv.max():.1f} plain={plain.std_uv.max():.1f}")

    # amplitude-outlier rejection: ONE giant electrode artifact must not blow out
    # the ±SD band (the real-data failure: a 21 mV event took trough SD 9 -> 331 µV)
    amp_scale = rng.uniform(0.6, 1.5, 200)      # realistic spike-amplitude spread
    clean2 = np.stack([bio * a + rng.normal(0, 2, L) for a in amp_scale])
    giant = (bio * 400.0)[None, :]              # one 400x electrode artifact
    snips2 = np.vstack([clean2, giant])
    amps2 = np.concatenate([-30.0 * amp_scale, [-12000.0]])
    w2 = np.full(201, 0.5)
    noamp = tpl.build_template(tpl.BIO, snips2, t_ms, amps2, w2, FS, robust=False)
    withamp = tpl.build_template(tpl.BIO, snips2, t_ms, amps2, w2, FS, robust=True,
                                 amp_mad_k=6.0)
    clean_only = tpl.build_template(tpl.BIO, clean2, t_ms, -30.0 * amp_scale,
                                    np.full(200, 0.5), FS, robust=True,
                                    amp_mad_k=6.0)
    check("amplitude-outlier drops the giant artifact", withamp.n == 200,
          f"n={withamp.n}")
    check("band collapses once the artifact is removed",
          withamp.std_uv.max() < noamp.std_uv.max() / 20.0,
          f"with={withamp.std_uv.max():.1f} without={noamp.std_uv.max():.1f}")
    check("mean matches the artifact-free template (giant had no effect)",
          np.allclose(withamp.mean_uv, clean_only.mean_uv, atol=1e-6),
          f"trough with={withamp.mean_uv.min():.1f} clean={clean_only.mean_uv.min():.1f}")


def test_stim_train():
    print("stim train (pulsed flood)")
    # 10 pulses 25 ms apart = one 250 ms train, then another train 5 s later
    train1 = np.arange(0, 0.25, 0.025)          # 0,25,...,225 ms
    on = np.concatenate([train1, 5.0 + train1])
    starts = pk.train_starts(on, 0.25)
    check("10 pulses collapse to 1 train start (x2 trains)",
          starts.size == 2 and abs(starts[0]) < 1e-9 and abs(starts[1] - 5.0) < 1e-9,
          str(np.round(starts, 3)))
    wins = pk.train_windows(starts, 0.002, 0.25, 0.015)
    # a peak at the LAST pulse (225 ms) and at the first pulse must both be locked
    locked, _ = pk.locked_to_events(np.array([0.001, 0.225, 0.30, 2.5]), wins)
    check("all pulses within train are locked; gaps are not",
          list(locked) == [True, True, False, False], str(list(locked)))

    # classify: a whole 10-transient train -> all artifact; a spontaneous spike
    # between trains -> bio (this is the leaking-artifact regression fix)
    L = 61
    tms = (np.arange(-30, 31) / FS) * 1000.0
    solar_shape = 40.0 * np.sin(np.linspace(0, np.pi, L))
    bio_shape = _ap(24, -30.0)
    bio_shape = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, bio_shape.size),
                          bio_shape)
    ts = tpl.TemplateSet({
        tpl.SOLAR: tpl.build_template(tpl.SOLAR, np.stack([solar_shape] * 20), tms,
                                      np.full(20, 40.), np.full(20, 2.), FS),
        tpl.BIO: tpl.build_template(tpl.BIO, np.stack([bio_shape] * 20), tms,
                                    np.full(20, -30.), np.full(20, 0.5), FS)})
    tt = np.concatenate([train1 + 0.002, [2.5]])     # 10 solar transients + 1 bio
    snips = np.stack([solar_shape] * 10 + [bio_shape])
    amps = np.array([40.] * 10 + [-30.])
    widths = np.array([2.] * 10 + [0.5])
    P = pk.Peaks(np.arange(11), tt, amps, np.sign(amps).astype(int), widths,
                 snips, tms, FS, 3.0)
    A = {"lock_pre_ms": 2.0, "lock_post_ms": 25.0, "event_tail_ms": 15.0,
         "stim_train_ms": 250.0, "corr_min": 0.6, "artifact_corr_min": 0.7}
    res = clf.classify_peaks(P, on, None, 0.0, ts, A, near_solar=np.zeros(11, bool))
    labs = list(res.labels)
    check("all 10 train transients -> artifact (none leak to bio)",
          all(l in (clf.SOLAR, clf.PIEZO) for l in labs[:10]), str(labs[:10]))
    check("spontaneous spike between trains -> bio", labs[10] == clf.BIO,
          str(labs[10]))


def test_seqfile():
    print("seq reader")
    import struct, seqfile, tempfile, os
    # synthesise the header: magic, frames=10, a=1000, on=15000, @16=0, off=10000
    hdr = b"1QSM" + struct.pack("<IIIII", 10, 1000, 15000, 0, 10000) + b"\x00" * 40
    p = os.path.join(tempfile.gettempdir(), "_t.seq")
    open(p, "wb").write(hdr)
    info = seqfile.read_seq(p)
    check("seq frames == 10", info and info["frames"] == 10, str(info))
    check("seq total == 250 ms", abs(info["total_ms"] - 250.0) < 1e-6,
          f"{info['total_ms']}")


def test_filtering():
    print("filtering")
    rng = np.random.default_rng(2)
    dur = 10.0
    N = int(FS * dur)
    x = rng.normal(0, 3.0, N)
    # artifact every 0.5 s (flood-locked)
    flood = np.arange(0.4, dur, 0.5)
    art = 60.0 * np.exp(-np.arange(0, int(0.008 * FS)) / (0.0015 * FS))
    for f in flood:
        i = int(f * FS)
        x[i:i + art.size] += art
    # a biological spike NOT locked to floods, in a gap
    ap = _ap(24, -45.0)
    bio_t = 0.25
    x[int(bio_t * FS):int(bio_t * FS) + ap.size] += ap
    sta, t_ms, n, p2p = pk.flood_triggered_average(x, FS, flood,
                                                   pre_ms=3, post_ms=15)
    resid = flt.subtract_flood_artifact(x, FS, flood, sta, t_ms)
    # residual near a flood should be much smaller than raw
    i = int(flood[5] * FS)
    raw_pp = np.ptp(x[i:i + art.size])
    res_pp = np.ptp(resid[i:i + art.size])
    check("artifact suppressed >3x", res_pp < raw_pp / 3.0,
          f"raw={raw_pp:.0f} res={res_pp:.0f}")
    # the biological spike survives (check its full excursion, either polarity)
    j = int(bio_t * FS)
    check("bio spike preserved", np.max(np.abs(resid[j:j + ap.size])) > 35.0,
          f"|max|={np.max(np.abs(resid[j:j+ap.size])):.0f}")


class _FakeRaw:
    """Minimal RawRecording stand-in for the loader-free matrix-filter test."""
    def __init__(self, mat, labels, fs):
        self._mat = np.asarray(mat, np.float32)
        self.labels = [str(l) for l in labels]
        self.fs = float(fs)
        self.n_samples = self._mat.shape[1]
        self.duration = self._mat.shape[1] / float(fs)
        self._index = {l: i for i, l in enumerate(self.labels)}

    def channel(self, lab):
        return self._mat[self._index[str(lab)]].astype(float)

    def matrix(self):
        return self._mat

    def free(self):
        pass


def test_matrix_filter():
    print("matrix filter (batch path)")
    rng = np.random.default_rng(5)
    dur = 6.0
    N = int(FS * dur)
    labels = ["a", "b", "c"]                      # scrambled order under test
    flood = np.arange(0.4, dur, 0.5)
    art = 60.0 * np.exp(-np.arange(0, int(0.008 * FS)) / (0.0015 * FS))
    mat = rng.normal(0, 3.0, (3, N)).astype(np.float32)
    for f in flood:                              # channel "b" = flood artifact
        i = int(f * FS)
        mat[1, i:i + art.size] += art.astype(np.float32)
    raw = _FakeRaw(mat, labels, FS)
    A = {"lock_pre_ms": 2.0, "event_tail_ms": 15.0, "stim_train_ms": 250.0,
         "hp_hz": 100.0, "reject_flood_locked": False}   # testing ordering, not rejection
    seen = {"done": 0, "total": 0}

    def det_mat(m, fs_, progress=None):          # 1 spike/row at t=row -> ordering
        out = []
        for k in range(m.shape[0]):
            if progress:
                progress(k + 1)
            out.append(np.array([float(k)]))
        return out

    def outer_prog(done, total):                 # GUI-style progress(done, total)
        seen["done"], seen["total"] = done, total

    res = flt.filter_recording_matrix(raw, flood, A, det_mat, progress=outer_prog)
    check("matrix: per-row spikes mapped in label order",
          all(res.bio_spikes[labels[k]][0] == k for k in range(3)),
          str({l: res.bio_spikes[l].tolist() for l in labels}))
    check("matrix: flood channel flagged & subtracted, clean ones not",
          "b" in res.artifact_labels and "a" not in res.artifact_labels
          and "c" not in res.artifact_labels, str(sorted(res.artifact_labels)))
    check("matrix: batched detector progress reached the GUI (done/total)",
          seen["done"] == 3 and seen["total"] == 3, str(seen))
    check("matrix: total spike count == n channels", res.total_spikes() == 3,
          str(res.total_spikes()))


def test_meanap_export():
    print("MEA-NAP spike export")
    import loaders
    import tempfile
    import os
    saved = {}

    class FakeIO:
        CONFIG = {}

        def save_spike_file(self, stem, spike_times, labels, fs, params,
                            auto_ground=None, noise_uv=None):
            saved.update(stem=stem, spike_times=list(spike_times),
                         labels=list(labels), fs=fs, params=dict(params),
                         noise_uv=(None if noise_uv is None
                                   else np.asarray(noise_uv, float)),
                         auto_ground=list(auto_ground or []),
                         spike_dir=self.CONFIG.get("SPIKE_DIR"))
            path = os.path.join(self.CONFIG["SPIKE_DIR"], stem + "_spikes.npz")
            return path, path[:-4] + ".mat"

    class FakeDet:
        DETECT_PARAMS = {"bp_low_hz": 600.0, "bp_high_hz": 8000.0}

    loaders._MODS.clear()
    loaders._MODS.update(io=FakeIO(), det=FakeDet())
    try:
        d = tempfile.gettempdir()
        spikes = {"47": np.array([0.1, 0.2]), "12": np.array([0.5])}
        path = loaders.save_meanap_spike_file(
            d, "rec1", spikes, 20000.0,
            noise_by_label={"47": 3.0, "12": 4.0},
            extra_params={"artifact_labels": ["47"]})
        check("export filename is <stem>_spikes.npz",
              os.path.basename(path) == "rec1_spikes.npz", path)
        check("labels follow the dict order", saved["labels"] == ["47", "12"],
              str(saved["labels"]))
        check("spike times passed through (seconds)",
              np.allclose(saved["spike_times"][0], [0.1, 0.2])
              and np.allclose(saved["spike_times"][1], [0.5]))
        check("noise_uv aligned to labels",
              np.allclose(saved["noise_uv"], [3.0, 4.0]), str(saved["noise_uv"]))
        check("params carry fs + source + extra",
              saved["params"]["fs"] == 20000.0 and "source" in saved["params"]
              and saved["params"].get("artifact_labels") == ["47"])
        check("SPIKE_DIR pointed at the chosen folder during write",
              saved["spike_dir"] == d, str(saved["spike_dir"]))
        check("SPIKE_DIR restored (None) after write",
              loaders._MODS["io"].CONFIG.get("SPIKE_DIR") is None)
    finally:
        loaders._MODS.clear()


def test_threshold_gate():
    print("prefilter threshold gate")
    rng = np.random.default_rng(11)
    dur = 4.0
    N = int(FS * dur)
    x = rng.normal(0, 3.0, N)
    big_t = np.array([0.5, 1.5, 2.5])          # large (biological-size) spikes
    small_t = np.array([1.0, 2.0, 3.0])         # tiny sub-threshold blips
    big = _ap(24, -70.0)
    small = _ap(24, -4.0)
    for t in big_t:
        i = int(t * FS)
        x[i:i + big.size] += big
    for t in small_t:
        i = int(t * FS)
        x[i:i + small.size] += small
    all_times = np.sort(np.concatenate([big_t, small_t])) + 0.0003
    kept = flt.threshold_gate(x, FS, all_times, low_hz=200.0, high_hz=3000.0,
                              order=2, k=5.0)
    check("gate keeps the large spikes", kept.size == 3
          and all(np.min(np.abs(kept - t)) < 1e-3 for t in big_t),
          f"n_kept={kept.size}")
    check("gate drops the sub-threshold blips",
          all(np.min(np.abs(kept - t)) > 1e-3 for t in small_t), str(kept))
    check("gate on empty input is safe",
          flt.threshold_gate(x, FS, np.array([]), k=5.0).size == 0)


def test_model_field():
    print("modeled solar field")
    arr = geo.ElectrodeArray(pitch_um=200.0)
    cell = geo.SolarCell(center_grid=(4.5, 4.5), side_um=200.0, pitch_um=200.0)
    Z, ext = fm.model_field_grid(arr, [cell], amp0_uv=80.0, lam_um=300.0, n=60)
    check("field grid shape", Z.shape == (60, 60), str(Z.shape))
    check("field peaks near the cell (== amp0 inside)",
          abs(Z.max() - 80.0) < 1e-6, f"max={Z.max():.1f}")
    check("field decays with distance", Z.min() < 0.5 * Z.max(),
          f"min={Z.min():.1f} max={Z.max():.1f}")
    # a farther point has a smaller field than a nearer one
    x0, x1, y0, y1 = ext
    xs = np.linspace(x0, x1, 60)
    ys = np.linspace(y0, y1, 60)
    cx = np.argmin(np.abs(xs - 4.5))
    near = Z[np.argmin(np.abs(ys - 4.5)), cx]
    far = Z[np.argmin(np.abs(ys - y0)), cx]
    check("nearer > farther field", near > far, f"near={near:.1f} far={far:.1f}")


def test_bursts():
    print("burst detection")
    import bursts
    fs = 1.0
    # two clear bursts (tight ISI) separated by a long gap, plus isolated spikes
    b1 = np.arange(0, 0.5, 0.02)           # 25 spikes @ 20 ms ISI -> one burst
    b2 = 10.0 + np.arange(0, 0.4, 0.02)    # 20 spikes -> one burst
    isolated = np.array([3.0, 5.0, 7.0])   # lone spikes, no burst
    t = np.concatenate([b1, isolated, b2])
    bs = bursts.detect_bursts(t, max_isi_s=0.1, min_spikes=5)
    check("detects the two bursts", len(bs) == 2, "n=%d" % len(bs))
    check("burst spans correct", abs(bs[0][0] - 0.0) < 1e-6 and bs[0][1] <= 0.5,
          str(np.round(bs[0], 3)))
    check("isolated spikes make no burst",
          bursts.burst_count(isolated, 0.1, 5) == 0)
    check("too-few-spikes run is not a burst",
          bursts.burst_count(np.array([0.0, 0.02, 0.04]), 0.1, 5) == 0)
    # burst rate per minute over 60 s = 2 bursts -> 2/min
    check("burst rate per minute", abs(
        bursts.burst_rate_per_min(t, 60.0, 0.1, 5) - 2.0) < 1e-6,
        "%.2f" % bursts.burst_rate_per_min(t, 60.0, 0.1, 5))


def test_auto_stim_span():
    print("auto stim-span detection")
    import pipeline
    rng = np.random.default_rng(21)
    dur = 30.0
    N = int(FS * dur)
    starts = np.array([1.0, 6.0, 11.0, 16.0])       # 4 stims, 5 s apart
    trans = _ap(16, -300.0)                           # a sharp transient

    def make(n_pulses, period_ms):
        x = rng.normal(0, 4.0, N)
        for s in starts:
            for k in range(n_pulses):
                i = int((s + k * period_ms * 1e-3) * FS)
                x[i:i + trans.size] += trans
        return x

    class _R:
        def __init__(self, x):
            self._x = x
            self.labels = ["a", "b", "c", "d"]
            self.fs = FS
            self.duration = dur

        def channel(self, l):
            return self._x

    A = {"hp_hz": 100.0}
    # 10-pulse, 25 ms period train -> last transient at 225 ms -> span ~0.225 s
    sp_train = pipeline.auto_stim_span(_R(make(10, 25.0)), starts, A)
    check("train span ~ last transient (0.2-0.28 s)",
          sp_train is not None and 0.2 <= sp_train <= 0.28, "%.3f" % (sp_train or 0))
    # single pulse -> very short span
    sp_single = pipeline.auto_stim_span(_R(make(1, 25.0)), starts, A)
    check("single-pulse span is short (< 30 ms)",
          sp_single is not None and sp_single < 0.03, "%.3f" % (sp_single or 0))
    # a longer 20-pulse train -> longer span than the 10-pulse
    sp_long = pipeline.auto_stim_span(_R(make(20, 25.0)), starts, A)
    check("longer train -> longer detected span", sp_long > sp_train,
          "long=%.3f vs train=%.3f" % (sp_long or 0, sp_train or 0))
    check("no events -> None",
          pipeline.auto_stim_span(_R(make(10, 25.0)), np.array([]), A) is None)
    # regression: a GIANT onset transient (like a 4 mV contact electrode) must not
    # collapse the span — its non-causal filter ringing bleeds into the pre-window,
    # which is why the baseline must come from the post-artifact tail, not the pre.
    def make_giant():
        x = rng.normal(0, 4.0, N)
        for s in starts:
            for kk in range(10):
                i = int((s + kk * 25.0 * 1e-3) * FS)
                amp = -6000.0 if kk == 0 else -300.0     # 20x first pulse
                x[i:i + 16] += _ap(16, amp)
        return x
    sp_giant = pipeline.auto_stim_span(_R(make_giant()), starts, A)
    check("giant onset transient does NOT collapse the span",
          sp_giant is not None and sp_giant > 0.2, "%.3f" % (sp_giant or 0))


def test_reject_flood_locked():
    print("flood-locked rejection (filter residual artifacts)")
    # trains at 0 and 5 s, 250 ms each. An artifact electrode has residual "spikes"
    # inside the train windows PLUS real biology between trains; a clean electrode has
    # only biology and must be left untouched.
    A = {"lock_pre_ms": 2.0, "stim_train_ms": 250.0, "event_tail_ms": 15.0,
         "reject_flood_locked": True}
    starts = np.array([0.0, 5.0])
    inside = np.array([0.01, 0.12, 0.24, 5.05, 5.20])     # within train windows
    between = np.array([1.0, 2.0, 3.0, 6.5, 8.0])          # real biology
    bio = {"art": np.sort(np.concatenate([inside, between])),
           "clean": between.copy()}
    flt.reject_flood_locked(bio, ["art"], starts, A)
    check("artifact electrode: flood-locked residuals dropped",
          np.allclose(np.sort(bio["art"]), np.sort(between)),
          f"left={np.round(bio['art'],2)}")
    check("clean (unflagged) electrode untouched",
          np.allclose(bio["clean"], between))
    # disabled -> no change
    bio2 = {"art": np.sort(np.concatenate([inside, between]))}
    flt.reject_flood_locked(bio2, ["art"], starts,
                            {**A, "reject_flood_locked": False})
    check("respects the disable flag",
          bio2["art"].size == inside.size + between.size)


def test_fieldmap():
    print("fieldmap")
    arr = geo.ElectrodeArray(pitch_um=200.0)
    sc = geo.SolarCell(center_grid=(4.5, 4.5), side_um=400.0, pitch_um=200.0)
    d = sc.distance_um(arr.grid)
    lam_true = 300.0
    resp = 80.0 * np.exp(-d / lam_true) + 2.0
    resp_by = {lab: resp[i] for i, lab in enumerate(arr.labels)}
    r = fm.compute_field_map(arr, resp_by, [sc], floor_uv=5.0)
    check("recovers lambda ~300µm", abs(r.fit[1] - lam_true) < 80.0,
          f"lam={r.fit[1]:.0f}µm")
    check("effective radius positive", r.r_eff > 0, f"r_eff={r.r_eff:.0f}µm")
    check("n_cells == 1", r.n_cells == 1)
    Z, ext = r.interpolate(arr, n=40)
    check("field map interpolates", Z.shape == (40, 40))
    # interpolate must not crash when fewer than 60 electrodes have a response
    resp_sub = {lab: resp_by[lab] for lab in arr.labels[:55]}
    rsub = fm.compute_field_map(arr, resp_sub, [sc], floor_uv=5.0)
    Zs, _ = rsub.interpolate(arr, n=30)
    check("interpolate ok with 55/60 electrodes", Zs.shape == (30, 30),
          f"n_labels={len(rsub.labels)}")

    # two cells: distance is to the nearest; response between them stays high
    sc2 = geo.SolarCell(center_grid=(2.5, 4.5), side_um=400.0, pitch_um=200.0)
    d2, _ = geo.nearest_cell_distance_um([sc, sc2], arr.grid)
    resp2 = 80.0 * np.exp(-d2 / lam_true) + 2.0
    resp2_by = {lab: resp2[i] for i, lab in enumerate(arr.labels)}
    r2 = fm.compute_field_map(arr, resp2_by, [sc, sc2], floor_uv=5.0)
    check("two-cell map n_cells == 2", r2.n_cells == 2)
    check("two-cell total > one-cell total", r2.total_uv > r.total_uv,
          f"{r2.total_uv:.0f} vs {r.total_uv:.0f}")
    # baseline subtraction lowers response
    base_by = {lab: 10.0 for lab in arr.labels}
    rb = fm.compute_field_map(arr, resp_by, [sc], floor_uv=5.0,
                              baseline_by_label=base_by)
    check("baseline subtraction reduces peak", rb.peak_uv < r.peak_uv,
          f"{rb.peak_uv:.0f} vs {r.peak_uv:.0f}")


def test_per_cell_max():
    """Each electrode is claimed by its NEAREST solar cell; each cell reports the
    biggest value among the electrodes it claimed (what the maps annotate)."""
    print("per-cell max voltage")
    arr = geo.ElectrodeArray(pitch_um=200.0)
    # three cells across the array, well separated
    cells = [geo.SolarCell(center_grid=c, side_um=300.0, pitch_um=200.0)
             for c in ((2.0, 2.0), (4.5, 6.5), (7.0, 3.0))]
    # plant a distinct, unmistakable maximum on one electrode per cell
    planted = {"27": 900.0, "53": 1500.0, "76": 300.0}   # label "CR" -> grid (C, 9-R)
    vals = {lab: 10.0 for lab in arr.labels}
    vals.update(planted)
    recs = geo.max_by_cell(cells, vals, arr)
    check("one record per cell", len(recs) == 3, f"n={len(recs)}")
    got = {r["label"] for r in recs}
    check("each cell finds its own planted peak", got == set(planted),
          f"got={sorted(got)}")
    by_lab = {r["label"]: r["value"] for r in recs}
    check("values match the planted maxima",
          all(abs(by_lab[l] - v) < 1e-9 for l, v in planted.items()),
          str(by_lab))
    check("every electrode is assigned to some cell",
          sum(r["n"] for r in recs) == len(arr.labels),
          "%d vs %d" % (sum(r["n"] for r in recs), len(arr.labels)))
    # non-finite values are skipped, not propagated
    vals2 = dict(vals)
    vals2["27"] = float("nan")
    recs2 = geo.max_by_cell(cells, vals2, arr)
    check("NaN electrode ignored", np.isfinite(recs2[0]["value"])
          and recs2[0]["label"] != "27", str(recs2[0]))
    # a distance cut-off can starve a far cell
    recs3 = geo.max_by_cell(cells, vals, arr, max_dist_um=1.0)
    check("distance cut-off reduces assignments",
          sum(r["n"] for r in recs3) < len(arr.labels),
          "%d assigned" % sum(r["n"] for r in recs3))
    check("no cells -> empty list", geo.max_by_cell([], vals, arr) == [])
    # FieldMapResult exposes the same numbers from its own assignment
    fmap = fm.compute_field_map(arr, vals, cells, floor_uv=5.0)
    peaks = fmap.cell_peaks
    check("FieldMapResult.cell_peaks agrees with max_by_cell",
          all(abs(peaks[i] - recs[i]["value"]) < 1e-9 for i in range(3)),
          str(peaks))
    check("cell_peak_labels agrees too",
          fmap.cell_peak_labels() == [r["label"] for r in recs],
          str(fmap.cell_peak_labels()))


def test_plateau():
    """flood_plateau recovers the SUSTAINED held level during illumination and
    ignores the fast capacitive edge transient; by_magnitude aggregation keeps sign.
    """
    print("sustained plateau")
    rng = np.random.default_rng(3)
    fs = FS
    dur = 8.0
    N = int(fs * dur)
    on_dur_s = 0.020                     # 20 ms illumination pulses
    pulses = np.arange(0.5, dur - 0.1, 0.5)
    # trace = held DC step of -120 µV during each pulse + a big fast edge spike
    y = rng.normal(0, 2.0, N)
    edge = 2000.0 * np.exp(-np.arange(0, int(0.003 * fs)) / (0.0004 * fs))  # ~0.4ms
    for f in pulses:
        i0 = int(f * fs)
        i1 = int((f + on_dur_s) * fs)
        y[i0:i1] += -120.0               # sustained plateau (held, DC)
        y[i0:i0 + edge.size] += edge     # fast onset transient on top
    plat, sta, t_ms, n = pk.flood_plateau(y, fs, pulses, on_dur_s, edge_ms=1.5)
    check("plateau uses all pulses", n == pulses.size, f"n={n}")
    check("plateau recovers the held level (~-120µV, not the 2mV spike)",
          -140.0 < plat < -100.0, f"plateau={plat:.1f}µV")
    _sta2, _t, _n2, p2p = pk.flood_triggered_average(y, fs, pulses, pre_ms=5,
                                                     post_ms=on_dur_s * 1e3 + 2)
    check("edge-spike p2p is far bigger than |plateau|", p2p > 8 * abs(plat),
          f"p2p={p2p:.0f} vs plateau={plat:.0f}")

    # a PURE fast transient with no sustained level -> plateau ~ 0
    y2 = rng.normal(0, 2.0, N)
    for f in pulses:
        i0 = int(f * fs)
        y2[i0:i0 + edge.size] += edge
    plat2, _s, _t, _n = pk.flood_plateau(y2, fs, pulses, on_dur_s, edge_ms=1.5)
    check("no held level -> plateau near 0", abs(plat2) < 15.0, f"plateau={plat2:.1f}")

    # unknown duration -> undefined plateau
    platx, _s, _t, _n = pk.flood_plateau(y, fs, pulses, 0.0)
    check("dur<=0 -> NaN plateau", not np.isfinite(platx), str(platx))

    # by_magnitude keeps the strongest deflection of EITHER polarity, with its sign
    arr = geo.ElectrodeArray(pitch_um=200.0)
    cell = [geo.SolarCell(center_grid=(4.5, 4.5), side_um=300.0, pitch_um=200.0)]
    signed = {lab: 5.0 for lab in arr.labels}
    signed["45"] = -300.0                # biggest magnitude, negative
    signed["55"] = 200.0                 # smaller magnitude, positive
    rec_mag = geo.max_by_cell(cell, signed, arr, by_magnitude=True)[0]
    check("by_magnitude picks largest |value| keeping sign",
          rec_mag["label"] == "45" and rec_mag["value"] == -300.0, str(rec_mag))
    rec_signed = geo.max_by_cell(cell, signed, arr, by_magnitude=False)[0]
    check("signed max picks the largest positive value",
          rec_signed["value"] == 200.0, str(rec_signed))


def test_footprint():
    print("footprint")
    arr = geo.ElectrodeArray(pitch_um=200.0)
    sph = geo.Spheroid(center_grid=(4.0, 5.0), radius_um=150.0, pitch_um=200.0)
    d = sph.distance_um(arr.grid)
    rate = 5.0 * np.exp(-(d / 350.0) ** 2)      # activity peaks at the soma
    rate_by = {lab: rate[i] for i, lab in enumerate(arr.labels)}
    r = fp.compute_footprint(arr, rate_by, spheroid=sph,
                             active_rate_hz=0.2, active_frac_of_max=0.1)
    check("some active electrodes", r.n_active() >= 3, f"n={r.n_active()}")
    cg_um = r.centroid_grid * arr.pitch_um
    err = np.linalg.norm(cg_um - sph.center_um)
    check("centroid near soma (<200µm)", err < 200.0, f"err={err:.0f}µm")
    check("reach finite", 0 < r.reach_um < 3000, f"reach={r.reach_um:.0f}µm")


if __name__ == "__main__":
    for fn in (test_geometry, test_peaks, test_templates_and_classify,
               test_bio_polarity_and_robust, test_stim_train, test_seqfile,
               test_filtering, test_matrix_filter, test_meanap_export,
               test_threshold_gate, test_model_field, test_bursts,
               test_auto_stim_span, test_reject_flood_locked,
               test_fieldmap, test_per_cell_max, test_plateau, test_footprint):
        fn()
    print()
    if _fail:
        print("FAILED:", _fail)
        raise SystemExit(1)
    print("ALL SELFTESTS PASSED")
