"""Data layer: .h5 discovery, duration, triggers, filtered trace, bursts.
Run:  python tests/test_data.py   (from the MEA Explorer folder)"""
import _env                                   # noqa: F401  (path + stub setup)
import os
import time
import numpy as np
import datasource as ds
import mea_nap_io as io
from meanap.pipeline import burst_detection as bd


def main():
    print("stub DATA_DIR (stale):", io.CONFIG["DATA_DIR"])
    rec = ds.Recording("burstrec")
    print("resolved h5          :", rec.h5_path)
    assert rec.h5_path and rec.h5_path.endswith(os.path.join("MEA 071826", "H5", "burstrec.h5"))
    assert abs(rec.duration - 120.0) < 1e-9, rec.duration
    on = rec.trigger_on_times(); assert on.size == 4, on
    print("duration %.1f s, %d triggers, band %s, order %d" % (
        rec.duration, on.size, rec.filter_band(), ds.TRACE_FILTER_ORDER))
    assert rec.filter_band() == tuple(float(v) for v in ds.TRACE_FILTER_BAND)

    # ---- filtered trace --------------------------------------------------
    lab = "21"
    w = rec.filtered_window(lab, 10.0, 12.0, max_points=2000)
    full, fs = w["full"], w["fs"]
    assert full.size == int(2.0 * fs) and w["t"].size <= 4000
    mcs = rec._ensure_mcs(); i = mcs.labels.index(lab)
    raw = mcs.get_channel_window(i, int(10 * fs), int(12 * fs))

    def bandpower(x, f0):
        X = np.abs(np.fft.rfft(x * np.hanning(x.size)))
        f = np.fft.rfftfreq(x.size, 1 / fs)
        return X[np.argmin(abs(f - f0))]
    assert bandpower(full, 50) < 0.02 * bandpower(raw, 50), "mains not removed"
    st = rec.spike_times_of(lab); st = st[(st >= 10.02) & (st < 11.98)]
    vals = [full[int(round(t * fs)) - w["s0"]] for t in st]
    assert np.mean(vals) < -3 * np.median(np.abs(full)) / 0.6745, "troughs not at spike times"
    edge = np.abs(full[:200]).max(); mid = np.percentile(np.abs(full), 99.5)
    assert edge < 3 * mid, "filtfilt edge transient"
    # windowed result == filtering the whole channel (padding is sufficient)
    from scipy import signal
    whole = np.asarray(mcs.get_channel_window(i, 0, mcs.n_samples), float)
    sos = signal.butter(ds.TRACE_FILTER_ORDER, rec.filter_band(), btype="bandpass", fs=fs, output="sos")
    ref = signal.sosfiltfilt(sos, whole)[int(10 * fs):int(12 * fs)]
    err = np.max(np.abs(full - ref)) / np.max(np.abs(ref))
    assert err < 1e-6, err
    print("filtered trace       : mains removed, troughs aligned, no edge transient, matches whole-channel filter (%.1e)" % err)

    # ---- bursts: identical to the pipeline --------------------------------
    for lab in rec.labels[:12]:
        t = rec.spike_times_of(lab)
        r = ds.detect_bursts(t, rec.fs, 5, "automatic", rec.duration)
        m = bd.single_channel_burst_detection({0: t}, 1, rec.fs, min_spikes=5,
                                              isi_threshold="automatic",
                                              recording_duration_s=rec.duration)
        bm = m["burst_matrices"][0]
        assert np.allclose(r["t_start"], np.asarray(bm["T_start"], float))
        assert np.allclose(r["t_end"], np.asarray(bm["T_end"], float))
        assert np.array_equal(r["size"], np.asarray(bm["S"], int))
        if len(m["bursting_units"]):
            assert abs(r["burst_rate_per_min"] - m["all_burstRates"][0]) < 1e-9
            assert abs(r["frac_in_bursts"] - m["all_fracsInBursts"][0]) < 1e-12
        for b in range(r["t_start"].size):
            mem = r["spike_times"][r["spike_burst"] == b]
            assert mem.size == r["size"][b] and mem[0] == r["t_start"][b] and mem[-1] == r["t_end"][b]
    print("bursts               : identical to single_channel_burst_detection on 12 electrodes")
    r = ds.detect_bursts(rec.spike_times_of("21"), rec.fs, 5, "automatic", rec.duration)
    print("electrode 21         : %d spikes, %d bursts, ISI_N %.1f ms (auto)" % (
        r["n_spikes"], r["t_start"].size, r["isin_th"] * 1000))
    rm = ds.detect_bursts(rec.spike_times_of("21"), rec.fs, 5, 0.02, rec.duration)
    assert not rm["auto"] and abs(rm["isin_th"] - 0.02) < 1e-12
    for n in (0, 3):
        assert ds.detect_bursts(np.sort(np.random.rand(n)), rec.fs, 5, "automatic", 10)["t_start"].size == 0
    big = np.sort(np.random.default_rng(1).uniform(0, 3600, 200000))
    t0 = time.time(); rb = ds.detect_bursts(big, 25000.0, 5, "automatic", 3600)
    print("perf                 : 200k spikes -> %.2f s (%d bursts)" % (time.time() - t0, rb["t_start"].size))
    print("\nDATA LAYER: ALL PASS")


if __name__ == "__main__":
    main()
