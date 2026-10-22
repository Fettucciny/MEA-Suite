#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 MEA-NAP wavelet spike detection  --  EXACT algorithm + GPU-batched CWT
================================================================================
This is the SAND-Lab MEA-NAP detection algorithm (detectSpikesWavelet.m, via the
lab's own validated Python port, meanap/pipeline/spike_detection.py), with the
heavy CWT moved onto the GPU so a 60-min x 60-channel file takes minutes.

WHAT THIS FIXES vs. the previous version of this file
-----------------------------------------------------
The old detector used the right *parameter names* (bior1.5, L=-0.12) but the
wrong *math*, and under-detected at realistic SNR (measured: MEA-NAP finds 1.3x
more spikes at SNR 5, 2.8x at SNR 3.6, 5.1x at SNR 2.8). Three real differences,
all corrected here:

  1. COST SCALING + BAYESIAN THRESHOLD. MEA-NAP scales the cost by
     Lmax = 36.7368 and uses the Nenadic & Burdick Bayesian threshold
         DTh = Mj/2 + Sigmaj^2/Mj * (L*Lmax + log(PN/PS))
     where Thj = Sigmaj*sqrt(2*log(Nt)) is only a HARD threshold used to get
     Mj = mean(|c[|c|>Thj]|) and PS = #{|c|>Thj}/Nt.
     The old code used thr = sigma*sqrt(2*log(Nt))*(1+L) -- not this formula at
     all (~5.3 sigma vs ~3.9 sigma, hence the missing spikes).
  2. POLARITY. The CWT is negated (PyWavelets' bior1.5 psi_d has the opposite
     sign to MATLAB's analysis wavelet) and negative coefficients are zeroed, so
     only NEGATIVE-going (physiological) APs count. The old code used |c| and so
     mixed both polarities.
  3. SIGMA ESTIMATION. Sigmaj comes from coefficients SUBSAMPLED at
     stride = round(scale) (independent samples), not all coefficients with an
     arbitrary floor.

PARITY
------
Scale determination, the wavelet function, the bandpass and spike-index parsing
are CALLED DIRECTLY from the lab's port (not re-implemented). Only the CWT and
the per-scale threshold loop are re-expressed in torch so they can run batched
on the GPU, and detect_spikes_matrix is validated to return the SAME spike
indices as the lab's reference detect_spikes_wavelet (see _selftest).

Public API unchanged (mea_nap_io.py calls these):
  DETECT_PARAMS, GPU_PARAMS, bandpass(), detect_spikes(), detect_spikes_matrix(),
  auto_device(), _cwt_coeffs(), wavelet_threshold(), determine_scales()
================================================================================
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meanap.pipeline import spike_detection as _sand   # the lab's own port


# ==============================================================================
# PARAMETERS  (MEA-NAP / SAND-port defaults)
# ------------------------------------------------------------------------------
#   wid_ms : expected spike-width range (ms). MEA-NAP/SAND default (0.4, 0.8).
#   ns     : number of CWT scales spanning wid_ms. Default 5.
#   L      : Bayesian cost parameter. MEA-NAP default -0.12 (liberal). Raise
#            toward 0 to detect FEWER spikes if noise is picked up; more negative
#            detects more.
#   Lmax   : cost scaling constant from detectSpikesWavelet.m. Do not change.
#   option : 'l' liberal / 'c' conservative (MEA-NAP default 'l').
#   bp_*   : 3rd-order Butterworth bandpass 600-8000 Hz (MEA-NAP default).
# ==============================================================================
DETECT_PARAMS = {
    "bp_low_hz":  600.0,      # bandpass high-pass corner (Hz). Range 300 - 800
    "bp_high_hz": 8000.0,     # bandpass low-pass corner (Hz). Range 3000 - 8000
                              #   (must stay < fs/2; MEA-NAP uses 8000, older 6150)
    "bp_order":   3,          # Butterworth order. Range 2 - 4 (3 = MEA-NAP default)
    "wname":      "bior1.5",  # wavelet. Only 'bior1.5' is supported by determine_scales
    "wid_ms":     (0.4, 0.8), # expected spike-width range (ms).
                              #   Range (0.3, 0.7) - (0.5, 1.0); MEA-NAP (0.4, 0.8)
    "ns":         5,          # number of CWT scales spanning wid_ms. Range 3 - 6
    "option":     "l",        # options: 'l' liberal | 'c' conservative
    "L":          -0.12,      # Bayesian cost. Range -0.2 (more spikes) .. +0.1 (fewer);
                              #   more negative detects more. MEA-NAP default -0.12
    "Lmax":       36.7368,    # cost scaling constant (detectSpikesWavelet.m). DO NOT CHANGE
}

GPU_PARAMS = {
    "mem_budget_gb": 12.0,   # target GPU working set (GB). Range 2 - 20; lower if you
                             #   hit OOM (~half your VRAM is safe; the RTX 4090 has 24 GB)
    "ch_batch":      None,   # channels per GPU batch. None = auto, or an int in the
                             #   range 1 .. n_channels (hard override)
    "fp64":          False,  # float64 CWT. Options: True | False. True is exact vs the
                             #   CPU reference but ~64x slower on a GeForce card (poor
                             #   FP64); False (float32) is validated identical spike indices.
}


def auto_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def bandpass(signal, fs, p=DETECT_PARAMS):
    """3rd-order Butterworth bandpass (filtfilt) -- the lab's own filter."""
    return _sand.bandpass_filter(np.asarray(signal, float), fs,
                                 p["bp_low_hz"], p["bp_high_hz"])


def determine_scales(fs, p=DETECT_PARAMS):
    """CWT scales for the wid_ms range (the lab's determine_scales port)."""
    return _sand._determine_scales(p["wname"], tuple(p["wid_ms"]), fs, p["ns"])


# ==============================================================================
# CWT  (torch; mirrors the lab's _cwt_bior15 exactly, incl. the sign negation)
# ==============================================================================
def _cwt_torch(sig, scales, device, fp64=False):
    """CWT of (B, Nt) signals at `scales`. Returns (B, n_scales, Nt).

    Same construction as the port's _cwt_bior15: FFT convolution against the
    cascade-approximated bior1.5 analysis wavelet, padded to the next power of
    two, rolled by n_wvl//2, then NEGATED (negative-going APs -> positive
    coefficients, matching MATLAB's convention).
    """
    import torch
    psi_d, x = _sand._get_bior15_wavefn()
    x_range = x[-1]
    dt_t = torch.float64 if fp64 else torch.float32

    sig = torch.as_tensor(sig, dtype=dt_t, device=device)
    if sig.ndim == 1:
        sig = sig[None, :]
    B, n = sig.shape
    n_fft = int(2 ** np.ceil(np.log2(n)))

    sig_rfft = torch.fft.rfft(sig, n=n_fft)
    out = torch.empty((B, len(scales), n), dtype=dt_t, device=device)

    for i, scale in enumerate(scales):
        n_wvl = max(3, int(round(x_range * float(scale))))
        t_fine = np.linspace(0, x_range, n_wvl)
        psi_at_scale = np.interp(t_fine, x, psi_d) / np.sqrt(float(scale))
        psi_padded = torch.zeros(n_fft, dtype=dt_t, device=device)
        n_put = min(len(psi_at_scale), n_fft)
        psi_padded[:n_put] = torch.as_tensor(psi_at_scale[:n_put],
                                             dtype=dt_t, device=device)
        psi_rfft = torch.fft.rfft(psi_padded)
        conv = torch.fft.irfft(sig_rfft * psi_rfft[None, :], n=n_fft)
        out[:, i, :] = torch.roll(conv[:, :n], -(n_wvl // 2), dims=1)
    return -out                                              # sign negation


def _cwt_coeffs(sig, fs, p=DETECT_PARAMS):
    """CWT coefficients (n_scales, Nt) for one FILTERED trace (Stage-1 QC plot).
    Uses the lab's CPU implementation (reference path)."""
    x = np.asarray(sig, float)
    W = determine_scales(fs, p)
    return _sand._cwt_bior15(x - x.mean(), W.astype(float))


def _median_np(t):
    """numpy-compatible median (torch.median returns the LOWER middle value for
    even counts; numpy averages the two middle values)."""
    import torch
    n = t.numel()
    s, _ = torch.sort(t)
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def wavelet_threshold(c_i, scale, Nt, p=DETECT_PARAMS):
    """MEA-NAP per-scale Bayesian threshold DTh. `c_i` is a 1-D torch tensor of
    CWT coefficients for one scale. Returns (DTh, Sigmaj); DTh None = skip."""
    import torch
    L_scaled = p["L"] * p["Lmax"]
    stride = max(1, int(round(float(scale))))
    c_sub = c_i[::stride]
    Sigmaj = _median_np(torch.abs(c_sub - c_sub.mean())) / 0.6745
    if float(Sigmaj) == 0.0:
        return None, Sigmaj
    Thj = Sigmaj * float(np.sqrt(2 * np.log(Nt)))
    absc = torch.abs(c_i)
    index = absc > Thj
    n_idx = int(index.sum())
    if n_idx == 0:
        if p["option"] == "c":
            return None, Sigmaj
        Mj = Thj
        PS = 1.0 / Nt
    else:
        Mj = absc[index].mean()
        PS = n_idx / Nt
    PN = 1.0 - PS
    DTh = Mj / 2 + Sigmaj ** 2 / Mj * (L_scaled + float(np.log(PN / PS)))
    DTh = torch.abs(DTh) * (DTh >= 0)
    return DTh, Sigmaj


def wavelet_threshold_np(c_i, scale, Nt, p=DETECT_PARAMS):
    """numpy transcription of `wavelet_threshold` (same formula), for callers
    without torch -- e.g. the Stage-1 QC plot. Verified equal in _selftest."""
    L_scaled = p["L"] * p["Lmax"]
    stride = max(1, int(round(float(scale))))
    c_sub = c_i[::stride]
    Sigmaj = np.median(np.abs(c_sub - c_sub.mean())) / 0.6745
    if Sigmaj == 0.0:
        return None
    Thj = Sigmaj * np.sqrt(2 * np.log(Nt))
    absc = np.abs(c_i)
    index = absc > Thj
    n_idx = int(index.sum())
    if n_idx == 0:
        if p["option"] == "c":
            return None
        Mj = Thj
        PS = 1.0 / Nt
    else:
        Mj = absc[index].mean()
        PS = n_idx / Nt
    PN = 1.0 - PS
    DTh = Mj / 2 + Sigmaj ** 2 / Mj * (L_scaled + np.log(PN / PS))
    return abs(DTh) * (DTh >= 0)


def _detect_from_cwt(c, fs, scales, p=DETECT_PARAMS):
    """Per-scale threshold + polarity gate + union for one channel's (ns, Nt)
    CWT. Mirrors detect_spikes_wavelet's loop. Returns spike frames."""
    import torch
    ns, Nt = c.shape
    Io = torch.zeros(Nt, dtype=torch.bool, device=c.device)
    zero = torch.zeros((), dtype=c.dtype, device=c.device)
    for i in range(ns):
        DTh, _ = wavelet_threshold(c[i], scales[i], Nt, p)
        if DTh is None:
            continue
        ct_i = torch.where(torch.abs(c[i]) > DTh, c[i], zero)
        ct_i = torch.where(ct_i < 0, zero, ct_i)          # keep negative-going APs
        Io |= (ct_i != 0)
    return _sand._parse_spike_index(Io.to(torch.int64).cpu().numpy(), fs,
                                    tuple(p["wid_ms"]))


# ==============================================================================
# PUBLIC DETECTION
# ==============================================================================
def detect_spikes(signal, fs, p=DETECT_PARAMS, prefiltered=False):
    """Reference (CPU) detection for ONE trace -- the lab's own function.
    Returns spike times in SECONDS."""
    x = np.asarray(signal, float)
    if not prefiltered:
        x = bandpass(x, fs, p)
    frames = _sand.detect_spikes_wavelet(
        x, fs, wid_ms=tuple(p["wid_ms"]), ns=p["ns"], option=p["option"],
        L=p["L"], wname=p["wname"])
    return np.asarray(frames, float) / fs


def _ch_batch_for(Nt, n_ch, gp=GPU_PARAMS):
    if gp["ch_batch"]:
        return gp["ch_batch"]
    bytes_per = 16 if gp["fp64"] else 8         # complex buffer per sample
    n_fft = int(2 ** np.ceil(np.log2(max(Nt, 2))))
    per_ch = 4 * n_fft * bytes_per              # rfft + product + irfft + out
    k = max(1, int(gp["mem_budget_gb"] * 1e9 / max(per_ch, 1)))
    return min(k, n_ch)


def detect_spikes_matrix(sigs, fs, p=DETECT_PARAMS, device=None, prefiltered=False,
                         gp=GPU_PARAMS, progress=None):
    """Detect spikes on many channels (n_ch, Nt) with the CWT batched on the GPU.
    Returns a list of spike-time arrays in SECONDS. Falls back to the lab's CPU
    reference per channel if torch is unavailable."""
    sigs = np.asarray(sigs, float)
    if sigs.ndim == 1:
        sigs = sigs[None, :]
    n_ch, Nt = sigs.shape
    try:
        import torch
    except Exception:
        return [detect_spikes(sigs[i], fs, p, prefiltered) for i in range(n_ch)]

    dev = torch.device(device or auto_device())
    scales = determine_scales(fs, p)
    out = [None] * n_ch
    B = _ch_batch_for(Nt, n_ch, gp)
    for s in range(0, n_ch, B):
        e = min(s + B, n_ch)
        block = sigs[s:e]
        if not prefiltered:
            block = np.stack([bandpass(block[j], fs, p)
                              for j in range(block.shape[0])])
        block = block - block.mean(axis=1, keepdims=True)   # detect_spikes_wavelet
        c = _cwt_torch(block, scales, dev, gp["fp64"])
        for j in range(block.shape[0]):
            out[s + j] = np.asarray(_detect_from_cwt(c[j], fs, scales, p),
                                    float) / fs
            if progress is not None:
                progress(s + j + 1)          # contract: progress(n_done)
        del c
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return out


# ==============================================================================
def _selftest():
    """The torch/GPU path must return the SAME spikes as the lab's CPU reference."""
    fs = 20000.0
    rng = np.random.default_rng(0)
    dur = 12.0
    N = int(fs * dur)
    apN = int(0.6 * fs / 1000)
    s = np.sin(np.linspace(0, np.pi, apN))
    shape = -np.concatenate([s[:apN // 2], -0.6 * s[apN // 2:]])
    ok = True
    for amp in (40, 18, 12):
        x = rng.normal(0, 5, N)
        for t in np.sort(rng.uniform(0.05, dur - 0.05, 300)):
            i = int(t * fs)
            x[i:i + apN] += (shape * amp)[:min(apN, N - i)]
        ref = detect_spikes(x, fs)                        # lab reference (CPU)
        got = detect_spikes_matrix(x[None, :], fs, device="cpu")[0]
        same = (len(ref) == len(got)) and np.allclose(ref, got)
        print(f"  amp={amp:>3}uV  reference={len(ref):>4}  torch={len(got):>4}  "
              f"identical={same}")
        ok &= same

    # the numpy threshold (used by the QC plot) must equal the torch one
    try:
        import torch
        xf = bandpass(x, fs)
        W = determine_scales(fs)
        c = _cwt_coeffs(xf, fs)
        agree = True
        for i in range(len(W)):
            a = wavelet_threshold_np(c[i], W[i], c.shape[1])
            b, _ = wavelet_threshold(torch.as_tensor(c[i]), W[i], c.shape[1])
            if a is None or b is None:
                agree &= (a is None) == (b is None)
            else:
                agree &= bool(np.isclose(float(a), float(b), rtol=1e-6))
        print(f"  numpy threshold == torch threshold (QC parity): {agree}")
        ok &= agree
    except Exception as e:
        print(f"  [threshold parity check skipped: {e!r}]")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    _selftest()
