# Polygon 1000, PolyScan2, optics and stimulation — working notes

Facts established with Jerry on the real rig (Fett), mostly on 2026-10-15.
"Observed" = Jerry tested it; "measured" = computed from his files;
"inferred" = reasoning not yet confirmed on the rig.

## 1. Hardware chain

- **Mightex Polygon 1000** DMD. `.seq` canvas is 1824×1216, 1 bit per pixel.
  The `.map` file also contains 912 and 1140, the DLP4500 mirror-array
  dimensions (inferred).
- **Optics:** Nikon 4X objective, DSI-K-TUBE-10 tube. PolyScan calibration file
  `..\config\120123_Calibration_4X.map` (2023-12-01, "3rd Party Camera").
- **AmScope camera:** native 5440×3648, run live at resolution index 2 =
  1824×1216 (3× binning), so camera px = canvas px under identity calibration.
  Saved exposure is 50 ms.
- **MEA:** 60 electrodes on an 8×8 grid with the corners absent. 200 µm pitch,
  30 µm diameter. Labels are column-row (11 top-left … 88 bottom-right; 11, 18,
  81 and 88 are printed labels, not electrodes).
- **TTL source:** Agilent 33250A function generator into the Polygon Trigger In.

## 2. `.seq` format

The reference is the docstring of `mpc/seqfile.py`. Summary:

- 128-byte header ("1QSM", version 1, ints 1000, 10000, 0, 15000);
- 31 × 132-byte records (each has u32 = 1 at +8);
- 4-byte gap;
- N frame blocks: [u32 0, u32 count (frame 0 only), u32 W, u32 H, u32 1] +
  bitmap (MSB-first, row-major) + u32 0;
- 8-byte footer.

Frame block = 277,272 B, so file size = 4232 + 277,272·N.

## 3. Optical scale (measured)

- From `config\1 Solar Cell.png`, a lattice fit to the electrodes: pitch
  233.0 native px gives **0.8583 µm/native px**, so **2.575 µm/canvas px**.
  Array rotation −0.7°. Camera FOV 4.67 × 3.13 mm.
  (`tools\measure_mea_pitch.py` reproduces this.)
- PolyScan's `.map` holds the float 2.70004 (and 1.35002, exactly half), 4.9 %
  higher. **Unresolved**: see HANDOFF open item 1.
- Electrode FWHM in the photo is 36 µm, against 30 µm nominal (blur at 4X). This
  is consistent with the scale.
- The **illuminated field is smaller than the DMD canvas**. Jerry observed that
  slivers in the canvas corners are not visible. RingOut's rings stop at r ≈
  800 px, which may mark the usable field edge (inferred).

## 4. PolyScan2 behaviour

**Master mode** (internal timing):

- Pattern Period = dwell of every frame. Batch Size and Loop Count apply.
- Observed: Pattern Period 1 ms with a 2-frame on/off file gave a 500 Hz flicker,
  which the camera integrates and which looked like a slowly "breathing" dot.
  Set Pattern Period to the intended frame duration.
- Output Trigger (Pulse Delay/Width) is a TTL *out* for sync. It does not gate
  the light.

**Slave mode with external TTL** (observed 2026-10-15):

- Each TTL pulse advances one frame. The sequence wraps forever.
- **EPSM Pause had no visible effect** at 1000 ms or 5000 ms.
- **Repeat/Loop Count had no effect**: playback did not stop.
- **Batch Size must be at least the sequence's frame count**, or only part of the
  sequence is loaded. This was the only setting that mattered.
- A 33250A burst of exactly 160 pulses ran the 160-frame sweep once and stopped.
- So under TTL, **run length = pulse count** and **dark gaps must be blank frames
  in the `.seq`** (or pauses in the TTL train).
- Unknown: whether frame 1 is shown before the first pulse (off-by-one). See
  HANDOFF open item 2.
- Inferred, not tested: after the TTL stops, the DMD keeps showing the current
  frame until the run is stopped. Choose counts that end on a harmless frame.

## 5. Agilent 33250A setup for TTL runs

- Waveform: **Pulse**, 50 Hz, width 1 ms. Levels: HiLevel 5 V, LoLevel 0 V.
- Output load: **High Z** (Utility → Output Setup → Load). With the 50 Ω setting
  the real output into a high-impedance input is about double.
- Burst mode: N Cycle. Set the trigger source to **Manual** *before* entering the
  count: the internal burst period maxes out at 500 s, which is shorter than
  long runs. Burst count can be 1 to 1,000,000.
- Timebase is 2 ppm, so a 55-min run is accurate to under 7 ms.
- Start the PolyScan Slave run first, then press Trigger.

55-minute runs at 50 Hz:

| Sequence | Frames/pass | #Cycles | Duration | Ends on |
|---|---|---|---|---|
| BR→TL sweep | 160 | 164,960 | 54:59.2 | last or first frame, both off-field slivers |
| Radial centre-out | 133 | 164,900 | 54:58.0 | blank frame 113/114 whichever the off-by-one |

Source: Agilent 33250A datasheet and User's Guide.

## 6. Sequence catalogue (in `..\config\` unless noted)

All speeds assume 50 Hz TTL and 2.575 µm/px. "Over electrodes" assumes the array
is centred in the field. Frame indices are 0-based (PolyScan and the settings
test number frames from 1).

| File | Frames | Pattern | Width | Step/frame | Speed | Lit per point | Over electrodes | Dark gap (looped) |
|---|---|---|---|---|---|---|---|---|
| `Sept2021_SweepBRtoTL.seq` (PolyScan-made) | 160, none blank | single 45° bar, bottom-right → top-left | 57 diag px ≈ 104 µm | 19 diag px = 13.4 px = 34.6 µm | 1.73 mm/s (40 Hz: 1.38) | 3 frames, 60 ms | frames 53–104 (1.04 s) | 108 fr = 2.16 s |
| `Oct2026_RadialCenterOut.seq` (ours) | 133 = 82 ring + 51 blank | ring from canvas centre (911.5, 607.5) to field corners | 40.3 px ≈ 104 µm | 13.4 px | 1.73 mm/s | 3 frames | frames 1–25 (0.50 s) | 108 fr = 2.16 s (matched) |
| `01212025_RingOut_20ms_50HzTTL_pause.seq` (previous lab member) | 100 = 20 ring + 80 blank | tiling rings around (911, 607), r < 800 px | 40 px ≈ 103 µm | 40 px = 103 µm | 5.15 mm/s | 1 frame, 20 ms | frames 1–8 | 92 fr = 1.84 s |
| settings test (`tools\make_settings_test_seq.py`) | 20 | numbered diagnostic frames, crosshair, 1 mm bar, border on frame 1 | – | – | – | – | – | – |

Notes:

- The sweep has a single bar. The "second bar entering at frame 92" seen
  earlier was an artifact of the old offset bug.
- Radial: pixels within 34 µm of the centre get 1 frame and those at 34–69 µm
  get 2. The nearest electrodes are 141 µm away, so every electrode gets the
  full 3 frames.
- Regenerate the radial file with `tools\make_radial_seq.py` (default output is
  byte-identical, md5 `bee2579fe9ae06170f57e9c166b373b7`).
- Raster electrode orders for analysis are in `electrode_order_BR_to_TL.json` and
  `electrode_order_RadialCenterOut.json` (MEA Explorer format; owned by MEA
  Explorer).

## 7. Solar-cell (photovoltaic) stimulation notes

These are discussion results, not yet tested.

**Observation.** Under flood illumination, the cell voltage shows a sharp
~0.5 ms transient and then decays.

**Explanation given.** It is most likely not the light: the DMD and LED switch
in µs and hold for the whole frame. Two candidates:

- **Electrode–electrolyte double-layer charging.** Measured time constants are
  0.15 ms for Pt and 3.1 ms for SIROF (Loudin et al., J Neural Eng; Palanker
  lab).
- **The MEA amplifier's high-pass filter.** A 300 Hz corner gives τ =
  1/(2πf) ≈ 0.53 ms.

**Discriminating tests suggested:**

- vary the on-time;
- look for an opposite-polarity transient at light-off;
- record DC-coupled or with a 0.1–1 Hz high-pass;
- check the light directly with a photodiode;
- measure the cell into a resistor, out of saline.

**Pulse design.** Charge Q(t) = Qmax(1 − e^(−t/τ)), so the pulse width should be
about 1–2 τ. Without a shunt, the electrodes only fully discharge at about 1 Hz;
at 10 Hz they don't, and at 33 Hz amplitude drops to about ¼. A shunt of tens of
MΩ across the cell (18–68 MΩ in Boinagrov et al.) allows 20–40 Hz. Charge per
pulse is capped at C·V; Voc rises only logarithmically with intensity.

**Sources:**

- Boinagrov et al., IEEE TBioCAS 2015, "Photovoltaic pixels for neural
  stimulation";
- Loudin et al., "Photodiode circuits for retinal prostheses";
- Mathieson et al., Nature Photonics 2012.
