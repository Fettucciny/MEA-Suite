# Polygon 1000 .seq — format and stimulation notes

_Copied from the claude.ai "MEA Suite" project on 2026-10-22; originally written 2026-10-15._

## Frame layout (corrected 2026-10-15)
Each frame block = 20-byte header (u32 0, u32 count on frame 0 else 0, u32 width 1824, u32 height 1216, u32 1) + width*height/8 bitmap (1 bpp, MSB-first, row-major) + 4-byte zero trailer. File: 128-byte header + 31×132-byte records + 4-byte gap + frames + 8-byte footer.
Earlier Pattern Creator code put the bitmap at +24. That still round-tripped byte-exactly, but every exported pattern was shifted 32 DMD px (~82 µm) to the right in PolyScan. Fixed in mpc/seqfile.py, with regression tests in tests/test_core.py. .seq files exported before the fix carry the shift and need re-exporting.

## Scale (4X, AmScope at 1824×1216 = DMD canvas, identity calibration)
0.8583 µm per native camera px (5440×3648), measured from the 200 µm electrode pitch, so 2.575 µm per canvas px. The PolyScan .map (120123_Calibration_4X.map) stores 2.70004, which is 4.9% higher and not yet reconciled.

## PolyScan2 behaviour under external TTL (observed by user, 2026-10-15)
- Each TTL pulse advances one frame and the sequence wraps indefinitely.
- EPSM Pause (Slave mode) had no visible effect at 1 s or 5 s. Repeat/Loop Count did not stop playback.
- Batch Size must be at least the sequence's frame count, or only part of the sequence is loaded. It is the only one of these settings that mattered.
- So under TTL, the number of passes comes from the TTL source (burst length), and dark gaps must be built into the sequence as blank frames or as pauses in the TTL train.

## config/Sept2021_SweepBRtoTL.seq
160 frames (Batch Size ≥ 160). A single 45° band, 57 px wide in x+y (~104 µm), steps 19 px in x+y per frame (34.6 µm). It starts on the bottom-right corner pixel and ends on the top-left one. At 50 Hz TTL: 1.73 mm/s (40 Hz: 1.38 mm/s). It is over the electrodes in frames 53–104 (52 frames); dark gap when looped is 108 frames = 2.16 s.

## config/Oct2026_RadialCenterOut.seq
133 frames (Batch Size ≥ 133): 82 ring frames growing from the canvas center (911.5, 607.5) with the same width and step as the sweep, then 51 blank frames. It is over the electrodes in frames 1–25. Dark gap when looped is 108 frames = 2.16 s, matching BR→TL. Assumes a 60-electrode 8×8 array centered in the field and 50 Hz TTL.
Matching raster order for MEA Explorer: `config/electrode_order_RadialCenterOut.json` (electrodes by distance from the array centre; the file's `rings` key lists each ring's distance in pitches and approximate arrival frame at 19 px/frame).

## config/Oct2026_SettingsTest_20frames.seq
20 diagnostic frames (Batch Size ≥ 20): each frame shows its number and a progress box, plus a center crosshair and a 1 mm scale bar (388.4 px at 2.575 µm/px). Frame 1 also has a full border to mark loop restarts.
