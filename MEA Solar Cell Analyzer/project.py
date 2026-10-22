"""Project state for the MEA Solar Cell Analyzer + JSON persistence.

Holds everything the GUI needs to remember between sessions: where the data is,
what CONDITION each recording is (so templates learn from the right files), where
the solar cell and spheroid sit on the array, the array pitch, and the analysis
parameters. Serialises to a single .mscproj JSON file.
"""
import json
import os
import numpy as np

import geometry

# ---- recording conditions -------------------------------------------------
# The condition tells the pipeline what a recording is FOR:
LASER_ONLY = "laser_only"        # laser on bare MEA, no cells -> PIEZO template + baseline
SOLAR_ONLY = "solar_only"        # laser on ONE solar cell, no culture -> SOLAR template + field
SOLAR_TWO = "solar_two"          # laser on TWO solar cells, no culture -> field (amplification)
SOLAR_CULTURE = "solar_culture"  # laser on solar cell(s) + culture -> SAMPLE to clean
CULTURE_ONLY = "culture_only"    # culture, no solar cell -> BIO template / footprint
UNSET = "unset"

CONDITIONS = [LASER_ONLY, SOLAR_ONLY, SOLAR_TWO, SOLAR_CULTURE, CULTURE_ONLY, UNSET]
CONDITION_LABELS = {
    LASER_ONLY:    "Laser only (bare MEA) - piezo/baseline",
    SOLAR_ONLY:    "Solar cell x1 (no culture) - solar/field",
    SOLAR_TWO:     "Solar cells x2 (no culture) - field/amplification",
    SOLAR_CULTURE: "Solar cell + culture - sample",
    CULTURE_ONLY:  "Culture only (no solar) - biological",
    UNSET:         "Unset",
}


DEFAULT_ANALYSIS = {
    # peak detection
    "hp_hz": 100.0,            # high-pass corner for broadband peak detection (Hz)
    "peak_k": 5.0,             # detection threshold in robust MAD units
    "refractory_ms": 1.0,      # minimum separation between peaks (ms)
    "snippet_pre_ms": 1.5,     # snippet window before the peak extremum (ms)
    "snippet_post_ms": 1.5,    # snippet window after the peak extremum (ms)
    # flood locking (spontaneous neurons -> flood-locked peaks are artifacts).
    # A solar cell fires TWICE per exposure — an ON transient at laser onset and an
    # OFF transient at laser offset — so the artifact window spans the whole flood
    # (onset - lock_pre .. offset + event_tail), not a single fixed post-onset gap.
    "lock_pre_ms": 2.0,        # window START before flood onset (ms)
    "event_tail_ms": 15.0,     # window END after flood OFFSET (ms) — covers the off-transient
    "lock_post_ms": 25.0,      # FALLBACK post-onset window (ms) when the flood offset/duration
                               #   is unknown (no OFF times in the digital events)
    "stim_span_mode": "auto",  # "auto" = measure the artifact duration per recording
                               #   from the data (adapts to any pulse count/width, single
                               #   pulses, mixed protocols); "fixed" = use stim_train_ms below.
    "stim_train_ms": 250.0,    # DURATION of the whole flood stimulation train (ms). The
                               #   flood.seq fires a burst of pulses (e.g. 10 pulses of
                               #   15ms on/10ms off = 250 ms), each driving a solar transient,
                               #   so the artifact spans this whole window, not one pulse.
                               #   Set from the .seq via 'Load flood .seq' or edit directly.
    "lock_min_fraction": 0.5,  # electrode is "flood-driven" if it responds on >= this
                               # fraction of floods (reproducible = artifact)
    # classification
    "corr_min": 0.6,           # min bio-template correlation to accept a biological peak
    "artifact_corr_min": 0.7,  # a non-flood-locked peak matching the solar/piezo peak
                               # shape this well (and better than bio) is rejected as an
                               # artifact — the shape backstop that keeps sharp artifacts
                               # out of the biological class. Raise toward 1.0 to reject
                               # fewer; lower to reject more.
    # template shape cleaning (so a class wavelet + its error band reflect the TRUE
    # class shape, not contaminating outliers)
    "bio_polarity": -1,        # BIOLOGICAL peak template keeps only this extremum
                               #   polarity. -1 is NOT an independent assumption: it
                               #   MIRRORS THE VALIDATED bior1.5 DETECTOR, which hard-
                               #   gates to negative-going APs (mea_nap_wavelet_detect
                               #   _detect_from_cwt: `where(ct_i < 0, 0, ct_i)  # keep
                               #   negative-going APs`). So the bio wavelet matches the
                               #   spikes the Filter tab actually detects. Set 0 to keep
                               #   both polarities (only sensible if the detector's own
                               #   polarity gate is changed to match).
    "template_amp_mad_k": 6.0, # PRIMARY error-band cleaner. A rare electrode artifact
                               #   (railed sample, stim edge) can be 100-1000x a real
                               #   spike; even one such event dominates the ±SD band while
                               #   leaving the mean clean. Peaks whose |amplitude| exceeds
                               #   median + this×MAD are dropped from the template average.
                               #   Verified on real data: one 21 mV artifact drove the
                               #   trough SD from ~9 µV to 331 µV; removing it restores it.
    "template_min_corr": 0.55, # a collected peak whose shape correlates below this with
                               #   the emerging class mean is an outlier and is dropped
                               #   from the template average.
    "template_keep_frac": 0.8, # but always keep at least this fraction of peaks, so a
                               #   noisy class still yields a template (top-correlated
                               #   peaks are kept if the corr threshold would drop too many).
    # biological spikes re-detection band (reuses the validated detector)
    # ---- Filter-tab detection tightening (MEA-NAP-style amplitude gate) --------
    "prefilter_enabled": True,  # AND the bior1.5 wavelet detections with an amplitude
                                #   threshold on a band-passed trace, so low-amplitude
                                #   non-biological detections are rejected (tighter criteria).
    "prefilter_low_hz": 200.0,  # Butterworth band-pass low corner (Hz)
    "prefilter_high_hz": 3000.0,  # Butterworth band-pass high corner (Hz)
    "prefilter_order": 2,       # Butterworth order (2 = MEA-NAP default)
    "prefilter_k": 5.0,         # keep a spike only if |band-passed amplitude| >= k×noise
    "reject_flood_locked": True,  # on artifact-contaminated electrodes, DROP any
                                #   re-detected spike inside a flood-train window —
                                #   residual solar/piezo transients the STA subtraction
                                #   couldn't fully remove on high-amplitude electrodes
                                #   (a spontaneous neuron is never time-locked to the laser).
    # ---- solar-cell field overlay --------------------------------------------
    "field_lambda_um": 300.0,   # decay length constant (µm) of the MODELED solar field
                                #   (field ∝ exp(-distance_to_cell_edge / λ)).
    # ---- burst detection (Filter-tab burst-rate heatmap) ---------------------
    "burst_max_isi_ms": 100.0,  # max inter-spike interval within a burst (ms)
    "burst_min_spikes": 5,      # minimum spikes to call a burst
}


class Project:
    def __init__(self):
        self.path = None
        self.name = "Untitled"
        self.data_dir = ""
        self.spike_dir = ""
        self.pitch_um = geometry.DEFAULT_PITCH_UM
        self.conditions = {}        # stem -> condition
        self.solars = {}            # stem -> list of solar-cell dicts (per recording)
        self.spheroid = None        # dict or None
        self.image = None           # microscope-image overlay dict or None
        self.analysis = dict(DEFAULT_ANALYSIS)
        self.templates_path = ""    # optional .npz with learned templates

    # -- condition helpers --------------------------------------------------
    def condition(self, stem):
        return self.conditions.get(stem, UNSET)

    def set_condition(self, stem, cond):
        self.conditions[stem] = cond

    def stems_with_condition(self, cond):
        return [s for s, c in self.conditions.items() if c == cond]

    # -- geometry helpers ---------------------------------------------------
    def array(self):
        return geometry.ElectrodeArray(pitch_um=self.pitch_um)

    def solar_list(self, stem):
        """Raw solar-cell dicts placed on one recording (stem)."""
        return self.solars.get(stem, [])

    def solar_cells(self, stem):
        """geometry.SolarCell objects for one recording (may be 0, 1 or 2+)."""
        out = []
        for s in self.solar_list(stem):
            out.append(geometry.SolarCell(
                center_grid=s["center_grid"], side_um=s["side_um"],
                orientation_deg=s.get("orientation_deg", 0.0),
                electrode_axis_deg=s.get("electrode_axis_deg"),
                pitch_um=self.pitch_um))
        return out

    def set_solar_list(self, stem, cells):
        """Store the solar-cell dicts for one recording (empty list -> removed)."""
        if cells:
            self.solars[stem] = [dict(c) for c in cells]
        else:
            self.solars.pop(stem, None)

    @staticmethod
    def make_solar_dict(center_grid, side_um, orientation_deg=0.0,
                        electrode_axis_deg=None):
        return {
            "center_grid": [float(center_grid[0]), float(center_grid[1])],
            "side_um": float(side_um),
            "orientation_deg": float(orientation_deg),
            "electrode_axis_deg": (None if electrode_axis_deg is None
                                   else float(electrode_axis_deg)),
        }

    def spheroid_obj(self):
        if not self.spheroid:
            return None
        s = self.spheroid
        if s.get("points_grid"):
            return geometry.SpheroidOutline(s["points_grid"],
                                            pitch_um=self.pitch_um)
        if s.get("center_grid") is not None:
            return geometry.Spheroid(center_grid=s["center_grid"],
                                     radius_um=s.get("radius_um", 200.0),
                                     pitch_um=self.pitch_um)
        return None

    def set_spheroid(self, center_grid, radius_um=200.0):
        self.spheroid = {
            "center_grid": [float(center_grid[0]), float(center_grid[1])],
            "radius_um": float(radius_um),
        }

    def set_spheroid_points(self, points_grid):
        """Store a hand-traced spheroid perimeter (list of grid-coord points)."""
        self.spheroid = {
            "points_grid": [[float(x), float(y)] for x, y in points_grid],
        }

    def set_image(self, path, um_per_px=1.0, rotation_deg=0.0, cx_um=0.0,
                  cy_um=0.0, opacity=0.6, flip_y=True):
        """Microscope-image overlay: path + alignment transform. The centre is in
        µm from the array centre; um_per_px is the image's physical scale."""
        self.image = {
            "path": path, "um_per_px": float(um_per_px),
            "rotation_deg": float(rotation_deg),
            "cx_um": float(cx_um), "cy_um": float(cy_um),
            "opacity": float(opacity), "flip_y": bool(flip_y),
        }

    # -- persistence --------------------------------------------------------
    def to_dict(self):
        return {
            "name": self.name,
            "data_dir": self.data_dir,
            "spike_dir": self.spike_dir,
            "pitch_um": self.pitch_um,
            "conditions": self.conditions,
            "solars": self.solars,
            "spheroid": self.spheroid,
            "image": self.image,
            "analysis": self.analysis,
            "templates_path": self.templates_path,
            "_format": "mea_solar_cell_analyzer/1",
        }

    @classmethod
    def from_dict(cls, d):
        p = cls()
        p.name = d.get("name", "Untitled")
        p.data_dir = d.get("data_dir", "")
        p.spike_dir = d.get("spike_dir", "")
        p.pitch_um = float(d.get("pitch_um", geometry.DEFAULT_PITCH_UM))
        p.conditions = dict(d.get("conditions", {}))
        p.solars = dict(d.get("solars", {}))
        # migrate a legacy single global solar cell (pre per-recording) so old
        # projects still show it: attach it to every tagged solar recording.
        if not p.solars and d.get("solar"):
            legacy = d["solar"]
            targets = [s for s, c in p.conditions.items()
                       if c in (SOLAR_ONLY, SOLAR_TWO, SOLAR_CULTURE)]
            for s in (targets or ["(legacy)"]):
                p.solars[s] = [legacy]
        p.spheroid = d.get("spheroid")
        p.image = d.get("image")
        p.analysis = dict(DEFAULT_ANALYSIS)
        p.analysis.update(d.get("analysis", {}))
        p.templates_path = d.get("templates_path", "")
        return p

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as fh:
            p = cls.from_dict(json.load(fh))
        p.path = path
        p.name = os.path.splitext(os.path.basename(path))[0]
        return p
