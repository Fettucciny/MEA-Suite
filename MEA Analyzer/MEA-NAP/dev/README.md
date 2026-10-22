# dev/ — verification tools for MEA-NAP

Run these from `MEA Analyzer\MEA-NAP\` (the folder above this one).

| tool | question it answers |
|---|---|
| `check_gui_keys.py [candidate_phase_lockin.py]` | Will the Analyzer window build? Every parameter the schema binds must exist in its module dict, or launch dies silently under pythonw. |
| `astdiff.py OLD.py NEW.py` | Which top-level functions actually changed between two revisions (AST hash, ignores comments/whitespace)? Prove plotting changes left the maths alone. |
| `compare_figures.py BEFORE_DIR AFTER_DIR` | Did the self-test figures / CSVs change byte-for-byte? Pixel report for PNGs that differ. |
| `radial_proto.py` | Prototype (NOT installed) of a true radial matched filter + model-selection test, on synthetic data. See HANDOFF_2026-10-22.md §2.4. |
| `gitignore.txt` | Proposed `.gitignore` for the suite root. |

Typical change workflow: run the self-test (`python mea_nap_phase_lockin.py`)
BEFORE editing and move `_phaselock_selftest` aside; edit; `py_compile`;
`check_gui_keys.py`; self-test again; `compare_figures.py`; `astdiff.py`
against the previous copy; then report exactly what changed.
