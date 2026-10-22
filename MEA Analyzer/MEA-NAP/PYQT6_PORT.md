# MEA Analyzer GUI — PyQt6 port (migration notes)

**Date:** 2026-07-23
**What changed:** `mea_nap_gui.py` was rewritten from **Tkinter/ttk** to **PyQt6**.
The old Tkinter version lives untouched in the sibling folder
`MEA Analyzer (Legacy)`. Nothing was lost — this is a fresh folder.

## Why
Tkinter relays out its whole widget tree synchronously on every `<Configure>`
event during a live window-resize, software-rendered with no compositor help.
With the nested `PanedWindow → Notebook → scroll-canvas` structure this produced
visible lag, low frame-rate and ghosting on drag. Qt uses retained-mode,
double-buffered, GPU-assisted painting (the same reason the PyQt Solar Cell
Analyzer resizes smoothly), so the port fixes the lag at the root instead of
working around it. It also fixed the parameter-tab text-clipping regression and
gives crisp high-DPI rendering automatically.

## Ground rule for this port
**The pipeline back-end is 100% unchanged.** `mea_nap_io`, `mea_nap_activity`,
`mea_nap_wavelet_detect`, `mea_nap_original_plots`, `mea_nap_spectral_propagation`,
`mea_nap_spkit_egm`, `mea_nap_phase_lockin`, `mea_nap_digital_events`,
`mea_nap_progress` and the `meanap/` package are byte-for-byte copies of Legacy.
Only the window layer was rebuilt. All parameter names, the settings-JSON schema,
and the run logic are identical, so existing configs and saved settings keep
working.

## Architecture mapping (Tk → Qt6)
- `tk.Tk` → `QMainWindow`; `tk.PanedWindow(opaqueresize=False)` → `QSplitter`.
- Parameter tabs: `ttk.Notebook` + per-tab `tk.Canvas` scroll → `QTabWidget` with
  `QScrollArea(setWidgetResizable=True)`. Qt manages the inner width, so the old
  manual `_defer_canvas_width` canvas juggling is gone entirely.
- Per-sample progress rings: PIL-rendered images on a `tk.Canvas` → a
  `RingWidget` painted with `QPainter` (antialiased, DPI-scaled). They reflow
  via a small `FlowLayout` instead of a hand-computed column grid.
- Collapsible Paths/Environment/Analyses/Parameters panels → `CollapsibleCard`
  (a `QFrame#card` with a toggle header).
- `tk.Text` log → `QPlainTextEdit`; `filedialog`/`messagebox` → `QFileDialog`/
  `QMessageBox`; `ttk.Style('clam')` → `mea_theme.apply_qt()` stylesheet.
- Worker bridge is **unchanged in spirit**: the pipeline still runs on a plain
  `threading.Thread` posting commands to a `queue.Queue`; a `QTimer` (100 ms)
  drains that queue on the GUI thread. `EmbeddedProgress` (the drop-in for
  `mea_nap_progress.BarProgress`) works exactly as before.
- A thin `QtVar` adapter gives each Qt editor a Tk-`Variable`-like `.get()/.set()`
  so the original value-format/parse and config save/load code is reused verbatim.

## Runtime requirement
The shared `.venv` now also needs **PyQt6** (added to `config\requirements.txt`,
unpinned so pip picks the right Python-3.14 wheel). It installs alongside PyQt5 —
different import name — so MEA Explorer / Pattern Creator / Solar Cell (still
PyQt5) are unaffected. Re-run `config\setup.bat` (or
`pip install PyQt6` into the venv) once before launching.

## For the other Cowork sessions
The plan is to migrate Explorer, Pattern Creator and Solar Cell from **PyQt5 to
PyQt6** one at a time in their own sessions. `mea_theme.apply_qt()` is now
binding-agnostic (tries PyQt6 → PyQt5 → PySide6 → PySide2 and resolves the
scoped `QPalette.ColorRole` enums), so it already works under either Qt on all
four programs — no theme change is needed when each one migrates.
