# MEA Solar Cell Analyzer — GUI theming & styling

_Last updated 2026-07-23 (Claude Cowork session). This is the single reference
for this program's shared MEA-Suite look and its **PyQt6** GUI. As of this update
the GUI has been migrated **PyQt5 → PyQt6** and wired to the shared theme kit
(`mea_theme.py`, accent **solar / #FBBF24 amber**). The GUI lives in `app.py`
(the tabbed window + entry point) and `electrode_map.py` (the click-to-place
electrode/solar-cell canvas)._

---

## 1. Where the assets are

All suite paths are relative to the suite root `…\MEA\MEA Analyzer\`.

| Asset | Location | What it is |
|-------|----------|------------|
| **Theme kit (master)** | `config\mea_theme.py` | The shared module: palette, per-program accents, `qss()`, `apply_qt()`, `apply_tk()`, `style_pyqtgraph()`, `parula()`. Edit colours here. |
| **Theme kit (local copy)** | `MEA Solar Cell Analyzer\mea_theme.py` | This program's own copy (kept in sync with the master). Imported at runtime by `app.py`. |
| **Reference GUI (PyQt6)** | `MEA Analyzer\MEA-NAP\mea_nap_gui.py` | The fully-migrated Analyzer GUI — the worked example of every migration point. |
| **Migration write-up** | `MEA Analyzer\MEA-NAP\PYQT6_PORT.md` | Why & how the Analyzer went Tk→Qt6; the widget mapping. |

The palette, accents and parula map in `mea_theme.py` are the **exact** values
baked into the launcher icons, splash screens and the MEA Suite selector, so
styling from the kit keeps every program visually identical.

---

## 2. This program's identity

- **Accent:** `solar` → **#FBBF24** (amber).
- Everything else (window gradient, panels, fields, text, plot colours) is shared
  across all four programs — only the accent differs.

Shared palette tokens (from `mea_theme.py`):

| token | value | use |
|-------|-------|-----|
| INDIGO / TEAL | #16213E / #0D4A56 | window gradient (top / bottom) |
| PANEL | #0E1626 | card / group-box background |
| PANEL_HI | #141F33 | raised / hovered panel, secondary buttons |
| FIELD | #0B1220 | input fields, plot-adjacent surfaces |
| LINE | #26374A | borders / dividers |
| TEXT / TEXT_MUTED | #E8EEF4 / #9FB4C8 | primary / secondary text |
| PLOT_BG / PLOT_FG | #0C121E / #C7D2DE | plot background / foreground |

Accents (one per program): analyzer #3B82F6 · explorer #34D399 · pattern #DB7FFF ·
**solar #FBBF24**.

---

## 3. How the theme is applied

`apply_qt()` is **binding-agnostic** — it resolves the scoped-enum differences
itself and works under PyQt6 or PyQt5. In `app.py`, right after the
`QApplication` is created (in `main()`):

```python
import mea_theme
app = QtWidgets.QApplication(sys.argv)
app.setStyle("Fusion")
mea_theme.apply_qt(app, "solar")        # palette + fonts + full stylesheet
```

and once at import time, before any plots are built:

```python
mea_theme.style_pyqtgraph("solar")      # dark pyqtgraph bg + antialias
```

The binding is selected before pyqtgraph is imported (PyQt6 preferred, PyQt5
fallback):

```python
if "PYQTGRAPH_QT_LIB" not in os.environ:
    for _binding in ("PyQt6", "PyQt5"):
        try:
            __import__(_binding)
            os.environ["PYQTGRAPH_QT_LIB"] = _binding
            break
        except Exception:
            continue
```

Both GUI files import Qt via `from pyqtgraph.Qt import QtWidgets, QtCore, QtGui`,
so they follow whichever binding `PYQTGRAPH_QT_LIB` selects.

Handy styling hooks provided by `qss()` (opt-in): `setProperty("primary", True)`
→ solid accent button; `QFrame` with `objectName == "card"` → rounded panel;
`QLabel` with `property muted == "true"` → dim secondary text. Tabs, checkboxes,
sliders, progress bars, scrollbars, menus, tables and tooltips are themed
automatically.

Note on plot images: the calibration report, filter PNGs and electrode heatmaps
keep their **own** deliberate scientific styling (their own dark background and
colormaps — e.g. viridis for the electrode heatmap). We do **not** apply a global
`matplotlib_rc()` here, so those exported figures are unchanged by the theme.

---

## 4. PyQt5 → PyQt6 migration — what was done

The GUI files were `app.py` and `electrode_map.py`. Both already used the
`pyqtgraph.Qt` abstraction, so the migration was contained to enum scoping, the
`exec_` rename, the binding-selection env, and wiring the shared theme.

1. **Binding via `pyqtgraph.Qt`** — kept. Only the `PYQTGRAPH_QT_LIB`
   auto-selection (above) changed, to prefer PyQt6.
2. **Scoped enums** — every enum uses the full Qt6 path:
   `Qt.Horizontal` → `Qt.Orientation.Horizontal` (opacity slider);
   `Qt.DashLine` → `Qt.PenStyle.DashLine` (field-decay / FOV-ring / solar-axis
   dashed lines);
   `QMessageBox.Yes/No` → `QMessageBox.StandardButton.Yes/No` (batch prompt);
   `QMessageBox.Warning/Information` → `QMessageBox.Icon.Warning/Information`
   (batch-done dialog).
3. **`dialog.exec_()` → `dialog.exec()`** (batch-done message box). The entry
   point already tolerates both via `getattr(app, "exec", None) or app.exec_`.
4. **High-DPI** — nothing to remove (no `AA_*` attributes were set); Qt6 handles
   it automatically.
5. **Mouse events** — the electrode canvas uses pyqtgraph's `MouseClickEvent`
   (`ev.scenePos()`), not raw `QMouseEvent.pos()`, so no `.position()` change was
   needed.
6. **Theme** — removed the old inline `THEME_QSS` block and the four
   `pg.setConfigOption(...)` calls; replaced with `mea_theme.apply_qt(app,
   "solar")` and `mea_theme.style_pyqtgraph("solar")`.

Verified: `py_compile` on all modules passes; the GUI builds under an offscreen
`QApplication` with the theme applied; and `selftest.py` still passes (loader-free
math/geometry checks, unaffected by the GUI change).

---

## 5. Keeping it in sync

`MEA Solar Cell Analyzer\mea_theme.py` is a copy of the master
`config\mea_theme.py`. If the palette or accents ever change, edit the **master**
and re-copy — don't fork the colours here. Don't hard-code hex values in `app.py`;
pull them from `mea_theme` (`mea_theme.PANEL`, `mea_theme.accent("solar")`, …) so
a future palette tweak is one edit.
