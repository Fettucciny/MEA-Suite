# MEA Explorer — GUI theming & styling

_Last updated 2026-07-23 (Claude Cowork session). This is the single reference
for restyling **MEA Explorer** to the shared MEA-Suite look and (when you're ready)
migrating it from **PyQt5 → PyQt6**. MEA Explorer is a PyQt5 + pyqtgraph app; `app.py` is the whole GUI._

---

## 1. Where the assets are

All paths are relative to the suite root `…\MEA\MEA Analyzer\`.

| Asset | Location | What it is |
|-------|----------|------------|
| **Theme kit (master)** | `config\mea_theme.py` | The shared module: palette, per-program accents, `qss()`, `apply_qt()`, `apply_tk()`, `style_pyqtgraph()`, `parula()`. Edit colours here. |
| **Theme kit (local copy)** | `MEA Explorer\mea_theme.py` | This program's own copy (kept in sync with the master). Import this one at runtime. |
| **Visual reference** | `config\theme_preview.png` | Rendered swatch of the palette + widgets. |
| **This program's icon** | `MEA Explorer\MEA Explorer.ico` | Window / launcher icon (accent-tinted). |
| **Suite icon** | `config\mea_suite.ico` | The MEA-monogram folder / selector icon. |
| **Reference GUI (PyQt6)** | `MEA Analyzer\MEA-NAP\mea_nap_gui.py` | The fully-migrated Analyzer GUI — copy its patterns. |
| **Migration write-up** | `MEA Analyzer\MEA-NAP\PYQT6_PORT.md` | Why & how the Analyzer went Tk→Qt6; the widget mapping. |

The palette, accents and parula map in `mea_theme.py` are the **exact** values
baked into the launcher icons, splash screens and the MEA Suite selector, so
styling from the kit keeps every program visually identical.

---

## 2. This program's identity

- **Accent:** `explorer` → **#34D399** (emerald).
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
solar #FBBF24.

---

## 3. Applying the theme (PyQt5 today, PyQt6 after migration)

`apply_qt()` is **binding-agnostic** — it tries PyQt6 → PyQt5 → PySide and resolves
the scoped-enum differences itself — so this exact call works both **before and
after** you migrate. Call it once, right after you create the `QApplication`:

```python
import mea_theme
app = QApplication(sys.argv)
mea_theme.apply_qt(app, "explorer")     # palette + fonts + full stylesheet
mea_theme.style_pyqtgraph("explorer")   # dark pyqtgraph bg + antialias (if used)
```

For any matplotlib figures:

```python
import matplotlib as mpl
mpl.rcParams.update(mea_theme.matplotlib_rc())
```

Handy styling hooks provided by `qss()`:

- `widget.setProperty("primary", True)` → solid **accent** button (re-polish after
  setting the property).
- `QFrame` with `objectName == "card"` → rounded panel with the panel background.
- `QLabel` with `property muted == "true"` → dim secondary text.
- Tabs, checkboxes, sliders, progress bars, scrollbars, menus, tables and tooltips
  are all themed automatically.

---

## 4. PyQt5 → PyQt6 migration checklist

The GUI file to migrate is **`app.py`**. The Analyzer port
(`MEA Analyzer\MEA-NAP\mea_nap_gui.py`) is a worked example of every point below.

1. **Imports:** `from PyQt5 …` → `from PyQt6 …`. `QAction`/`QShortcut` moved from
   `QtWidgets` to **`QtGui`**.
2. **Scoped enums** — the big one. Qt6 requires the full path:
   `Qt.AlignCenter` → `Qt.AlignmentFlag.AlignCenter`,
   `Qt.Horizontal` → `Qt.Orientation.Horizontal`,
   `QSizePolicy.Expanding` → `QSizePolicy.Policy.Expanding`,
   `QPainter.Antialiasing` → `QPainter.RenderHint.Antialiasing`, etc.
3. **`app.exec_()` → `app.exec()`** (and `dialog.exec_()` → `dialog.exec()`).
4. **High-DPI is automatic** in Qt6 — delete any `AA_EnableHighDpiScaling` /
   `AA_UseHighDpiPixmaps` / manual DPI attributes.
5. **`QMouseEvent.pos()` / `.globalPos()`** → `.position().toPoint()` /
   `.globalPosition().toPoint()` (they now return `QPointF`).
6. **pyqtgraph**: works on PyQt6; just make sure `pyqtgraph>=0.13`. Set the binding
   before importing if needed via `os.environ["PYQTGRAPH_QT_LIB"]="PyQt6"`.
7. **Enum `.value`**: where you stored an enum as int, use `.value` on Qt6.
8. Keep `mea_theme.apply_qt(app, "explorer")` exactly as-is — no theme change
   needed across the migration.

Reusable widget patterns worth lifting from the Analyzer port: `CollapsibleCard`
(rounded fold-away panel), `RingWidget` (QPainter progress ring), `FlowLayout`
(wrap-to-width cell grid), and the `QtVar` adapter (Tk-`Variable`-like
`.get()/.set()` around a Qt editor — handy if you keep any generic
value-serialization code).

Runtime: the shared `.venv` now has **PyQt6** installed alongside PyQt5
(`config\requirements.txt`), so both bindings are importable during the transition —
migrate this program whenever you're ready without touching the others.

---

## 5. Keeping it in sync

`MEA Explorer\mea_theme.py` is a copy of the master `config\mea_theme.py`. If the
palette or accents ever change, edit the **master** and re-copy — don't fork the
colours here. Don't hard-code hex values in `app.py`; pull them from
`mea_theme` (`mea_theme.PANEL`, `mea_theme.accent("explorer")`, …) so a future
palette tweak is one edit.
