# MEA Analyzer — shared UI theme

> **PyQt6 (2026-07-23):** MEA Analyzer's GUI is now **PyQt6**, not Tkinter — see
> `MEA-NAP\PYQT6_PORT.md`. It is themed with `mea_theme.apply_qt(app, "analyzer")`
> (call once, right after `QApplication` is created); the `apply_tk` path below is
> kept only for reference / the Legacy build. `apply_qt` is binding-agnostic
> (PyQt6 / PyQt5 / PySide) so the same call works before and after the other
> programs migrate to Qt6.

# MEA Analyzer — shared UI theme

_Last updated 2026-07-23 (Claude Cowork session). This documents the shared look-and-feel kit so this program's GUI can match the launchers, splash screens, and the MEA Suite selector._

## What this is

`mea_theme.py` is a small, dependency-light module (dropped into this program's folder: `MEA-NAP\mea_theme.py (next to mea_nap_gui.py)`) that carries the whole MEA Suite look — the indigo->teal dark palette, this program's accent colour (**#3B82F6 (royal blue)**), Segoe UI fonts, rounded 'card' panels, and the parula plot colormap. It imports no GUI toolkit at load time, so importing it is always safe; each toolkit is imported lazily inside the apply function.

## How to apply (PyQt6 — this program)

Right after you create the `QApplication`:

```python
import mea_theme
app = QApplication(sys.argv)
mea_theme.apply_qt(app, "analyzer")       # palette + fonts + full stylesheet
```

## How to apply (Tkinter / ttk)

Right after you create the root window:

```python
import mea_theme
root = tk.Tk()
mea_theme.apply_tk(root, "analyzer")      # ttk styles + option database
```

Named ttk styles the kit provides:

```python
ttk.Button(root, text="Run", style="Accent.TButton")   # solid accent button
ttk.Frame(root, style="Card.TFrame")                    # card panel
ttk.Label(root, text="hint", style="Muted.TLabel")      # dim text
```

For the MEA-NAP matplotlib figures, match them to the theme with:

```python
import matplotlib as mpl
mpl.rcParams.update(mea_theme.matplotlib_rc())
```

Note: MEA Analyzer already has its own dark mode. `apply_tk` restyles the ttk
widgets to the shared palette; adopt it wholesale, or just import the palette
constants (mea_theme.PANEL, .TEXT, .accent("analyzer"), ...) to align the
existing dark mode with the rest of the suite.

## Palette (shared)

| token | value | use |
|-------|-------|-----|
| INDIGO / TEAL | #16213E / #0D4A56 | window gradient (top/bottom) |
| PANEL | #0E1626 | card / group-box background |
| PANEL_HI | #141F33 | raised / hovered panel, secondary buttons |
| FIELD | #0B1220 | input fields, plot-adjacent surfaces |
| LINE | #26374A | borders / dividers |
| TEXT / TEXT_MUTED | #E8EEF4 / #9FB4C8 | primary / secondary text |
| PLOT_BG / PLOT_FG | #0C121E / #C7D2DE | plot background / foreground |

Accents (one per program): analyzer #3B82F6 (blue), explorer #34D399 (emerald),
pattern #DB7FFF (magenta), solar #FBBF24 (amber).

## Keeping it in sync

This file is a copy of the shared kit; the master lives in the suite `config\mea_theme.py`, and a visual reference is `config\theme_preview.png`. If the palette or accents change, update the master and re-copy — don't fork the colours here. The accents, palette, and parula map are the exact values used by the icons, splash screens, and the MEA Suite launcher.
