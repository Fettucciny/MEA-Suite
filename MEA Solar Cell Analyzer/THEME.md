# MEA Solar Cell Analyzer — shared UI theme

_Last updated 2026-07-23 (Claude Cowork session). This documents the shared look-and-feel kit so this program's GUI can match the launchers, splash screens, and the MEA Suite selector._

## What this is

`mea_theme.py` is a small, dependency-light module (dropped into this program's folder: `mea_theme.py (this folder, next to app.py)`) that carries the whole MEA Suite look — the indigo->teal dark palette, this program's accent colour (**#FBBF24 (amber)**), Segoe UI fonts, rounded 'card' panels, and the parula plot colormap. It imports no GUI toolkit at load time, so importing it is always safe; each toolkit is imported lazily inside the apply function.

## How to apply (PyQt5 / pyqtgraph)

Right after you create the QApplication (once, before building widgets):

```python
import mea_theme
app = QApplication(sys.argv)
mea_theme.apply_qt(app, "solar")          # palette + fonts + full stylesheet
mea_theme.style_pyqtgraph()               # dark plot bg + antialias (call once)
```

Opt-in widget roles the stylesheet understands (set with setProperty, then
re-polish, or just via Qt Designer):

```python
btn.setProperty("primary", True)          # solid accent button
lbl.setProperty("muted", True)            # secondary/dim text
frame.setObjectName("card")               # rounded card panel (like the selector)
mainwindow.setObjectName("background")    # apply the window gradient
# after changing a dynamic property: w.style().unpolish(w); w.style().polish(w)
```

For plots / heatmaps use the shared parula colormap so images match the
SpKit / Phase-Lock figures:

```python
import numpy as np, pyqtgraph as pg
lut = mea_theme.parula_lut(256)           # (256,3) uint8
img.setLookupTable(lut)                    # pg.ImageItem
# or a ColorMap:
cm = pg.ColorMap(np.linspace(0,1,256), np.column_stack([lut, np.full(256,255)]))
```

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
