"""Section export: image (PNG/SVG) and spike/raw data (npz/csv)."""
import os
import csv
import numpy as np


def _axis_items(plot_item):
    out = []
    for name in ("left", "bottom", "right", "top"):
        try:
            ax = plot_item.getAxis(name)
        except Exception:
            continue
        if ax is not None and ax.isVisible():
            out.append(ax)
    return out


def _tick_strings(ax):
    """The tick labels this axis will actually draw (for width sizing)."""
    out = []
    for level in (getattr(ax, "_tickLevels", None) or []):
        for item in level:
            try:
                out.append(str(item[1]))
            except Exception:
                pass
    return out or ["00000"]


def _reserve_axis_space(plot_item):
    """Pin each axis to the size its current fonts actually need.

    pyqtgraph sizes an AxisItem from `textHeight`/`textWidth`, which are only
    refreshed while it paints — so straight after a font change it still
    reports the old size and the enlarged labels get clipped out of the
    export (and its own _adjustSize() undercounts the axis label). Measuring
    with QFontMetrics and pinning the size with setHeight()/setWidth() is
    deterministic. `_release_axis_space` undoes it.
    """
    from PyQt6 import QtGui, QtWidgets

    for ax in _axis_items(plot_item):
        tick_font = ax.style.get("tickFont") or QtGui.QFont()
        fm = QtGui.QFontMetrics(tick_font)
        tick_len = abs(int(ax.style.get("tickLength", 0) or 0))
        label_h = ax.label.boundingRect().height() if ax.labelText else 0.0
        shows = bool(ax.style.get("showValues", True))
        pad = 8
        if ax.orientation in ("bottom", "top"):
            text_h = fm.height() if shows else 0
            ax.setHeight(text_h + tick_len + label_h + pad)
        else:
            widest = (max(fm.horizontalAdvance(s) for s in _tick_strings(ax))
                      if shows else 0)
            ax.setWidth(widest + tick_len + label_h + pad)
    try:
        plot_item.layout.activate()
    except Exception:
        pass
    QtWidgets.QApplication.processEvents()


def _release_axis_space(plot_item):
    """Hand axis sizing back to pyqtgraph (undo _reserve_axis_space)."""
    from PyQt6 import QtWidgets

    for ax in _axis_items(plot_item):
        try:
            if ax.orientation in ("bottom", "top"):
                ax.setHeight(None)
            else:
                ax.setWidth(None)
        except Exception:
            pass
    try:
        plot_item.layout.activate()
    except Exception:
        pass
    QtWidgets.QApplication.processEvents()


def _target_rect(plot_item):
    """Scene rect to export: the plot plus its axis labels.

    A PlotItem's own sceneBoundingRect stops at the layout cell, but the axis
    *labels* ("Time (s)", "Electrode") hang a few pixels outside it — which is
    why they were cropped out of every exported image. Uniting with the
    children rect brings them back in; the size guard keeps a stray child from
    blowing the canvas up.
    """
    rect = plot_item.sceneBoundingRect()
    try:
        kids = plot_item.mapRectToScene(plot_item.childrenBoundingRect())
        united = rect.united(kids)
        if (united.width() <= rect.width() * 3
                and united.height() <= rect.height() * 3):
            rect = united
    except Exception:
        pass
    return rect.adjusted(-2, -2, 2, 2)


def _background_brush(plot_item):
    """The plot's background colour.

    pyqtgraph paints the background on the *view*, not the scene, so rendering
    the scene alone would come out transparent — and the light-grey axis text
    would be invisible on it.
    """
    from PyQt6 import QtGui

    try:
        views = plot_item.scene().views()
        if views:
            brush = views[0].backgroundBrush()
            if brush is not None and brush.color().alpha() > 0:
                return brush
    except Exception:
        pass
    try:
        import pyqtgraph as pg
        return QtGui.QBrush(pg.mkColor(pg.getConfigOption("background")))
    except Exception:
        return QtGui.QBrush(QtGui.QColor("#0C121E"))


def natural_size(plot_item):
    """(width, height) in px the plot exports at with no scaling."""
    rect = _target_rect(plot_item)
    return max(1.0, rect.width()), max(1.0, rect.height())


def _render(plot_item, path, svg=False, width_px=None, dpi=None):
    """Paint the plot (with its axis labels) to PNG or SVG.

    Both formats go through QPainter on the scene rather than pyqtgraph's
    exporters: the SVGExporter raises on this scene, and the ImageExporter
    crops to a rect that excludes the axis labels.

    `width_px` re-renders at that pixel width (height follows the aspect
    ratio). Everything — including text — scales uniformly, so the layout is
    unchanged and only the resolution goes up. `dpi` is written into the file
    so Word/Illustrator place it at the right physical size.
    """
    from PyQt6 import QtGui, QtCore

    rect = _target_rect(plot_item)
    w, h = max(1.0, rect.width()), max(1.0, rect.height())
    scale = (float(width_px) / w) if width_px else 1.0
    scale = max(0.05, min(scale, 40.0))            # sanity clamp
    ow, oh = max(1.0, w * scale), max(1.0, h * scale)
    dest = QtCore.QRectF(0, 0, ow, oh)
    bg = _background_brush(plot_item)

    if svg:
        from PyQt6.QtSvg import QSvgGenerator
        dev = QSvgGenerator()
        dev.setFileName(path)
        dev.setSize(QtCore.QSize(int(round(ow)), int(round(oh))))
        dev.setViewBox(dest)
        dev.setTitle("MEA Explorer raster")
        if dpi:
            dev.setResolution(int(dpi))            # must precede the painter
        painter = QtGui.QPainter(dev)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            painter.fillRect(dest, bg)
            plot_item.scene().render(painter, dest, rect)
        finally:
            painter.end()
        return path

    img = QtGui.QImage(int(round(ow)), int(round(oh)),
                       QtGui.QImage.Format.Format_ARGB32)
    img.fill(bg.color())
    if dpi:
        dpm = int(round(float(dpi) / 0.0254))      # dots per metre
        img.setDotsPerMeterX(dpm)
        img.setDotsPerMeterY(dpm)
    painter = QtGui.QPainter(img)
    try:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(
            QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        plot_item.scene().render(painter, dest, rect)
    finally:
        painter.end()
    img.save(path)
    return path


def export_image(plot_item, path, label_pt=None, tick_pt=None,
                 width_px=None, dpi=None, hide_tick_values=()):
    """Save the current raster view as PNG or SVG.

    `label_pt` / `tick_pt` override the axis-label and tick-mark font sizes,
    `hide_tick_values` names axes ("left", "bottom", ...) whose tick labels to
    omit, and `width_px` / `dpi` set the output resolution. All of these apply
    to the exported file only — the on-screen view is restored afterwards,
    even if the export raises.
    """
    from PyQt6 import QtGui

    saved = []
    hidden = []
    for name in (hide_tick_values or ()):
        try:
            ax = plot_item.getAxis(name)
        except Exception:
            continue
        if ax is not None:
            hidden.append((ax, bool(ax.style.get("showValues", True))))
            ax.setStyle(showValues=False)

    if label_pt or tick_pt:
        for ax in _axis_items(plot_item):
            saved.append((ax, dict(ax.labelStyle), ax.style.get("tickFont"),
                          ax.labelText, ax.labelUnits))
            if tick_pt:
                f = QtGui.QFont()
                f.setPointSize(int(tick_pt))
                ax.setTickFont(f)
            if label_pt and ax.labelText:
                style = dict(ax.labelStyle)
                style["font-size"] = "{}pt".format(int(label_pt))
                # NB: pass the text through. setLabel() only re-shows the
                # label when text/units are given — style-only calls leave it
                # hidden, which silently dropped the axis label from exports.
                ax.setLabel(text=ax.labelText,
                            units=(ax.labelUnits or None), **style)
    if saved or hidden:
        _reserve_axis_space(plot_item)

    try:
        _render(plot_item, path,
                svg=os.path.splitext(path)[1].lower() == ".svg",
                width_px=width_px, dpi=dpi)
    finally:
        for ax, old_show in hidden:
            try:
                ax.setStyle(showValues=old_show)
            except Exception:
                pass
        for ax, old_style, old_font, old_text, old_units in saved:
            try:
                ax.setTickFont(old_font)
                if old_text:
                    ax.setLabel(text=old_text, units=(old_units or None),
                                **old_style)
                else:
                    ax.labelStyle = old_style
                    ax.label.setHtml(ax.labelString())
                ax.picture = None
                ax.update()
            except Exception:
                pass
        if saved or hidden:
            try:
                _release_axis_space(plot_item)
            except Exception:
                pass
    return path


def export_spikes_npz(rec, t0, t1, path, labels=None):
    """Per-channel spike subset over [t0, t1) -> compressed .npz.

    Saves an object array of per-channel time arrays plus labels/t0/t1/fs.
    """
    labels = list(labels) if labels is not None else list(rec.labels)
    idx = [rec.labels.index(l) for l in labels]
    windows = rec.window_spikes(t0, t1)
    subset = np.empty(len(idx), dtype=object)
    for k, j in enumerate(idx):
        subset[k] = np.asarray(windows[j], dtype=np.float64)
    np.savez_compressed(
        path,
        spike_times=subset,
        labels=np.asarray(labels),
        t0=np.float64(t0),
        t1=np.float64(t1),
        fs=np.float64(rec.fs),
        stem=str(rec.stem),
    )
    return path


def export_spikes_csv(rec, t0, t1, path, labels=None):
    """Tidy channel,time_s rows over [t0, t1) -> .csv."""
    labels = list(labels) if labels is not None else list(rec.labels)
    idx = [rec.labels.index(l) for l in labels]
    windows = rec.window_spikes(t0, t1)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["channel", "time_s"])
        for k, j in enumerate(idx):
            lab = labels[k]
            for t in windows[j]:
                w.writerow([lab, "{:.6f}".format(float(t))])
    return path


def export_raw_npz(rec, t0, t1, path, labels):
    """Raw voltage windows for the given channels over [t0, t1) -> .npz.

    Triggers the (heavy) MCSFile load if not already resident. Intended for a
    small number of channels; guard the channel count in the UI.
    """
    labels = list(labels)
    data = {}
    tref = None
    for lab in labels:
        t, uv = rec.channel_waveform(lab, t0, t1, max_points=10**9)  # no decimation
        data["ch_" + lab] = uv.astype(np.float32)
        if tref is None:
            tref = t
    np.savez_compressed(
        path,
        t_s=(tref if tref is not None else np.empty(0)),
        labels=np.asarray(labels),
        t0=np.float64(t0),
        t1=np.float64(t1),
        fs=np.float64(rec.fs),
        stem=str(rec.stem),
        **data,
    )
    return path


def export_bursts_csv(stem, label, result, path):
    """One row per detected burst on a single electrode -> .csv.

    `result` is datasource.detect_bursts(...). The detection settings are
    repeated on every row so the file is self-describing on its own.
    """
    th = result.get("isin_th", float("nan"))
    th_ms = "" if not np.isfinite(th) else "{:.3f}".format(th * 1000.0)
    mode = "automatic" if result.get("auto") else "manual"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["recording", "electrode", "burst", "t_start_s", "t_end_s",
                    "duration_ms", "n_spikes", "rate_in_burst_hz",
                    "min_spikes_N", "isin_threshold_ms", "threshold_mode"])
        for i in range(int(result["t_start"].size)):
            fr = result["in_burst_fr"][i]
            w.writerow([stem, label, i + 1,
                        "{:.6f}".format(result["t_start"][i]),
                        "{:.6f}".format(result["t_end"][i]),
                        "{:.3f}".format(result["duration_ms"][i]),
                        int(result["size"][i]),
                        "" if not np.isfinite(fr) else "{:.3f}".format(fr),
                        result["min_spikes"], th_ms, mode])
    return path
