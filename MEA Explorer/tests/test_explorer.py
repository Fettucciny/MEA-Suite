"""GUI (offscreen): main window + electrode explorer end to end.
Run:  python tests/test_explorer.py   (from the MEA Explorer folder)
Writes tests/out/explorer.png and tests/out/main.png for a visual check."""
import _env                                   # noqa: F401
import os, time, tempfile
import numpy as np
import app                                    # the main window module
import datasource as ds
import export
from PyQt6 import QtWidgets

TMP = tempfile.mkdtemp()
app.SETTINGS_FILE = os.path.join(TMP, "settings.json")   # never touch the real settings


def wait(qa, cond, t=20):
    end = time.time() + t
    while time.time() < end:
        qa.processEvents(); time.sleep(0.01)
        if cond():
            return True
    return False


def main():
    qa = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import mea_theme; mea_theme.apply_qt(qa, "explorer")
    ds.set_spike_dir(os.path.join(_env.FIXTURES, "MEA 071826", "Output", "SpikeData"))
    rec = ds.Recording("burstrec")
    win = app.RasterWindow(rec, "burstrec"); win.resize(1400, 800); win.show()

    win.open_explorer()                       # toolbar path: first visible electrode
    ex = win._explorer; assert ex is not None
    assert ex.label == rec.labels[win.vis_idx()[0]]
    ex.resize(1400, 800)
    assert wait(qa, lambda: ex.result is not None), "no burst result"
    r = ex.result
    assert ex.model.rowCount() == r["t_start"].size
    (x0, x1), _ = ex.spike_plot.viewRange()
    if r["t_start"].size:
        assert x0 <= r["t_start"][0] <= x1, "did not auto-zoom to first burst"
    assert wait(qa, lambda: ex._trace is not None), "no filtered trace"
    assert ex.trace_curve.xData.size > 0 and len(ex.mark_bst.data) > 0
    print("explorer             : %s -> %d spikes, %d bursts, trace drawn" % (ex.label, r["n_spikes"], r["t_start"].size))

    win.open_explorer("21"); assert win._explorer is ex and ex.label == "21"   # map Shift-click path
    assert wait(qa, lambda: ex.result is not None and ex.result["n_spikes"] == rec.spike_times_of("21").size)

    ex.rb_manual.setChecked(True); ex.sp_ms.setValue(20.0)                     # manual threshold
    assert wait(qa, lambda: ex.result is not None and not ex.result["auto"] and abs(ex.result["isin_th"] - 0.02) < 1e-9)
    assert win.explorer_opts["auto_threshold"] is False and win.explorer_opts["manual_ms"] == 20.0
    ex.rb_auto.setChecked(True); assert wait(qa, lambda: ex.result["auto"])

    ex.zoom_to_burst(2); (x0, x1), _ = ex.spike_plot.viewRange()
    assert x0 <= ex.result["t_start"][2] <= x1
    ex._step_burst(+1); qa.processEvents()

    ex.fit_all()                                                               # > 30 s -> hint, no trace
    assert wait(qa, lambda: "Zoom in" in ex.hint.textItem.toPlainText())
    assert ex.trace_curve.xData is None or ex.trace_curve.xData.size == 0

    out = os.path.join(TMP, "b.csv")
    export.export_bursts_csv(rec.stem, ex.label, ex.result, out)
    lines = open(out).read().splitlines()
    assert len(lines) == ex.result["t_start"].size + 1 and lines[0].startswith("recording,electrode,burst")

    ex.zoom_to_burst(3)
    assert wait(qa, lambda: ex._trace is not None and ex._trace["t"].size > 0 and not ex.hint.textItem.toPlainText(), 10)
    for _ in range(30): qa.processEvents(); time.sleep(0.01)
    ex.grab().save(os.path.join(_env.OUT, "explorer.png")); win.grab().save(os.path.join(_env.OUT, "main.png"))

    rec.h5_path = None; ex.zoom_to_burst(1)                                    # missing raw file
    assert wait(qa, lambda: "not found" in ex.hint.textItem.toPlainText())
    assert not ex.locate_btn.isHidden()
    rec.set_raw_path(ds.resolve_h5("burstrec"))
    win._on_raw_located(os.path.dirname(rec.h5_path)); ex._refresh_view()
    assert wait(qa, lambda: ex.locate_btn.isHidden())
    assert app.load_settings().get("data_dir") == os.path.dirname(rec.h5_path)

    win.toggle_qc(False); qa.processEvents()                                   # new recording closes it
    assert win._explorer is None
    win.open_explorer("21"); ex2 = win._explorer; assert ex2 is not None and ex2.rec is win.rec
    ex2.close(); wait(qa, lambda: win._explorer is None, 2); assert win._explorer is None
    win.close()
    print("screenshots          : %s" % _env.OUT)
    print("\nGUI: ALL PASS")


if __name__ == "__main__":
    main()
