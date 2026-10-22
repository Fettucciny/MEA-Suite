#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 mea_nap_progress  --  THREADED GUI progress window (one bar per sample)
================================================================================
A Tk window listing every sample (H5 file / recording) with a horizontal
progress bar and percentage to the right of its name, so you can see which
sample is being processed and how far along it is.

Used by Stage 1 (spike detection, bar fills per channel) and Stage 2 (activity +
network analysis, bar fills per region/lag).

WHY THREADED (this is the fix for the freezing window)
------------------------------------------------------
The previous version drew from the main thread and only repainted when the
pipeline happened to call set(). During a long step (200 surrogates on a 55-min
window) the main thread never returned to Tk's event loop, so Windows greyed the
window out as "(Not Responding)".

Here, Tk lives in its OWN daemon thread and runs its own mainloop, repainting on
its own timer regardless of what the pipeline is doing. The pipeline only ever
drops a message on a queue (non-blocking, microseconds); the GUI thread drains
that queue every POLL_MS and redraws.

Tkinter is NOT thread-safe, so the rule is strict and load-bearing: the root and
every widget are created INSIDE the GUI thread, and the pipeline thread touches
NOTHING but the queue.

Fails soft: if Tk is unavailable (headless/SSH) every call is a no-op and the run
proceeds with the tqdm bars. PROGRESS_PARAMS["enabled"]=False disables it.

Public API (unchanged, so mea_nap_io / mea_nap_activity call sites need no edits):
    p = make(names, title, subtitle)
    p.start(name); p.set(name, frac, note=...); p.done/skipped/failed(name); p.close()
================================================================================
"""

import queue
import threading

PROGRESS_PARAMS = {
    "enabled":         True,
    "bar_w_px":        240,    # progress-bar width
    "bar_h_px":        16,
    "name_chars":      24,     # name column width
    "rows_visible":    14,     # rows shown before the list scrolls
    "width_px":        760,
    "poll_ms":         60,     # GUI redraw / queue-drain interval
    "close_when_done": False,  # True = close the window automatically at the end
    "startup_timeout_s": 5.0,  # how long to wait for the window to come up
}

_STATE_COLOR = {
    "pending": "#d9d9d9",
    "running": "#1f77b4",
    "done":    "#2ca02c",
    "skipped": "#ff7f0e",
    "failed":  "#d62728",
}
_TRACK = "#ececec"


class BarProgress:
    """Threaded progress window: one horizontal bar per sample."""

    def __init__(self, names, title="Progress", subtitle=""):
        self.names = list(names)
        self.ok = False
        self.applied = 0                      # commands applied by the GUI thread
        self._q = queue.Queue()
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._gui_thread, args=(title, subtitle), daemon=True)
        self._thread.start()
        # wait until the window is up (or has failed) so ok is meaningful
        self._ready.wait(timeout=PROGRESS_PARAMS["startup_timeout_s"])

    # ---- GUI thread ---------------------------------------------------------
    def _gui_thread(self, title, subtitle):
        try:
            import tkinter as tk
            from tkinter import ttk
        except Exception:
            self._ready.set()
            return
        try:
            root = tk.Tk()
            root.title(title)
            rows_h = min(len(self.names), PROGRESS_PARAMS["rows_visible"]) * 30 + 92
            root.geometry(f"{PROGRESS_PARAMS['width_px']}x{rows_h}")

            head = tk.Frame(root); head.pack(fill="x", padx=10, pady=(8, 2))
            self._head_lbl = tk.Label(head, text=subtitle or title, anchor="w",
                                      font=("Segoe UI", 10, "bold"))
            self._head_lbl.pack(fill="x")
            self._overall = tk.Label(head, text="", anchor="w", fg="#555555",
                                     font=("Segoe UI", 8))
            self._overall.pack(fill="x")

            outer = tk.Frame(root); outer.pack(fill="both", expand=True, padx=6,
                                               pady=6)
            canvas = tk.Canvas(outer, highlightthickness=0)
            sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
            body = tk.Frame(canvas)
            body.bind("<Configure>",
                      lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas.create_window((0, 0), window=body, anchor="nw")
            canvas.configure(yscrollcommand=sb.set)
            canvas.pack(side="left", fill="both", expand=True)
            sb.pack(side="right", fill="y")

            W = PROGRESS_PARAMS["bar_w_px"]; H = PROGRESS_PARAMS["bar_h_px"]
            self._rows = {}
            for nm in self.names:
                row = tk.Frame(body); row.pack(fill="x", pady=3)
                tk.Label(row, text=nm, font=("Segoe UI", 9), anchor="w",
                         width=PROGRESS_PARAMS["name_chars"]).pack(side="left",
                                                                   padx=(4, 6))
                cv = tk.Canvas(row, width=W + 2, height=H + 2,
                               highlightthickness=0)          # bar, right of name
                cv.pack(side="left")
                cv.create_rectangle(1, 1, W + 1, H + 1, fill=_TRACK, outline="")
                fill = cv.create_rectangle(1, 1, 1, H + 1,
                                           fill=_STATE_COLOR["pending"], outline="")
                pct = tk.Label(row, text="", font=("Segoe UI", 8), width=5,
                               anchor="e", fg="#333333")
                pct.pack(side="left", padx=(6, 6))
                note = tk.Label(row, text="pending", font=("Segoe UI", 8),
                                anchor="w", fg="#777777")
                note.pack(side="left", fill="x", expand=True)
                self._rows[nm] = {"cv": cv, "fill": fill, "pct": pct, "note": note,
                                  "frac": 0.0, "state": "pending"}
            root.protocol("WM_DELETE_WINDOW", lambda: self._closed.set())
            self._root = root
            self.ok = True
        except Exception:
            self.ok = False
            self._ready.set()
            return
        self._ready.set()

        def poll():
            drained = 0
            try:                                   # drain everything pending
                while True:
                    cmd = self._q.get_nowait()
                    self._apply(cmd)
                    drained += 1
            except queue.Empty:
                pass
            self.applied += drained
            if self._closed.is_set():
                try:
                    root.destroy()
                except Exception:
                    pass
                return
            root.after(PROGRESS_PARAMS["poll_ms"], poll)

        root.after(PROGRESS_PARAMS["poll_ms"], poll)
        try:
            root.mainloop()
        except Exception:
            pass
        self.ok = False

    def _apply(self, cmd):
        kind = cmd[0]
        if kind == "head":
            self._head_lbl.config(text=cmd[1]); return
        _, name, frac, state, note = cmd
        r = self._rows.get(name)
        if r is None:
            return
        if frac is not None:
            r["frac"] = max(0.0, min(float(frac), 1.0))
        if state:
            r["state"] = state
        if note is not None:
            r["note"].config(text=note)
        W = PROGRESS_PARAMS["bar_w_px"]; H = PROGRESS_PARAMS["bar_h_px"]
        r["cv"].coords(r["fill"], 1, 1, 1 + W * r["frac"], H + 1)
        r["cv"].itemconfig(r["fill"], fill=_STATE_COLOR.get(r["state"], "#1f77b4"))
        r["pct"].config(text=f"{int(round(r['frac'] * 100))}%"
                        if r["state"] == "running" else "")
        done = sum(1 for x in self._rows.values()
                   if x["state"] in ("done", "skipped", "failed"))
        self._overall.config(text=f"{done}/{len(self._rows)} complete")

    # ---- pipeline thread: enqueue only, never touch Tk -----------------------
    def _put(self, name, frac, state=None, note=None):
        if self.ok:
            self._q.put(("set", name, frac, state, note))

    def start(self, name, note="running"):
        self._put(name, 0.0, "running", note)

    def set(self, name, frac, state=None, note=None):
        self._put(name, frac, state, note)

    def done(self, name, note="done"):
        self._put(name, 1.0, "done", note)

    def skipped(self, name, note="skipped"):
        self._put(name, 1.0, "skipped", note)

    def failed(self, name, note="failed"):
        self._put(name, 1.0, "failed", note)

    def close(self):
        if not self.ok:
            return
        if PROGRESS_PARAMS["close_when_done"]:
            self._closed.set()
            self._thread.join(timeout=2.0)
        else:
            self._q.put(("head", "finished — close this window when you're done"))
        self.ok = False


class _NullProgress:
    """No-op stand-in so callers never need to branch."""
    ok = False
    applied = 0
    def start(self, *a, **k): pass
    def set(self, *a, **k): pass
    def done(self, *a, **k): pass
    def skipped(self, *a, **k): pass
    def failed(self, *a, **k): pass
    def close(self): pass


def make(names, title="Progress", subtitle=""):
    """BarProgress if a GUI is available and enabled, else a silent no-op."""
    if not PROGRESS_PARAMS["enabled"]:
        return _NullProgress()
    p = BarProgress(names, title, subtitle)
    return p if p.ok else _NullProgress()


if __name__ == "__main__":
    import time
    names = [f"sample_{i}_D{d}" for i in (1, 2) for d in (1, 2, 3)]
    p = make(names, "Demo", "bar-per-sample demo")
    for nm in names:
        p.start(nm)
        for k in range(11):
            p.set(nm, k / 10, note=f"channel {k*6}/60")
            time.sleep(0.05)
        p.done(nm, "60 ch, 12,345 spikes")
    p.close()
