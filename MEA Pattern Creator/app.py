#!/usr/bin/env python3
"""
MEA Pattern Creator -- entry point (PyQt6 + shared MEA-Suite theme).

Launches the GUI for building Polygon 1000 stimulation sequences from a live
AmScope camera feed.  Run via "MEA Pattern Creator.exe" in the parent folder
(which uses the shared per-machine venv), or directly:

    python app.py
"""

import os
import sys
import traceback

# make `import mpc...` and `import mea_theme` work no matter where we launch from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    # High-DPI scaling is automatic in Qt6 -- no attributes to set.
    from PyQt6 import QtWidgets, QtGui

    QtWidgets.QApplication.setApplicationName("MEA Pattern Creator")
    app = QtWidgets.QApplication(sys.argv)

    # Shared MEA-Suite look (magenta "pattern" accent). apply_qt is
    # binding-agnostic, so this is the only theming call the app needs.
    try:
        import mea_theme
        mea_theme.apply_qt(app, "pattern")
    except Exception:
        traceback.print_exc()

    # Window / taskbar icon (accent-tinted, ships beside this script).
    try:
        ico = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "MEA Pattern Creator.ico")
        if os.path.exists(ico):
            app.setWindowIcon(QtGui.QIcon(ico))
    except Exception:
        pass

    from mpc.gui.main_window import MainWindow
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # write a crash log next to the app so the launcher can surface it
        log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "error.log")
        with open(log, "w") as fh:
            fh.write(traceback.format_exc())
        traceback.print_exc()
        raise
