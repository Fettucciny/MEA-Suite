"""Run every MEA Explorer test in its own process (Qt can only start once).

    python tests/run_tests.py          # from the MEA Explorer folder

Needs the shared venv (PyQt6, pyqtgraph, scipy, numpy) and the real MEA-NAP
folder (auto-discovered like the app does; or set MEANAP_REAL_DIR).
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ok = True
for name in ("test_data.py", "test_explorer.py"):
    print("=" * 70); print(name); print("=" * 70)
    rc = subprocess.call([sys.executable, os.path.join(HERE, name)], cwd=HERE)
    ok &= rc == 0
    print("-> %s\n" % ("PASS" if rc == 0 else "FAIL (exit %d)" % rc))
sys.exit(0 if ok else 1)
