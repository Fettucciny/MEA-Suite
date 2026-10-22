"""Shared test bootstrap: stub loaders first, real MEA-NAP second, app folder
on the path, Qt offscreen. Import this at the top of every test module."""
import os, sys

TESTS = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(TESTS)                              # ...\MEA Explorer
STUBS = os.path.join(TESTS, "stubs")
FIXTURES = os.path.join(TESTS, "fixtures")
OUT = os.path.join(TESTS, "out")
os.makedirs(OUT, exist_ok=True)

os.environ["MEANAP_DIR"] = STUBS                          # config.py puts this first
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")
sys.path.insert(0, APP)

import config                                             # noqa: E402
config.ensure_loaders_on_path()                           # -> stubs at sys.path[0]

# The stubs cover only io / digital_events / original_plots. The real
# mea_nap_wavelet_detect + meanap/ come from the real MEA-NAP folder, found the
# same way the app finds it (or MEANAP_REAL_DIR to force one).
_real = os.environ.get("MEANAP_REAL_DIR", "")
if not _real:
    _saved = os.environ.pop("MEANAP_DIR")                 # don't rediscover the stubs
    _real = config._discover_loaders_dir()
    os.environ["MEANAP_DIR"] = _saved
if not os.path.isfile(os.path.join(_real, "mea_nap_wavelet_detect.py")):
    raise FileNotFoundError(
        "Real MEA-NAP folder not found (need mea_nap_wavelet_detect.py + meanap/). "
        "Set MEANAP_REAL_DIR. Tried: {}".format(_real))
if _real not in sys.path:
    sys.path.append(_real)
REAL_MEANAP = _real
