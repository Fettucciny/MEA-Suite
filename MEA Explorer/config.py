"""Configuration + loader-path wiring for the MEA raster viewer.

This project runs from its own folder ("MEA Explorer"), separate from the tested
loaders. It adds the loaders' folder to sys.path and imports them normally;
nothing about the HDF5 / spike-file formats is reimplemented here.

The loaders' folder has been renamed/moved at least once (Scripts -> MEA
Analyzer), so instead of trusting a single hard-coded path we AUTO-DISCOVER it:
env override -> known paths -> search near this file. Set the MEANAP_DIR
environment variable to force a specific location.
"""
import os
import sys

# Primary known location of the loaders (folder with mea_nap_io.py, meanap/...).
DEFAULT_LOADERS_DIR = r"C:\Users\Yikan\Nextcloud2\MEA\MEA Analyzer\MEA Analyzer\MEA-NAP"

# Older/alternate absolute locations to try if the primary has moved.
_FALLBACK_DIRS = [
    r"C:\Users\Yikan\Nextcloud2\MEA\MEA Analyzer\Scripts\MEA-NAP",
    r"C:\Users\Yikan\Nextcloud2\MEA Data\New MEA 06172026\Scripts\MEA-NAP",
]

# Optional data/spike overrides (empty => use the loaders' own io.CONFIG).
DATA_DIR = os.environ.get("MEANAP_DATA_DIR", "")
SPIKE_DIR = os.environ.get("MEANAP_SPIKE_DIR", "")


def _looks_like_loaders(d):
    return bool(d) and os.path.isfile(os.path.join(d, "mea_nap_io.py"))


def _discover_loaders_dir():
    # 1. explicit override
    env = os.environ.get("MEANAP_DIR", "")
    if _looks_like_loaders(env):
        return env
    # 2. known + fallback absolute paths
    for d in [DEFAULT_LOADERS_DIR] + _FALLBACK_DIRS:
        if _looks_like_loaders(d):
            return d
    # 3. search near this file: the app root is the parent of "MEA Explorer".
    #    Find a "MEA-NAP" folder directly under, or one level below, that root
    #    (or its parent) — so a renamed container folder is still found.
    here = os.path.dirname(os.path.abspath(__file__))     # ...\MEA Explorer
    for base in (os.path.dirname(here), os.path.dirname(os.path.dirname(here))):
        if not base or not os.path.isdir(base):
            continue
        cand = os.path.join(base, "MEA-NAP")
        if _looks_like_loaders(cand):
            return cand
        try:
            for name in sorted(os.listdir(base)):
                p = os.path.join(base, name)
                if os.path.isdir(p):
                    cand = os.path.join(p, "MEA-NAP")
                    if _looks_like_loaders(cand):
                        return cand
        except OSError:
            pass
    return DEFAULT_LOADERS_DIR    # give up; ensure_loaders_on_path raises clearly


LOADERS_DIR = _discover_loaders_dir()


def ensure_loaders_on_path():
    """Put the loaders folder on sys.path so `import mea_nap_io` and
    `import meanap...` resolve. Raises a clear error if it can't be found."""
    d = LOADERS_DIR
    if not _looks_like_loaders(d):
        raise FileNotFoundError(
            "Could not find the MEA-NAP loaders (mea_nap_io.py).\n"
            "Last tried:\n  {}\n"
            "Set the MEANAP_DIR environment variable to the folder that "
            "contains mea_nap_io.py.".format(d)
        )
    if d not in sys.path:
        sys.path.insert(0, d)
