"""Configuration + loader-path wiring for the MEA Solar Cell Analyzer.

This project runs from its own folder ("MEA Solar Cell Analyzer"), separate from
the tested MEA-NAP loaders. It adds the loaders' folder to sys.path and imports
them normally; nothing about the HDF5 / spike-file / digital-event formats is
reimplemented here -- exactly the same reuse contract the "MEA Explorer" viewer
uses.

Folder layout on the machine this was built for::

    MEA Analyzer\                     (shared root: .venv + launchers)
      MEA Solar Cell Analyzer.bat     double-click launcher (this project)
      MEA Solar Cell Analyzer\        THIS folder
      MEA Analyzer\
        MEA-NAP\                      the loaders (mea_nap_io.py, ...)

Discovery order: MEANAP_DIR env override -> known/fallback absolute paths ->
search for a "MEA-NAP" folder near this file. Set MEANAP_DIR to force it.
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

# Optional data/spike overrides (empty => use the loaders' own io.CONFIG, or the
# folder chosen at runtime in the GUI).
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
    # 3. search near this file. This project's folder sits next to the code
    #    container that holds "MEA-NAP", so walk up a couple levels and look for a
    #    "MEA-NAP" folder directly under, or one level below, each ancestor.
    here = os.path.dirname(os.path.abspath(__file__))
    ancestors = [here,
                 os.path.dirname(here),
                 os.path.dirname(os.path.dirname(here))]
    for base in ancestors:
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
