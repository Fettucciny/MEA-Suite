r"""Every parameter the window binds must exist in its module dict, or the
Analyzer dies silently under pythonw while building the parameter tabs.

    python dev\check_gui_keys.py              # checks the live modules
    python dev\check_gui_keys.py some_module.py   # checks a candidate copy of
                                                   # mea_nap_phase_lockin.py
Exit code 0 = all good.
"""
import os, re, sys, ast
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

def dict_keys_from_source(path, name):
    """Keys of a module-level dict literal, read from source (values may be
    expressions such as 2 * 60 - only the keys matter here)."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    for n in tree.body:
        if (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == name
                and isinstance(n.value, ast.Dict)):
            keys = set()
            for k in n.value.keys:
                if isinstance(k, ast.Constant):
                    keys.add(k.value)
            return keys
    return None

gui_src = open(os.path.join(HERE, "mea_nap_gui.py"), encoding="utf-8").read()
# _P(<alias>, "<DICTNAME>", "<key>", ...)
refs = re.findall(r'_P\(\s*\w+\s*,\s*"([A-Z_]+)"\s*,\s*"([^"]+)"', gui_src)
by_dict = {}
for d, k in refs:
    by_dict.setdefault(d, set()).add(k)

# which module owns which dict
owners = {"PHASE_LOCKIN": "mea_nap_phase_lockin.py", "SPKIT_EGM": "mea_nap_spkit_egm.py",
          "DETECT_PARAMS": "mea_nap_wavelet_detect.py", "QC_PARAMS": "mea_nap_io.py",
          "CONFIG": "mea_nap_io.py", "SWEEP_CALIB": "mea_nap_digital_events.py",
          "DIGITAL_PARAMS": "mea_nap_digital_events.py", "PROP_PARAMS": "mea_nap_spectral_propagation.py",
          "MEANAP_PLOT_PARAMS": "mea_nap_original_plots.py"}
override = sys.argv[1] if len(sys.argv) > 1 else None
bad = 0
for d, keys in sorted(by_dict.items()):
    path = os.path.join(HERE, owners[d]) if d in owners else ""
    if d == "PHASE_LOCKIN" and override:
        path = override
    if not path or not os.path.isfile(path):
        # dicts that live in modules not listed above (activity params etc.)
        found = None
        for f in os.listdir(HERE):
            if f.endswith(".py") and dict_keys_from_source(os.path.join(HERE, f), d) is not None:
                found = os.path.join(HERE, f); break
        if found is None:
            print(f"  ?  {d:<22} could not locate the owning module ({len(keys)} keys unchecked)")
            continue
        path = found
    have = dict_keys_from_source(path, d)
    if have is None:
        print(f"  ?  {d:<22} not a literal dict in {os.path.basename(path)} ({len(keys)} keys unchecked)")
        continue
    missing = sorted(keys - have)
    flag = "OK " if not missing else "BAD"
    bad += len(missing)
    print(f"  {flag} {d:<22} {len(keys):3d} key(s) bound in the window"
          + (f"   MISSING from {os.path.basename(path)}: {missing}" if missing else ""))
print("\nall schema keys resolve" if not bad else f"\n{bad} key(s) would raise KeyError at launch")
sys.exit(1 if bad else 0)
