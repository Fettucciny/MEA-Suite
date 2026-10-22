_CORN = {"11", "18", "81", "88"}
def mcs_coord(label):
    s = str(label).strip()
    if len(s) == 2 and s.isdigit():
        c, r = int(s[0]), int(s[1])
        if 1 <= c <= 8 and 1 <= r <= 8: return float(c), float(9 - r)
    return None
def _mcs60_labels():
    return [f"{c}{r}" for c in range(1, 9) for r in range(1, 9) if f"{c}{r}" not in _CORN]
