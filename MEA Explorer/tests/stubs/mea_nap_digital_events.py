import os, numpy as np
def read_digital_events(path, stream_index=None):
    if not os.path.exists(path): raise FileNotFoundError(path)   # real one opens the h5
    return {"on_s": np.array([20.0, 50.0, 80.0, 110.0]), "off_s": np.array([21., 51., 81., 111.]),
            "intervals": None, "source": "stub", "fs": 25000.0, "detail": "stub"}
