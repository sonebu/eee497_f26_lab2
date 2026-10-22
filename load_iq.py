import numpy as np

fs = 2048000


def load_iq(path):
    raw = np.fromfile(path, dtype=np.uint8).astype(float)
    x = (raw[0::2] - 127.0) / 128.0 + 1j * (raw[1::2] - 127.0) / 128.0
    return x, float(fs)
