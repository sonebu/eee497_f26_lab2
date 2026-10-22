import numpy as np


def rectangular(n):
    return np.ones(n)


def hann(n):
    k = np.arange(n)
    return 0.5 - 0.5 * np.cos(2 * np.pi * k / n)


def blackman(n):
    k = np.arange(n)
    return 0.42 - 0.5 * np.cos(2 * np.pi * k / n) + 0.08 * np.cos(4 * np.pi * k / n)


def peak_bin_hz(block, fs, f_low, f_high):
    """Frequency of the tallest bin between f_low and f_high, in Hz."""
    n = len(block)
    X = np.fft.fft(np.asarray(block))
    k0 = int(f_low * n / fs)
    k1 = int(f_high * n / fs)
    k = k0 + np.argmax(np.abs(X[k0:k1]))
    return k * fs / n


def zero_crossing_hz(samples, fs):
    """Frequency from sign changes of a real sequence. Two crossings per cycle."""
    x = np.asarray(samples, float)
    x = x - np.mean(x)
    sign = np.sign(x)
    sign[sign == 0] = 1.0
    crossings = np.count_nonzero(sign[1:] * sign[:-1] < 0)
    return crossings * fs / (2.0 * len(x))


def parabolic_offset(left, peak, right):
    """Offset of the peak, in bins. left, peak, right are the three bin values."""
    a = np.log(abs(left) + 1e-30)
    b = np.log(abs(peak) + 1e-30)
    c = np.log(abs(right) + 1e-30)
    return 0.5 * (a - c) / (a - 2.0 * b + c)


def bandpass(x, fs, shift_hz, bandwidth_hz, taps=1024):
    """Keep a band of width bandwidth_hz centered at shift_hz.

    Mix shift_hz down to zero, lowpass, and mix back up. A tone that went
    in at some frequency comes out at that same frequency.
    """
    x = np.asarray(x)
    if taps % 2 == 0:
        taps += 1
    m = np.arange(taps) - (taps - 1) / 2.0
    h = (bandwidth_hz / fs) * np.sinc((bandwidth_hz / fs) * m)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(taps) / (taps - 1))
    h = h * w
    h = h / np.sum(h)
    n = np.arange(len(x))
    down = np.exp(-2j * np.pi * shift_hz * n / fs)
    y = np.convolve(x * down, h, mode="same")
    return y * np.conj(down)


def block_power(x, n_blk, hop):
    """Mean power of each block, and the sample index where the block starts."""
    powers = []
    starts = []
    for s in range(0, len(x) - n_blk, hop):
        blk = x[s:s + n_blk]
        powers.append(np.mean(np.abs(blk) ** 2))
        starts.append(s)
    return np.array(powers), np.array(starts)
