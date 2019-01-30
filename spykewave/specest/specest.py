# specest.py - SpykeWave spectral estimation methods
# 
# Created: January 22 2019
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-01-30 13:49:36>

import numpy as np
import scipy.signal as signal
import scipy.signal.windows as windows


__all__ = ["mtmfft"]

def mtmfft(data, dt=0.001, taper=windows.hann, pad="nextpow2", padtype="zero",
           polyorder=None, taperopt={}, fftAxis=1, tapsmofrq=None):

    if data.ndim > 2:
        raise np.AxisError("Number of data dimensions must be <= 2")

    # move fft/samples dimension into first place
    data = np.moveaxis(np.atleast_2d(data), fftAxis, 1)

    nSamples = data.shape[1]
    nChannels = data.shape[0]
    T = nSamples * dt
    fsample = 1 / dt

    if polyorder:
        raise NotImplementedError("Detrending has not been implemented yet.")

    # padding
    if pad:
        padWidth = np.zeros((data.ndim, 2), dtype=int)
        if pad == "nextpow2":
            padWidth[1, 0] = nextpow2(nSamples) - nSamples
        else:
            padWidth[1, 0] = np.ceil((pad - T) / dt).astype(int)
        if padtype == "zero":
            data = np.pad(data, pad_width=padWidth,
                          mode="constant", constant_values=0)

        # update number of samples
        nSamples = data.shape[1]

    if taper == windows.dpss and (not taperopt):
        nTaper = np.int(np.floor(tapsmofrq * T))
        taperopt = {"NW": tapsmofrq, "Kmax": nTaper}

    # compute taper in shape nTaper x nSamples
    win = np.atleast_2d(taper(nSamples, **taperopt))

    # construct frequency axis
    df = fsample / nSamples
    freq = np.arange(0, np.floor(nSamples / 2) + 1) * df

    # compute spectra
    spec = np.zeros((win.shape[0],) + (nChannels,) +
                    (freq.size,), dtype=complex)
    for wIdx, tap in enumerate(win):

        if data.ndim > 1:
            tap = np.tile(tap, (nChannels, 1))

        # taper x chan x freq
        spec[wIdx, ...] = np.fft.rfft(data * tap, axis=1)

    return freq, spec, win


def nextpow2(number):
    n = 1
    while n < number:
        n *= 2
    return n


def create_test_data(frequencies=(6.5, 22.75, 67.2, 100.5)):
    freq = np.array(frequencies)
    amp = np.ones(freq.shape)
    phi = np.random.rand(len(freq)) * 2 * np.pi
    signal = np.random.rand(2000) + 0.3
    dt = 0.001
    t = np.arange(signal.size) * dt
    for idx, f in enumerate(freq):
        signal += amp[idx] * np.sin(2 * np.pi * f * t + phi[idx])
    return signal


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()
    data = create_test_data()
    freq, spec, win = mtmfft(data, dt=0.001, pad="nextpow2",
                             taper=windows.hann,
                             tapsmofrq=2)
    fig, ax = plt.subplots(3)
    ax[0].plot(data)
    ax[1].plot(freq, np.squeeze(np.mean(np.absolute(spec), axis=0)), '.-')

    ax[1].set_xlim([-0.5, 105.5])
    ax[2].plot(win.T)
    plt.draw()
