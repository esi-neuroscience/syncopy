# -*- coding: utf-8 -*-
##
# Created: 2019-02-25 13:08:56
# Last modified by: Joscha Schmiedt [joscha.schmiedt@esi-frankfurt.de]
# Last modification time: <2019-03-14 17:34:18>


# Builtin/3rd party package imports
import numpy as np
import scipy.signal as signal
# Add spykewave package to Python search path
import os
import sys
import matplotlib.pyplot as plt
spw_path = os.path.abspath(".." + os.sep + "..")
if spw_path not in sys.path:
    sys.path.insert(0, spw_path)

from syncopy import __dask__
import dask
import dask.array as da
from dask.distributed import get_client
from dask.distributed import Client, LocalCluster

# Import Spykewave
import syncopy as spy


def generate_artifical_data(nTrials=100, nChannels=64):
    dt = 0.001
    t = np.arange(0, 3, dt) - 1.0
    sig = np.cos(2 * np.pi * (7 * (np.heaviside(t, 1) * t - 1) + 10) * t)
    sig = np.repeat(sig[np.newaxis, :], axis=0, repeats=nChannels)
    sig = np.tile(sig, [1, nTrials])
    sig += np.random.randn(*sig.shape) * 0.5
    sig = np.float32(sig)

    trialdefinition = np.zeros((nTrials, 3), dtype='int')
    for iTrial in range(nTrials):
        trialdefinition[iTrial, :] = np.array([iTrial * t.size, (iTrial + 1) * t.size, 1000])

    return spy.AnalogData(data=sig, dimord=["channel", "time"],
                          channel='channel', samplerate=1 / dt,
                          trialdefinition=trialdefinition)


if __name__ == "__main__":
    print("Generate data")
    dat = generate_artifical_data(nTrials=50, nChannels=6)
    print("Save data")
    dat.save('example.spy')
    del dat
    print("Load data")
    data = spy.AnalogData(filename="example.spy")
    import socket
    cluster = LocalCluster(ip=socket.gethostname(),
                           n_workers=6,
                           threads_per_worker=1,
                           memory_limit="2G",
                           processes=False)
    client = Client(cluster)

    print("Calculate spectra")
    # result = spy.freqanalysis(data)
    out = spy.SpectralData()
    mtmfft = spy.MultiTaperFFT(1 / data.samplerate)
    mtmfft.initialize(data)
    result = dask.delayed(mtmfft.compute(data, out))
    # spec.visualize('test.pdf')
    out = result.compute()
    # plt.ion()
    # plt.plot(out.freq, np.squeeze(np.absolute(out.trials[0]).T))

    print("Calculate wavelet spectra")
    outWavelet = spy.SpectralData()
    wavelet = spy.WaveletTransform(1 / data.samplerate)
    wavelet.initialize(data)
    wavelet.compute(data, outWavelet)
    plt.pcolormesh(outWavelet.time[0], outWavelet.freq,
                   np.absolute(outWavelet.trials[0][:, 0, 0, :]).T)
    plt.show()
