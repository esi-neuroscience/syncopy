# -*- coding: utf-8 -*-
##
# Created: 2019-02-25 13:08:56
# Last modified by: Joscha Schmiedt [joscha.schmiedt@esi-frankfurt.de]
# Last modification time: <2019-03-15 15:28:11>


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


def generate_artifical_data(nTrials=200, nChannels=512):
    dt = 0.001
    t = np.arange(0, 3, dt, dtype=np.float32) - 1.0
    sig = np.sin(2 * np.pi * (50 + 10 * np.sin(2 * np.pi * t))) * np.heaviside(t, 1)
    sig = np.repeat(sig[np.newaxis, :], axis=0, repeats=nChannels)
    sig = np.tile(sig, [1, nTrials])
    # sig += np.random.randn(*sig.shape) * 0.5
    sig = np.float32(sig)

    trialdefinition = np.zeros((nTrials, 3), dtype='int')
    for iTrial in range(nTrials):
        trialdefinition[iTrial, :] = np.array([iTrial * t.size, (iTrial + 1) * t.size, 1000])

    return spy.AnalogData(data=sig, dimord=["channel", "time"],
                          channel='channel', samplerate=1 / dt,
                          trialdefinition=trialdefinition)


if __name__ == "__main__":
    print("Generate data")
    dat = generate_artifical_data(nTrials=20, nChannels=3)
    print("Save data")
    dat.save('example2.spy')
    dat = None
    del dat
    print("Load data")
    data = spy.AnalogData(filename="example2.spy")
    # import socket
    # cluster = LocalCluster(ip=socket.gethostname(),
    #                        n_workers=3,
    #                        threads_per_worker=1,
    #                        memory_limit="3G",
    #                        processes=False)
    # client = Client(cluster)

    # print("Calculate spectra")
    # # result = spy.freqanalysis(data)
    # out = spy.SpectralData()
    # mtmfft = spy.MultiTaperFFT(1 / data.samplerate)
    # mtmfft.initialize(data)
    # # result = dask.delayed(mtmfft.compute(data, out))
    # result = mtmfft.compute(data, out, useDask=False)
    # spec.visualize('test.pdf')
    # out = result.compute()
    # plt.ion()
    # plt.plot(out.freq, np.squeeze(np.absolute(out.trials[0]).T))

    # print("Calculate wavelet spectra")
    outWavelet = spy.SpectralData()
    # import pdb
    # pdb.set_trace()
    wavelet = spy.WaveletTransform(1 / data.samplerate, stepsize=10)
    wavelet.initialize(data)
    wavelet.compute(data, outWavelet, useDask=False)

    #
    fix, ax = plt.subplots(2, 1)
    ax[0].pcolormesh(outWavelet.time[0], outWavelet.freq,
                     np.absolute(outWavelet.trials[0][:, 0, 0, :]).T)
    ax[1].plot(data.time[0], np.mean(data.trials[0], axis=0))
    # ax[0].set_yscale('log')
    ax[0].set_ylim([0, 100])
    plt.show()
