# -*- coding: utf-8 -*-
# 
# 
# 
# Created: 2019-02-25 13:08:56
# Last modified by: Joscha Schmiedt [joscha.schmiedt@esi-frankfurt.de]
# Last modification time: <2019-04-29 15:22:16>



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
spy.cleanup()

def generate_artifical_data(nTrials=200, nChannels=512):
    dt = 0.001
    t = np.arange(0, 3, dt, dtype=np.float32) - 1.0
    sig = np.sin(2 * np.pi * (50 + 10 * np.sin(2 * np.pi * t))) * np.heaviside(t, 1)
    sig += 0.3 * np.sin(2 * np.pi * 4 * t)
    sig = np.repeat(sig[:, np.newaxis], axis=1, repeats=nChannels)
    sig = np.tile(sig, [nTrials, 1])
    sig += np.random.randn(*sig.shape) * 0.5
    sig = np.float32(sig)

    trialdefinition = np.zeros((nTrials, 3), dtype='int')
    for iTrial in range(nTrials):
        trialdefinition[iTrial, :] = np.array([iTrial * t.size, (iTrial + 1) * t.size, 1000])

    return spy.AnalogData(data=sig, dimord=["time", "channel"],
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
    # data = spy.AnalogData(filename="~/Projects/SyNCoPy/example2.spy")
    data = spy.AnalogData(filename="example2.spy")

    method = "with_dask"
    if method == "with_dask":
        import socket
        cluster = LocalCluster(ip=socket.gethostname(),
                               n_workers=3,
                               threads_per_worker=1,
                               memory_limit="4G",
                               processes=False)
        client = Client(cluster)

    print("Calculate tapered spectra")
    out = spy.SpectralData()
    mtmfft = spy.MultiTaperFFT(1 / data.samplerate, output="abs")
    mtmfft.initialize(data)
    result = mtmfft.compute(data, out, methodName=method)
    # out.save("mtmfft_spectrum")

    # print("Calculate wavelet spectra")
    # outWavelet = spy.SpectralData()
    # wavelet = spy.WaveletTransform(1 / data.samplerate, stepsize=10, output="abs")
    # wavelet.initialize(data)
    # wavelet.compute(data, outWavelet, methodName="sequentially")

    # #
    # fig, ax = plt.subplots(3, 1)
    # ax[0].pcolormesh(outWavelet.time[0], outWavelet.freq,
    #                  outWavelet.trials[0][:, 0, :, 0].T)
    # ax[0].set_ylim([0, 100])
    # ax[0].set_ylabel('Frequency (Hz)')

    # ax[1].plot(data.time[0], np.mean(data.trials[0], axis=1))
    # ax[1].set_xlabel('Time (s)')

    # ax[2].plot(out.freq, out.trials[0][0, 0, :, 0])
    # ax[2].plot(outWavelet.freq,
    #            np.squeeze(np.mean(outWavelet.trials[0][:, 0, :, 0], axis=0)))
    # ax[2].set_xlabel('Frequency (Hz)')
    # ax[2].set_ylabel('Power')
    # plt.show()
