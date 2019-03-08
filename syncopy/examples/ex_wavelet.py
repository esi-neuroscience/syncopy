# -*- coding: utf-8 -*-
##
# Created: 2019-02-25 13:08:56
# Last modified by: Joscha Schmiedt [joscha.schmiedt@esi-frankfurt.de]
# Last modification time: <2019-03-08 16:31:58>


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
if __dask__:
    import dask
    import dask.array as da
    from dask.distributed import get_client

# Import Spykewave
import syncopy as spy


def generate_artifical_data(nTrials=2, nChannels=2):
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
    data = generate_artifical_data(nChannels=5, nTrials=37)
    spec, lastres = spy.wavelet(data)
    import matplotlib.pyplot as plt
    plt.ion()
    plt.close('all')
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(data.time[0], data.trials[0].T)
    ax[1].pcolormesh(spec.time[0], spec.freq,
                     np.absolute(np.squeeze(spec.trials[3][:, 0, 0, :])).T)
    ax[1].set_yscale('log')
    ax[1].set_ylim([1, 100])
