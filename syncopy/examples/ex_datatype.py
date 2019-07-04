# -*- coding: utf-8 -*-
#
# Example script illustrating usage of SyNCoPy data objects
#
# Created: 2019-02-25 13:08:56
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-07-04 11:53:58>

# Builtin/3rd party package imports
import numpy as np

# Add SynCoPy package to Python search path
import os
import sys
spy_path = os.path.abspath(".." + os.sep + "..")
if spy_path not in sys.path:
    sys.path.insert(0, spy_path)

# Import SynCoPy
import syncopy as spy


# Dummy function for generating a docstring and function links
def ex_datatype():
    """
    Docstring...
    """


def time2sample(t, dt=0.001):
    return np.asarray(t / dt, dtype=np.uint64)


def sample2time(s, dt=0.001):
    return s * dt


# This prevents Sphinx from executing the script
if __name__ == "__main__":

    # Set path to data directory
    # datadir = ".." + os.sep + ".." + os.sep + ".." + os.sep + ".." + os.sep + "Data"\
    #           + os.sep + "testdata" + os.sep
    datadir = os.path.join(os.sep, "mnt", "hpx", "it",
                           "dev", "SpykeWave", "testdata")
    basename = "MT_RFmapping_session-168a1"

    files = [basename + "_xWav.lfp",
             basename + "_xWav.mua",
             basename + "_Raws.lfp",
             basename + "_Raws.mua"]
    channelLabels = []
    channelLabels.extend(["ecogLfp_{:03d}".format(idx) for idx in range(256)])
    channelLabels.extend(["ecogMua_{:03d}".format(idx) for idx in range(256)])
    channelLabels.extend(["vprobeLfp_{:03d}".format(idx) for idx in range(24)])
    channelLabels.extend(["vprobeMua_{:03d}".format(idx) for idx in range(24)])

    data = spy.AnalogData(filename=[os.path.join(datadir, file) for file in files],
                          filetype="esi")

    # define trials from photodiode onsets
    pdFile = os.path.join(datadir, basename + ".dpd")
    pdData = np.memmap(pdFile,
                       dtype=np.uint64,
                       mode='r',
                       offset=128).reshape((2, -1))
    # photodiode was recorded with 1017.25... Hz
    pdDt = 24 / 24414.0625

    # Trials start 200 ms before stimulus onset
    prestimTime = 0.2
    tOnset = sample2time(pdData[0, pdData[1, :] == 1], dt=pdDt) - prestimTime
    tOffset = sample2time(pdData[0, pdData[1, :] == 0], dt=pdDt) + 0.2
    iOnset = time2sample(tOnset, dt=1 / data.samplerate)
    iOffset = time2sample(tOffset, dt=1 / data.samplerate)
    iZero = time2sample(prestimTime, dt=1 / data.samplerate)

    trl = np.stack((iOnset, iOffset, np.ones(iOnset.shape, dtype=np.uint64) * iZero), axis=1)

    trl = trl[trl[:, 1] - trl[:, 0] > 500, :].astype(int)
    data = spy.AnalogData(filename=[os.path.join(datadir, file) for file in files],
                          filetype="esi", trialdefinition=trl, channel=channelLabels)

    data.save(os.path.join(datadir, basename))
