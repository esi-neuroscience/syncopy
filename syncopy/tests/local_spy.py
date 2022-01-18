# -*- coding: utf-8 -*-
#
# Simple script for testing Syncopy w/o pip-installing it
#

# Builtin/3rd party package imports
import numpy as np

# Add SynCoPy package to Python search path
import os
import sys
spy_path = os.path.abspath(".." + os.sep + "..")
if spy_path not in sys.path:
    sys.path.insert(0, spy_path)

# Import package
import syncopy as spy

# Import artificial data generator
from syncopy.tests.misc import generate_artificial_data
from syncopy.tests import synth_data


def call_con(data, method, **kwargs):

    res = spy.connectivity(data=data,
                           method=method,
                           **kwargs)
    return res


def call_freq(data, method, **kwargs):
    res = spy.freqanalysis(data=data, method=method, **kwargs)

    return res


# Prepare code to be executed using, e.g., iPython's `%run` magic command
if __name__ == "__main__":

    nSamples = 2500
    nChannels = 4
    nTrials = 10
    fs = 200

    foilim = [5, 80]
    foi = np.arange(5, 80, 1)
    # this still gives type(tsel) = slice :)
    sdict1 = {'channels' : ['channel01', 'channel03'], 'toilim' : [-.221, 1.12]}

    # AR(2) Network test data
    AdjMat = synth_data.mk_RandomAdjMat(nChannels)
    trls = [100 * synth_data.AR2_network(AdjMat) for _ in range(nTrials)]
    tdat1 = spy.AnalogData(trls, samplerate=fs)

    # phase difusion test data
    f1, f2 = 10, 40
    trls = []
    for _ in range(nTrials):

        p1 = synth_data.phase_evo(f1, eps=.01, nChannels=nChannels, nSamples=nSamples)
        p2 = synth_data.phase_evo(f2, eps=0.001, nChannels=nChannels, nSamples=nSamples)
        trls.append(
            1 * np.cos(p1) + 1 * np.cos(p2) + 0.6 * np.random.randn(
                nSamples, nChannels))

    tdat2 = spy.AnalogData(trls, samplerate=1000)


    # Test stuff within here...
    data1 = generate_artificial_data(nTrials=5, nChannels=16, equidistant=False, inmemory=False)
    data2 = generate_artificial_data(nTrials=5, nChannels=16, equidistant=True, inmemory=False)



    # client = spy.esi_cluster_setup(interactive=False)
    # data1 + data2

    # sys.exit()
    # spec = spy.freqanalysis(artdata, method="mtmfft", taper="dpss", output="pow")

