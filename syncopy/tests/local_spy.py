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


# Prepare code to be executed using, e.g., iPython's `%run` magic command
if __name__ == "__main__":

    nTrials = 20

    nSamples = 1000
    fs = 500

    trls = []
    AdjMat = np.zeros((2, 2))
    # coupling from 0 to 1
    AdjMat[0, 1] = .15
    for _ in range(nTrials):

        # defaults AR(2) parameters yield 40Hz peak
        alphas = [.55, -.8]
        trl = synth_data.AR2_network(AdjMat, nSamples=nSamples,
                                     alphas=alphas)

        trls.append(trl)

    ad2 = spy.AnalogData(trls, samplerate=fs)
    spec = spy.freqanalysis(ad2, tapsmofrq=2, keeptrials=False)
    foi = np.linspace(40, 160, 25)
    spec2 = spy.freqanalysis(ad2, method='wavelet', keeptrials=False, foi=foi)
    coh = spy.connectivityanalysis(ad2, method='coh', tapsmofrq=5)
    gr = spy.connectivityanalysis(ad2, method='granger', tapsmofrq=10, polyremoval=0)

    # show new plotting
    ad2.singlepanelplot(trials=12, toilim=[0, 0.35])

    # mtmfft spectrum
    spec.singlepanelplot()
    # time freq singlepanel needs single channel
    spec2.singlepanelplot(channel=0, toilim=[0, 0.35])

    coh.singlepanelplot(channel_i=0, channel_j=1)

    gr.singlepanelplot(channel_i=0, channel_j=1, foilim=[40, 160])
    gr.singlepanelplot(channel_i=1, channel_j=0, foilim=[40, 160])

    # test top-level interface
    spy.singlepanelplot(ad2, trials=2, toilim=[-.2, .2])
