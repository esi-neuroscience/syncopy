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
    f1, f2 = 20, 40
    A1, A2 = 2.3, 1

    trls = []
    for _ in range(nTrials):

        sig1 = A1 * np.cos(f1 * 2 * np.pi * np.arange(nSamples) / fs)
        sig1 += A2 * np.cos(f2 * 2 * np.pi * np.arange(nSamples) / fs)
        sig2 = np.random.randn(nSamples)
        trls.append(np.vstack([sig1, sig2]).T)
    ad1 = spy.AnalogData(trls, samplerate=500)
    #spy.preprocessing(ad1, filter_class='d')

    trls = []
    AdjMat = np.zeros((2, 2))
    AdjMat[0, 1] = .25
    for _ in range(nTrials):

        # defaults AR(2) parameters yield 40Hz peak
        alphas = [.74, -.46]  # broad peak at 60Hz
        alphas = [0.24, -.46]
        alphas = [.55, -.8]
        trl = synth_data.AR2_network(AdjMat, nSamples=nSamples,
                                     alphas=alphas)
        #trl = synth_data.AR2_network(None, nSamples=nSamples,
        #                             alphas=alphas)

        trls.append(trl)
    print(trl.mean())
    ad2 = spy.AnalogData(trls, samplerate=2000)

    spec = spy.freqanalysis(ad1, tapsmofrq=2, keeptrials=False)
    foi = np.linspace(10, 60, 25)
    spec2 = spy.freqanalysis(ad1, method='wavelet', keeptrials=False, foi=foi)
    # coh = spy.connectivityanalysis(ad2, method='coh', tapsmofrq=5)
    # gr = spy.connectivityanalysis(ad2, method='granger', tapsmofrq=10, polyremoval=0)
