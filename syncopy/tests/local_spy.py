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
    alphas = [.55, -.8]
    adata = synth_data.AR2_network(nTrials, samplerate=fs,
                                   AdjMat=AdjMat,
                                   nSamples=nSamples,
                                   alphas=alphas)

    foi = np.linspace(40, 160, 25)
    spec = spy.freqanalysis(adata, tapsmofrq=2, keeptrials=False, foi=foi)

