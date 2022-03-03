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

    mock_up = np.arange(24).reshape((8, 3))
    ad1 = spy.AnalogData([mock_up] * 5)

    nTrials = 50
    nSamples = 500
    fs = 500
    f1, f2 = 20, 40
    A1, A2 = 2, 3
    trls = []
    for _ in range(nTrials):
        sig1 = A1 * np.cos(f1 * 2 * np.pi * np.arange(nSamples) / fs)
        sig1 += A2 * np.cos(f2 * 2 * np.pi * np.arange(nSamples) / fs)
        sig2 = np.random.randn(nSamples)
        trls.append(np.vstack([sig1, sig2]).T)
    ad1 = spy.AnalogData(trls, samplerate=500)
    #spy.preprocessing(ad1, filter_class='d')
