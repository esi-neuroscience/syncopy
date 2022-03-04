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
    nSamples = 200
    trls = []
    for _ in range(nTrials):
        # defaults AR(2) parameters yield 40Hz peak
        trls.append(synth_data.AR2_network(None, nSamples=nSamples))
    ad1 = spy.AnalogData(trls, samplerate=200)
    gr = spy.connectivityanalysis(ad1, method='granger', taper='dpss', tapsmofrq=3,
                                  foilim=[0, 100])
