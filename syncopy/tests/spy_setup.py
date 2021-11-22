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

# Prepare code to be executed using, e.g., iPython's `%run` magic command
if __name__ == "__main__":

    # Test stuff within here...
    data1 = generate_artificial_data(nTrials=5, nChannels=16, equidistant=False, inmemory=False)
    data2 = generate_artificial_data(nTrials=5, nChannels=16, equidistant=True, inmemory=False)
    # client = spy.esi_cluster_setup(interactive=False)
    # data1 + data2

    sys.exit()
    spec = spy.freqanalysis(artdata, method="mtmfft", taper="dpss", output="pow")

