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

from pynwb import NWBHDF5IO


# Prepare code to be executed using, e.g., iPython's `%run` magic command
if __name__ == "__main__":

    nwbFilePath = "/home/fuertingers/Documents/job/SyNCoPy/Data/tt.nwb"
    # nwbFilePath = "/home/fuertingers/Documents/job/SyNCoPy/Data/test.nwb"

    nwbio = NWBHDF5IO(nwbFilePath, "r", load_namespaces=True)
    nwbfile = nwbio.read()

