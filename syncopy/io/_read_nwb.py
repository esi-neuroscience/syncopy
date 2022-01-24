# -*- coding: utf-8 -*-
#
# Load data from NWB file
#

# Builtin/3rd party package imports
from genericpath import exists
import numpy as np

# Local imports
from syncopy import __nwb__
from syncopy.shared.errors import SPYError
from syncopy.shared.parsers import io_parser

# Conditional imports
if __nwb__:
    from pynwb import NWBHDF5IO

# Global consistent error message if NWB is missing
nwbErrMsg = "\nSyncopy <core> WARNING: Could not import 'pynwb'. \n" +\
          "{} requires a working pyNWB installation. \n" +\
          "Please consider installing 'pynwb', e.g., via conda: \n" +\
          "\tconda install -c conda-forge pynwb\n" +\
          "or using pip:\n" +\
          "\tpip install pynwb"

__all__ = ["read_nwb"]


def read_nwb(filename):
    """
    Coming soon...
    """

    # Abort if NWB is not installed
    if not __nwb__:
        raise SPYError(nwbErrMsg.format("read_nwb"))

    nwbFilePath, nwbName = io_parser(filename, varname="filename", isfile=True, exists=True)

    nwbio = NWBHDF5IO(nwbFilePath, "r", load_namespaces=True)
    nwbfile = nwbio.read()

