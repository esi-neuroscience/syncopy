# -*- coding: utf-8 -*-
#
# Load data from NWB file
#

# Builtin/3rd party package imports
import h5py
import subprocess
import numpy as np

# Local imports
from syncopy import __nwb__
from syncopy.datatype.continuous_data import AnalogData
from syncopy.shared.errors import SPYError, SPYValueError, SPYWarning
from syncopy.shared.parsers import io_parser

# Conditional imports
if __nwb__:
    import pynwb

# Global consistent error message if NWB is missing
nwbErrMsg = "\nSyncopy <core> WARNING: Could not import 'pynwb'. \n" +\
          "{} requires a working pyNWB installation. \n" +\
          "Please consider installing 'pynwb', e.g., via conda: \n" +\
          "\tconda install -c conda-forge pynwb\n" +\
          "or using pip:\n" +\
          "\tpip install pynwb"

__all__ = ["read_nwb"]


def read_nwb(filename, memuse=3000):
    """
    Coming soon...

    memuse : scalar
        Approximate in-memory cache size (in MB) for writing data to disk
    """

    # Abort if NWB is not installed
    if not __nwb__:
        raise SPYError(nwbErrMsg.format("read_nwb"))

    # Check if file exists
    nwbFullName, nwbBaseName = io_parser(filename, varname="filename", isfile=True, exists=True)

    # First, perform some basal validation w/NWB
    subprocess.run(["python", "-m", "pynwb.validate", nwbFullName], check=True)

    nwbio = pynwb.NWBHDF5IO(nwbFullName, "r", load_namespaces=True)
    nwbfile = nwbio.read()

    # Electrodes: nwbfile.acquisition['ElectricalSeries_1'].electrodes[:]

    # Trials: if "epochs" in nwbfile.fields.keys()

    nSamples = 0
    nChannels = 0
    chanNames = []
    tStarts = []
    sRates = []
    dTypes = []

    hasTrials = "epochs" in nwbfile.fields.keys()


    for acqName, acqValue in nwbfile.acquisition.items():

        if isinstance(acqValue, pynwb.ecephys.ElectricalSeries):

            channels = acqValue.electrodes[:].location
            if channels.unique().size == 1:
                SPYWarning("No channel names found for {}".format(acqName))
            else:
                chanNames += channels.to_list()

            dTypes.append(acqValue.data.dtype)
            if acqValue.channel_conversion is not None:
                dTypes.append(acqValue.channel_conversion.dtype)

            tStarts.append(acqValue.starting_time)
            sRates.append(acqValue.rate)
            nChannels += acqValue.data.shape[1]
            nSamples = max(nSamples, acqValue.data.shape[0])

        elif str(acqValue.__class__) == "TTLs":
            pass

        else:
            lgl = "supported NWB data class"
            raise SPYValueError(lgl, varname=acqName, actual=str(acqValue.__class__))


    if hasTrials:
        if all(tStarts) is None or all(sRates) is None:
            lgl = "acquisition timings defined by `starting_time` and `rate`"
            act = "`starting_time` or `rate` not set"
            raise SPYValueError(lgl, varname="starting_time/rate", actual=act)
        if np.unique(tStarts).size > 1 or np.unique(sRates).size > 1:
            lgl = "acquisitions with unique `starting_time` and `rate`"
            act = "`starting_time` or `rate` different across acquisitions"
            raise SPYValueError(lgl, varname="starting_time/rate", actual=act)
        epochs = nwbfile.epochs[:]
        trl = np.zeros((epochs.shape[0], 3), dtype=np.intp)
        trl[:, :2] = epochs * sRates[0]
    else:
        trl = np.array([[0, nSamples, 0]])

    angData = AnalogData()
    angShape = [None, None]
    angShape[angData._defaultDimord.index("time")] = nSamples
    angShape[angData._defaultDimord.index("channel")] = nChannels

    h5ang = h5py.File(angData.filename, mode="w")
    angDset = h5ang.create_dataset("data", dtype=np.result_type(*dTypes), shape=angShape)

    # Compute actually available memory (divide by 2 since we're working with an add'l tmp array)
    memuse *= 1024**2 / 2

    for acqName, acqValue in nwbfile.acquisition.items():

        if isinstance(acqValue, pynwb.ecephys.ElectricalSeries):

            # Given memory cap, compute how many data blocks can be grabbed per swipe
            nSamp = int(memuse / (np.prod(angDset.shape[1:]) * angDset.dtype.itemsize))
            rem = int(angDset.shape[0] % nSamp)
            nBlocks = [nSamp] * int(angDset.shape[0] // nSamp) + [rem] * int(rem > 0)

            # If channel-specific gains are set, load them now
            if acqValue.channel_conversion is not None:
                gains = acqValue.channel_conversion.data[()]

            # Write data block-wise to `angDset` (use `del` to wipe blocks from memory)
            for m, M in enumerate(nBlocks):
                tmp = acqValue.data[m * nSamp: m * nSamp + M, :]
                if acqValue.channel_conversion is not None:
                    tmp *= gains
                angDset[m * nSamp: m * nSamp + M, :] = tmp
                del tmp

