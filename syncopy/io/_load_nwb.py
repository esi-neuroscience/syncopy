# -*- coding: utf-8 -*-
#
# Load data from NWB file
#

# Builtin/3rd party package imports
import os
import sys
import h5py
import subprocess
import numpy as np
from tqdm import tqdm

# Local imports
from syncopy import __nwb__
from syncopy.datatype.continuous_data import AnalogData
from syncopy.datatype.discrete_data import EventData
from syncopy.shared.errors import SPYError, SPYValueError, SPYWarning, SPYInfo
from syncopy.shared.parsers import io_parser, scalar_parser

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

__all__ = ["load_nwb"]


def load_nwb(filename, memuse=3000):
    """
    Read contents of NWB files

    Parameters
    ----------
    filename : str
        Name of (may include full path to) NWB file (e.g., `"/path/to/mydata.nwb"`).
    memuse : scalar
        Approximate in-memory cache size (in MB) for reading data from disk

    Returns
    -------
    angData : syncopy.AnalogData
        Any NWB `TimeSeries`-like data is imported into an :class:`~syncopy.AnalogData`
        object
    evtData : syncopy.EventData
        If the NWB file contains TTL pulse data, an additional :class:`~syncopy.EventData`
        object is instantiated
    """

    # Abort if NWB is not installed
    if not __nwb__:
        raise SPYError(nwbErrMsg.format("read_nwb"))

    # Check if file exists
    nwbPath, nwbBaseName = io_parser(filename, varname="filename", isfile=True, exists=True)
    nwbFullName = os.path.join(nwbPath, nwbBaseName)

    # Ensure `memuse` makes sense`
    try:
        scalar_parser(memuse, varname="memuse", lims=[0, np.inf])
    except Exception as exc:
        raise exc

    # First, perform some basal validation w/NWB
    try:
        this_python = os.path.join(os.path.dirname(sys.executable),'python')
        subprocess.run([this_python, "-m", "pynwb.validate", nwbFullName], check=True)
    except subprocess.CalledProcessError as exc:
        err = "NWB file validation failed. Original error message: {}"
        raise SPYError(err.format(str(exc)))

    # Load NWB meta data from disk
    nwbio = pynwb.NWBHDF5IO(nwbFullName, "r", load_namespaces=True)
    nwbfile = nwbio.read()

    # Allocate lists for storing temporary NWB info: IMPORTANT use lists to preserve
    # order of data chunks/channels
    nSamples = 0
    nChannels = 0
    chanNames = []
    tStarts = []
    sRates = []
    dTypes = []
    angSeries = []
    ttlVals = []
    ttlChanStates = []
    ttlChans = []
    ttlDtypes = []

    # If the file contains `epochs`, use it to infer trial information
    hasTrials = "epochs" in nwbfile.fields.keys()

    # Access all (supported) `acquisition` fields in the file
    for acqName, acqValue in nwbfile.acquisition.items():

        # Actual extracellular analog time-series data
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
            angSeries.append(acqValue)

        # TTL event pulse data
        elif ".TTLs" in str(acqValue.__class__):

            if acqValue.name == "TTL_PulseValues":
                ttlVals.append(acqValue)
            elif acqValue.name == "TTL_ChannelStates":
                ttlChanStates.append(acqValue)
            elif acqValue.name == "TTL_Channels":
                ttlChans.append(acqValue)
            else:
                lgl = "TTL data exported via `esi-oephys2nwb`"
                act = "unformatted TTL data '{}'"
                raise SPYValueError(lgl, varname=acqName, actual=act.format(acqValue.description))

            ttlDtypes.append(acqValue.data.dtype)
            ttlDtypes.append(acqValue.timestamps.dtype)

        # Unsupported
        else:
            lgl = "supported NWB data class"
            raise SPYValueError(lgl, varname=acqName, actual=str(acqValue.__class__))

    # If the NWB data is split up in "trials" (i.e., epochs), ensure things don't
    # get too wild (uniform sampling rates and timing offsets)
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
        trl[:, :2] = (epochs - tStarts[0]) * sRates[0]
        msg = "Found {} trials".format(trl.shape[0])
    else:
        trl = np.array([[0, nSamples, 0]])
        msg = "No trial information found. Proceeding with single all-encompassing trial"

    # Print status update to inform user
    SPYInfo(msg)
    SPYInfo("Creating AnalogData object...")

    # If TTL data was found, ensure we have exactly one set of values and associated
    # channel markers
    if max(len(ttlVals), len(ttlChans)) > min(len(ttlVals), len(ttlChans)):
        lgl = "TTL pulse values and channel markers"
        act = "pulses: {}, channels: {}".format(str(ttlVals), str(ttlChans))
        raise SPYValueError(lgl, varname=ttlVals[0].name, actual=act)
    if len(ttlVals) > 1:
        lgl = "one set of TTL pulses"
        act = "{} TTL data sets".format(len(ttlVals))
        raise SPYValueError(lgl, varname=ttlVals[0].name, actual=act)

    # Use provided TTL data to initialize `EventData` object
    evtData = None
    if len(ttlVals) > 0:
        msg = "Creating separate EventData object for embedded TTL pulse data..."
        SPYInfo(msg)
        evtData = EventData(dimord=EventData._defaultDimord)
        h5evt = h5py.File(evtData.filename, mode="w")
        evtDset = h5evt.create_dataset("data", dtype=np.result_type(*ttlDtypes),
                                       shape=(ttlVals[0].data.size, 3))
        # Column 1: sample indices
        # Column 2: TTL pulse values
        # Column 3: TTL channel markers
        if 'resolution' in ttlChans[0].__nwbfields__:
            ts_resolution = ttlChans[0].resolution
        else:
            ts_resolution = ttlChans[0].timestamps__resolution
            
        evtDset[:, 0] = ((ttlChans[0].timestamps[()] - tStarts[0]) / ts_resolution).astype(np.intp)
        evtDset[:, 1] = ttlVals[0].data[()]
        evtDset[:, 2] = ttlChans[0].data[()]
        evtData.data = evtDset
        evtData.samplerate = float(1 / ts_resolution)
        if hasTrials:
            evtData.trialdefinition = trl
        else:
            evtData.trialdefinition = np.array([[np.nanmin(evtDset[:,0]), np.nanmax(evtDset[:,0]), 0]])
            msg = "No trial information found. Proceeding with single all-encompassing trial"

    # Allocate `AnalogData` object and use generated HDF5 file-name to manually
    # allocate a target dataset for reading the NWB data
    angData = AnalogData(dimord=AnalogData._defaultDimord)
    angShape = [None, None]
    angShape[angData.dimord.index("time")] = nSamples
    angShape[angData.dimord.index("channel")] = nChannels
    h5ang = h5py.File(angData.filename, mode="w")
    angDset = h5ang.create_dataset("data", dtype=np.result_type(*dTypes), shape=angShape)

    # Compute actually available memory (divide by 2 since we're working with an add'l tmp array)
    pbarDesc = "Reading data in blocks of {} GB".format(round(memuse / 1000, 2))
    memuse *= 1024**2 / 2
    chanCounter = 0

    # Process analog time series data and save stuff block by block (if necessary)
    pbar = tqdm(angSeries, position=0)
    for acqValue in pbar:

        # Show dataset name in progress bar label
        pbar.set_description("Loading {} from disk".format(acqValue.name))

        # Given memory cap, compute how many data blocks can be grabbed per swipe:
        # `nSamp` is the no. of samples that can be loaded into memory without exceeding `memuse`
        # `rem` is the no. of remaining samples, s. t. ``nSamp + rem = angDset.shape[0]`
        # `blockList` is a list of samples to load per swipe, i.e., `[nSamp, nSamp, ..., rem]`
        nSamp = int(memuse / (np.prod(angDset.shape[1:]) * angDset.dtype.itemsize))
        rem = int(angDset.shape[0] % nSamp)
        blockList = [nSamp] * int(angDset.shape[0] // nSamp) + [rem] * int(rem > 0)

        # If channel-specific gains are set, load them now
        if acqValue.channel_conversion is not None:
            gains = acqValue.channel_conversion[()]

        # Write data block-wise to `angDset` (use `del` to wipe blocks from memory)
        # Use 'unsafe' casting to allow `tmp` array conversion int -> float
        endChan = chanCounter + acqValue.data.shape[1]
        for m, M in enumerate(tqdm(blockList, desc=pbarDesc, position=1, leave=False)):
            tmp = acqValue.data[m * nSamp: m * nSamp + M, :]
            if acqValue.channel_conversion is not None:
                np.multiply(tmp, gains, out=tmp, casting="unsafe")
            angDset[m * nSamp: m * nSamp + M, chanCounter : endChan] = tmp
            del tmp

        # Update channel counter for next `acqValue``
        chanCounter += acqValue.data.shape[1]

    # Finalize angData
    angData.data = angDset
    angData.channel = chanNames
    angData.samplerate = sRates[0]
    angData.trialdefinition = trl

    # Write logs
    msg = "Read data from NWB file {}".format(nwbFullName)
    angData.log = msg
    if evtData is not None:
        evtData.log = msg

    return angData, evtData
