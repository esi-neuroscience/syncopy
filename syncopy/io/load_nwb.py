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
from syncopy.shared.errors import SPYError, SPYTypeError, SPYValueError, SPYWarning, SPYInfo
from syncopy.shared.parsers import io_parser, scalar_parser, filename_parser

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


def load_nwb(filename, memuse=3000, container=None):
    """
    Read contents of NWB files

    Parameters
    ----------
    filename : str
        Name of (may include full path to) NWB file (e.g., `"/path/to/mydata.nwb"`).
    memuse : scalar
        Approximate in-memory cache size (in MB) for reading data from disk
    container : str
        Name of syncopy container folder to create the syncopy data in

    Returns
    -------
    objdict : dict
        Any NWB `TimeSeries`-like data is imported into an :class:`~syncopy.AnalogData`
        object. If the NWB file contains TTL pulse data, an additional
        :class:`~syncopy.EventData` object is instantiated. The syncopy
        objects are returned as a dictionary whose keys are the base-names
        (sans path) of the corresponding files.
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

            dTypes.append(acqValue.data.dtype)
            if acqValue.channel_conversion is not None:
                dTypes.append(acqValue.channel_conversion.dtype)

            tStarts.append(acqValue.starting_time)
            sRates.append(acqValue.rate)
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

    # Check for filename
    if container is not None:
        if not isinstance(container, str):
            raise SPYTypeError(container, varname="container", expected="str")
        if not os.path.splitext(container)[1] == ".spy":
            container += ".spy"
        if not os.path.isdir(container):
            os.makedirs(container)
        fileInfo = filename_parser(container)
        filebase = os.path.join(fileInfo["folder"],
                                fileInfo["container"],
                                fileInfo["basename"])

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
    objectDict = {}
    if len(ttlVals) > 0:
        msg = "Creating separate EventData object for embedded TTL pulse data..."
        SPYInfo(msg)
        if container is not None:
            filename = filebase + ".event"
        else:
            filename = None

        evtData = EventData(dimord=["sample","eventid","chans"], filename=filename)
        h5evt = h5py.File(evtData.filename, mode="w")
        evtDset = h5evt.create_dataset("data", dtype=int,
                                       shape=(ttlVals[0].data.size, 3))
        # Column 1: sample indices
        # Column 2: TTL pulse values
        # Column 3: TTL channel markers
        if 'resolution' in ttlChans[0].__nwbfields__:
            ts_resolution = ttlChans[0].resolution
        else:
            ts_resolution = ttlChans[0].timestamps__resolution

        evtDset[:, 0] = ((ttlChans[0].timestamps[()] - tStarts[0]) / ts_resolution).astype(np.intp)
        evtDset[:, 1] = ttlVals[0].data[()].astype(int)
        evtDset[:, 2] = ttlChans[0].data[()].astype(int)
        evtData.data = evtDset
        evtData.samplerate = float(1 / ts_resolution)
        if hasTrials:
            evtData.trialdefinition = trl
        else:
            evtData.trialdefinition = np.array([[np.nanmin(evtDset[:,0]), np.nanmax(evtDset[:,0]), 0]])
            msg = "No trial information found. Proceeding with single all-encompassing trial"

        # Write logs
        log_msg = "Read data from NWB file {}".format(nwbFullName)
        evtData.log = log_msg
        objectDict[os.path.basename(evtData.filename)] = evtData

    # Compute actually available memory
    pbarDesc = "Reading data in blocks of {} GB".format(round(memuse / 1000, 2))
    memuse *= 1024**2

    # Process analog time series data and convert stuff block by block (if necessary)
    pbar = tqdm(angSeries, position=0, disable=None)
    for acqValue in pbar:
        # Show dataset name in progress bar label
        pbar.set_description("Loading {} from disk".format(acqValue.name))

        # Allocate `AnalogData` object and use generated HDF5 file-name to manually
        # allocate a target dataset for reading the NWB data
        if container is not None:
            filename = filebase + "_" + acqValue.name + ".analog"
        else:
            filename = None

        angData = AnalogData(dimord=AnalogData._defaultDimord, filename=filename)
        angShape = [None, None]
        angShape[angData.dimord.index("time")] = acqValue.data.shape[0]
        angShape[angData.dimord.index("channel")] = acqValue.data.shape[1]
        h5ang = h5py.File(angData.filename, mode="w")
        angDset = h5ang.create_dataset("data", dtype=np.result_type(*dTypes), shape=angShape)

        # If channel-specific gains are set, load them now
        if acqValue.channel_conversion is not None:
            gains = acqValue.channel_conversion[()]
            if np.all(gains ==  gains[0]):
                gains = gains[0]

        # Given memory cap, compute how many data blocks can be grabbed per swipe:
        # `nSamp` is the no. of samples that can be loaded into memory without exceeding `memuse`
        # `rem` is the no. of remaining samples, s. t. ``nSamp + rem = angDset.shape[0]`
        # `blockList` is a list of samples to load per swipe, i.e., `[nSamp, nSamp, ..., rem]`
        nSamp = int(memuse / (acqValue.data.shape[1] * angDset.dtype.itemsize))
        rem = int(angDset.shape[0] % nSamp)
        blockList = [nSamp] * int(angDset.shape[0] // nSamp) + [rem] * int(rem > 0)

        for m, M in enumerate(tqdm(blockList, desc=pbarDesc, position=1, leave=False, disable=None)):
            st_samp, end_samp = m * nSamp, m * nSamp + M
            angDset[st_samp : end_samp, :] = acqValue.data[st_samp : end_samp, :]
            if acqValue.channel_conversion is not None:
                angDset[st_samp : end_samp, :] *= gains

        # Finalize angData
        angData.data = angDset
        channels = acqValue.electrodes[:].location
        if channels.unique().size == 1:
            SPYWarning("No channel names found for {}".format(acqName))
            angData.channel = None
        else:
            angData.channel = channels.to_list()
        angData.samplerate = sRates[0]
        angData.trialdefinition = trl
        angData.info = {'starting_time' : tStarts[0]}
        angData.log = log_msg
        objectDict[os.path.basename(angData.filename)] = angData

    return objectDict
