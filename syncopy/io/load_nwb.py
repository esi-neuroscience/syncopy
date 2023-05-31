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


def load_nwb(filename, memuse=3000, container=None, validate=False):
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
    scalar_parser(memuse, varname="memuse", lims=[0, np.inf])

    # First, perform some basal validation w/NWB
    if validate:
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
    hasEpochs = "epochs" in nwbfile.fields.keys()
    hasTrials = "trials" in nwbfile.fields.keys()
    hasSpikedata = "units" in nwbfile.fields.keys()
    hasAcquisitions = "acquisition" in nwbfile.fields.keys()

    # Access all (supported) `acquisition` fields in the file
    for acqName, acqValue in nwbfile.acquisition.items():

        # Actual extracellular analog time-series data
        if isinstance(acqValue, pynwb.ecephys.ElectricalSeries):

            channels = acqValue.electrodes[:].location
            if channels.unique().size == 1:
                SPYWarning("No unique channel names found for {}".format(acqName))

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

    # TODO: Parse Spike Data (units and maybe waveforms) here.
    spikes_by_unit = None
    if hasSpikedata:
        SPYWarning("Spike data found in NWB file. This data is not yet supported by Syncopy.")
        units = nwbfile.units.to_dataframe()
        spikes_by_unit = {
                n: units.loc[n, "spike_times"] for n in units.index
        }
        # see https://github.com/pynapple-org/pynapple/blob/main/pynapple/io/neurosuite.py#L289

    # If the NWB data is split up in "trials" (or epochs), ensure things don't
    # get too wild (uniform sampling rates and timing offsets).
    if hasTrials or hasEpochs:
        if len(tStarts) < 1 or len(sRates) < 1:
            if hasSpikedata and not hasAcquisitions:
                tStarts.append(0.0)  # TODO: unclear where to get this from, fill in defaults for now.
                sRates.append(20000.0)
            else:
                raise SPYError("Found acquisitions and trials but no valid timing/samplerate data in NWB file. Data in file not supported.")
        if all(tStarts) is None or all(sRates) is None:
            lgl = "acquisition timings defined by `starting_time` and `rate`"
            act = "`starting_time` or `rate` not set"
            raise SPYValueError(lgl, varname="starting_time/rate", actual=act)
        if np.unique(tStarts).size > 1 or np.unique(sRates).size > 1:
            lgl = "acquisitions with unique `starting_time` and `rate`"
            act = "`starting_time` or `rate` different across acquisitions"
            raise SPYValueError(lgl, varname="starting_time/rate", actual=act)

        if hasTrials:
            time_intervals = nwbfile.trials[:]
            time_intervals_source = "trials"
        else:
            time_intervals = nwbfile.epochs[:]
            time_intervals_source = "epochs"
        SPYWarning(f"time_intervals taken from {time_intervals_source} has shape {time_intervals.shape} and type {type(time_intervals)}.")
        if not type(time_intervals) is np.ndarray:
            time_intervals = time_intervals.to_numpy()
            SPYWarning("converted to numpy")
        trl = np.zeros((time_intervals.shape[0], 3), dtype=np.intp) # TODO: check dtype?
        SPYWarning(f"tStarts: {tStarts}.")
        SPYWarning(f"sRates: {sRates}.")
        trial_start_stop = (time_intervals - tStarts[0]) * sRates[0]  # use offset relative to first acquisition
        SPYWarning(f"trial_start_stop has shape {trial_start_stop.shape}: {trial_start_stop}.")
        trl[:, 0:2] = trial_start_stop[:, 0:2]

        # If we found trials, we may be able to load the offset field from the trials
        # table. This is not guaranteed to work, though, as the offset field is only present if the
        # file was exported by Syncopy. If the field is not present, we do not do anything here, we just
        # proceed with the default zero offset.
        if hasTrials and "offset" in nwbfile.trials.colnames:
            df = nwbfile.trials.to_dataframe()
            SPYWarning(f"Type of nwbfile.trials[offset] is {type(nwbfile.trials['offset'])}")
            trl[:, 2] = df["offset"] * sRates[0]

        msg = "Found {} trials".format(trl.shape[0])
    else:
        trl = np.array([[0, nSamples, 0]])
        msg = "No trial information found. Proceeding with single all-encompassing trial"

    # Print status update to inform user
    log_msg = "Read data from NWB file {}".format(nwbFullName)

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
            SPYWarning("No unique channel names found for acquisition {}".format(acqName))
            angData.channel = None
        else:
            angData.channel = channels.to_list()
        angData.samplerate = sRates[0]
        angData.trialdefinition = trl
        angData.info = {'starting_time' : tStarts[0]}
        angData.log = log_msg
        objectDict[os.path.basename(angData.filename)] = angData

    if hasSpikedata and spikes_by_unit is not None:
        SPYWarning("TODO: add SpikeData instance here.")

    # Close NWB file
    nwbio.close()



    return objectDict
