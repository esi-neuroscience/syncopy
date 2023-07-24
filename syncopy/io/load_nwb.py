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
from syncopy.datatype.continuous_data import AnalogData
from syncopy.datatype.discrete_data import EventData, SpikeData
from syncopy.shared.errors import (
    SPYError,
    SPYTypeError,
    SPYValueError,
    SPYWarning,
    SPYInfo,
)
from syncopy.shared.parsers import io_parser, scalar_parser, filename_parser
from syncopy import __pynwb__

__all__ = ["load_nwb"]


if __pynwb__:
    import pynwb


def _is_valid_nwb_file(filename):
    try:
        this_python = os.path.join(os.path.dirname(sys.executable), "python")
        subprocess.run([this_python, "-m", "pynwb.validate", filename], check=True)
        return True, None
    except subprocess.CalledProcessError as exc:
        err = f"NWB file validation failed. Original error message: {str(exc)}"
        return False, err


def load_nwb(
    filename,
    memuse=3000,
    container=None,
    validate=False,
    default_spike_data_samplerate=None,
):
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
    default_spike_data_samplerate : float, optional
        The samplerate for spike data, in Hz. If not provided, the samplerate is read from the NWB file, but
        this is not guaranteed to work as some NWB files which contain only spike data do not store a
        samplerate. If this is `None` and no samplerate is found in the file, this function will raise an
        error, and you will have to provide the samplerate manually.

    Returns
    -------
    objdict : dict
        Any NWB `TimeSeries`-like data is imported into an :class:`~syncopy.AnalogData`
        object. If the NWB file contains TTL pulse data, an additional
        :class:`~syncopy.EventData` object is instantiated. The syncopy
        objects are returned as a dictionary whose keys are the base-names
        (sans path) of the corresponding files.
    """
    if not __pynwb__:
        raise SPYError("NWB support is not available. Please install the 'pynwb' package.")

    # Check if file exists
    nwbPath, nwbBaseName = io_parser(filename, varname="filename", isfile=True, exists=True)
    nwbFullName = os.path.join(nwbPath, nwbBaseName)

    # Ensure `memuse` makes sense`
    scalar_parser(memuse, varname="memuse", lims=[0, np.inf])

    # First, perform some basal validation w/NWB if requested.
    if validate:
        is_valid, err = _is_valid_nwb_file(nwbFullName)
        if not is_valid:
            raise SPYError(err)

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

    # Access LFPs in ecephys processing module, if any.
    try:
        lfp = nwbfile.processing["ecephys"]["LFP"]["ElectricalSeries"]

        if isinstance(lfp, pynwb.ecephys.ElectricalSeries):

            channel_names = lfp.electrodes[:].location
            if channel_names.unique().size == 1:
                SPYWarning("No unique channel names found for LFP.")

            dTypes.append(lfp.data.dtype)
            if lfp.channel_conversion is not None:
                dTypes.append(lfp.channel_conversion.dtype)

            tStarts.append(lfp.starting_time)
            sRates.append(lfp.rate)
            nSamples = max(nSamples, lfp.data.shape[0])
            angSeries.append(lfp)
    except KeyError:
        pass

    # Access all (supported) `acquisition` fields in the file
    for acqName, acqValue in nwbfile.acquisition.items():

        # Actual extracellular analog time-series data
        if isinstance(acqValue, pynwb.ecephys.ElectricalSeries):

            channel_names = acqValue.electrodes[:].location
            if channel_names.unique().size == 1:
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
            lgl = "supported NWB Acquisition data class"
            raise SPYValueError(lgl, varname=acqName, actual=str(acqValue.__class__))

    # Load Spike Data from units. The data gets turned into a SpikeData object later.
    spikes_by_unit = None
    units = None
    if hasSpikedata:
        units = nwbfile.units.to_dataframe()
        spikes_by_unit = {n: units.loc[n, "spike_times"] for n in units.index}

    # If the NWB data is split up in "trials" (or epochs), ensure things don't
    # get too wild (uniform sampling rates and timing offsets).
    if hasTrials or hasEpochs:
        if len(tStarts) < 1 or len(sRates) < 1:
            if (
                hasSpikedata and not hasAcquisitions
            ):  # There may be no samplerate read from acquisitions because there are no acquisitions, but only spike data.
                samplerate = default_spike_data_samplerate
                if samplerate is None:
                    if "samplerate" in units.columns:
                        samplerate = units.loc[:, "samplerate"].unique()[0]
                        sRates.append(samplerate)
                        tStarts.append(0.0)
                    else:
                        raise SPYError(
                            "Could not read samplerate for spike data from NWB file. Please provide a samplerate manually via parameter 'default_spike_data_samplerate'."
                        )
            else:
                raise SPYError(
                    "Found acquisitions and trials but no valid timing/samplerate data in NWB file. Data in file not supported."
                )
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
        else:
            time_intervals = nwbfile.epochs[:]
        if not type(time_intervals) is np.ndarray:
            time_intervals = time_intervals.to_numpy()
        trl = np.zeros((time_intervals.shape[0], 3), dtype=np.intp)
        trial_start_stop = (time_intervals - tStarts[0]) * sRates[
            0
        ]  # use offset relative to first acquisition
        trl[:, 0:2] = trial_start_stop[:, 0:2]

        # If we found trials, we may be able to load the offset field from the trials
        # table. This is not guaranteed to work, though, as the offset field is only present if the
        # file was exported by Syncopy. If the field is not present, we do not do anything here, we just
        # proceed with the default zero offset.
        if hasTrials and "offset" in nwbfile.trials.colnames:
            df = nwbfile.trials.to_dataframe()
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
        filebase = os.path.join(fileInfo["folder"], fileInfo["container"], fileInfo["basename"])

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

        evtData = EventData(dimord=["sample", "eventid", "chans"], filename=filename)
        h5evt = h5py.File(evtData.filename, mode="w")
        evtDset = h5evt.create_dataset("data", dtype=int, shape=(ttlVals[0].data.size, 3))
        # Column 1: sample indices
        # Column 2: TTL pulse values
        # Column 3: TTL channel markers
        if "resolution" in ttlChans[0].__nwbfields__:
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
            evtData.trialdefinition = np.array([[np.nanmin(evtDset[:, 0]), np.nanmax(evtDset[:, 0]), 0]])
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
        numDataChannels = acqValue.data.shape[1] if acqValue.data.ndim > 1 else 1
        angShape[angData.dimord.index("channel")] = numDataChannels
        h5ang = h5py.File(angData.filename, mode="w")
        angDset = h5ang.create_dataset("data", dtype=np.result_type(*dTypes), shape=angShape)

        # If channel-specific gains are set, load them now
        if acqValue.channel_conversion is not None:
            gains = acqValue.channel_conversion[()]
            if np.all(gains == gains[0]):
                gains = gains[0]

        # Given memory cap, compute how many data blocks can be grabbed per swipe:
        # `nSamp` is the no. of samples that can be loaded into memory without exceeding `memuse`
        # `rem` is the no. of remaining samples, s. t. ``nSamp + rem = angDset.shape[0]`
        # `blockList` is a list of samples to load per swipe, i.e., `[nSamp, nSamp, ..., rem]`
        nSamp = int(memuse / (numDataChannels * angDset.dtype.itemsize))
        rem = int(angDset.shape[0] % nSamp)
        blockList = [nSamp] * int(angDset.shape[0] // nSamp) + [rem] * int(rem > 0)

        for m, M in enumerate(tqdm(blockList, desc=pbarDesc, position=1, leave=False, disable=None)):
            st_samp, end_samp = m * nSamp, m * nSamp + M
            angDset[st_samp:end_samp, :] = acqValue.data[st_samp:end_samp, :]
            if acqValue.channel_conversion is not None:
                angDset[st_samp:end_samp, :] *= gains

        # Finalize angData
        angData.data = angDset
        channel_names = acqValue.electrodes[:].location

        if channel_names.size != numDataChannels:
            SPYWarning(
                f"Found {channel_names.size} channel names for data with {numDataChannels} channels in NWB file. Discarding channel names."
            )
            angData.channel = None

        if channel_names.unique().size == 1 and channel_names.size > 1:
            SPYWarning(
                "No unique channel names found for acquisition {}. Discarding channel names.".format(acqName)
            )
            angData.channel = None
        else:
            angData.channel = channel_names.to_list()
        angData.samplerate = sRates[0]
        angData.trialdefinition = trl
        angData.info = {"starting_time": tStarts[0]}
        angData.log = log_msg
        objectDict[os.path.basename(angData.filename)] = angData

    if hasSpikedata and spikes_by_unit is not None:
        dsetname = "nwbspike"  # TODO: Can we get a name for this somwhere in the NWB file?
        if container is not None:
            filename = filebase + "_" + dsetname + ".spike"
        else:
            filename = None

        spData = SpikeData(dimord=SpikeData._defaultDimord, filename=filename)

        # Convert spike times to Syncopy format: load one vector for time, unit, and channel, repectively.
        spike_times = np.sort(np.concatenate([np.array(i) for i in spikes_by_unit.values()]))
        spike_units = np.concatenate([np.array([i] * len(spikes_by_unit[i])) for i in spikes_by_unit.keys()])
        spike_channels = np.array([0] * len(spike_times))  # single channel, map all to channel 0.

        # Try to get the samplerate from the NWB file
        samplerate = sRates[0]
        spike_data_sampleidx = np.column_stack(
            (np.rint(spike_times * samplerate), spike_channels, spike_units)
        )
        hdf5_file = h5py.File(spData.filename, mode="w")

        spDset = hdf5_file.create_dataset("data", data=spike_data_sampleidx, dtype=np.int64)

        # Finally, assign the dataset to the SpikeData object.
        spData.data = spDset

        # Fill other fields
        spData.channel = [
            "channel0"
        ]  # No channel information is saved in NWB files for spike data, only unit information.
        spData.samplerate = samplerate
        spData.trialdefinition = trl
        spData.info = {"starting_time": tStarts[0]}
        spData.log = log_msg

        # Add loaded Syncopy data object to list of objects to return
        objectDict[os.path.basename(spData.filename)] = spData

    # Close NWB file
    nwbio.close()

    return objectDict
