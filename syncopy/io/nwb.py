# -*- coding: utf-8 -*-
#
# Common routines for saving Syncopy data objects to Neurodata Withour Borders (NWB) files.
#
# Note that NWB is a very general container format, and that a software can read NBW files does
# not mean that it can interpret the data stored by a different software into an NWB file.
#
# The NWB format is documented at https://pynwb.readthedocs.io/en/stable/
#
# We have support for exporting a subset of the Syncopy data objects to NWB files. Specifically,
# AnalogData, TimeLockData, and SpikeData can be exported and re-read.
#
# For NWB files created by other software, our loaders can be used as a rough guide, but users
# will have to adapt the code to their specific needs and files.

# Builtin/3rd party package imports
import numpy as np
from datetime import datetime
from uuid import uuid4
import pytz
import os
import shutil

from syncopy import __pynwb__


if __pynwb__:
    import pynwb
    from pynwb import NWBFile
    from pynwb.ecephys import LFP, ElectricalSeries
    from hdmf.common import (
        DynamicTableRegion,
    )  # hdmf is a dependency of pynwb, so this should be available.

# Local imports

__all__ = []


def _get_nwbfile_template(channels=None):
    """
    Get a template NWBFile object with some basic metadata. No data or electrodes are added.

    This NWBFile object is not tied to any filesystem file. This template contains the value 'unknown' for most fields, but
    users are free to modify the NWBFile object before writing it to disk, see the pynwb documentation for details.

    Parameters
    ----------
    channels: list of str, the channel names. The length of this list determines the number of channels. The channel names are used as labels for the electrodes.
    the correct amount of electrodes for the data's channel count must already exist, or be added later before
    adding data.
    """
    start_time_no_tz = datetime.now()
    tz = pytz.timezone("Europe/Berlin")
    start_time = tz.localize(start_time_no_tz)

    nwbfile = NWBFile(
        session_description="unknown",  # required
        identifier=str(uuid4()),  # required
        session_start_time=start_time,  # required and relevant, use something like `datetime(2018, 4, 25, 2, 30, 3, tzinfo=tz.gettz("US/Pacific"))` for real data.
        session_id="session_0001",  # optional. remember that one file is for one session.
        experimenter=[
            "unknown",
        ],  # optional, name of experimenters
        lab="unknown",  # optional, the research lab where the experiment was performed
        institution="unknown",  # optional
        experiment_description="unknown",  # optional
        related_publications="",  # put a DOI here, if any. e.g., "https://doi.org/###"
    )
    # When creating your own NWBFile, you can also add subject information by setting `nwbfile.subject` to a `pynwb.file.Subject` instance, see docs.

    nwbfile, _ = _add_electrodes(nwbfile, channels)
    return nwbfile


def _add_electrodes(nwbfile, channels):
    """Add the channel information (an 'electrode_table_region') to the NWBFile object `nwbfile`.

    Parameters
    ----------
    nwbfile: `pynwb.NWBFile` object, the NWBFile instance to which to add the channel information.

    channels: list of str, the channel names. The length of this list determines the number of channels. The channel names are used as labels for the electrodes.

    Returns
    -------
    nwbfile: `pynwb.NWBFile` object with the channel information added. Note that this is the same object as the input `nwbfile`.

    electrodes: `hdmf.common.table.DynamicTableRegion`, an electrode table region object. It represents a reference to
      the part of the electrode table that was added by this function call (the respective rows, one row per electrode).
      We assume the table is empty before calling this function, so
      the region spans the entire electrode table. The full table is of type `hdmf.common.table.DynamicTable`.
    """
    # Add the channel information.
    ## First add the recording array
    nchannels_per_shank = len(channels)
    nshanks = 1  # We assume 1 shank.

    device = nwbfile.create_device(
        name="array", description="Unknown array", manufacturer="Unknown manufacturer"
    )
    nwbfile.add_electrode_column(name="label", description="label of electrode")

    # the number of shanks in the array, each is placed in a separate electrode group. We assume 1 shank.
    electrode_counter = 0  # Total number of electectrodes in the electrode table. This is used to create the electrode table region.

    for ishank in range(nshanks):
        # create an electrode group for this shank
        electrode_group = nwbfile.create_electrode_group(
            name="shank{}".format(ishank),
            description="electrode group for shank {}".format(ishank),
            device=device,
            location="unknown brain area",
        )
        # add electrodes to the electrode table
        for ielec in range(nchannels_per_shank):
            nwbfile.add_electrode(
                x=0.0,
                y=0.0,
                z=0.0,
                imp=float("nan"),
                filtering="unknown",
                group=electrode_group,
                label="shank{}elec{}".format(ishank, ielec),
                # location="unknown brain area (shank {}, elec {})".format(ishank, ielec),
                location=channels[ielec],
            )
            electrode_counter += 1

    electrodes = nwbfile.create_electrode_table_region(
        region=list(range(electrode_counter)),  # reference row indices 0 to N-1 in the electrode table.
        description="all electrodes",
    )
    return nwbfile, electrodes


def _analog_timelocked_to_nwbfile(
    atdata,
    nwbfile=None,
    with_trialdefinition=True,
    is_raw=True,
    elec_series_name="ElectricalSeries",
):
    """Convert `AnalogData` or `TimeLockData` into a `pynwb.NWBFile` instance,
    for writing to files in Neurodata Without Borders (NWB) file format.
    An NWBFile represents a single session of an experiment.

    Parameters
    ----------
    atdata : :class:`syncopy.AnalogData` or :class:`syncopy.TimeLockData`object, the data object to be converted to NWB.

    nwbfile : :class:`pynwb.NWBFile` object or None. If `None`, a new NWBFile will be created.
        It is highly recommended to create your own NWBFile object and pass it to this function,
        as this will allow you to add metadata to the file. If this is `None`, all metadata fields will be set to `'unknown'`.

    with_trialdefinition : Boolean, whether to save the trial definition in the NWB file.

    is_raw : Boolean, whether this is raw data (that should never change), or LFP data that originates from some preprocessing,
        e.g., down-sampling and detrending. Determines where data is stored in the NWB container, to make it easier for
        other software to interprete what the data represents. If `is_raw` is `True`, the `ElectricalSeries` is stored
        directly in an acquisition of the :class:`pynwb.NWBFile`. If False, it is stored inside an `LFP` instance in a
        processing group called `ecephys`.
        Note that for the Syncopy NWB reader, the data should be stored as raw, so this is currently the default.

    Returns
    -------
    :class:`pynwb.NWBFile` object, the NWBFile instance that contains the data.

    Notes
    -----
    This internal function is provided such that you can use it to create an NWBFile instance, and then modify it before writing it to disk.
    """
    # See https://pynwb.readthedocs.io/en/stable/tutorials/domain/ecephys.html
    # It is also worth veryfying that the web tool nwbexplorer can read the produced files, see http://nwbexplorer.opensourcebrain.org/.

    if nwbfile is None:
        nwbfile = _get_nwbfile_template(atdata.channel)

    electrode_region = DynamicTableRegion(
        name="electrodes",
        data=list(range(len(atdata.channel))),
        description="All electrodes.",
        table=nwbfile.electrodes,
    )

    # Now that we have an NWBFile and channels, we can add the data.
    time_series_with_rate = ElectricalSeries(
        name=elec_series_name,
        data=atdata.data,
        electrodes=electrode_region,
        starting_time=0.0,
        rate=atdata.samplerate,  # Fixed sampling rate.
        description="Electrical time series dataset",
        comments="Exported by Syncopy",
    )

    if (
        is_raw
    ):  # raw measurements from instruments, not to be changed. Not downsampled, detrended, or anything. This is not enforced technically, but it is a convention.
        nwbfile.add_acquisition(time_series_with_rate)
    else:  # LFP, data used for analysis, or the result of an analysis.
        lfp = LFP(electrical_series=time_series_with_rate)
        ecephys_module = nwbfile.create_processing_module(name="ecephys", description=atdata._log)
        ecephys_module.add(lfp)

    # Add trial definition, if possible and requested.
    _add_trials_to_nwbfile(nwbfile, atdata.trialdefinition, atdata.samplerate, do_add=with_trialdefinition)

    return nwbfile


def _add_trials_to_nwbfile(nwbfile, trialdefinition, samplerate, do_add=True, save_as="both"):
    """Add trial definition to an existing NWBFile.

    Parameters
    ----------
    nwbfile : :class:`pynwb.NWBFile` object, the NWBFile instance that contains the data.

    trialdefinition : :class:`numpy.ndarray` of shape (N, 3), the trial definition in Syncopy format. Each row contains the start time, stop time, and offset of a trial.

    samplerate: float, the sampling rate of the data in Hz

    do_add : Boolean, whether to add the trial definition to the NWB file. If `False`, this function does nothing.

    save_as : str, how to store the trials in the NWB file. Must be one of `'both'`, `'epochs'`, or `'trials'`. Determines into which interval table of the NWB file the time intervals are saved: epochs table, trials table, or both.

    Returns
    -------
    None
    """
    # Add the trial definition, if any.
    if trialdefinition is None or not do_add:
        return

    if nwbfile.trials is None or not "offset" in nwbfile.trials.colnames:
        nwbfile.add_trial_column(
            name="offset",
            description="The offset of the trial.",
        )
    for trial_idx in range(trialdefinition.shape[0]):
        td = trialdefinition[trial_idx, :].astype(np.float64) / samplerate  # Compute time from sample number.
        if save_as == "both" or save_as == "trials":
            nwbfile.add_trial(start_time=td[0], stop_time=td[1], offset=td[2])
        if save_as == "both" or save_as == "epochs":
            nwbfile.add_epoch(start_time=td[0], stop_time=td[1], tags="trial {}".format(trial_idx))


def _spikedata_to_nwbfile(sdata, nwbfile=None, with_trialdefinition=True, unit_info=None):
    """Convert SpikeData into pynwb.NWBFile instance, for writing to files in Neurodata Without Borders (NWB) file format.
    An NWBFile represents a single session of an experiment.

    Parameters
    ----------
    sdata : :class:`syncopy.AnalogData` or :class:`syncopy.TimeLockData`object, the data object to be converted to NWB.

    nwbfile : :class:`pynwb.NWBFile` object or None. If `None`, a new NWBFile will be created. It is highly recommended to create
     your own NWBFile object and pass it to this function, as this will allow you to add metadata to the file. If this is `None`, all metadata fields will be set to `'unknown'`.

    with_trialdefinition : Boolean, whether to save the trial definition to the NWB file.

    unit_info : dict of dicts or None, metadata for the units (neurons). The outer dict must have the two keys 'location' and 'group', holding one dict each. Inner dicts are of type `<int, str>` and map numeric unit ids to their location and group, respectively. Both location and group are freeform strings.

    Returns
    -------
    :class:`pynwb.NWBFile` object, the NWBFile instance that contains the data. Note that channel information is lost, as it is
    not stored in unit data structure in the NWB file. With spike data, the channel is not relevant anymore, as spike sorting has already been performed
    and thus neurons have been identified. The unit (neuron) is the relevant entity here.

    Also note that NWB format does not save spikes as spike indices, but as spike times. Thus, the spike times are converted to spike times before saving.
    Due to that, the NWB file also does not save the samplerate. While this is okay for saving the data, it is not okay for reading it back in, as the
    samplerate is needed to convert the spike times back to spike indices. Thus, the samplerate is saved as a unit metadata field, and the Syncopy NWB reader
    will use this to convert the spike times back to spike indices when creating the `spy.SpikeData` instance.

    Notes
    -----
    This internal function is provided such that you can use it to create an NWBFile instance, and then modify it before writing it to disk.
    """
    # See https://pynwb.readthedocs.io/en/stable/tutorials/domain/ecephys.html
    # It is also worth veryfying that the web tool nwbexplorer can read the produced files, see http://nwbexplorer.opensourcebrain.org/.

    num_channels = 1

    if nwbfile is None:
        nwbfile = _get_nwbfile_template(channels=["Channel0"])

    # Now that we have an NWBFile and channels, we can add the data.
    # cf. https://github.com/pynapple-org/pynapple/blob/main/pynapple/io/neurosuite.py#L212 to be
    # compatible with Neurosuite/Pynapple.
    # electrode_region = nwbfile.electrodes.create_region("electrodes", region=list(range(len(sdata.channel))), description="All electrodes.")
    # electrode_region = DynamicTableRegion(name='electrodes', data=list(range(num_channels)), description='All electrodes.', table=nwbfile.electrodes)

    data_single_channel = np.delete(sdata.data, obj=1, axis=1)  # Delete unused channel column.

    units = np.unique(data_single_channel[:, 1])

    if unit_info is None:
        unit_info = {"location": dict(), "group": dict()}

    # See how they do in Pynapple for compatibility:
    # https://github.com/pynapple-org/pynapple/blob/main/pynapple/io/neurosuite.py#L212

    # Extra fields for Pynapple compatibility
    nwbfile.add_unit_column("location", "the anatomical location of this unit")
    nwbfile.add_unit_column("group", "the group of the unit")

    # Extra fields for Syncopy compatibility, so we can restore the samplerate when reading the file.
    nwbfile.add_unit_column(
        "samplerate",
        "the samplerate of the unit. this is the same as the samplerate of the data, and identical across all units.",
    )

    for unit_idx in units:
        nwbfile.add_unit(
            id=unit_idx,
            spike_times=data_single_channel[np.where(data_single_channel[:, 1] == unit_idx), 0].flatten()
            / sdata.samplerate,
            electrodes=list(range(num_channels)),
            location=unit_info["location"].get(unit_idx, "unknown"),
            group=unit_info["group"].get(unit_idx, "unknown"),
            samplerate=sdata.samplerate,
        )

    # Add trial definition, if possible and requested.
    _add_trials_to_nwbfile(nwbfile, sdata.trialdefinition, sdata.samplerate, do_add=with_trialdefinition)

    return nwbfile


def _nwb_copy_pynapple(nwbfilepath, targetdir):
    """
    Copy an existing NWB file to a new directory, in which it will be placed in a new sub directory named 'pynapplenwb', as required for the Pynapple loader.

    This is a convenience function so we can test the Pynapple loader for NWB files created by Syncopy.

    Parameters
    ----------
    nwbfilepath : str, path to the NWB file to copy.
    targetdir : str, path to the existing directory in which to create the subdirectory 'pynapplenwb'. The targetdir has to exist and be writable.
    """
    if not os.path.isfile(nwbfilepath):
        raise ValueError("NWB file '{}' does not exist.".format(nwbfilepath))
    if not os.path.isdir(targetdir):
        raise ValueError("Target directory '{}' does not exist.".format(targetdir))
    subdir = os.path.join(targetdir, "pynapplenwb")
    os.makedirs(subdir, exist_ok=True)
    shutil.copy(nwbfilepath, subdir)
