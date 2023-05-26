# -*- coding: utf-8 -*-
#
# Load data from NWB file
#

# Builtin/3rd party package imports
import numpy as np
from datetime import datetime
from uuid import uuid4
import pytz
import syncopy as spy
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.ecephys import LFP, ElectricalSeries, EventWaveform, SpikeEventSeries
from pynwb.core import DynamicTableRegion

# Local imports

__all__ = []


def _get_nwbfile_template(num_channels=None):
    """
    Get a template NWBFile object with some basic metadata. No data or electrodes are added.

    This NWBFile object is not tied to any filesystem file.

    Parameters
    ----------
    num_channels: int, number of channels, used for adding electrodes. If None, no electrodes are added. In that case, the correct amount of electrodes for the data's channel count must already exist.
    """
    start_time_no_tz = datetime.now()
    tz = pytz.timezone('Europe/Berlin')
    start_time = tz.localize(start_time_no_tz)

    nwbfile = NWBFile(
                session_description="unknown",          # required
                identifier=str(uuid4()),                # required
                session_start_time=start_time,      # required and relevant, use something like `datetime(2018, 4, 25, 2, 30, 3, tzinfo=tz.gettz("US/Pacific"))` for real data.
                session_id="session_0001",              # optional. remember that one file is for one session.
                experimenter=[ "unknown", ],            # optional, name of experimenters
                lab="unknown",                          # optional, the research lab where the experiment was performed
                institution="unknown",                  # optional
                experiment_description="unknown",       # optional
                related_publications="",                # put a DOI here, if any. e.g., "https://doi.org/###"
            )
            # When creating your own NWBFile, you can also add subject information by setting `nwbfile.subject` to a `pynwb.file.Subject` instance, see docs.


    if num_channels is not None:
        assert nwbfile.electrodes is None or len(nwbfile.electrodes.colnames) == 0
        nwbfile, _ = _add_electrodes(nwbfile, num_channels)
    return nwbfile


def _add_electrodes(nwbfile, nchannels_per_shank, nshanks = 1):
    """Add the channel information (an 'electrode_table_region') to the NWBFile object `nwbfile`.

    Parameters
    ----------

    nchannels_per_shank: int, number of channels per shank. If you have only one shank, this is the total number of channels.

    nshanks: int, number of shanks. Default is 1.

    Returns
    -------
    nwbfile: NWBFile object with the channel information added. Note that this is the same object as the input `nwbfile`.

    electrodes: an `electrode_table_region` object that can be used to add data to the NWBFile object. See the NWB documentation for details.
    """
    # Add the channel information.
    ## First add the recording array
    device = nwbfile.create_device(name="array", description="Unknown array", manufacturer="Unknown manufacturer")
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
                group=electrode_group,
                label="shank{}elec{}".format(ishank, ielec),
                location="unknown brain area (shank {}, elec {})".format(ishank, ielec),
            )
            electrode_counter += 1

    electrodes = nwbfile.create_electrode_table_region(
        region=list(range(electrode_counter)),  # reference row indices 0 to N-1 in the electrode table.
        description="all electrodes",
        )
    return nwbfile, electrodes


def _analog_timelocked_to_nwbfile(atdata, nwbfile=None, with_trialdefinition=True, is_raw=True):
        """Convert `AnalogData` or `TimeLockData` into a `pynwb.NWBFile` instance, for writing to files in Neurodata Without Borders (NWB) file format.
        An NWBFile represents a single session of an experiment.

        Parameters
        ----------
        atdata : :class:`syncopy.AnalogData` or :class:`syncopy.TimeLockData`object, the data object to be converted to NWB.

        nwbfile : :class:`pynwb.NWBFile` object or None. If `None`, a new NWBFile will be created. It is highly recommended to create
         your own NWBFile object and pass it to this function, as this will allow you to add metadata to the file. If this is `None`, all metadata fields will be set to `'unknown'`.

        with_trialdefinition : Boolean, whether to save the trial definition in the NWB file.

        is_raw : Boolean, whether this is raw data (that should never change), as opposed to LFP data that originates from some processing, e.g., down-sampling and
         detrending. Determines where data is stored in the NWB container, to make it easier for other software to interprete what the data represents. If `is_raw` is `True`,
         the `ElectricalSeries` is stored directly in an acquisition of the :class:`pynwb.NWBFile`. If False, it is stored inside an `LFP` instance in a processing group called `ecephys`.
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
            nwbfile = _get_nwbfile_template(num_channels=len(atdata.channel))

        electrode_region = DynamicTableRegion('electrodes', list(range(len(atdata.channel))), 'All electrodes.', nwbfile.electrodes)


        # Now that we have an NWBFile and channels, we can add the data.
        time_series_with_rate = ElectricalSeries(
            name="ElectricalSeries",
            data=atdata.data,
            electrodes=electrode_region,
            starting_time=0.0,
            rate=atdata.samplerate, # Fixed sampling rate.
            description="Electrical time series dataset",
            comments="Exported by Syncopy",
        )

        if is_raw:  # raw measurements from instruments, not to be changed. Not downsampled, detrended, or anything.
            nwbfile.add_acquisition(time_series_with_rate)
        else: # LFP, data used for analysis.
            lfp = LFP(electrical_series=time_series_with_rate)
            ecephys_module = nwbfile.create_processing_module(
                name="ecephys", description="processed extracellular electrophysiology data"
            )
            ecephys_module.add(lfp)


        # Add trial definition, if possible and requested.
        _add_trials_to_nwbfile(nwbfile, atdata.trialdefinition, atdata.samplerate, do_add=with_trialdefinition)

        return nwbfile


def _add_trials_to_nwbfile(nwbfile, trialdefinition, samplerate, do_add=True):
    """Add trial definition to an existing NWBFile.

    Parameters
    ----------
    nwbfile : :class:`pynwb.NWBFile` object, the NWBFile instance that contains the data.

    trialdefinition : :class:`numpy.ndarray` of shape (N, 3), the trial definition. Each row contains the start time, stop time, and offset of a trial.

    samplerate: float, the sampling rate of the data in Hz

    do_add : Boolean, whether to add the trial definition to the NWB file. If `False`, the trial definition is only returned, but not added to the NWB file.

    Returns
    -------
    :class:`pynwb.epoch.TimeIntervals` object, the trial definition.
    """
    # Add the trial definition, if any.
    if trialdefinition is None or not do_add:
        return

    nwbfile.add_trial_column(
        name="offset",
        description="The offset of the trial.",
    )
    for trial_idx in range(trialdefinition.shape[0]):
        td = trialdefinition[trial_idx, :].astype(np.float64) / samplerate # Compute time from sample number.
        if do_add:
            nwbfile.add_trial(start_time=td[0], stop_time=td[1], offset=td[2])
            nwbfile.add_epoch(start_time=td[0], stop_time=td[1])


def _spikedata_to_nwbfile(sdata, nwbfile=None, with_trialdefinition=True):
        """Convert SpikeData into pynwb.NWBFile instance, for writing to files in Neurodata Without Borders (NWB) file format.
        An NWBFile represents a single session of an experiment.

        Parameters
        ----------
        sdata : :class:`syncopy.AnalogData` or :class:`syncopy.TimeLockData`object, the data object to be converted to NWB.

        nwbfile : :class:`pynwb.NWBFile` object or None. If `None`, a new NWBFile will be created. It is highly recommended to create
         your own NWBFile object and pass it to this function, as this will allow you to add metadata to the file. If this is `None`, all metadata fields will be set to `'unknown'`.

        with_trialdefinition : Boolean, whether to save the trial definition in the NWB file.

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
            nwbfile = _get_nwbfile_template(num_channels=len(sdata.channel))

        # Now that we have an NWBFile and channels, we can add the data.
        # cf. https://github.com/pynapple-org/pynapple/blob/main/pynapple/io/neurosuite.py#L212 to be
        # compatible with Neurosuite/Pynapple.
        electrode_region = nwbfile.electrodes.create_region("electrodes", region=list(range(len(sdata.channel))), description="All electrodes.")
        num_units = sdata.shape[sdata.dimord.index("unit")]
        for unit_idx in range(num_units):
            nwbfile.add_unit(
                id=unit_idx,
                spike_times = sdata.data[unit_idx],
                electrodes=electrode_region
                )

        # Add trial definition, if possible and requested.
        _add_trials_to_nwbfile(nwbfile, sdata.trialdefinition, sdata.samplerate, do_add=with_trialdefinition)

        return nwbfile


