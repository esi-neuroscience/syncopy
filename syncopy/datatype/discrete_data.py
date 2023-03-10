# -*- coding: utf-8 -*-
#
# Syncopy's abstract base class for discrete data + regular children
#

# Builtin/3rd party package imports
import numpy as np
from abc import ABC
from collections.abc import Iterator
import inspect


# Local imports
from .base_data import BaseData, FauxTrial
from .methods.definetrial import definetrial
from syncopy.shared.parsers import scalar_parser, array_parser
from syncopy.shared.errors import SPYValueError, SPYError, SPYTypeError
from syncopy.shared.tools import best_match

__all__ = ["SpikeData", "EventData"]


class DiscreteData(BaseData, ABC):
    """
    Abstract class for non-uniformly sampled data where only time-stamps are recorded.

    Notes
    -----
    This class cannot be instantiated. Use one of the children instead.
    """

    _infoFileProperties = BaseData._infoFileProperties + ("samplerate", )
    _hdfFileAttributeProperties = BaseData._hdfFileAttributeProperties + ("samplerate",)
    _selectionKeyWords = BaseData._selectionKeyWords + ('latency',)

    @property
    def data(self):
        """
        Array-like object representing data without trials.

        Trials are concatenated along the time axis.
        """

        if getattr(self._data, "id", None) is not None:
            if self._data.id.valid == 0:
                lgl = "open HDF5 file"
                act = "backing HDF5 file {} has been closed"
                raise SPYValueError(legal=lgl, actual=act.format(self.filename),
                                    varname="data")
        return self._data

    @data.setter
    def data(self, inData):
        """ Also checks for integer type of data """
        # this comes from BaseData
        self._set_dataset_property(inData, "data")

        if inData is not None:
            if not np.issubdtype(self.data.dtype, np.integer):
                raise SPYTypeError(self.data.dtype, 'data', "integer like")

    def __str__(self):
        # Get list of print-worthy attributes
        ppattrs = [attr for attr in self.__dir__()
                   if not (attr.startswith("_") or attr in ["log", "trialdefinition"])]
        ppattrs = [attr for attr in ppattrs
                   if not (inspect.ismethod(getattr(self, attr))
                           or isinstance(getattr(self, attr), Iterator))]

        ppattrs.sort()

        # Construct string for pretty-printing class attributes
        dsep = " by "
        hdstr = "Syncopy {clname:s} object with fields\n\n"
        ppstr = hdstr.format(clname=self.__class__.__name__)
        maxKeyLength = max([len(k) for k in ppattrs])
        printString = "{0:>" + str(maxKeyLength + 5) + "} : {1:}\n"
        for attr in ppattrs:
            value = getattr(self, attr)
            if hasattr(value, 'shape') and attr == "data" and self.sampleinfo is not None:
                tlen = np.unique([sinfo[1] - sinfo[0] for sinfo in self.sampleinfo])
                if tlen.size == 1:
                    trlstr = "of length {} ".format(str(tlen[0]))
                else:
                    trlstr = ""
                dsize = np.prod(self.data.shape)*self.data.dtype.itemsize/1024**2
                dunit = "MB"
                if dsize > 1000:
                    dsize /= 1024
                    dunit = "GB"
                valueString = "{} trials {}defined on ".format(str(len(self.trials)), trlstr)
                valueString += "[" + " x ".join([str(numel) for numel in value.shape]) \
                              + "] {dt:s} {tp:s} " +\
                              "of size {sz:3.2f} {szu:s}"
                valueString = valueString.format(dt=self.data.dtype.name,
                                                 tp=self.data.__class__.__name__,
                                                 sz=dsize,
                                                 szu=dunit)
            elif hasattr(value, 'shape'):
                valueString = "[" + " x ".join([str(numel) for numel in value.shape]) \
                              + "] element " + str(type(value))
            elif isinstance(value, list):
                if attr == "dimord" and value is not None:
                    valueString = dsep.join(dim for dim in self.dimord)
                else:
                    valueString = "{0} element list".format(len(value))
            elif isinstance(value, dict):
                msg = "dictionary with {nk:s}keys{ks:s}"
                keylist = value.keys()
                showkeys = len(keylist) < 7
                valueString = msg.format(nk=str(len(keylist)) + " " if not showkeys else "",
                                         ks=" '" + "', '".join(key for key in keylist) + "'" if showkeys else "")
            else:
                valueString = str(value)
            ppstr += printString.format(attr, valueString)
        ppstr += "\nUse `.log` to see object history"
        return ppstr

    @property
    def sample(self):
        """Indices of all recorded samples"""
        if self.data is None:
            return None
        return self.data[:, self.dimord.index("sample")]

    @property
    def samplerate(self):
        """float: underlying sampling rate of non-uniform data acquisition"""
        return self._samplerate

    @samplerate.setter
    def samplerate(self, sr):
        if sr is None:
            self._samplerate = None
            return

        try:
            scalar_parser(sr, varname="samplerate", lims=[1, np.inf])
        except Exception as exc:
            raise exc
        self._samplerate = sr

    @property
    def time(self):
        """list(float): trigger-relative time of each event """
        if self.samplerate is not None and self.sampleinfo is not None:
            return [(trl[:,self.dimord.index("sample")] - self.sampleinfo[tk,0] + self._t0[tk]) / self.samplerate \
                    for tk, trl in enumerate(self.trials)]

    @property
    def trialid(self):
        """:class:`numpy.ndarray` of trial id associated with the sample"""
        return self._trialid

    @trialid.setter
    def trialid(self, trlid):
        if trlid is None:
            self._trialid = None
            return

        if self.data is None:
            SPYError("SyNCoPy core - trialid: Cannot assign `trialid` without data. " +
                     "Please assign data first")
            return
        if (self.data.shape[0] == 0) and (trlid.shape[0] == 0):
            self._trialid = np.array(trlid, dtype=int)
            return
        scount = np.nanmax(self.data[:, self.dimord.index("sample")])
        try:
            array_parser(trlid, varname="trialid", dims=(self.data.shape[0],),
                         hasnan=False, hasinf=False, ntype="int_like", lims=[-1, scount])
        except Exception as exc:
            raise exc
        self._trialid = np.array(trlid, dtype=int)

    @property
    def trialtime(self):
        """list(:class:`numpy.ndarray`): trigger-relative sample times in s"""
        if self.samplerate is not None and self.sampleinfo is not None:
            sample0 = self.sampleinfo[:, 0] - self._t0[:]
            sample0 = np.append(sample0, np.nan)[self.trialid]
            return (self.data[:, self.dimord.index("sample")] - sample0) / self.samplerate

    # Helper function that grabs a single trial
    def _get_trial(self, trialno):
        return self._data[self._trialslice[trialno], :]

    # Helper function that spawns a `FauxTrial` object given actual trial information
    def _preview_trial(self, trialno):
        """
        Generate a `FauxTrial` instance of a trial

        Parameters
        ----------
        trialno : int
            Number of trial the `FauxTrial` object is intended to mimic

        Returns
        -------
        faux_trl : :class:`syncopy.datatype.base_data.FauxTrial`
            An instance of :class:`syncopy.datatype.base_data.FauxTrial` mainly
            intended to be used in `noCompute` runs of
            :meth:`syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
            to avoid loading actual trial-data into memory.

        Notes
        -----
        If an active in-place selection is found, the generated `FauxTrial` object
        respects it (e.g., if only 2 of 10 channels are selected in-place, `faux_trl`
        reports to only contain 2 channels)

        See also
        --------
        syncopy.datatype.base_data.FauxTrial : class definition and further details
        syncopy.shared.computational_routine.ComputationalRoutine : Syncopy compute engine
        """
        trlSlice = self._trialslice[trialno]
        trialIdx = np.arange(trlSlice.start, trlSlice.stop) #np.where(self.trialid == trialno)[0]
        nCol = len(self.dimord)
        idx = [[], slice(0, nCol)]
        if self.selection is not None: # selections are harmonized, just take `.time`
            idx[0] = trialIdx[self.selection.time[self.selection.trial_ids.index(trialno)]].tolist()
        else:
            idx[0] = trialIdx.tolist()

        shp = [len(idx[0]), nCol]

        return FauxTrial(shp, tuple(idx), self.data.dtype, self.dimord)

    # Helper function that extracts by-trial timing-related indices
    def _get_time(self, trials, toi=None, toilim=None):
        """
        Get relative by-trial indices of time-selections.

        Parameters
        ----------
        trials : list
            List of trial-indices to perform selection on.
        toi : None or list
            Time-points to be selected (in seconds) on a by-trial scale.
        toilim : None or list
            Time-window to be selected (in seconds) on a by-trial scale.

        Returns
        -------
        timing : list of lists
            List of by-trial sample-indices corresponding to provided
            time-selection. If both `toi` and `toilim` are `None`, `timing`
            is a list of universal (i.e., ``slice(None)``) selectors.

        Notes
        -----
        This class method is intended to be solely used by
        :class:`syncopy.datatype.selector.Selector` objects and thus has purely
        auxiliary character. Therefore, all input sanitization and error checking
        is left to :class:`syncopy.datatype.selector.Selector` and not
        performed here.

        See also
        --------
        syncopy.datatype.selector.Selector : Syncopy data selectors
        """
        timing = []
        if toilim is not None:
            allTrials = self.trialtime
            for trlno in trials:
                trlTime = allTrials[self._trialslice[trlno]]
                _, selTime = best_match(trlTime, toilim, span=True)
                selTime = selTime.tolist()
                if len(selTime) > 1 and np.diff(trlTime).min() > 0:
                    timing.append(slice(selTime[0], selTime[-1] + 1, 1))
                else:
                    timing.append(selTime)

        elif toi is not None:
            allTrials = self.trialtime
            for trlno in trials:
                trlTime = allTrials[self._trialslice[trlno]]
                _, arrayIdx = best_match(trlTime, toi)
                # squash duplicate values then readd
                _, xdi = np.unique(trlTime[arrayIdx], return_index=True)
                arrayIdx = arrayIdx[xdi] # we assume sorted data
                selTime = []
                for t in arrayIdx:
                    selTime += np.where(trlTime[t] == trlTime)[0].tolist()
                # convert to slice if possible
                if len(selTime) > 1:
                    timeSteps = np.diff(selTime)
                    if timeSteps.min() == timeSteps.max() == 1:
                        selTime = slice(selTime[0], selTime[-1] + 1, 1)
                timing.append(selTime)

        else:
            timing = [slice(None)] * len(trials)

        return timing

    def __init__(self, data=None, samplerate=None, trialid=None, **kwargs):

        # set as instance attribute to allow (un-)registering of additional datasets
        self._hdfFileDatasetProperties = BaseData._hdfFileDatasetProperties + ("data",)

        # Assign (default) values
        self._trialid = None
        self._samplerate = None
        self._data = None

        self.samplerate = samplerate
        self.trialid = trialid

        # Call initializer
        super().__init__(data=data, **kwargs)

        if self.data is not None:

            if self.data.size == 0:
                # initialization with empty data not allowed
                raise SPYValueError("non empty data set", 'data')

            # In case of manual data allocation (reading routine would leave a
            # mark in `cfg`), fill in missing info
            if self.sampleinfo is None:
                # Fill in dimensional info
                definetrial(self, kwargs.get("trialdefinition"))


class SpikeData(DiscreteData):
    """Spike times of multi- and/or single units

    This class can be used for representing spike trains. The data is always
    stored as a two-dimensional [nSpikes x 3] array on disk with the columns
    being ``["sample", "channel", "unit"]``.

    Data is only read from disk on demand, similar to HDF5 files.
    """

    _infoFileProperties = DiscreteData._infoFileProperties + ("channel", "unit",)
    _defaultDimord = ["sample", "channel", "unit"]
    _stackingDimLabel = "sample"
    _selectionKeyWords = DiscreteData._selectionKeyWords + ('channel', 'unit',)

    def _compute_unique_idx(self):
        """
        Use `np.unique` on whole(!) dataset to compute globally
        available channel and unit indices only once

        This function gets triggered by the constructor
        `if data is not None` or latest when channel/unit
        labels are assigned with the respective setters.
        """

        # after data was added via selection or loading from file
        # this function gets re-triggered by the channel/unit setters!
        if self.data is None:
            return

        # this is costly and loads the entire hdf5 dataset into memory!
        self.channel_idx = np.unique(self.data[:, self.dimord.index("channel")])
        self.unit_idx = np.unique(self.data[:, self.dimord.index("unit")])

    @property
    def channel(self):
        """ :class:`numpy.ndarray` : list of original channel names for each unit"""

        return self._channel

    @channel.setter
    def channel(self, chan):
        if self.data is None:
            if chan is not None:
                raise SPYValueError(f"non-empty SpikeData", "cannot assign `channel` without data. " +
                                    "Please assign data first")
            # No labels for no data is fine
            self._channel = chan
            return

        # there is data
        elif chan is None:
            raise SPYValueError("channel labels, cannot set `channel` to `None` with existing data.")

        # if we landed here, we have data and new labels

        # in case of selections and/or loading from file
        # the constructor was called with data=None, hence
        # we have to compute the unique indices here
        if self.channel_idx is None:
            self._compute_unique_idx()

        # we need as many labels as there are distinct channels
        nChan = self.channel_idx.size

        if nChan != len(chan):
            raise SPYValueError(f"exactly {nChan} channel label(s)")
        array_parser(chan, varname="channel", ntype="str", dims=(nChan, ))
        self._channel = np.array(chan)

    def _default_channel_labels(self):

        """
        Creates the default channel labels
        """

        # channel entries in self.data are 0-based
        chan_max = self.channel_idx.max()
        channel_labels = np.array(["channel" + str(int(i + 1)).zfill(len(str(chan_max)) + 1)
                                   for i in self.channel_idx])
        return channel_labels

    @property
    def unit(self):
        """ :class:`numpy.ndarray(str)` : unit names"""

        return self._unit

    @unit.setter
    def unit(self, unit):
        if self.data is None:
            if unit is not None:
                raise SPYValueError(f"non-empty SpikeData", "cannot assign `unit` without data. " +
                                    "Please assign data first")
            # empy labels for empty data is fine
            self._unit = unit
            return

        # there is data
        elif unit is None:
            raise SPYValueError("unit labels, cannot set `unit` to `None` with existing data.")

        # in case of selections and/or loading from file
        # the constructor was called with data=None, hence
        # we have to compute this here
        if self.unit_idx is None:
            self._compute_unique_idx()

        if unit is None and self.data is not None:
            raise SPYValueError("Cannot set `unit` to `None` with existing data.")
        elif self.data is None and unit is not None:
            raise SPYValueError("Syncopy - SpikeData - unit: Cannot assign `unit` without data. " +
                  "Please assign data first")
        elif unit is None:
            self._unit = None
            return

        nunit = self.unit_idx.size
        if nunit != len(unit):
            raise SPYValueError(f"exactly {nunit} unit label(s)")
        array_parser(unit, varname="unit", ntype="str", dims=(nunit,))

        self._unit = np.array(unit)

    def _default_unit_labels(self):

        """
        Creates the default unit labels
        """

        unit_max = self.unit_idx.max()
        return np.array(["unit" + str(int(i + 1)).zfill(len(str(unit_max)) + 1)
                         for i in self.unit_idx])

    # Helper function that extracts by-trial unit-indices
    def _get_unit(self, trials, units=None):
        """
        Get relative by-trial indices of unit selections

        Parameters
        ----------
        trials : list
            List of trial-indices to perform selection on
        units : None or list
            List of unit-indices to be selected

        Returns
        -------
        indices : list of lists
            List of by-trial sample-indices corresponding to provided
            unit-selection. If `units` is `None`, `indices` is a list of universal
            (i.e., ``slice(None)``) selectors.

        Notes
        -----
        This class method is intended to be solely used by
        :class:`syncopy.datatype.selector.Selector` objects and thus has purely
        auxiliary character. Therefore, all input sanitization and error checking
        is left to :class:`syncopy.datatype.selector.Selector` and not
        performed here.

        See also
        --------
        syncopy.datatype.selector.Selector : Syncopy data selectors
        """
        if units is not None:
            indices = []
            for trlno in trials:
                thisTrial = self.data[self._trialslice[trlno], self.dimord.index("unit")]
                trialUnits = []
                for unit in units:
                    trialUnits += list(np.where(thisTrial == unit)[0])
                if len(trialUnits) > 1:
                    steps = np.diff(trialUnits)
                    if steps.min() == steps.max() == 1:
                        trialUnits = slice(trialUnits[0], trialUnits[-1] + 1, 1)
                indices.append(trialUnits)
        else:
            indices = [slice(None)] * len(trials)

        return indices

    @property
    def waveform(self):
        """The waveform of the spikes in the data.

        This is a tiny part of the raw data around the time point at which a spike was detected
        (by the recording hardware/software: keep in mind that the spike data Syncopy is working
        with is already pre-processed). From this sequence of samples, one can derive
        the waveform type of the spike (via classification), which may allow to
        derive the type of neuron that produced the spike.
        """
        return self._waveform

    @waveform.setter
    def waveform(self, waveform):
        """Set a waveform dataset from a numpy array, `None`, or an `h5py.Dataset` instance."""
        if self.data is None:
            if waveform is not None:
                raise SPYValueError(legal="non-empty SpikeData", varname="waveform",
                actual="empty SpikeData main dataset (None). Cannot assign `waveform` without data. Please assign data first.")
        if waveform is None:
            self._set_dataset_property(waveform, 'waveform') # None
            self._unregister_dataset("waveform", del_attr=False)
            return

        if waveform.ndim < 2:
            raise SPYValueError(legal="waveform data with at least 2 dimensions", varname="waveform", actual=f"data with {waveform.ndim} dimensions")

        if waveform.shape[0] != self.data.shape[0]:
            raise SPYValueError(f"waveform shape[0]={waveform.shape[0]} must equal nSpikes={self.data.shape[0]}. " +
                                "Please create one waveform per spike in data.", varname="waveform", actual=f"wrong size waveform with shape {waveform.shape}")
        self._set_dataset_property(waveform, 'waveform')

    # "Constructor"
    def __init__(self,
                 data=None,
                 filename=None,
                 trialdefinition=None,
                 samplerate=None,
                 channel=None,
                 unit=None,
                 dimord=None):
        """Initialize a :class:`SpikeData` object.

        Parameters
        ----------
            data : [nSpikes x 3] :class:`numpy.ndarray`

            filename : str
                path to filename or folder (spy container)
            trialdefinition : :class:`EventData` object or nTrials x 3 array
                [start, stop, trigger_offset] sample indices for `M` trials
            samplerate : float
                sampling rate in Hz
            channel : str or list/array(str)
                original channel names
            unit : str or list/array(str)
                names of all units
            dimord : list(str)
                ordered list of dimension labels

        1. `filename` + `data` : create hdf dataset incl. sampleinfo @filename
        2. `filename` no `data` : read from file (spy, hdf5 file)
        3. just `data` : try to attach data (error checking done by
           :meth:`SpikeData.data.setter`)

        See also
        --------
        :func:`syncopy.definetrial`

        """

        # instance attribute to allow modification
        self._hdfFileAttributeProperties = DiscreteData._hdfFileAttributeProperties + ("channel", "unit")

        self._unit = None
        self.unit_idx = None
        self._channel = None
        self.channel_idx = None

        # Call parent initializer
        super().__init__(data=data,
                         filename=filename,
                         trialdefinition=trialdefinition,
                         samplerate=samplerate,
                         dimord=dimord)

        self._hdfFileDatasetProperties +=  ("waveform",)

        self._waveform = None

        # for fast lookup and labels
        self._compute_unique_idx()

        # constructor gets `data=None` for
        # empty inits, selections and loading from file
        # can't set any labels in that case
        if channel is not None:
            # setter raises exception if data=None
            self.channel = channel
        elif data is not None:
            # data but no given labels
            self.channel = self._default_channel_labels()

        # same for unit
        if unit is not None:
            # setter raises exception if data=None
            self.unit = unit
        elif data is not None:
            self.unit = self._default_unit_labels()


class EventData(DiscreteData):
    """Timestamps and integer codes of experimental events

    This class can be used for representing events during an experiment, e.g.
    stimulus was turned on, etc. These usually occur at non-regular time points
    and have associated event codes.

    Data is only read from disk on demand, similar to HDF5 files.
    """

    _defaultDimord = ["sample", "eventid"]
    _stackingDimLabel = "sample"
    _selectionKeyWords = DiscreteData._selectionKeyWords + ('eventid',)

    @property
    def eventid(self):
        """numpy.ndarray(int): integer event code assocated with each event"""
        if self.data is None:
            return None
        return np.unique(self.data[:, self.dimord.index("eventid")])

    # Helper function that extracts by-trial eventid-indices
    def _get_eventid(self, trials, eventids=None):
        """
        Get relative by-trial indices of event-id selections

        Parameters
        ----------
        trials : list
            List of trial-indices to perform selection on
        eventids : None or list
            List of event-id-indices to be selected

        Returns
        -------
        indices : list of lists
            List of by-trial sample-indices corresponding to provided
            event-id-selection. If `eventids` is `None`, `indices` is a list of
            universal (i.e., ``slice(None)``) selectors.

        Notes
        -----
        This class method is intended to be solely used by
        :class:`syncopy.datatype.selector.Selector` objects and thus has purely
        auxiliary character. Therefore, all input sanitization and error checking
        is left to :class:`syncopy.datatype.selector.Selector` and not
        performed here.

        See also
        --------
        syncopy.datatype.selector.Selector : Syncopy data selectors
        """
        if eventids is not None:
            indices = []
            for trlno in trials:
                thisTrial = self.data[self._trialslice[trlno], self.dimord.index("eventid")]
                trialEvents = []
                for event in eventids:
                    trialEvents += list(np.where(thisTrial == event)[0])
                if len(trialEvents) > 1:
                    steps = np.diff(trialEvents)
                    if steps.min() == steps.max() == 1:
                        trialEvents = slice(trialEvents[0], trialEvents[-1] + 1, 1)
                indices.append(trialEvents)
        else:
            indices = [slice(None)] * len(trials)

        return indices

    # "Constructor"
    def __init__(self,
                 data=None,
                 filename=None,
                 trialdefinition=None,
                 samplerate=None,
                 dimord=None):
        """Initialize a :class:`EventData` object.

        Parameters
        ----------
        data : [nEvents x 2] :class:`numpy.ndarray`

        filename : str
            path to filename or folder (spy container)
        trialdefinition : :class:`EventData` object or nTrials x 3 array
            [start, stop, trigger_offset] sample indices for `M` trials
        samplerate : float
            sampling rate in Hz
        dimord : list(str)
            ordered list of dimension labels

        1. `filename` + `data` : create hdf dataset incl. sampleinfo @filename
        2. `filename` no `data` : read from file(spy, hdf5)
        3. just `data` : try to attach data (error checking done by
           :meth:`EventData.data.setter`)

        See also
        --------
        :func:`syncopy.definetrial`

        """
        if dimord is not None:
            # ensure that event data can have extra dimord columns
            if len(dimord) != len(self._defaultDimord):
                for col in self._defaultDimord:
                    if col not in dimord:
                        base = "dimensional label {}"
                        lgl = base.format("'" + col + "'")
                        raise SPYValueError(legal=lgl, varname="dimord")
                self._defaultDimord = dimord

        # Call parent initializer
        super().__init__(data=data,
                         filename=filename,
                         trialdefinition=trialdefinition,
                         samplerate=samplerate,
                         dimord=dimord)

        self._hdfFileAttributeProperties = BaseData._hdfFileAttributeProperties + ("samplerate",)
