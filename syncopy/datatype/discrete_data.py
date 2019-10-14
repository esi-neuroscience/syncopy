# -*- coding: utf-8 -*-
# 
# SynCoPy DiscreteData abstract class + regular children
# 
# Created: 2019-03-20 11:20:04
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-10-11 13:38:26>

# Builtin/3rd party package imports
import numpy as np
from abc import ABC

# Local imports
from .base_data import BaseData, Indexer
from .data_methods import definetrial
from syncopy.shared.parsers import scalar_parser, array_parser
from syncopy.shared.errors import SPYValueError

__all__ = ["SpikeData", "EventData"]


class DiscreteData(BaseData, ABC):
    """Abstract class for non-uniformly sampled data where only time-stamps are recorded

    Notes
    -----
    This class cannot be instantiated. Use one of the children instead.
    """

    _infoFileProperties = BaseData._infoFileProperties + ("_hdr", "samplerate", )
    _hdfFileAttributeProperties = BaseData._hdfFileAttributeProperties + ("samplerate",)


    @property
    def hdr(self):
        """dict with information about raw data

        This property is empty for data created by Syncopy.
        """
        return self._hdr

    @property
    def sample(self):
        """Indices of all recorded samples"""
        if self.data is None:
            return None
        return np.unique(self.data[:, self.dimord.index("sample")])

    @property
    def samplerate(self):
        """float: underlying sampling rate of non-uniformly data acquisition"""
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
    def trialid(self):
        """:class:`numpy.ndarray` of trial id associated with the sample"""
        return self._trialid

    @trialid.setter
    def trialid(self, trlid):
        if trlid is None:
            self._trialid = None
            return
        
        if self.data is None:
            print("SyNCoPy core - trialid: Cannot assign `trialid` without data. " +
                  "Please assing data first")
            return
        scount = np.nanmax(self.data[:, self.dimord.index("sample")])
        try:
            array_parser(trlid, varname="trialid", dims=(self.data.shape[0],),
                         hasnan=False, hasinf=False, ntype="int_like", lims=[-1, scount])
        except Exception as exc:
            raise exc
        self._trialid = np.array(trlid, dtype=int)

    @property
    def trials(self):
        """list-like([sample x (>=2)] :class:`numpy.ndarray`) : trial slices of :attr:`data` property"""
        if self.trialid is not None:
            valid_trls = np.unique(self.trialid[self.trialid >= 0])
            return Indexer(map(self._get_trial, valid_trls),
                           valid_trls.size)
        else:
            return None

    @property
    def trialtime(self):
        """list(:class:`numpy.ndarray`): trigger-relative sample times in s"""
        if self.samplerate is not None and self.sampleinfo is not None:
            return [((t + self._t0[tk]) / self.samplerate \
                    for t in range(0, int(self.sampleinfo[tk, 1] - self.sampleinfo[tk, 0]))) \
                    for tk in self.trialid]

    # Selector method
    def selectdata(self, trials=None, deepcopy=False, **kwargs):
        """Select parts of the data (:func:`syncopy.selectdata`)        
        """

    # Helper function that grabs a single trial
    def _get_trial(self, trialno):
        return self._data[self.trialid == trialno, :]
    
    # Helper function that extracts by-trial timing-related indices
    def _get_time(self, trials, toi=None, toilim=None):
        """
        Get relative by-trial indices of time-selections
        
        Parameters
        ----------
        trials : list
            List of trial-indices to perform selection on
        toi : None or list
            Time-points to be selected (in seconds) on a by-trial scale. 
        toilim : None or list
            Time-window to be selected (in seconds) on a by-trial scale
            
        Returns
        -------
        timing : list of lists
            List of by-trial sample-indices corresponding to provided 
            time-selection. If both `toi` and `toilim` are `None`, `timing`
            is a list of universal (i.e., ``slice(None)``) selectors. 
            
        Notes
        -----
        This class method is intended to be solely used by 
        :class:`syncopy.datatype.base_data.Selector` objects and thus has purely 
        auxiliary character. Therefore, all input sanitization and error checking
        is left to :class:`syncopy.datatype.base_data.Selector` and not 
        performed here. 
        
        See also
        --------
        syncopy.datatype.base_data.Selector : Syncopy data selectors
        """
        timing = []
        if toilim is not None:
            allTrials = self.trialtime
            allSamples = self.data[:, self.dimord.index("sample")]
            for trlno in trials:
                thisTrial = allSamples[self.trialid == trlno]
                trlSample = np.arange(*self.sampleinfo[trlno, :])
                trlTime = np.array(list(allTrials[np.where(self.trialid == trlno)[0][0]]))
                minSample = trlSample[np.where(trlTime >= toilim[0])[0][0]]
                maxSample = trlSample[np.where(trlTime <= toilim[1])[0][-1]]
                selSample = trlSample[np.intersect1d(np.where(trlSample >= minSample)[0], 
                                                     np.where(trlSample <= maxSample)[0])]
                idxList = []
                for smp in selSample:
                    idxList += list(np.where(thisTrial == smp)[0])
                if len(idxList) > 1:
                    sampSteps = np.diff(idxList)
                    if sampSteps.min() == sampSteps.max() == 1:
                        idxList = slice(idxList[0], idxList[-1] + 1, 1)
                timing.append(idxList)
                
        elif toi is not None:
            allTrials = self.trialtime
            allSamples = self.data[:, self.dimord.index("sample")]
            for trlno in trials:
                thisTrial = allSamples[self.trialid == trlno]
                trlSample = np.arange(*self.sampleinfo[trlno, :])
                trlTime = np.array(list(allTrials[np.where(self.trialid == trlno)[0][0]]))
                selSample = [min(trlTime.size - 1, idx) 
                                       for idx in np.searchsorted(trlTime, toi, side="left")]
                for k, idx in enumerate(selSample):
                    if np.abs(trlTime[idx - 1] - toi[k]) < np.abs(trlTime[idx] - toi[k]):
                        selSample[k] = trlSample[idx -1]
                    else:
                        selSample[k] = trlSample[idx]
                idxList = []
                for smp in selSample:
                    idxList += list(np.where(thisTrial == smp)[0])
                if len(idxList) > 1:
                    sampSteps = np.diff(idxList)
                    if sampSteps.min() == sampSteps.max() == 1:
                        idxList = slice(idxList[0], idxList[-1] + 1, 1)
                timing.append(idxList)
                
        else:
            timing = [slice(None)] * len(trials)
            
        return timing

    def __init__(self, samplerate=None, trialid=None, **kwargs):

        # Assign (default) values
        self._trialid = None
        self._samplerate = None                           
        self._hdr = None

        # Call initializer
        super().__init__(**kwargs)

        self.samplerate = samplerate
        self.trialid = trialid

        # If a super-class``__init__`` attached data, be careful
        if self.data is not None:

            # In case of manual data allocation (reading routine would leave a
            # mark in `cfg`), fill in missing info
            if len(self.cfg) == 0:

                # Fill in dimensional info
                definetrial(self, kwargs.get("trialdefinition"))


class SpikeData(DiscreteData):
    """Spike times of multi- and/or single units

    This class can be used for representing spike trains. The data is always
    stored as a two-dimensional [nSpikes x 3] array on disk with the columns
    being ``["sample", "channel", "unit"]``. 

    Data is only read from disk on demand, similar to memory maps and HDF5
    files.

    """

    _infoFileProperties = DiscreteData._infoFileProperties + ("channel", "unit",)
    _hdfFileAttributeProperties = DiscreteData._hdfFileAttributeProperties + ("channel",)
    _defaultDimord = ["sample", "channel", "unit"]
    
    @property
    def channel(self):
        """ :class:`numpy.ndarray` : list of original channel names for each unit"""        
        # if data exists but no user-defined channel labels, create them on the fly
        if self._channel is None and self._data is not None:
            channelIndices = np.unique(self.data[:, self.dimord.index("channel")])
            return np.array(["channel" + str(int(i)).zfill(len(str(channelIndices.max())))
                             for i in channelIndices])
            
        return self._channel

    @channel.setter
    def channel(self, chan):
        if chan is None:
            self._channel = None
            return
        
        if self.data is None:
            raise SPYValueError("Syncopy: Cannot assign `channels` without data. " +
                  "Please assign data first")    

        nchan = np.unique(self.data[:, self.dimord.index("channel")]).size
        try:
            array_parser(chan, varname="channel", ntype="str", dims=(nchan,))
        except Exception as exc:
            raise exc
        self._channel = np.array(chan)

    @property
    def unit(self):
        """ :class:`numpy.ndarray(str)` : unit names"""
        if self.data is not None and self._unit is None:
            unitIndices = np.unique(self.data[:, self.dimord.index("unit")])
            return np.array(["unit" + str(int(i)).zfill(len(str(unitIndices.max())))
                             for i in unitIndices])
        return self._unit

    @unit.setter
    def unit(self, unit):
        if unit is None:
            self._unit = None
            return
        
        if self.data is None:
            raise SPYValueError("Syncopy - SpikeData - unit: Cannot assign `unit` without data. " +
                  "Please assign data first")
                        
        nunit = np.unique(self.data[:, self.dimord.index("unit")]).size
        try:
            array_parser(unit, varname="unit", ntype="str", dims=(nunit,))
        except Exception as exc:
            raise exc
        self._unit = np.array(unit)
        
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
        :class:`syncopy.datatype.base_data.Selector` objects and thus has purely 
        auxiliary character. Therefore, all input sanitization and error checking
        is left to :class:`syncopy.datatype.base_data.Selector` and not 
        performed here. 
        
        See also
        --------
        syncopy.datatype.base_data.Selector : Syncopy data selectors
        """
        if units is not None:
            indices = []
            allUnits = self.data[:, self.dimord.index("unit")]
            for trlno in trials:
                thisTrial = allUnits[self.trialid == trlno]
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
        2. `filename` no `data` : read from file or memmap (spy, hdf5, npy file
           array -> memmap)
        3. just `data` : try to attach data (error checking done by
           :meth:`SpikeData.data.setter`)

        See also
        --------
        :func:`syncopy.definetrial`

        """

        self._unit = None
        self._channel = None
        
        # Call parent initializer
        super().__init__(data=data,
                         filename=filename,
                         trialdefinition=trialdefinition,
                         samplerate=samplerate,
                         dimord=dimord)

        self.channel = channel
        self.unit = unit


class EventData(DiscreteData):
    """Timestamps and integer codes of experimental events

    This class can be used for representing events during an experiment, e.g.
    stimulus was turned on, etc. These usually occur at non-regular time points
    and have associated event codes.

    Data is only read from disk on demand, similar to memory maps and HDF5
    files.

    """        
    
    _defaultDimord = ["sample", "eventid"]
    
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
        :class:`syncopy.datatype.base_data.Selector` objects and thus has purely 
        auxiliary character. Therefore, all input sanitization and error checking
        is left to :class:`syncopy.datatype.base_data.Selector` and not 
        performed here. 
        
        See also
        --------
        syncopy.datatype.base_data.Selector : Syncopy data selectors
        """
        if eventids is not None:
            indices = []
            allEvents = self.data[:, self.dimord.index("eventid")]
            for trlno in trials:
                thisTrial = allEvents[self.trialid == trlno]
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
        2. `filename` no `data` : read from file or memmap (spy, hdf5, npy file
           array -> memmap)
        3. just `data` : try to attach data (error checking done by
           :meth:`EventData.data.setter`)

        See also
        --------
        :func:`syncopy.definetrial`

        """

        # Call parent initializer
        super().__init__(data=data,
                         filename=filename,
                         trialdefinition=trialdefinition,
                         samplerate=samplerate,
                         dimord=dimord)
