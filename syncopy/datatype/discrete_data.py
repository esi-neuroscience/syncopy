# -*- coding: utf-8 -*-
#
# SynCoPy DiscreteData abstract class + regular children
# 
# Created: 2019-03-20 11:20:04
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-04-03 10:25:37>

# Builtin/3rd party package imports
import numpy as np
from abc import ABC

# Local imports
from .base_data import BaseData, Indexer
from .data_methods import _selectdata_discrete, definetrial
from syncopy.utils import scalar_parser, array_parser, SPYValueError

__all__ = ["SpikeData", "EventData"]


class DiscreteData(BaseData, ABC):

    @property
    def hdr(self):
        return self._hdr

    @property
    def sample(self):
        return self._dimlabels.get("sample")
    
    @property
    def samplerate(self):
        return self._samplerate
    
    @samplerate.setter
    def samplerate(self, sr):
        try:
            scalar_parser(sr, varname="samplerate", lims=[1, np.inf])
        except Exception as exc:
            raise exc
        self._samplerate = sr
        
    @property
    def trialid(self):
        return self._trialid
    
    @trialid.setter
    def trialid(self, trlid):
        if self.data is None:
            print("SyNCoPy core - trialid: Cannot assign `trialid` without data. "+\
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
        if self.trialid is not None:
            valid_trls = np.unique(self.trialid[self.trialid >= 0])
            return Indexer(map(self._get_trial, valid_trls),
                           valid_trls.size)
        else:
            return None

    @property
    def trialtime(self):
        return [range(-self.t0[tk],
                      self.sampleinfo[tk, 1] - self.sampleinfo[tk, 0] - self.t0[tk]) \
                for tk in self.trialid] if self.trialid is not None else None
    
    # Selector method
    def selectdata(self, trials=None, deepcopy=False, **kwargs):
        """
        Docstring mostly pointing to ``selectdata``
        """
        return _selectdata_discrete(self, trials, deepcopy, **kwargs)

    # Helper function that grabs a single trial
    def _get_trial(self, trialno):
        return self._data[self.trialid == trialno, :]
    
    # Make instantiation persistent in all subclasses
    def __init__(self, **kwargs):

        # Hard constraint: required no. of data-dimensions
        self._ndim = 2
        
        # Assign (default) values
        self._trialid = None
        if kwargs.get("samplerate") is not None:
            self.samplerate = kwargs["samplerate"]      # use setter for error-checking
        else:
            self._samplerate = None
        self._hdr = None

        # Call initializer
        super().__init__(**kwargs)

        # If a super-class``__init__`` attached data, be careful
        if self.data is not None:

            # In case of manual data allocation (reading routine would leave a
            # mark in `cfg`), fill in missing info
            if len(self.cfg) == 0:
                
                # Fill in dimensional info
                definetrial(self, kwargs.get("trialdefinition"))
                

class SpikeData(DiscreteData):

    @property
    def channel(self):
        return self._dimlabels.get("channel")

    @channel.setter
    def channel(self, chan):
        if self.data is None:
            print("SyNCoPy core - channel: Cannot assign `channels` without data. "+\
                  "Please assing data first")
            return
        nchan = np.unique(self.data[:, self.dimord.index("channel")]).size
        try:
            array_parser(chan, varname="channel", ntype="str", dims=(nchan,))
        except Exception as exc:
            raise exc
        self._dimlabels["channel"] = np.array(chan)
    
    @property
    def unit(self):
        return self._dimlabels.get("unit")

    @unit.setter
    def unit(self, unit):
        if self.data is None:
            print("SyNCoPy core - unit: Cannot assign `unit` without data. "+\
                  "Please assing data first")
            return
        nunit = np.unique(self.data[:, self.dimord.index("unit")]).size
        try:
            array_parser(unit, varname="unit", ntype="str", dims=(nunit,))
        except Exception as exc:
            raise exc
        self._dimlabels["unit"] = np.array(unit)
    
    # "Constructor"
    def __init__(self,
                 data=None,
                 filename=None,
                 filetype=None,
                 trialdefinition=None,
                 samplerate=None,
                 channel="channel",
                 unit="unit",
                 mode="w",
                 dimord=["sample", "channel", "unit"]):
        """
        Docstring

        filename + data = create memmap @filename
        filename no data = read from file or memmap
        just data = try to attach data (error checking done by data.setter)
        """

        # The one thing we check right here and now
        expected = ["sample", "channel", "unit"]
        if not set(dimord).issubset(expected):
            base = "dimensional labels {}"
            lgl = base.format("'" + "' x '".join(str(dim) for dim in expected) + "'")
            act = base.format("'" + "' x '".join(str(dim) for dim in dimord) + "'")
            raise SPYValueError(legal=lgl, varname="dimord", actual=act)

        # Call parent initializer
        super().__init__(data=data,
                         filename=filename,
                         filetype=filetype,
                         trialdefinition=trialdefinition,
                         samplerate=samplerate,
                         channel=channel,
                         unit=unit,
                         mode=mode,
                         dimord=dimord)

        # If a super-class``__init__`` attached data, be careful
        if self.data is not None:

            # In case of manual data allocation (reading routine would leave a
            # mark in `cfg`), fill in missing info
            if len(self.cfg) == 0:
                
                # If necessary, construct list of channel labels (parsing is done by setter)
                if isinstance(channel, str):
                    channel = [channel + str(int(i)) for i in np.unique(self.data[:,self.dimord.index("channel")])]
                self.channel = np.array(channel)

                # If necessary, construct list of unit labels (parsing is done by setter)
                if isinstance(unit, str):
                    unit = [unit + str(int(i)) for i in np.unique(self.data[:,self.dimord.index("unit")])]
                self.unit = np.array(unit)

        # Dummy assignment: if we have no data but channel labels, assign bogus to tigger setter warning
        else:
            if channel is not None:
                self.channel = ['channel']
            if unit is not None:
                self.unit = ['unit']


class EventData(DiscreteData):

    @property
    def eventid(self):
        return self._dimlabels.get("eventid")
    
    # "Constructor"
    def __init__(self,
                 data=None,
                 filename=None,
                 filetype=None,
                 trialdefinition=None,
                 samplerate=None,
                 mode="w",
                 dimord=["sample", "eventid"]):
        """
        Docstring

        filename + data = create memmap @filename
        filename no data = read from file or memmap
        just data = try to attach data (error checking done by data.setter)
        """

        # The one thing we check right here and now
        expected = ["sample", "eventid"]
        if not set(dimord).issubset(expected):
            base = "dimensional labels {}"
            lgl = base.format("'" + "' x '".join(str(dim) for dim in expected) + "'")
            act = base.format("'" + "' x '".join(str(dim) for dim in dimord) + "'")
            raise SPYValueError(legal=lgl, varname="dimord", actual=act)

        # Call parent initializer
        super().__init__(data=data,
                         filename=filename,
                         filetype=filetype,
                         trialdefinition=trialdefinition,
                         samplerate=samplerate,
                         mode=mode,
                         dimord=dimord)
