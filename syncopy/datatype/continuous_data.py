# -*- coding: utf-8 -*-
#
# SynCoPy ContinuousData abstract class + regular children
# 
# Created: 2019-03-20 11:11:44
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-05-09 13:56:59>

# Builtin/3rd party package imports
import h5py
import shutil
import numpy as np
from abc import ABC
from copy import copy
from numpy.lib.format import open_memmap

# Local imports
from .base_data import BaseData, VirtualData
from .data_methods import _selectdata_continuous, definetrial
from syncopy.shared import scalar_parser, array_parser, io_parser
from syncopy.shared.errors import SPYValueError

__all__ = ["AnalogData", "SpectralData"]


class ContinuousData(BaseData, ABC):

    @property
    def _shapes(self):
        if self.sampleinfo is not None:
            sid = self.dimord.index("time")
            shp = [list(self.data.shape) for k in range(self.sampleinfo.shape[0])]
            for k, sg in enumerate(self.sampleinfo):
                shp[k][sid] = sg[1] - sg[0]
            return [tuple(sp) for sp in shp]

    @property
    def channel(self):
        return self._dimlabels.get("channel")

    @channel.setter
    def channel(self, chan):
        if self.data is None:
            print("SyNCoPy core - channel: Cannot assign `channels` without data. "+\
                  "Please assing data first")
            return
        nchan = self.data.shape[self.dimord.index("channel")]
        try:
            array_parser(chan, varname="channel", ntype="str", dims=(nchan,))
        except Exception as exc:
            raise exc
        self._dimlabels["channel"] = np.array(chan)

    @property
    def samplerate(self):
        return self._samplerate

    @samplerate.setter
    def samplerate(self, sr):
        try:
            scalar_parser(sr, varname="samplerate", lims=[1, np.inf])
        except Exception as exc:
            raise exc
        self._samplerate = float(sr)

    @property
    def time(self):
        return [np.arange(-self.t0[tk], end - start - self.t0[tk]) * 1/self.samplerate \
                for tk, (start, end) in enumerate(self.sampleinfo)] if self.samplerate is not None else None

    # Selector method
    def selectdata(self, trials=None, deepcopy=False, **kwargs):
        """
        Docstring mostly pointing to ``selectdata``
        """
        return _selectdata_continuous(self, trials, deepcopy, **kwargs)

    # Helper function that reads a single trial into memory
    @staticmethod
    def _copy_trial(trialno, filename, dimord, sampleinfo, hdr):
        idx = [slice(None)] * len(dimord)
        idx[dimord.index("time")] = slice(int(sampleinfo[trialno, 0]), int(sampleinfo[trialno, 1]))
        idx = tuple(idx)
        if hdr is None:
            # For pre-processed npy files
            return np.array(open_memmap(filename, mode="c")[idx])
        else:
            # For VirtualData objects
            dsets = []
            for fk, fname in enumerate(filename):
                dsets.append(np.memmap(fname, offset=int(hdr[fk]["length"]),
                                       mode="r", dtype=hdr[fk]["dtype"],
                                       shape=(hdr[fk]["M"], hdr[fk]["N"]))[idx])
            return np.vstack(dsets)

    # Helper function that grabs a single trial
    def _get_trial(self, trialno):
        idx = [slice(None)] * len(self.dimord)
        sid = self.dimord.index("time")
        idx[sid] = slice(int(self.sampleinfo[trialno, 0]), int(self.sampleinfo[trialno, 1]))
        return self._data[tuple(idx)]
        
    # Make instantiation persistent in all subclasses
    def __init__(self, **kwargs):

        # Assign (blank) values
        if kwargs.get("samplerate") is not None:
            self.samplerate = kwargs["samplerate"]      # use setter for error-checking
        else:
            self._samplerate = None
            
        # Call initializer
        super().__init__(**kwargs)

        # If a super-class``__init__`` attached data, be careful
        if self.data is not None:

            # In case of manual data allocation (reading routine would leave a
            # mark in `cfg`), fill in missing info
            if len(self.cfg) == 0:
                
                # First, fill in dimensional info
                definetrial(self, kwargs.get("trialdefinition"))

                # If necessary, construct list of channel labels (parsing is done by setter)
                channel = kwargs.get("channel")
                if isinstance(channel, str):
                    channel = [channel + str(i + 1) for i in range(self.data.shape[self.dimord.index("channel")])]
                self.channel = np.array(channel)

        # Dummy assignment: if we have no data but channel labels, assign bogus to tigger setter warning
        else:
            if isinstance(kwargs.get("channel"), (list, np.ndarray)):
                self.channel = ['channel']

class AnalogData(ContinuousData):

    @property
    def hdr(self):
        return self._hdr

    # "Constructor"
    def __init__(self,
                 data=None,
                 filename=None,
                 filetype=None,
                 trialdefinition=None,
                 samplerate=None,
                 channel="channel",
                 mode="w",
                 dimord=["time", "channel"]):
        """
        Docstring

        filename + data = create memmap @filename
        filename no data = read from file or memmap
        just data = try to attach data (error checking done by data.setter)
        """

        # The one thing we check right here and now
        expected = ["time", "channel"]
        if not set(dimord).issubset(expected):
            base = "dimensional labels {}"
            lgl = base.format("'" + "' x '".join(str(dim) for dim in expected) + "'")
            act = base.format("'" + "' x '".join(str(dim) for dim in dimord) + "'")
            raise SPYValueError(legal=lgl, varname="dimord", actual=act)

        # Hard constraint: required no. of data-dimensions
        self._ndim = 2

        # Assign default (blank) values
        self._hdr = None

        # Call parent initializer
        super().__init__(data=data,
                         filename=filename,
                         filetype=filetype,
                         trialdefinition=trialdefinition,
                         samplerate=samplerate,
                         channel=channel,
                         mode=mode,
                         dimord=dimord)

    # Overload ``clear`` method to account for `VirtualData` memmaps
    def clear(self):
        if isinstance(self.data, np.memmap):
            filename, mode = self.data.filename, self.data.mode
            self.data.flush()
            self._data = None
            self._data = open_memmap(filename, mode=mode)
        elif hasattr(self.data, "clear"):       # `VirtualData`
            self.data.clear()
        return

    # Overload ``copy`` method to account for `VirtualData` memmaps
    def copy(self, deep=False):
        cpy = copy(self)
        if deep:
            if isinstance(self.data, VirtualData):
                print("SyNCoPy core - copy: Deep copy not possible for " +
                      "VirtualData objects. Please use `save_spy` instead. ")
                return
            elif isinstance(self.data, (np.memmap, h5py.Dataset)):
                self.data.flush()
                filename = self._gen_filename()
                shutil.copyfile(self._filename, filename)
                cpy.data = filename
        return cpy

    
class SpectralData(ContinuousData):

    @property
    def taper(self):
        return self._dimlabels.get("taper")

    @taper.setter
    def taper(self, tpr):
        if self.data is None:
            print("SyNCoPy core - taper: Cannot assign `taper` without data. "+\
                  "Please assing data first")
            return
        ntap = self.data.shape[self.dimord.index("taper")]
        try:
            array_parser(tpr, varname="taper", ntype="str", dims=(ntap,))
        except Exception as exc:
            raise exc
        self._dimlabels["taper"] = np.array(tpr)

    @property
    def freq(self):
        return self._dimlabels.get("freq")

    @freq.setter
    def freq(self, freq):
        if self.data is None:
            print("SyNCoPy core - freq: Cannot assign `freq` without data. "+\
                  "Please assing data first")
            return
        nfreq = self.data.shape[self.dimord.index("freq")]
        try:
            array_parser(freq, varname="freq", dims=(nfreq,), hasnan=False, hasinf=False)
        except Exception as exc:
            raise exc
        self._dimlabels["freq"] = np.array(freq)
    
    # "Constructor"
    def __init__(self,
                 data=None,
                 filename=None,
                 filetype=None,
                 trialdefinition=None,
                 samplerate=None,
                 channel="channel",
                 taper=None,
                 freq=None,
                 mode="w",
                 dimord=["time", "taper", "freq", "channel"]):

        # The one thing we check right here and now
        expected = ["time", "taper", "freq", "channel"]
        if not set(dimord).issubset(expected):
            base = "dimensional labels {}"
            lgl = base.format("'" + "' x '".join(str(dim) for dim in expected) + "'")
            act = base.format("'" + "' x '".join(str(dim) for dim in dimord) + "'")
            raise SPYValueError(legal=lgl, varname="dimord", actual=act)

        # Hard constraint: required no. of data-dimensions
        self._ndim = 4

        # Call parent initializer
        super().__init__(data=data,
                         filename=filename,
                         filetype=filetype,
                         trialdefinition=trialdefinition,
                         samplerate=samplerate,
                         channel=channel,
                         taper=taper,
                         freq=freq,
                         mode=mode,
                         dimord=dimord)

        # If __init__ attached data, be careful
        if self.data is not None:

            # In case of manual data allocation (reading routine would leave a
            # mark in `cfg`), fill in missing info
            if len(self.cfg) == 0:
                self.freq = np.arange(self.data.shape[self.dimord.index("freq")])
                self.taper = np.array(["dummy_taper"] * self.data.shape[self.dimord.index("taper")])

        # Dummy assignment: if we have no data but freq/taper labels,
        # assign bogus to tigger setter warnings
        else:
            if freq is not None:
                self.freq = [1]
            if taper is not None:
                self.taper = ['taper']
