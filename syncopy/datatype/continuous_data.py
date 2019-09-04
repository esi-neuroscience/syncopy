# -*- coding: utf-8 -*-
# 
# SynCoPy ContinuousData abstract class + regular children
# 
# Created: 2019-03-20 11:11:44
# Last modified by: Joscha Schmiedt [joscha.schmiedt@esi-frankfurt.de]
# Last modification time: <2019-09-04 15:38:09>
"""Uniformly sampled (continuous data).

This module holds classes to represent data with a uniformly sampled time axis.

"""
# Builtin/3rd party package imports
import h5py
import shutil
import inspect
import numpy as np
from abc import ABC
from copy import copy
from numpy.lib.format import open_memmap

# Local imports
from .base_data import BaseData, VirtualData, FauxTrial
from .data_methods import _selectdata_continuous, definetrial
from syncopy.shared.parsers import scalar_parser, array_parser, io_parser
from syncopy.shared.errors import SPYValueError, SPYIOError
import syncopy as spy

__all__ = ["AnalogData", "SpectralData"]


class ContinuousData(BaseData, ABC):
    """Abstract class for uniformly sampled data

    Notes
    -----
    This class cannot be instantiated. Use one of the children instead.

    """
    
    _infoFileProperties = BaseData._infoFileProperties + ("samplerate", "channel",)
    _hdfFileProperties = BaseData._hdfFileProperties + ("samplerate", "channel",)
        
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
        """ :class:`numpy.ndarray` : list of recording channel names """
        return self._channel

    @channel.setter
    def channel(self, chan):
        if self.data is None:
            print("SyNCoPy core - channel: Cannot assign `channels` without data. " +
                  "Please assing data first")
            return
        nchan = self.data.shape[self.dimord.index("channel")]
        try:
            array_parser(chan, varname="channel", ntype="str", dims=(nchan,))
        except Exception as exc:
            raise exc
        self._channel = np.array(chan)

    @property
    def samplerate(self):
        """float: sampling rate of uniformly sampled data in Hz"""
        return self._samplerate

    @samplerate.setter
    def samplerate(self, sr):
        try:
            scalar_parser(sr, varname="samplerate", lims=[np.finfo('float').eps, np.inf])
        except Exception as exc:
            raise exc
        self._samplerate = float(sr)

    @property
    def time(self):
        """list(float): trigger-relative time axes of each trial """
        if self.samplerate is not None and self.sampleinfo is not None:
            return [np.arange(self._t0[tk], end - start - self._t0[tk]) * 1/self.samplerate \
                    for tk, (start, end) in enumerate(self.sampleinfo)]

    # Selector method
    def selectdata(self, trials=None, deepcopy=False, **kwargs):
        """
        Docstring mostly pointing to ``selectdata``
        """
        return _selectdata_continuous(self, trials, deepcopy, **kwargs)

    # Helper function that reads a single trial into memory
    @staticmethod
    def _copy_trial(trialno, filename, dimord, sampleinfo, hdr):
        """
        # FIXME: currently unused - check back to see if we need this functionality
        """
        idx = [slice(None)] * len(dimord)
        idx[dimord.index("time")] = slice(int(sampleinfo[trialno, 0]), int(sampleinfo[trialno, 1]))
        idx = tuple(idx)
        if hdr is None:
            # Generic case: data is either a HDF5 dataset or memmap
            try:
                with h5py.File(filename, mode="r") as h5f:
                    h5keys = list(h5f.keys())
                    cnt = [h5keys.count(dclass) for dclass in spy.datatype.__all__
                           if not inspect.isfunction(getattr(spy.datatype, dclass))]
                    if len(h5keys) == 1:
                        arr = h5f[h5keys[0]][idx]
                    else:
                        arr = h5f[spy.datatype.__all__[cnt.index(1)]][idx]
            except:
                try:
                    arr = np.array(open_memmap(filename, mode="c")[idx])
                except:
                    raise SPYIOError(filename)
            return arr
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
            
        See also
        --------
        syncopy.datatype.base_data.FauxTrial : class definition and further details
        syncopy.shared.computational_routine.ComputationalRoutine : Syncopy compute engine
        """
        shp = list(self.data.shape)
        idx = [slice(None)] * len(self.dimord)
        tidx = self.dimord.index("time")
        stop = int(self.sampleinfo[trialno, 1])
        start = int(self.sampleinfo[trialno, 0])
        shp[tidx] = stop - start
        idx[tidx] = slice(start, stop)
        return FauxTrial(shp, tuple(idx), self.data.dtype)
    
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

        # Dummy assignment: if we have no data but channel labels, assign bogus to tigger setter warning
        else:
            if isinstance(kwargs.get("channel"), (list, np.ndarray)):
                self.channel = ['channel']

class AnalogData(ContinuousData):
    """Multi-channel, uniformly-sampled, analog (real float) data

    This class can be used for representing any analog signal data with a time
    and a channel axis such as local field potentials, firing rates, eye
    position etc.

    The data is always stored as a two-dimensional array on disk. On disk, Trials are
    concatenated along the time axis. 

    Data is only read from disk on demand, similar to memory maps and HDF5
    files.
    """
    
    _infoFileProperties = ContinuousData._infoFileProperties + ("_hdr",)
    
    @property
    def hdr(self):
        """dict with information about raw data
        
        This property is empty for data created by Syncopy.
        """
        return self._hdr

    # "Constructor"
    def __init__(self,
                 data=None,
                 filename=None,
                 trialdefinition=None,
                 samplerate=None,
                 channel="channel",
                 mode="w",
                 dimord=["time", "channel"]):
        """Initialize an :class:`AnalogData` object.
        
        Parameters
        ----------
            data : 2D :class:numpy.ndarray    
                multi-channel time series data with uniform sampling            
            filename : str
                path to filename or folder (spy container)
            trialdefinition : :class:`EventData` object or Mx3 array 
                [start, stop, trigger_offset] sample indices for `M` trials
            samplerate : float
                sampling rate in Hz
            channel : str or list/array(str)
            mode : str
                write mode for data. 'r' for read-only, 'w' for writable
            dimord : list(str)
                ordered list of dimension labels

        1. `filename` + `data` : create hdf dataset incl. sampleinfo @filename
        2. `filename` no `data` : read from file or memmap (spy, hdf5, npy file array -> memmap)
        3. just `data` : try to attach data (error checking done by :meth:`AnalogData.data.setter`)
        
        See also
        --------
        :func:`syncopy.definetrial`
        
        """

        # The one thing we check right here and now
        expected = ["time", "channel"]
        if not set(dimord) == set(expected):
            base = "dimensional labels {}"
            lgl = base.format("'" + "' x '".join(str(dim) for dim in expected) + "'")
            act = base.format("'" + "' x '".join(str(dim) for dim in dimord) + "'")
            raise SPYValueError(legal=lgl, varname="dimord", actual=act)

        # Assign default (blank) values
        self._hdr = None

        # Call parent initializer
        super().__init__(data=data,
                         filename=filename,
                         trialdefinition=trialdefinition,
                         samplerate=samplerate,
                         channel=channel,
                         mode=mode,
                         dimord=dimord)

    # Overload ``copy`` method to account for `VirtualData` memmaps
    def copy(self, deep=False):
        """Create a copy of the data object in memory.

        Parameters
        ----------
            deep : bool
                If `True`, a copy of the underlying data file is created in the temporary Syncopy folder

        
        Returns
        -------
            AnalogData
                in-memory copy of AnalogData object

        See also
        --------
        save_spy

        """

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
    """Multi-channel, real or complex spectral data

    This class can be used for representing any data with a frequency, channel,
    and optionally a time axis. The datatype can be complex or float.

    """
    
    _infoFileProperties = ContinuousData._infoFileProperties + ("taper", "freq",)
    
    @property
    def taper(self):
        return self._taper

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
        self._taper = np.array(tpr)

    @property
    def freq(self):
        """:class:`numpy.ndarray`: frequency axis in Hz """
        return self._freq

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
        self._freq = np.array(freq)
    
    # "Constructor"
    def __init__(self,
                 data=None,
                 filename=None,
                 trialdefinition=None,
                 samplerate=None,
                 channel="channel",
                 taper=None,
                 freq=None,
                 mode="w",
                 dimord=["time", "taper", "freq", "channel"]):

        # The one thing we check right here and now
        expected = ["time", "taper", "freq", "channel"]
        if not set(dimord) == set(expected):
            base = "dimensional labels {}"
            lgl = base.format("'" + "' x '".join(str(dim) for dim in expected) + "'")
            act = base.format("'" + "' x '".join(str(dim) for dim in dimord) + "'")
            raise SPYValueError(legal=lgl, varname="dimord", actual=act)

        # Call parent initializer
        super().__init__(data=data,
                         filename=filename,
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
