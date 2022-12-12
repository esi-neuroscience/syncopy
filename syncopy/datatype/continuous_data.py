# -*- coding: utf-8 -*-
#
# Syncopy's abstract base class for continuous data + regular children
#

"""Uniformly sampled (continuous data).

This module holds classes to represent data with a uniformly sampled time axis.

"""
# Builtin/3rd party package imports
import inspect
import numpy as np
from abc import ABC
from collections.abc import Iterator

# Local imports
from .base_data import BaseData, FauxTrial, _definetrial
from .methods.definetrial import definetrial
from syncopy.shared.parsers import scalar_parser, array_parser
from syncopy.shared.errors import SPYValueError, SPYWarning
from syncopy.shared.tools import best_match
from syncopy.plotting import sp_plotting, mp_plotting


__all__ = ["AnalogData", "SpectralData", "CrossSpectralData", "TimeLockData"]


class ContinuousData(BaseData, ABC):
    """Abstract class for uniformly sampled data

    Notes
    -----
    This class cannot be instantiated. Use one of the children instead.

    """

    _infoFileProperties = BaseData._infoFileProperties + ("samplerate", "channel",)
    _hdfFileDatasetProperties = BaseData._hdfFileDatasetProperties + ("data",)
    # all continuous data types have a time axis
    _selectionKeyWords = BaseData._selectionKeyWords + ('latency',)

    @property
    def data(self):
        """array-like object representing data without trials

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

        self._set_dataset_property(inData, "data")

        if inData is None:
            return

    def __str__(self):
        # Get list of print-worthy attributes
        ppattrs = [attr for attr in self.__dir__()
                   if not (attr.startswith("_") or attr in ["log", "trialdefinition"])]
        ppattrs = [attr for attr in ppattrs
                   if not (inspect.ismethod(getattr(self, attr))
                           or isinstance(getattr(self, attr), Iterator))]
        if self.__class__.__name__ == "CrossSpectralData":
            ppattrs.remove("channel")
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
    def _shapes(self):
        if self.sampleinfo is not None:
            shp = [list(self.data.shape) for k in range(self.sampleinfo.shape[0])]
            for k, sg in enumerate(self.sampleinfo):
                shp[k][self._stackingDim] = sg[1] - sg[0]
            return [tuple(sp) for sp in shp]

    @property
    def channel(self):
        """ :class:`numpy.ndarray` : list of recording channel names """
        # if data exists but no user-defined channel labels, create them on the fly
        if self._channel is None and self._data is not None:
            nChannel = self.data.shape[self.dimord.index("channel")]
            return np.array(["channel" + str(i + 1).zfill(len(str(nChannel)))
                           for i in range(nChannel)])
        return self._channel

    @channel.setter
    def channel(self, channel):

        if channel is None:
            self._channel = None
            return

        if self.data is None:
            raise SPYValueError("Syncopy: Cannot assign `channels` without data. " +
                  "Please assign data first")

        try:
            array_parser(channel, varname="channel", ntype="str",
                         dims=(self.data.shape[self.dimord.index("channel")],))
        except Exception as exc:
            raise exc

        self._channel = np.array(channel)

    @property
    def samplerate(self):
        """float: sampling rate of uniformly sampled data in Hz"""
        return self._samplerate

    @samplerate.setter
    def samplerate(self, sr):
        if sr is None:
            self._samplerate = None
            return

        try:
            scalar_parser(sr, varname="samplerate", lims=[np.finfo('float').eps, np.inf])
        except Exception as exc:
            raise exc
        self._samplerate = float(sr)

    @property
    def time(self):
        """list(float): trigger-relative time axes of each trial """
        if self.samplerate is not None and self.sampleinfo is not None:
            return [(np.arange(0, stop - start) + self._t0[tk]) / self.samplerate \
                    for tk, (start, stop) in enumerate(self.sampleinfo)]

    # Helper function that grabs a single trial
    def _get_trial(self, trialno):
        idx = [slice(None)] * len(self.dimord)
        idx[self._stackingDim] = slice(int(self.sampleinfo[trialno, 0]), int(self.sampleinfo[trialno, 1]))
        return self._data[tuple(idx)]

    def _is_empty(self):
        return super()._is_empty() or self.samplerate is None

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
        shp = list(self.data.shape)
        idx = [slice(None)] * len(self.dimord)
        stop = int(self.sampleinfo[trialno, 1])
        start = int(self.sampleinfo[trialno, 0])
        shp[self._stackingDim] = stop - start
        idx[self._stackingDim] = slice(start, stop)

        # process existing data selections
        if self.selection is not None:

            # time-selection is most delicate due to trial-offset
            tsel = self.selection.time[self.selection.trial_ids.index(trialno)]
            if isinstance(tsel, slice):
                if tsel.start is not None:
                    tstart = tsel.start
                else:
                    tstart = 0
                if tsel.stop is not None:
                    tstop = tsel.stop
                else:
                    tstop = stop - start

                # account for trial offsets an compute slicing index + shape
                start = start + tstart
                stop = start + (tstop - tstart)
                idx[self._stackingDim] = slice(start, stop)
                shp[self._stackingDim] = stop - start

            else:
                idx[self._stackingDim] = [tp + start for tp in tsel]
                shp[self._stackingDim] = len(tsel)

            # process the rest
            dims = list(self.dimord)
            dims.pop(self._stackingDim)
            for dim in dims:
                sel = getattr(self.selection, dim)
                if sel is not None:
                    dimIdx = self.dimord.index(dim)
                    idx[dimIdx] = sel
                    if isinstance(sel, slice):
                        begin, end, delta = sel.start, sel.stop, sel.step
                        if sel.start is None:
                            begin = 0
                        elif sel.start < 0:
                            begin = shp[dimIdx] + sel.start
                        if sel.stop is None:
                            end = shp[dimIdx]
                        elif sel.stop < 0:
                            end = shp[dimIdx] + sel.stop
                        if sel.step is None:
                            delta = 1
                        shp[dimIdx] = int(np.ceil((end - begin) / delta))
                        idx[dimIdx] = slice(begin, end, delta)
                    elif isinstance(sel, list):
                        shp[dimIdx] = len(sel)
                    else:
                        shp[dimIdx] = 1

        return FauxTrial(shp, tuple(idx), self.data.dtype, self.dimord)

    # Helper function that extracts timing-related indices
    def _get_time(self, trials, toi=None, toilim=None):
        """
        Get relative by-trial indices of time-selections
        `toi` is legacy.. `toilim ` is used by selections via `latency`

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
            for trlno in trials:
                _, selTime = best_match(self.time[trlno], toilim, span=True)
                selTime = selTime.tolist()
                if len(selTime) > 1:
                    timing.append(slice(selTime[0], selTime[-1] + 1, 1))
                else:
                    timing.append(selTime)

        elif toi is not None:
            for trlno in trials:
                _, selTime = best_match(self.time[trlno], toi)
                selTime = selTime.tolist()
                if len(selTime) > 1:
                    timeSteps = np.diff(selTime)
                    if timeSteps.min() == timeSteps.max() == 1:
                        selTime = slice(selTime[0], selTime[-1] + 1, 1)
                timing.append(selTime)

        else:
            timing = [slice(None)] * len(trials)

        return timing

    # Make instantiation persistent in all subclasses
    def __init__(self, data=None, channel=None, samplerate=None, **kwargs):

        self._channel = None
        self._samplerate = None
        self._data = None

        self.samplerate = samplerate     # use setter for error-checking

        # Call initializer
        super().__init__(data=data, **kwargs)

        self.channel = channel

        if self.data is not None:

            # In case of manual data allocation (reading routine would leave a
            # mark in `cfg`), fill in missing info
            if self.sampleinfo is None:

                # First, fill in dimensional info
                definetrial(self, kwargs.get("trialdefinition"))

    # plotting, only virtual in the abc
    def singlepanelplot(self):
        raise NotImplementedError

    def multipanelplot(self):
        raise NotImplementedError


class AnalogData(ContinuousData):
    """Multi-channel, uniformly-sampled, analog (real float) data

    This class can be used for representing any analog signal data with a time
    and a channel axis such as local field potentials, firing rates, eye
    position etc.

    The data is always stored as a two-dimensional array on disk. On disk, Trials are
    concatenated along the time axis.

    Data is only read from disk on demand, similar to HDF5 files.
    """

    _infoFileProperties = ContinuousData._infoFileProperties
    _defaultDimord = ["time", "channel"]
    _stackingDimLabel = "time"
    _selectionKeyWords = ContinuousData._selectionKeyWords + ('channel',)

    # "Constructor"
    def __init__(self,
                 data=None,
                 filename=None,
                 trialdefinition=None,
                 samplerate=None,
                 channel=None,
                 dimord=None):
        """Initialize an :class:`AnalogData` object.

        Parameters
        ----------
            data : 2D :class:numpy.ndarray or HDF5 dataset
                multi-channel time series data with uniform sampling
            filename : str
                path to target filename that should be used for writing
            trialdefinition : :class:`EventData` object or Mx3 array
                [start, stop, trigger_offset] sample indices for `M` trials
            samplerate : float
                sampling rate in Hz
            channel : str or list/array(str)
            dimord : list(str)
                ordered list of dimension labels

        1. `filename` + `data` : create hdf dataset incl. sampleinfo @filename
        2. just `data` : try to attach data (error checking done by :meth:`AnalogData.data.setter`)

        See also
        --------
        :func:`syncopy.definetrial`

        """

        # FIXME: I think escalating `dimord` to `BaseData` should be sufficient so that
        # the `if any(key...) loop in `BaseData.__init__()` takes care of assigning a default dimord
        if dimord is None:
            dimord = self._defaultDimord

        # Call parent initializer
        super().__init__(data=data,
                         filename=filename,
                         trialdefinition=trialdefinition,
                         samplerate=samplerate,
                         channel=channel,
                         dimord=dimord)

        # set as instance attribute to allow modification
        self._hdfFileAttributeProperties = BaseData._hdfFileAttributeProperties + ("samplerate", "channel",)

    # implement plotting
    def singlepanelplot(self, shifted=True, **show_kwargs):

        figax = sp_plotting.plot_AnalogData(self, shifted, **show_kwargs)
        return figax

    def multipanelplot(self, **show_kwargs):

        figax = mp_plotting.plot_AnalogData(self, **show_kwargs)
        return figax


class SpectralData(ContinuousData):
    """
    Multi-channel, real or complex spectral data

    This class can be used for representing any data with a frequency, channel,
    and optionally a time axis. The datatype can be complex or float.
    """

    _infoFileProperties = ContinuousData._infoFileProperties + ("taper", "freq",)
    _defaultDimord = ["time", "taper", "freq", "channel"]
    _stackingDimLabel = "time"
    _selectionKeyWords = ContinuousData._selectionKeyWords + ('channel', 'frequency', 'taper',)

    @property
    def taper(self):
        """ :class:`numpy.ndarray` : list of window functions used """
        if self._taper is None and self._data is not None:
            nTaper = self.data.shape[self.dimord.index("taper")]
            return np.array(["taper" + str(i + 1).zfill(len(str(nTaper)))
                            for i in range(nTaper)])
        return self._taper

    @taper.setter
    def taper(self, tpr):

        if tpr is None:
            self._taper = None
            return

        if self.data is None:
            print("Syncopy core - taper: Cannot assign `taper` without data. "+\
                  "Please assing data first")

        try:
            array_parser(tpr, dims=(self.data.shape[self.dimord.index("taper")],),
                         varname="taper", ntype="str", )
        except Exception as exc:
            raise exc

        self._taper = np.array(tpr)

    @property
    def freq(self):
        """:class:`numpy.ndarray`: frequency axis in Hz """
        # if data exists but no user-defined frequency axis, create one on the fly
        if self._freq is None and self._data is not None:
            return np.arange(self.data.shape[self.dimord.index("freq")])
        return self._freq

    @freq.setter
    def freq(self, freq):

        if freq is None:
            self._freq = None
            return

        if self.data is None:
            print("Syncopy core - freq: Cannot assign `freq` without data. "+\
                  "Please assing data first")
            return
        try:

            array_parser(freq, varname="freq", hasnan=False, hasinf=False,
                         dims=(self.data.shape[self.dimord.index("freq")],))
        except Exception as exc:
            raise exc

        self._freq = np.array(freq)

    # Helper function that extracts frequency-related indices
    def _get_freq(self, foi=None, foilim=None):
        """
        `foi` is legacy, we use `foilim` for frequency selection
        Error checking is performed by `Selector` class
        """
        if foilim is not None:
            _, selFreq = best_match(self.freq, foilim, span=True)
            selFreq = selFreq.tolist()
            if len(selFreq) > 1:
                selFreq = slice(selFreq[0], selFreq[-1] + 1, 1)

        elif foi is not None:
            _, selFreq = best_match(self.freq, foi)
            selFreq = selFreq.tolist()
            if len(selFreq) > 1:
                freqSteps = np.diff(selFreq)
                if freqSteps.min() == freqSteps.max() == 1:
                    selFreq = slice(selFreq[0], selFreq[-1] + 1, 1)

        else:
            selFreq = slice(None)

        return selFreq

    # "Constructor"
    def __init__(self,
                 data=None,
                 filename=None,
                 trialdefinition=None,
                 samplerate=None,
                 channel=None,
                 taper=None,
                 freq=None,
                 dimord=None):

        self._taper = None
        self._freq = None

        # FIXME: See similar comment above in `AnalogData.__init__()`
        if dimord is None:
            dimord = self._defaultDimord

        # Call parent initializer
        super().__init__(data=data,
                         filename=filename,
                         trialdefinition=trialdefinition,
                         samplerate=samplerate,
                         channel=channel,
                         taper=taper,
                         freq=freq,
                         dimord=dimord)

        self._hdfFileAttributeProperties = BaseData._hdfFileAttributeProperties +\
        ("samplerate", "channel", "freq",)

        # If __init__ attached data, be careful
        if self.data is not None:

            # In case of manual data allocation (reading routine would leave a
            # mark in `cfg`), fill in missing info
            if len(self.cfg) == 0:
                self.freq = freq
                self.taper = taper

        # Dummy assignment: if we have no data but freq/taper labels,
        # assign bogus to trigger setter warnings
        else:
            if freq is not None:
                self.freq = [1]
            if taper is not None:
                self.taper = ['taper']

    # implement plotting
    def singlepanelplot(self, logscale=True, **show_kwargs):

        figax = sp_plotting.plot_SpectralData(self, logscale, **show_kwargs)
        return figax

    def multipanelplot(self, **show_kwargs):

        figax = mp_plotting.plot_SpectralData(self, **show_kwargs)
        return figax


class CrossSpectralData(ContinuousData):
    """
    Multi-channel real or complex spectral connectivity data

    This class can be used for representing channel-channel interactions involving
    frequency and optionally time or lag. The datatype can be complex or float.
    """

    # Adapt `infoFileProperties` and `hdfFileAttributeProperties` from `ContinuousData`
    _infoFileProperties = BaseData._infoFileProperties +\
        ("samplerate", "channel_i", "channel_j", "freq", )
    _defaultDimord = ["time", "freq", "channel_i", "channel_j"]
    _stackingDimLabel = "time"
    _selectionKeyWords = ContinuousData._selectionKeyWords + ('channel_i', 'channel_j', 'frequency',)
    _channel_i = None
    _channel_j = None
    _samplerate = None
    _data = None

    # Steal frequency-related stuff from `SpectralData`
    _get_freq = SpectralData._get_freq
    freq = SpectralData.freq

    # override channel property to avoid accidental access
    @property
    def channel(self):
        return "see channel_i and channel_j"

    @channel.setter
    def channel(self, channel):
        if channel is None:
            pass
        else:
            msg = f"CrossSpectralData has no 'channel' to set but dimord: {self._dimord}"
            raise NotImplementedError(msg)

    @property
    def channel_i(self):
        """ :class:`numpy.ndarray` : list of recording channel names """
        # if data exists but no user-defined channel labels, create them on the fly
        if self._channel_i is None and self._data is not None:
            nChannel = self.data.shape[self.dimord.index("channel_i")]
            return np.array(["channel" + str(i + 1).zfill(len(str(nChannel)))
                             for i in range(nChannel)])

        return self._channel_i

    @channel_i.setter
    def channel_i(self, channel_i):
        """ :class:`numpy.ndarray` : list of channel labels """
        if channel_i is None:
            self._channel_i = None
            return

        if self.data is None:
            raise SPYValueError("Syncopy: Cannot assign `channels` without data. " +
                  "Please assign data first")

        try:
            array_parser(channel_i, varname="channel_i", ntype="str",
                         dims=(self.data.shape[self.dimord.index("channel_i")],))
        except Exception as exc:
            raise exc

        self._channel_i = np.array(channel_i)

    @property
    def channel_j(self):
        """ :class:`numpy.ndarray` : list of recording channel names """
        # if data exists but no user-defined channel labels, create them on the fly
        if self._channel_j is None and self._data is not None:
            nChannel = self.data.shape[self.dimord.index("channel_j")]
            return np.array(["channel" + str(i + 1).zfill(len(str(nChannel)))
                             for i in range(nChannel)])

        return self._channel_j

    @channel_j.setter
    def channel_j(self, channel_j):
        """ :class:`numpy.ndarray` : list of channel labels """
        if channel_j is None:
            self._channel_j = None
            return

        if self.data is None:
            raise SPYValueError("Syncopy: Cannot assign `channels` without data. " +
                  "Please assign data first")

        try:
            array_parser(channel_j, varname="channel_j", ntype="str",
                         dims=(self.data.shape[self.dimord.index("channel_j")],))
        except Exception as exc:
            raise exc

        self._channel_j = np.array(channel_j)

    def __init__(self,
                 data=None,
                 filename=None,
                 channel_i=None,
                 channel_j=None,
                 samplerate=None,
                 freq=None,
                 dimord=None):

        # Set dimensional labels
        self.dimord = dimord
        # set frequencies
        self.freq = freq

        # Call parent initializer
        super().__init__(data=data,
                         filename=filename,
                         samplerate=samplerate,
                         freq=freq,
                         dimord=dimord)

        # set as instance attribute to allow modification
        self._hdfFileAttributeProperties = BaseData._hdfFileAttributeProperties +\
        ("samplerate", "channel_i", "channel_j", "freq", )

    def singlepanelplot(self, **show_kwargs):

        sp_plotting.plot_CrossSpectralData(self, **show_kwargs)


class TimeLockData(ContinuousData):

    """
    Multi-channel, uniformly-sampled, time-locked data.
    """

    _infoFileProperties = ContinuousData._infoFileProperties
    _defaultDimord = ["time", "channel"]
    _selectionKeyWords = ContinuousData._selectionKeyWords + ('channel',)
    _stackingDimLabel = "time"

    # "Constructor"
    def __init__(self,
                 data=None,
                 filename=None,
                 trialdefinition=None,
                 samplerate=None,
                 channel=None,
                 dimord=None):

        """
        Initialize an :class:`TimeLockData` object.

        Parameters
        ----------
        data : 2D :class:numpy.ndarray or HDF5 dataset
            multi-channel time series data with uniform sampling
        filename : str
            path to target filename that should be used for writing
        samplerate : float
            sampling rate in Hz
        channel : str or list/array(str)
        dimord : list(str)
            ordered list of dimension labels

        See also
        --------
        :func:`syncopy.definetrial`
        """

        if dimord is None:
            dimord = self._defaultDimord

        # Call parent initializer
        # trialdefinition has to come from a CR!
        super().__init__(data=data,
                         filename=filename,
                         trialdefinition=trialdefinition,
                         samplerate=samplerate,
                         channel=channel,
                         dimord=dimord)

        # A `h5py.Dataset` holding the average of `data`, or `None` if not computed yet.
        self._avg = None

        # A `h5py.Dataset` holding variance of `data`, or `None` if not computed yet.
        self._var = None

        # A `h5py.Dataset` holding covariance of `data`, or `None` if not computed yet.
        self._cov = None

        # set as instance attribute to allow modification
        self._hdfFileDatasetProperties = ContinuousData._hdfFileDatasetProperties + ("avg", "var", "cov",)

    @property
    def avg(self):
        return self._avg

    @property
    def var(self):
        return self._var

    @property
    def cov(self):
        return self._cov

    @ContinuousData.trialdefinition.setter
    def trialdefinition(self, trldef):
        """
        Override trialdefinition setter, which is special for time-locked data:
        all trials have to have the same length and relative timings.

        So the trialdefinition has the same offsets everywhere, and it has the general
        simple structure:
                              [[0, nSamples, offset],
                              [nSamples, 2 * nSamples, offset],
                              [2 * nSamples, 3 * nSamples, offset],
                              ...]
        """

        # first harness all parsers here
        _definetrial(self, trialdefinition=trldef)

        # now check for additional conditions

        # FIXME: not clear, is timelocked data to be expected
        # to have same offsets?!
        # if not np.unique(trldef[:, 2]).size == 1:
        #     lgl = "equal offsets for timelocked data"
        #     act = "different offsets"
        #     raise SPYValueError(lgl, varname="trialdefinition", actual=act)

        # diff-diff should give 0 -> same number of samples for each trial
        if not np.all(np.diff(trldef, axis=0, n=2) == 0):
            lgl = "all trials of same length for timelocked data"
            act = "unequal sized trials defined"
            raise SPYValueError(lgl, varname="trialdefinition", actual=act)

    # TODO - overload `time` property, as there is only one by definition!

    # implement plotting
    def singlepanelplot(self, shifted=True, **show_kwargs):

        figax = sp_plotting.plot_AnalogData(self, shifted, **show_kwargs)
        return figax

    def multipanelplot(self, **show_kwargs):

        figax = mp_plotting.plot_AnalogData(self, **show_kwargs)
        return figax
