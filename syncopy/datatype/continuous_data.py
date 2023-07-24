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
from .base_data import BaseData, FauxTrial
from .methods.definetrial import definetrial
from .base_data import BaseData
from syncopy.shared.parsers import scalar_parser, array_parser
from syncopy.shared.errors import SPYValueError, SPYError
from syncopy.shared.tools import best_match
from syncopy.plotting import sp_plotting, mp_plotting
from syncopy.io.nwb import _analog_timelocked_to_nwbfile
from .util import TimeIndexer


from syncopy import __pynwb__

if __pynwb__:  # pragma: no cover
    from pynwb import NWBHDF5IO


__all__ = ["AnalogData", "SpectralData", "CrossSpectralData", "TimeLockData"]


class ContinuousData(BaseData, ABC):
    """Abstract class for uniformly sampled data

    Notes
    -----
    This class cannot be instantiated. Use one of the children instead.

    """

    _infoFileProperties = BaseData._infoFileProperties + (
        "samplerate",
        "channel",
    )
    _hdfFileDatasetProperties = BaseData._hdfFileDatasetProperties + ("data",)
    # all continuous data types have a time axis
    _selectionKeyWords = BaseData._selectionKeyWords + ("latency",)

    @property
    def data(self):
        """
        HDF5 dataset property representing contiguous
        data without trialdefinition.

        Trials are concatenated along the time axis.
        """

        if getattr(self._data, "id", None) is not None:
            if self._data.id.valid == 0:
                lgl = "open HDF5 file"
                act = "backing HDF5 file {} has been closed"
                raise SPYValueError(legal=lgl, actual=act.format(self.filename), varname="data")
        return self._data

    @data.setter
    def data(self, inData):

        self._set_dataset_property(inData, "data")

        if inData is None:
            return

    @property
    def is_time_locked(self):

        # check for equal offsets
        if not np.unique(self.trialdefinition[:, 2]).size == 1:
            return False

        # check for equal sample sizes of the trials
        if not np.unique(np.diff(self.sampleinfo, axis=1)).size == 1:
            return False

        return True

    def __str__(self):
        # Get list of print-worthy attributes
        ppattrs = [
            attr
            for attr in self.__dir__()
            if not (attr.startswith("_") or attr in ["log", "trialdefinition"])
        ]
        ppattrs = [
            attr
            for attr in ppattrs
            if not (inspect.ismethod(getattr(self, attr)) or isinstance(getattr(self, attr), Iterator))
        ]
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
            if hasattr(value, "shape") and attr == "data" and self.sampleinfo is not None:
                tlen = np.unique(np.diff(self.sampleinfo))
                if tlen.size == 1:
                    trlstr = "of length {} ".format(str(tlen[0]))
                else:
                    trlstr = ""
                dsize = np.prod(self.data.shape) * self.data.dtype.itemsize / 1024**2
                dunit = "MB"
                if dsize > 1000:
                    dsize /= 1024
                    dunit = "GB"
                valueString = "{} trials {}defined on ".format(str(len(self.trials)), trlstr)
                valueString += (
                    "["
                    + " x ".join([str(numel) for numel in value.shape])
                    + "] {dt:s} {tp:s} "
                    + "of size {sz:3.2f} {szu:s}"
                )
                valueString = valueString.format(
                    dt=self.data.dtype.name,
                    tp=self.data.__class__.__name__,
                    sz=dsize,
                    szu=dunit,
                )
            elif hasattr(value, "shape"):
                valueString = (
                    "[" + " x ".join([str(numel) for numel in value.shape]) + "] element " + str(type(value))
                )
            elif isinstance(value, list):
                if attr == "dimord" and value is not None:
                    valueString = dsep.join(dim for dim in self.dimord)
                else:
                    valueString = "{0} element list".format(len(value))
            elif isinstance(value, dict):
                msg = "dictionary with {nk:s}keys{ks:s}"
                keylist = value.keys()
                showkeys = len(keylist) < 7
                valueString = msg.format(
                    nk=str(len(keylist)) + " " if not showkeys else "",
                    ks=" '" + "', '".join(key for key in keylist) + "'" if showkeys else "",
                )
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
        """:class:`numpy.ndarray` : list of recording channel names"""
        # if data exists but no user-defined channel labels, create them on the fly
        if self._channel is None and self._data is not None:
            nChannel = self.data.shape[self.dimord.index("channel")]
            # default labels
            return np.array(["channel" + str(i + 1).zfill(len(str(nChannel))) for i in range(nChannel)])
        return self._channel

    @channel.setter
    def channel(self, channel):

        if channel is None:
            self._channel = None
            return

        if self.data is None:
            raise SPYValueError(
                "Syncopy: Cannot assign `channels` without data. " + "Please assign data first"
            )

        array_parser(
            channel,
            varname="channel",
            ntype="str",
            dims=(self.data.shape[self.dimord.index("channel")],),
        )

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

        scalar_parser(sr, varname="samplerate", lims=[np.finfo("float").eps, np.inf])
        self._samplerate = float(sr)
        # we need a new TimeIndexer
        if self.trialdefinition is not None:
            self._time = TimeIndexer(self.trialdefinition, self.samplerate, list(self._trial_ids))

    @BaseData.trialdefinition.setter
    def trialdefinition(self, trldef):

        # all-to-all trialdefinition
        if trldef is None:
            self._trialdefinition = np.array([[0, self.data.shape[self.dimord.index("time")], 0]])
            self._trial_ids = [0]
        else:
            scount = self.data.shape[self.dimord.index("time")]
            array_parser(trldef, varname="trialdefinition", dims=2)
            if trldef.shape[-1] < 3:
                lgl = "trialdefinition with at least 3 columns: [start, stop, offset]"
                act = f"got only {trldef.shape[-1]} columns"
                raise SPYValueError(lgl, "trialdefinition", act)

            array_parser(
                trldef[:, :2],
                varname="sampleinfo",
                hasnan=False,
                hasinf=False,
                ntype="int_like",
                lims=[0, scount],
            )

            self._trialdefinition = trldef.copy()
            self._trial_ids = np.arange(self.sampleinfo.shape[0])

        self._time = TimeIndexer(self.trialdefinition, self.samplerate, list(self._trial_ids))

    @property
    def time(self):
        """indexable iterable of the time arrays"""
        if self.samplerate is not None and self.sampleinfo is not None:
            return self._time

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
            tsel = self.selection.time[trialno]
            if isinstance(tsel, slice):
                if tsel.start is not None:
                    tstart = tsel.start
                else:
                    tstart = 0
                if tsel.stop is not None:
                    tstop = tsel.stop
                else:
                    tstop = stop - start

                # account for trial offsets and compute slicing index + shape
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

    # Make instantiation persistent in all subclasses
    def __init__(self, data=None, channel=None, samplerate=None, **kwargs):

        self._channel = None
        self._samplerate = None
        self._data = None
        self._time = None

        self.samplerate = samplerate  # use setter for error-checking

        # Call initializer
        super().__init__(data=data, **kwargs)

        # catches channel propagation
        # from concatenation of syncopy data objects
        if self._channel is None:
            self.channel = channel
        # overwrites channels from concatenation if desired
        elif channel is not None:
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
    _selectionKeyWords = ContinuousData._selectionKeyWords + ("channel",)

    # "Constructor"
    def __init__(
        self,
        data=None,
        filename=None,
        trialdefinition=None,
        samplerate=None,
        channel=None,
        dimord=None,
    ):
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
        super().__init__(
            data=data,
            filename=filename,
            trialdefinition=trialdefinition,
            samplerate=samplerate,
            channel=channel,
            dimord=dimord,
        )

        # set as instance attribute to allow modification
        self._hdfFileAttributeProperties = BaseData._hdfFileAttributeProperties + (
            "samplerate",
            "channel",
        )

    # implement plotting
    def singlepanelplot(self, shifted=True, **show_kwargs):

        figax = sp_plotting.plot_AnalogData(self, shifted, **show_kwargs)
        return figax

    def multipanelplot(self, **show_kwargs):

        figax = mp_plotting.plot_AnalogData(self, **show_kwargs)
        return figax

    def save_nwb(self, outpath, nwbfile=None, with_trialdefinition=True, is_raw=True):
        """Save AnalogData in Neurodata Without Borders (NWB) file format.
        An NWBFile represents a single session of an experiment.

        Parameters
        ----------
        outpath : str, path-like. Where to save the NWB file, including file name and `.nwb` extension.
            All directories in the path must exist. Example: `'mydata.nwb'`.

        nwbfile : :class:`~pynwb.file.NWBFile` instance
            Set to an existing instance to add an LFP signal with `is_raw=False`

        with_trialdefinition : Boolean, whether to save the trial definition in the NWB file.

        is_raw : Boolean, whether this is raw data (that should never change), as opposed to LFP data that
            typically originates from some preprocessing, e.g., down-sampling and detrending. Determines where
            data is stored in the NWB container, to make it easier for other software to interprete what
            the data represents. If `is_raw` is `True`, the ``ElectricalSeries`` is stored directly in an
            acquisition of the :class:`pynwb.NWBFile`. If False, it is stored inside an `LFP` instance in
            a processing group called `ecephys`.

        Returns
        -------
        nwbfile : :class:`~pynwb.file.NWBFile` instance
           Can be used to further add meta-information or even data via the pynwb API.
           To save use the :class:`pynwb.NWBHDF5IO` interface.

        Notes
        -----
        Due to the very general architecture of the NWB format, many fields need to be interpreted
        by software reading the format. Thus,
        providing a generic function to save Syncopy data in NWB format is possible only if you know who will read it.
        Depending on your target software, you may need to manually format the data using pynwb before writing
        it to disk, or manually open it using pynwb before using it with the target software.

        In place selections are ignored, the full dataset is exported. Create a new Syncopy data object from a selection
        before calling this function if you want to export a subset only.

        The Syncopy NWB reader only supports the NWB raw data format.

        This function requires the optional 'pynwb' dependency to be installed.
        """
        if not __pynwb__:
            raise SPYError("NWB support is not available. Please install the 'pynwb' package.")

        nwbfile = _analog_timelocked_to_nwbfile(
            self,
            nwbfile=nwbfile,
            with_trialdefinition=with_trialdefinition,
            is_raw=is_raw,
        )
        # Write the file to disk.
        with NWBHDF5IO(outpath, "w") as io:
            io.write(nwbfile)
        return nwbfile


class SpectralData(ContinuousData):
    """
    Multi-channel, real or complex spectral data

    This class can be used for representing any data with a frequency, channel,
    and optionally a time axis. The datatype can be complex or float.
    """

    _infoFileProperties = ContinuousData._infoFileProperties + (
        "taper",
        "freq",
    )
    _hdfFileAttributeProperties = BaseData._hdfFileAttributeProperties + (
        "samplerate",
        "channel",
        "freq",
    )
    _defaultDimord = ["time", "taper", "freq", "channel"]
    _stackingDimLabel = "time"
    _selectionKeyWords = ContinuousData._selectionKeyWords + (
        "channel",
        "frequency",
        "taper",
    )

    @property
    def taper(self):
        """:class:`numpy.ndarray` : list of window functions used"""
        if self._taper is None and self._data is not None:
            nTaper = self.data.shape[self.dimord.index("taper")]
            return np.array(["taper" + str(i + 1).zfill(len(str(nTaper))) for i in range(nTaper)])
        return self._taper

    @taper.setter
    def taper(self, tpr):

        if tpr is None:
            self._taper = None
            return

        if self.data is None:
            print("Syncopy core - taper: Cannot assign `taper` without data. " + "Please assing data first")

        try:
            array_parser(
                tpr,
                dims=(self.data.shape[self.dimord.index("taper")],),
                varname="taper",
                ntype="str",
            )
        except Exception as exc:
            raise exc

        self._taper = np.array(tpr)

    @property
    def freq(self):
        """:class:`numpy.ndarray`: frequency axis in Hz"""
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
            print("Syncopy core - freq: Cannot assign `freq` without data. " + "Please assing data first")
            return

        array_parser(
            freq,
            varname="freq",
            hasnan=False,
            hasinf=False,
            dims=(self.data.shape[self.dimord.index("freq")],),
        )
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
    def __init__(
        self,
        data=None,
        filename=None,
        trialdefinition=None,
        samplerate=None,
        channel=None,
        taper=None,
        freq=None,
        dimord=None,
    ):

        self._taper = None
        self._freq = None

        # FIXME: See similar comment above in `AnalogData.__init__()`
        if dimord is None:
            dimord = self._defaultDimord

        # Call parent initializer
        super().__init__(
            data=data,
            filename=filename,
            trialdefinition=trialdefinition,
            samplerate=samplerate,
            channel=channel,
            dimord=dimord,
        )

        # If __init__ attached data, be careful
        if self.data is not None:
            # In case of manual data allocation (reading routine would leave a
            # mark in `cfg`), fill in missing info
            if len(self.cfg) == 0:
                # concat operations will set this!
                if self.freq is None:
                    self.freq = freq
                if self.taper is None:
                    self.taper = taper

        # Dummy assignment: if we have no data but freq/taper labels,
        # assign bogus to trigger setter warnings
        else:
            self.freq = freq
            self.taper = taper

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
    _infoFileProperties = BaseData._infoFileProperties + (
        "samplerate",
        "channel_i",
        "channel_j",
        "freq",
    )
    _hdfFileAttributeProperties = BaseData._hdfFileAttributeProperties + (
        "samplerate",
        "channel_i",
        "channel_j",
        "freq",
    )
    _defaultDimord = ["time", "freq", "channel_i", "channel_j"]
    _stackingDimLabel = "time"
    _selectionKeyWords = ContinuousData._selectionKeyWords + (
        "channel_i",
        "channel_j",
        "frequency",
    )
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
        """:class:`numpy.ndarray` : list of recording channel names"""
        # if data exists but no user-defined channel labels, create them on the fly
        if self._channel_i is None and self._data is not None:
            nChannel = self.data.shape[self.dimord.index("channel_i")]
            return np.array(["channel" + str(i + 1).zfill(len(str(nChannel))) for i in range(nChannel)])

        return self._channel_i

    @channel_i.setter
    def channel_i(self, channel_i):
        """:class:`numpy.ndarray` : list of channel labels"""
        if channel_i is None:
            self._channel_i = None
            return

        if self.data is None:
            raise SPYValueError(
                "Syncopy: Cannot assign `channels` without data. " + "Please assign data first"
            )

        try:
            array_parser(
                channel_i,
                varname="channel_i",
                ntype="str",
                dims=(self.data.shape[self.dimord.index("channel_i")],),
            )
        except Exception as exc:
            raise exc

        self._channel_i = np.array(channel_i)

    @property
    def channel_j(self):
        """:class:`numpy.ndarray` : list of recording channel names"""
        # if data exists but no user-defined channel labels, create them on the fly
        if self._channel_j is None and self._data is not None:
            nChannel = self.data.shape[self.dimord.index("channel_j")]
            return np.array(["channel" + str(i + 1).zfill(len(str(nChannel))) for i in range(nChannel)])

        return self._channel_j

    @channel_j.setter
    def channel_j(self, channel_j):
        """:class:`numpy.ndarray` : list of channel labels"""
        if channel_j is None:
            self._channel_j = None
            return

        if self.data is None:
            raise SPYValueError(
                "Syncopy: Cannot assign `channels` without data. " + "Please assign data first"
            )

        try:
            array_parser(
                channel_j,
                varname="channel_j",
                ntype="str",
                dims=(self.data.shape[self.dimord.index("channel_j")],),
            )
        except Exception as exc:
            raise exc

        self._channel_j = np.array(channel_j)

    def __init__(
        self,
        data=None,
        filename=None,
        channel_i=None,
        channel_j=None,
        samplerate=None,
        freq=None,
        dimord=None,
    ):

        self._freq = None
        # Set dimensional labels
        self.dimord = dimord

        # Call parent initializer
        super().__init__(data=data, filename=filename, samplerate=samplerate, dimord=dimord)

        if freq is not None:
            # set frequencies
            self.freq = freq

    def singlepanelplot(self, **show_kwargs):

        return sp_plotting.plot_CrossSpectralData(self, **show_kwargs)


class TimeLockData(ContinuousData):

    """
    Multi-channel, uniformly-sampled, time-locked data.
    """

    _infoFileProperties = ContinuousData._infoFileProperties
    _defaultDimord = ["time", "channel"]
    _selectionKeyWords = ContinuousData._selectionKeyWords + ("channel",)
    _stackingDimLabel = "time"

    # "Constructor"
    def __init__(
        self,
        data=None,
        filename=None,
        trialdefinition=None,
        samplerate=None,
        channel=None,
        dimord=None,
    ):

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
        super().__init__(
            data=data,
            filename=filename,
            trialdefinition=trialdefinition,
            samplerate=samplerate,
            channel=channel,
            dimord=dimord,
        )

        # A `h5py.Dataset` holding the average of `data`, or `None` if not computed yet.
        self._avg = None

        # A `h5py.Dataset` holding variance of `data`, or `None` if not computed yet.
        self._var = None

        # A `h5py.Dataset` holding covariance of `data`, or `None` if not computed yet.
        self._cov = None

        # set as instance attribute to allow modification
        self._hdfFileDatasetProperties = ContinuousData._hdfFileDatasetProperties + (
            "avg",
            "var",
            "cov",
        )

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

        # we need parent setter for basic validation
        ContinuousData.trialdefinition.fset(self, trldef)

        # now check for additional conditions

        if not self.is_time_locked:
            lgl = "trialdefinition with equally sized trials and common offsets"
            act = "not timelock compatible trialdefinition"
            raise SPYValueError(lgl, "trialdefinition", act)

    # TODO - overload `time` property, as there is only one by definition!
    # implement plotting
    def singlepanelplot(self, shifted=True, **show_kwargs):

        figax = sp_plotting.plot_AnalogData(self, shifted, **show_kwargs)
        return figax

    def multipanelplot(self, **show_kwargs):

        figax = mp_plotting.plot_AnalogData(self, **show_kwargs)
        return figax

    def save_nwb(self, outpath, with_trialdefinition=True, is_raw=True):
        """Save TimeLockData in Neurodata Without Borders (NWB) file format.
        An NWBFile represents a single session of an experiment.

        Parameters
        ----------
        outpath : str, path-like. Where to save the NWB file, including file name and `.nwb` extension. All directories in the path must exist. Example: `'mydata.nwb'`.

        with_trialdefinition : Boolean, whether to save the trial definition in the NWB file.

        is_raw : Boolean, whether this is raw data (that should never change), as opposed to LFP data that originates from some processing, e.g., down-sampling and
         detrending. Determines where data is stored in the NWB container, to make it easier for other software to interprete what the data represents. If `is_raw` is `True`,
         the `ElectricalSeries` is stored directly in an acquisition of the :class:`pynwb.NWBFile`. If False, it is stored inside an `LFP` instance in a processing group called `ecephys`.
         Note that for the Syncopy NWB reader, the data should be stored as raw, so this is currently the default.

        Returns
        -------
        None, called for side effect of writing the NWB file to disk.

        Notes
        -----
        Due to the very general architecture of the NWB format, many fields need to be interpreted by software reading the format. Thus,
        providing a generic function to save Syncopy data in NWB format is possible only if you know who will read it.
        Depending on your target software, you may need to manually format the data using pynwb before writing it to disk, or manually
        open it using pynwb before using it with the target software.

        Selections are ignored, the full data is exported. Create a new Syncopy data object before calling this function if you want to export a subset only.

        This function requires the optional 'pynwb' dependency to be installed.
        """
        if not __pynwb__:
            raise SPYError("NWB support is not available. Please install the 'pynwb' package.")

        nwbfile = _analog_timelocked_to_nwbfile(
            self, nwbfile=None, with_trialdefinition=with_trialdefinition, is_raw=is_raw
        )
        # Write the file to disk.
        with NWBHDF5IO(outpath, "w") as io:
            io.write(nwbfile)
