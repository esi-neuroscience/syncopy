# -*- coding: utf-8 -*-
# 
# SynCoPy ContinuousData abstract class + regular children
# 
# Created: 2019-03-20 11:11:44
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2020-04-15 12:04:21>
"""Uniformly sampled (continuous data).

This module holds classes to represent data with a uniformly sampled time axis.

"""
# Builtin/3rd party package imports
import h5py
import os
import inspect
import numpy as np
from abc import ABC
from collections.abc import Iterator
from numpy.lib.format import open_memmap

# Local imports
from .base_data import BaseData, FauxTrial
from .methods.definetrial import definetrial
from .methods.selectdata import selectdata
from syncopy.shared.parsers import scalar_parser, array_parser
from syncopy.shared.errors import SPYValueError, SPYIOError, SPYError, SPYTypeError, SPYWarning
from syncopy.shared.tools import best_match, layout_subplot_panels
from syncopy.plotting.spy_plotting import pltErrMsg, pltConfig
from syncopy import __plt__
if __plt__:
    import matplotlib.pyplot as plt
    import matplotlib as mpl

__all__ = ["AnalogData", "SpectralData"]


class ContinuousData(BaseData, ABC):
    """Abstract class for uniformly sampled data

    Notes
    -----
    This class cannot be instantiated. Use one of the children instead.

    """
    
    _infoFileProperties = BaseData._infoFileProperties + ("samplerate", "channel",)
    _hdfFileAttributeProperties = BaseData._hdfFileAttributeProperties + ("samplerate", "channel",)
    _hdfFileDatasetProperties = BaseData._hdfFileDatasetProperties + ("data",)
    
    @property
    def data(self):
        """array-like object representing data without trials
        
        Trials are concatenated along the time axis.
        """

        if getattr(self._data, "id", None) is not None:
            if self._data.id.valid == 0:
                lgl = "open HDF5 container"
                act = "backing HDF5 container {} has been closed"
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
                   if not (attr.startswith("_") or attr in ["log", "trialdefinition", "hdr"])]
        ppattrs = [attr for attr in ppattrs
                   if not (inspect.ismethod(getattr(self, attr))
                           or isinstance(getattr(self, attr), Iterator))]
        
        ppattrs.sort()

        # Construct string for pretty-printing class attributes
        dsep = "' x '"
        dinfo = ""
        hdstr = "Syncopy {clname:s} object with fields\n\n"
        ppstr = hdstr.format(diminfo=dinfo + "'"  + \
                             dsep.join(dim for dim in self.dimord) + "' " if self.dimord is not None else "Empty ",
                             clname=self.__class__.__name__)
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
            sid = self.dimord.index("time")
            shp = [list(self.data.shape) for k in range(self.sampleinfo.shape[0])]
            for k, sg in enumerate(self.sampleinfo):
                shp[k][sid] = sg[1] - sg[0]
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

    # # Helper function that reads a single trial into memory
    # @staticmethod
    # def _copy_trial(trialno, filename, dimord, sampleinfo, hdr):
    #     """
    #     # FIXME: currently unused - check back to see if we need this functionality
    #     """
    #     idx = [slice(None)] * len(dimord)
    #     idx[dimord.index("time")] = slice(int(sampleinfo[trialno, 0]), int(sampleinfo[trialno, 1]))
    #     idx = tuple(idx)
    #     if hdr is None:
    #         # Generic case: data is either a HDF5 dataset or memmap
    #         try:
    #             with h5py.File(filename, mode="r") as h5f:
    #                 h5keys = list(h5f.keys())
    #                 cnt = [h5keys.count(dclass) for dclass in spy.datatype.__all__
    #                        if not inspect.isfunction(getattr(spy.datatype, dclass))]
    #                 if len(h5keys) == 1:
    #                     arr = h5f[h5keys[0]][idx]
    #                 else:
    #                     arr = h5f[spy.datatype.__all__[cnt.index(1)]][idx]
    #         except:
    #             try:
    #                 arr = np.array(open_memmap(filename, mode="c")[idx])
    #             except:
    #                 raise SPYIOError(filename)
    #         return arr
    #     else:
    #         # For VirtualData objects
    #         dsets = []
    #         for fk, fname in enumerate(filename):
    #             dsets.append(np.memmap(fname, offset=int(hdr[fk]["length"]),
    #                                    mode="r", dtype=hdr[fk]["dtype"],
    #                                    shape=(hdr[fk]["M"], hdr[fk]["N"]))[idx])
    #         return np.vstack(dsets)

    # Helper function that grabs a single trial
    def _get_trial(self, trialno):
        idx = [slice(None)] * len(self.dimord)
        sid = self.dimord.index("time")
        idx[sid] = slice(int(self.sampleinfo[trialno, 0]), int(self.sampleinfo[trialno, 1]))
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
        
        # process existing data selections
        if self._selection is not None:
            
            # time-selection is most delicate due to trial-offset
            tsel = self._selection.time[self._selection.trials.index(trialno)]
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
                idx[tidx] = slice(start, stop)
                shp[tidx] = stop - start
                
            else:
                idx[tidx] = [tp + start for tp in tsel]
                shp[tidx] = len(tsel)

            # process the rest                
            for dim in ["channel", "freq", "taper"]:
                sel = getattr(self._selection, dim)
                if sel:
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
                    else:
                        shp[dimIdx] = len(sel)
                        
        return FauxTrial(shp, tuple(idx), self.data.dtype, self.dimord)
    
    # Helper function that extracts timing-related indices
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
        
        # Call initializer
        super().__init__(data=data, **kwargs)
        
        self.channel = channel
        self.samplerate = samplerate     # use setter for error-checking   
        self.data = data
        
        if self.data is not None:

            # In case of manual data allocation (reading routine would leave a
            # mark in `cfg`), fill in missing info
            if len(self.cfg) == 0:
                
                # First, fill in dimensional info
                definetrial(self, kwargs.get("trialdefinition"))


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
    _defaultDimord = ["time", "channel"]
    
    @property
    def hdr(self):
        """dict with information about raw data
        
        This property is empty for data created by Syncopy.
        """
        return self._hdr

    # Selector method
    def selectdata(self, trials=None, channels=None, toi=None, toilim=None):
        """
        Create new `AnalogData` object from selection
        
        Please refere to :func:`syncopy.selectdata` for detailed usage information. 
        
        Examples
        --------
        >>> ang2chan = ang.selectdata(channels=["channel01", "channel02"])
        
        See also
        --------
        syncopy.selectdata : create new objects via deep-copy selections
        """
        return selectdata(self, trials=trials, channels=channels, toi=toi, toilim=toilim)

    # Visualize data using a single-panel plot
    def singleplot(self, trials="all", channels="all", toilim=None, avg_channels=True,
                   title=None, grid=None, **kwargs):
        """
        Coming soon...
        
        if trials is `None`, use "raw" data
        
        if overlay is True -> new plots use existing figure via plt.gcf().gca()
        if overlay is plt axis -> use this axis for plotting
        
        if avg_trials but len(ax.trialPanels) > 1 -> raise Error (don't overlay
        multi-trial plot over avg-trial plot!)
        
        kwargs can contain:
            fig
            nrow
            ncol
        """
        
        # FIXME: maybe summarize this part in a `plotting_parser(...)`?
        if not __plt__:
            raise SPYError(pltErrMsg.format("singleplot"))
        
        # See if figure has been already created
        fig = kwargs.get("fig", None)
        
        # If `trials` is `None`, values of other selectors need to match
        if trials is None:
            if channels is None:
                lgl = "one of `channels` or `trials` to be not `None`"
                act = "both `channels` and `trials` are `None`"
                raise SPYValueError(legal=lgl, varname="trials/channels", actual=act)
            if toilim is not None:
                lgl = "`trials` to be not `None` to perform timing selection"
                act = "`toilim` was provided but `trials` is `None`"
                raise SPYValueError(legal=lgl, varname="trials/toilim", actual=act)
            if avg_trials:
                msg = "`trials` is `None` but `avg_trials` is `True`. " +\
                    "Cannot perform trial averaging without trial specification - " +\
                    "setting ``avg_trials = False``. " 
                SPYWarning(msg)
                avg_trials = False
            if hasattr(fig, "trialPanels"):
                lgl = "`trials` to be not `None` to append to multi-trial plot"
                act = "multi-trial plot overlay was requested but `trials` is `None`"
                raise SPYValueError(legal=lgl, varname="trials/overlay", actual=act)
        
        # Don't overlay multi-trial plot on top of avg-trial plot
        if hasattr(fig, "trialPanels") and avg_trials:
            lgl = "overlay of multi-trial plot"
            act = "trial averaging was requested for multi-trial plot overlay"
            raise SPYValueError(legal=lgl, varname="trials/avg_trials", actual=act)
        
        # Ensure any optional keywords controlling plotting appearance make sense
        if title is not None:
            if not isinstance(title, str):
                raise SPYTypeError(title, varname="title", expected="str")
        if grid is not None:
            if not isinstance(grid, bool):
                raise SPYTypeError(grid, varname="grid", expected="bool")
        
        # Pass provided selections on to `Selector` class which performs error 
        # checking and generates required indexing arrays
        self._selection = {"trials": trials, 
                           "channels": channels, 
                           "toilim": toilim}
        
        # Adjust selector for special case of not using any trials
        if trials is None:
            nTrials = 0
        else:    
            trList = self._selection.trials
            nTrials = len(trList)

        # Prepare indexing list respecting potential non-default `dimord`s
        idx = [slice(None), slice(None)]
        chanIdx = self.dimord.index("channel")
        timeIdx = self.dimord.index("time")
        idx[chanIdx] = self._selection.channel

        # If we're overlaying a multi-panel plot, ensure panel-count matches up
        if hasattr(fig, "trialPanels"):
            if nTrials != len(fig.trialPanels):
                lgl = "number of trials to plot matching existing panels in figure"
                act = "{} panels but {} trials for plotting".format(len(fig.trialPanels), 
                                                                    nTrials)
                raise SPYValueError(legal=lgl, varname="trials/figure panels", actual=act)
            
        # If required, construct subplot panel layout or vet provided layout
        if nTrials > 1 and not avg_trials and fig is None:
            nrow = kwargs.get("nrow", None)
            ncol = kwargs.get("ncol", None)
            nrow, ncol = layout_subplot_panels(nTrials, nrow=nrow, ncol=ncol)
        
        # Used for non-overlayed figure titles (both for `avg_trials` = `True`/`False`)
        chArr = self.channel[self._selection.channel]
        nChan = chArr.size
        if nChan > 1:
            chanStr = "Average of {} channels".format(nChan)
        else:
            chanStr = "{}".format(chArr[0])
        
        # Single panel
        if avg_trials or nTrials == 1:
            
            # FIXME: Use `TimelockData` to do this?
            
            # Ensure provided timing selection can actually be averaged (leverage 
            # the fact that `toilim` selections exclusively generate slices)
            tLengths = np.zeros((nTrials,), dtype=np.intp)
            for k, tsel in enumerate(self._selection.time):
                start, stop = tsel.start, tsel.stop
                if start is None:
                    start = 0
                if stop is None:
                    stop = self._get_time([trList[k]], 
                                          toilim=[-np.inf, np.inf])[0].stop
                tLengths[k] = stop - start
                
            # For averaging, all `toilim` selections must be of identical length. 
            # If they aren't close any freshly opened figures and complain appropriately
            if np.unique(tLengths).size > 1:
                lgl = "time-selections of equal length for averaging"
                act = "time-selections of varying length"
                raise SPYValueError(legal=lgl, varname="toilim/avg_trials", actual=act)

            # Compute channel-/trial-average time-course: 2D array with slice/list
            # selection does not require fancy indexing - no need to check this here
            pltArr = np.zeros((tLengths[0],), dtype=self.data.dtype)
            for k, trlno in enumerate(trList):
                idx[timeIdx] = self._selection.time[k]
                pltArr += self._get_trial(trlno)[tuple(idx)].mean(axis=chanIdx).squeeze()
            pltArr /= nTrials
            
            # Prepare new axis or fetch existing
            if fig:
                ax, = fig.get_axes()
            else:
                fig, ax = plt.subplots(1, tight_layout=True,
                                          figsize=pltConfig["singleAvgTrialFigSize"])
                fig.objCount = 0
                ax.set_xlabel("time [s]", size=pltConfig["singleLabelSize"])
                ax.tick_params(axis="both", labelsize=pltConfig["singleTickSize"])
                
            # The actual plotting command is literally one line...
            time = self.time[trList[0]][self._selection.time[0]]
            ax.plot(time , pltArr, label=os.path.basename(self.filename))
            ax.set_xlim([time[0], time[-1]])
            
            # If grid-line modifier is set, apply it now
            if grid is not None:
                ax.grid(grid)
            
            # If no plots were present in the current figure, use a fancy title, 
            # otherwise, the title just references the no. of overlaid objects            
            if fig.objCount == 0:
                if title is None:
                    if nTrials > 1:
                        trStr = "{0}across {1} trials".format("averaged " if nChan == 1 else "",
                                                            nTrials)
                    else:
                        trStr = "Trial #{}".format(trList[0])
                        title = "{}, {}".format(chanStr, trStr)
                ax.set_title(title, size=pltConfig["singleTitleSize"])
            else:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels)
                if title is None:
                    title = "Overlay of {} datasets".format(len(handles))
                ax.set_title(title, size=pltConfig["singleTitleSize"])
         
        # Multi-panel   
        elif nTrials > 1:
            
            # Prepare new axes or fetch existing (layout is optimized for one to 
            # two panel rows)
            if fig:
                ax_arr = fig.get_axes()
                nrow, ncol = ax_arr[0].numRows, ax_arr[0].numCols
            else:
                (fig, ax_arr) = plt.subplots(nrow, ncol, constrained_layout=False,
                                             gridspec_kw={"wspace": 0, "hspace": 0.35,
                                                          "left": 0.05, "right": 0.97},
                                             figsize=pltConfig["singleMultiTrialFigSize"],
                                             sharey=True, squeeze=False)

                # Show xlabel only on bottom panel row
                for col in range(ncol):
                    ax_arr[-1, col].set_xlabel("time [s]", size=pltConfig["singleLabelSize"])
                    
                # Flatten axis array to make counting a little easier in here
                ax_arr = ax_arr.flatten(order="C")
                
                # Format axes that will actually contain something
                for k, trlno in enumerate(trList):
                    ax_arr[k].set_title("Trial #{}".format(trlno), size=pltConfig["singleTitleSize"])
                    ax_arr[k].tick_params(axis="both", labelsize=pltConfig["singleTickSize"])
                
                # Make any remainders as unobtrusive as possible
                for j in range(k + 1, nrow * ncol):
                    ax_arr[j].set_xticks([])
                    ax_arr[j].set_yticks([])
                    ax_arr[j].set_xlabel("")
                    for spine in ax_arr[j].spines.values():
                        spine.set_visible(False)
                ax_arr[min(k + 1, nrow * ncol - 1)].spines["left"].set_visible(True)
                        
                fig.objCount = 0
                fig.trialPanels = list(trList)
                    
            # Cycle through panels to plot by-trial channel(-averaged) data
            for k, trlno in enumerate(trList):
                idx[timeIdx] = self._selection.time[k]
                time = self.time[trList[k]][self._selection.time[k]]
                ax_arr[k].plot(time, 
                               self._get_trial(trlno)[tuple(idx)].mean(axis=chanIdx).squeeze(),
                               label=os.path.basename(self.filename))
                ax_arr[k].set_xlim([time[0], time[-1]])
                ax_arr[k].set_xticks(ax_arr[k].get_xticks()[int(k > 0):])
                if grid is not None:
                    ax_arr[k].grid(grid)
                
            # If we're overlaying datasets, adjust panel- and sup-titles: include
            # legend in top-right axis (note: `ax_arr` is row-major flattened)
            if fig.objCount == 0:
                if title is None:
                    title = chanStr
                fig.suptitle(title, size=pltConfig["singleTitleSize"])
            else:
                for k, trlno in enumerate(trList):
                    ax_arr[k].set_title("{0}/#{1}".format(ax_arr[k].get_title(), trlno))
                ax = ax_arr[ncol - 1]
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels)
                if title is None:
                    title = "Overlay of {} datasets".format(len(handles))
                fig.suptitle(title, size=pltConfig["singleTitleSize"])

        # Single panel "raw"                
        else:
            
            # Prepare new axis or fetch existing
            if fig:
                ax, = fig.get_axes()
            else:
                fig, ax = plt.subplots(1, tight_layout=True,
                                          figsize=pltConfig["singleAvgTrialFigSize"])
                fig.objCount = 0
                ax.set_xlabel("samples", size=pltConfig["singleLabelSize"])
                ax.tick_params(axis="both", labelsize=pltConfig["singleTickSize"])

            # Plot entire time-course of selected channel(s)                
            ax.plot(self.data[tuple(idx)].mean(axis=chanIdx).squeeze())
            if grid is not None:
                ax.grid(grid)
            
            # Set plot title depending on dataset overlay
            if fig.objCount == 0:
                if title is None:
                    title = chanStr
                ax.set_title(title, size=pltConfig["singleTitleSize"])
            else:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels)
                if title is None:
                    title = "Overlay of {} datasets".format(len(handles))
                ax.set_title(title, size=pltConfig["singleTitleSize"])
                
        # Increment overlay-counter and draw figure
        fig.objCount += 1
        plt.draw()
        
    # Visualize data using a multi-panel plot
    def multiplot(self, trials="all", channels="all", toilim=None, avg_trials=True,
                  avg_channels=False, title=None, grid=None, **kwargs):
        """
        Coming soon...
        
        avg_channel = False -> multi-panel plot
        avg_trial = False -> multi panel plot
        
        avg_channel = avg_trial = False -> multi-panel multi-line plot
        """
        
        # FIXME: maybe summarize this part in a `plotting_parser(...)`?
        if not __plt__:
            raise SPYError(pltErrMsg.format("multiplot"))
        
        # Pass provided selections on to `Selector` class which performs error 
        # checking and generates required indexing arrays
        self._selection = {"trials": trials, 
                           "channels": channels, 
                           "toilim": toilim}
        
        # Ensure binary flags are in fact binary
        vNames = ["avg_trials", "avg_channels"]
        for k, flag in enumerate([avg_trials, avg_channels]):
            if not isinstance(flag, bool):
                raise SPYTypeError(flag, varname=vNames[k], expected="bool")

        # Ensure any optional keywords controlling plotting appearance make sense
        if title is not None:
            if not isinstance(title, str):
                raise SPYTypeError(title, varname="title", expected="str")
        if grid is not None:
            if not isinstance(grid, bool):
                raise SPYTypeError(grid, varname="grid", expected="bool")

        # Get trial/channel count ("raw" plotting constitutes a special case)
        if trials is None:
            nTrials = 0
            if toilim is not None:
                lgl = "`trials` to be not `None` to perform timing selection"
                act = "`toilim` was provided but `trials` is `None`"
                raise SPYValueError(legal=lgl, varname="trials/toilim", actual=act)
            if avg_trials:
                msg = "`trials` is `None` but `avg_trials` is `True`. " +\
                    "Cannot perform trial averaging without trial specification - " +\
                    "setting ``avg_trials = False``. " 
                SPYWarning(msg)
                avg_trials = False
        else:    
            trList = self._selection.trials
            nTrials = len(trList)
        chArr = self.channel[self._selection.channel]
        nChan = chArr.size

        # If we're overlaying a multi-panel plot, ensure settings match up
        fig = kwargs.get("fig", None)
        if hasattr(fig, "nTrialPanels"):
            if nTrials != fig.nTrialPanels:
                lgl = "number of trials to plot matching existing panels in figure"
                act = "{} panels but {} trials for plotting".format(fig.nTrialPanels, 
                                                                    nTrials)
                raise SPYValueError(legal=lgl, varname="trials/figure panels", actual=act)
            if avg_trials:
                lgl = "overlay of multi-trial plot"
                act = "trial averaging was requested for multi-trial plot overlay"
                raise SPYValueError(legal=lgl, varname="trials/avg_trials", actual=act)
            if trials is None:
                lgl = "`trials` to be not `None` to append to multi-trial plot"
                act = "multi-trial plot overlay was requested but `trials` is `None`"
                raise SPYValueError(legal=lgl, varname="trials/overlay", actual=act)
        if hasattr(fig, "nChanPanels"):
            if nChan != fig.nChanPanels:
                lgl = "number of channels to plot matching existing panels in figure"
                act = "{} panels but {} channels for plotting".format(fig.nChanPanels, 
                                                                      nChan)
                raise SPYValueError(legal=lgl, varname="channels/figure panels", actual=act)
            if avg_channels:
                lgl = "overlay of multi-channel plot"
                act = "channel averaging was requested for multi-channel plot overlay"
                raise SPYValueError(legal=lgl, varname="channels/avg_channels", actual=act)
        if hasattr(fig, "chanOffsets"):
            if avg_channels:
                lgl = "multi-channel plot"
                act = "channel averaging was requested for multi-channel plot overlay"
                raise SPYValueError(legal=lgl, varname="channels/avg_channels", actual=act)
            if nChan != len(fig.chanOffsets):
                lgl = "channel-count matching existing multi-channel panels in figure"
                act = "{} channels per panel but {} channels for plotting".format(len(fig.chanOffsets), 
                                                                                  nChan)
                raise SPYValueError(legal=lgl, varname="channels/channels per panel", actual=act)

        # Prepare indexing list respecting potential non-default `dimord`s
        idx = [slice(None), slice(None)]
        chanIdx = self.dimord.index("channel")
        timeIdx = self.dimord.index("time")
        idx[chanIdx] = self._selection.channel

        # Either construct subplot panel layout/vet provided layout or fetch existing
        if fig is None:
            
            # Determine no. of required panels
            if avg_trials and not avg_channels:
                npanels = nChan 
            elif not avg_trials and avg_channels:
                npanels = nTrials
            elif not avg_trials and not avg_channels:
                npanels = int(nTrials == 0) * nChan + nTrials
            else:
                msg = "Averaging across both trials and channels results in " +\
                    "single-panel plot. Please use `singleplot` instead"
                SPYWarning(msg)
                return
            
            # Construct subplot panel layout or vet provided layout
            nrow = kwargs.get("nrow", None)
            ncol = kwargs.get("ncol", None)
            nrow, ncol = layout_subplot_panels(npanels, nrow=nrow, ncol=ncol)
            (fig, ax_arr) = plt.subplots(nrow, ncol, constrained_layout=False,
                                         gridspec_kw={"wspace": 0, "hspace": 0.35,
                                                      "left": 0.05, "right": 0.97},
                                         figsize=pltConfig["multiFigSize"],
                                         sharey=True, squeeze=False)

            # Show xlabel only on bottom row of panels
            if nTrials > 0:
                xLabel = "time [s]"
            else:
                xLabel = "samples"
            for col in range(ncol):
                ax_arr[-1, col].set_xlabel(xLabel, size=pltConfig["multiLabelSize"])
                
            # Omit first x-tick in all panels except first panel-row
            for row in range(nrow):
                for col in range(1, ncol):
                    ax_arr[row, col].xaxis.get_major_locator().set_params(prune="lower")
                    
            # Flatten axis array to make counting a little easier in here and make
            # any surplus panels as unobtrusive as possible
            ax_arr = ax_arr.flatten(order="C")
            for ax in ax_arr:
                ax.tick_params(axis="both", labelsize=pltConfig["multiTickSize"])
                ax.autoscale(enable=True, axis="x", tight=True)
            for k in range(npanels, nrow * ncol):
                ax_arr[k].set_xticks([])
                ax_arr[k].set_yticks([])
                ax_arr[k].set_xlabel("")
                for spine in ax_arr[k].spines.values():
                    spine.set_visible(False)
            ax_arr[min(npanels, nrow * ncol - 1)].spines["left"].set_visible(True)
            
            # Monkey-patch object-counter to newly created figure
            fig.objCount = 0

        # Get existing layout
        else:
            ax_arr = fig.get_axes()
            nrow, ncol = ax_arr[0].numRows, ax_arr[0].numCols
            
        # Panels correspond to channels
        if avg_trials and not avg_channels:
            
            # FIXME: Use `TimelockData` to do this?
            
            # Ensure provided timing selection can actually be averaged (leverage 
            # the fact that `toilim` selections exclusively generate slices)
            tLengths = np.zeros((nTrials,), dtype=np.intp)
            for k, tsel in enumerate(self._selection.time):
                start, stop = tsel.start, tsel.stop
                if start is None:
                    start = 0
                if stop is None:
                    stop = self._get_time([trList[k]], 
                                          toilim=[-np.inf, np.inf])[0].stop
                tLengths[k] = stop - start
                
            # For averaging, all `toilim` selections must be of identical length. 
            # If they aren't close any freshly opened figures and complain appropriately
            if np.unique(tLengths).size > 1:
                if fig.objCount == 0:
                    plt.close(fig)
                lgl = "time-selections of equal length for averaging"
                act = "time-selections of varying length"
                raise SPYValueError(legal=lgl, varname="toilim/avg_trials", actual=act)

            # Compute trial-averaged time-courses: 2D array with slice/list
            # selection does not require fancy indexing - no need to check this here
            pltArr = np.zeros((tLengths[0], nChan), dtype=self.data.dtype)
            for k, trlno in enumerate(trList):
                idx[timeIdx] = self._selection.time[k]
                pltArr += self._get_trial(trlno)[tuple(idx)]
            pltArr /= nTrials
                    
            # Cycle through channels and plot trial-averaged time-courses (time-
            # axis must be identical for all channels, set up `idx` just once)
            idx[timeIdx] = self._selection.time[0]
            time = self.time[trList[k]][self._selection.time[0]]
            for k, chan in enumerate(chArr):
                ax_arr[k].plot(time, pltArr[:, k], label=os.path.basename(self.filename))
                if grid is not None:
                    ax_arr[k].grid(grid)
                
            # If we're overlaying datasets, adjust panel- and sup-titles: include
            # legend in top-right axis (note: `ax_arr` is row-major flattened)
            if fig.objCount == 0:
                for k, chan in enumerate(chArr):
                    ax_arr[k].set_title(chan, size=pltConfig["multiTitleSize"])
                fig.nChanPanels = nChan
                if title is None:
                    if nTrials > 1:
                        title = "Average of {} trials".format(nTrials)
                    else:
                        title = "Trial #{}".format(trList[0])
                fig.suptitle(title, size=pltConfig["singleTitleSize"])
            else:
                for k, chan in enumerate(chArr):
                    ax_arr[k].set_title("{0}/{1}".format(ax_arr[k].get_title(), chan))
                ax = ax_arr[ncol - 1]
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels)
                if title is None:
                    title = "Overlay of {} datasets".format(len(handles))
                fig.suptitle(title, size=pltConfig["singleTitleSize"])
                
        # Panels correspond to trials
        elif not avg_trials and avg_channels:
            
            # Cycle through panels to plot by-trial channel-averages
            for k, trlno in enumerate(trList):
                idx[timeIdx] = self._selection.time[k]
                time = self.time[trList[k]][self._selection.time[k]]
                ax_arr[k].plot(time, 
                               self._get_trial(trlno)[tuple(idx)].mean(axis=chanIdx).squeeze(),
                               label=os.path.basename(self.filename))
                if grid is not None:
                    ax_arr[k].grid(grid)

            # If we're overlaying datasets, adjust panel- and sup-titles: include
            # legend in top-right axis (note: `ax_arr` is row-major flattened)
            if fig.objCount == 0:
                for k, trlno in enumerate(trList):
                    ax_arr[k].set_title("Trial #{}".format(trlno), size=pltConfig["multiTitleSize"])
                fig.nTrialPanels = nTrials
                if title is None:
                    if nChan > 1:
                        title = "Average of {} channels".format(nChan)
                    else:
                        title = chArr[0]
                fig.suptitle(title, size=pltConfig["singleTitleSize"])
            else:
                for k, trlno in enumerate(trList):
                    ax_arr[k].set_title("{0}/#{1}".format(ax_arr[k].get_title(), trlno))
                ax = ax_arr[ncol - 1]
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels)
                if title is None:
                    title = "Overlay of {} datasets".format(len(handles))
                fig.suptitle(title, size=pltConfig["singleTitleSize"])

        # Panels correspond to channels (if `trials` is `None`) otherwise trials
        elif not avg_trials and not avg_channels:
            
            # Plot each channel in separate panel
            if trials is None:
                chanSec = np.arange(self.channel.size)[self._selection.channel]
                for k, chan in enumerate(chanSec):
                    idx[chanIdx] = chan
                    ax_arr[k].plot(self.data[tuple(idx)].squeeze(),
                                   label=os.path.basename(self.filename))
                    if grid is not None:
                        ax_arr[k].grid(grid)
                        
                # If we're overlaying datasets, adjust panel- and sup-titles: include
                # legend in top-right axis (note: `ax_arr` is row-major flattened)
                if fig.objCount == 0:
                    for k, chan in enumerate(chArr):
                        ax_arr[k].set_title(chan, size=pltConfig["multiTitleSize"])
                    fig.nChanPanels = nChan
                    if title is None:
                        title = "Entire Data Timecourse"
                    fig.suptitle(title, size=pltConfig["singleTitleSize"])
                else:
                    for k, chan in enumerate(chArr):
                        ax_arr[k].set_title("{0}/{1}".format(ax_arr[k].get_title(), chan))
                    ax = ax_arr[ncol - 1]
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels)
                    if title is None:
                        title = "Overlay of {} datasets".format(len(handles))
                    fig.suptitle(title, size=pltConfig["singleTitleSize"])
            
            # Each trial gets its own panel w/multiple channels per panel
            else:

                # Prepare reshaping index to convert the (N,)-`chanOffset` array 
                # to a row/column vector depending on `dimord`
                rIdx = [1, 1]
                rIdx[chanIdx] = nChan
                rIdx = tuple(rIdx)
                
                # If required, compute max amplitude across provided trials + channels
                if not hasattr(fig, "chanOffsets"):
                    maxAmps = np.zeros((nTrials,), dtype=self.data.dtype)
                    tickOffsets = maxAmps.copy()
                    for k, trlno in enumerate(trList):
                        idx[timeIdx] = self._selection.time[k]
                        pltArr = np.abs(self._get_trial(trlno)[tuple(idx)])
                        maxAmps[k] = pltArr.max()
                        tickOffsets[k] = pltArr.mean()
                    fig.chanOffsets = np.cumsum([0] + [maxAmps.max()] * (nChan - 1))
                    fig.tickOffsets = fig.chanOffsets + tickOffsets.mean()
                
                # Cycle through panels to plot by-trial multi-channel time-courses
                for k, trlno in enumerate(trList):
                    idx[timeIdx] = self._selection.time[k]
                    time = self.time[trList[k]][self._selection.time[k]]
                    pltArr = self._get_trial(trlno)[tuple(idx)]
                    ax_arr[k].plot(time, 
                                   (pltArr + fig.chanOffsets.reshape(rIdx)).reshape(time.size, nChan), 
                                   color=plt.rcParams["axes.prop_cycle"].by_key()["color"][fig.objCount],
                                   label=os.path.basename(self.filename))
                    if grid is not None:
                        ax_arr[k].grid(grid)

                # If we're overlaying datasets, adjust panel- and sup-titles: include
                # legend in top-right axis (note: `ax_arr` is row-major flattened)
                # Note: y-axis is shared across panels, so `yticks` need only be set once
                if fig.objCount == 0:
                    for k, trlno in enumerate(trList):
                        ax_arr[k].set_title("Trial #{}".format(trlno), size=pltConfig["multiTitleSize"])
                    ax_arr[0].set_yticks(fig.tickOffsets)
                    ax_arr[0].set_yticklabels(chArr)
                    for i in range(k + 1, nrow * ncol):
                        ax_arr[i].tick_params(axis="both", length=0, width=0)
                    fig.nTrialPanels = nTrials
                    if title is None:
                        if nChan > 1:
                            title = "{} channels".format(nChan)
                        else:
                            title = chArr[0]
                    fig.suptitle(title, size=pltConfig["singleTitleSize"])
                else:
                    for k, trlno in enumerate(trList):
                        ax_arr[k].set_title("{0}/#{1}".format(ax_arr[k].get_title(), trlno))
                    ax_arr[0].set_yticklabels([" "] * chArr.size)
                    ax = ax_arr[ncol - 1]
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles[ : : (nChan + 1)], 
                              labels[ : : (nChan + 1)])
                    if title is None:
                        title = "Overlay of {} datasets".format(len(handles))
                    fig.suptitle(title, size=pltConfig["singleTitleSize"])
        
        # Increment overlay-counter and draw figure
        fig.objCount += 1
        plt.draw()

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
        if data is not None and dimord is None:
            dimord = self._defaultDimord            

        # Assign default (blank) values
        self._hdr = None

        # Call parent initializer
        super().__init__(data=data,
                         filename=filename,
                         trialdefinition=trialdefinition,
                         samplerate=samplerate,
                         channel=channel,
                         dimord=dimord)

    # # Overload ``copy`` method to account for `VirtualData` memmaps
    # def copy(self, deep=False):
    #     """Create a copy of the data object in memory.

    #     Parameters
    #     ----------
    #         deep : bool
    #             If `True`, a copy of the underlying data file is created in the temporary Syncopy folder

        
    #     Returns
    #     -------
    #         AnalogData
    #             in-memory copy of AnalogData object

    #     See also
    #     --------
    #     save_spy

    #     """

    #     cpy = copy(self)
        
    #     if deep:
    #         if isinstance(self.data, VirtualData):
    #             print("SyNCoPy core - copy: Deep copy not possible for " +
    #                   "VirtualData objects. Please use `save_spy` instead. ")
    #             return
    #         elif isinstance(self.data, (np.memmap, h5py.Dataset)):
    #             self.data.flush()
    #             filename = self._gen_filename()
    #             shutil.copyfile(self._filename, filename)
    #             cpy.data = filename
    #     return cpy


class SpectralData(ContinuousData):
    """Multi-channel, real or complex spectral data

    This class can be used for representing any data with a frequency, channel,
    and optionally a time axis. The datatype can be complex or float.

    """
    
    _infoFileProperties = ContinuousData._infoFileProperties + ("taper", "freq",)
    _defaultDimord = ["time", "taper", "freq", "channel"]
    
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
            return
        
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

    # Selector method
    def selectdata(self, trials=None, channels=None, toi=None, toilim=None,
                   foi=None, foilim=None, tapers=None):
        """
        Create new `SpectralData` object from selection
        
        Please refere to :func:`syncopy.selectdata` for detailed usage information. 
        
        Examples
        --------
        >>> spcBand = spc.selectdata(foilim=[10, 40])
        
        See also
        --------
        syncopy.selectdata : create new objects via deep-copy selections
        """
        return selectdata(self, trials=trials, channels=channels, toi=toi, 
                          toilim=toilim, foi=foi, foilim=foilim, tapers=tapers)
    
    # Helper function that extracts frequency-related indices
    def _get_freq(self, foi=None, foilim=None):
        """
        Coming soon... 
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
        if data is not None and dimord is None:
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
