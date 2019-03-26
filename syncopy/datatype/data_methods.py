# -*- coding: utf-8 -*-
#
# Base functions for interacting with SyNCoPy data objects
# 
# Created: 2019-02-25 11:30:46
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-03-26 16:44:15>

# Builtin/3rd party package imports
import numbers
import sys
import numpy as np

# Local imports
from syncopy.utils import (SPYTypeError, SPYValueError, data_parser,
                           array_parser, scalar_parser)

__all__ = ["selectdata", "redefinetrial"]


def selectdata(obj, trials=None, deepcopy=False, exact_match=False, **kwargs):
    """
    Docstring coming soon(ish)

    (tuple) -> value-range selection (e.g., freq=(5,10) frequency range b/w 5-10Hz)
    slice -> index-range selection (e.g. freq=slice(5,10), frequencies no. 5 - 9)
    [list-like] -> multi-index-selection (e.g. freq=[5,7,10], frequencies no. 5, 7, 10)
    float -> single-value selection (e.g. freq=5.0, frequency of 5Hz)
    int -> single-index selection (e.g., freq=5, 4th frequency in spectrum)
    """

    # Depending on input object, pass things right on to actual working routines
    if any(["ContinuousData" in str(base) for base in obj.__class__.__bases__]):
        return _selectdata_continuous(obj, trials, deepcopy, exact_match, **kwargs)
    elif any(["DiscreteData" in str(base) for base in obj.__class__.__bases__]):
        raise NotImplementedError("Coming soon")
    else:
        raise SPYTypeError(obj, varname="obj", expected="SyNCoPy data object")
    

def _selectdata_continuous(obj, trials, deepcopy, exact_match, **kwargs):

    # Make sure provided object is inherited from `ContinuousData`
    if not any(["ContinuousData" in str(base) for base in obj.__class__.__mro__]):
        raise SPYTypeError(obj, varname="obj", expected="SpkeWave ContinuousData object")
        
    # Convert provided selectors to array indices
    trials, selectors = _makeidx(obj, trials, deepcopy, exact_match, **kwargs)

    # Make sure our Boolean switches are actuall Boolean
    if not isinstance(deepcopy, bool):
        raise SPYTypeError(deepcopy, varname="deepcopy", expected="bool")
    if not isinstance(exact_match, bool):
        raise SPYTypeError(exact_match, varname="exact_match", expected="bool")

    # If time-based selection is requested, make some necessary preparations
    if "time" in selectors.keys():
        time_sel = selectors.pop("time")
        time_ref = np.array(obj.time[trials[0]])
        time_slice = [None, None]
        if isinstance(time_sel, tuple):
            if len(time_sel) != 2:
                raise SPYValueError(legal="two-element tuple",
                                    actual="tuple of length {}".format(str(len(time_sel))),
                                    varname="time")
            for tk, ts in enumerate(time_sel):
                if ts is not None:
                    if not exact_match:
                        time_slice[tk] = time_ref[np.abs(time_ref - ts).argmin()]
                    else:
                        try:
                            time_slice[tk] = list(time_ref).index(ts)
                        except:
                            raise SPYValueError(legal="exact time-point", actual=ts)
            time_slice = slice(*time_slice)
        elif isinstance(time_sel, slice):
            if not len(range(*time_sel.indices(time_ref.size))):
                lgl = "non-empty time-selection"
                act = "empty selector"
                raise SPYValueError(legal=lgl, varname=lbl, actual=act)
            time_slice = slice(time_sel.start, time_sel.stop, time_sel.step)
        elif isinstance(time_sel, (list, np.ndarray)):
            if not set(time_sel).issubset(range(time_ref.size))\
               or np.unique(np.diff(time_sel)).size != 1:
                vname = "contiguous list of time-points"
                raise SPYValueError(legal=lgl, varname=vname)
            time_slice = slice(time_sel[0], time_sel[-1] + 1)
        else:
            raise SPYTypeError(time_sel, varname="time-selection",
                               expected="tuple, slice or list-like")
    else:
        time_slice = slice(0, None)

        # SHALLOWCOPY
        sampleinfo = np.empty((trials.size, 2))
        for sk, trl in enumerate(trials):
            sinfo = range(*obj.sampleinfo[trl, :])[time_slice]
            sampleinfo[sk, :] = [sinfo.start, sinfo.stop - 1]
        
            
    # Build array-multi-index and shape of target array based on dimensional selectors
    idx = [slice(None)] * len(obj.dimord)
    target_shape = list(obj.data.shape)
    for lbl, selector in selectors.items():
        id = obj.dimord.index(lbl)
        idx[id] = selector
        if isinstance(selector, slice):
            target_shape[id] = len(range(*selector.indices(obj.data.shape[id])))
        elif isinstance(selector, int):
            target_shape[id] = 1
        else:
            if not deepcopy:
                deepcopy = True
            target_shape[id] = len(selector)
    tid = obj.dimord.index("time")
    idx[tid] = time_slice
    
    # Allocate shallow copy for target
    target = obj.copy()

    # First, we handle deep copies of `obj`
    if deepcopy:

        # Re-number trials: offset correction + close gaps b/w trials
        sampleinfo = obj.sampleinfo[trials, :] - obj.sampleinfo[trials[0], 0]
        stop = 0
        for sk in range(sampleinfo.shape[0]):
            sinfo = range(*sampleinfo[sk, :])[time_slice]
            nom_len = sinfo.stop - sinfo.start
            start = min(sinfo.start, stop)
            real_len = min(nom_len, sinfo.stop - stop)
            sampleinfo[sk, :] = [start, start + nom_len]
            stop = start + real_len + 1
            
        # Based on requested trials, set shape of target array (accounting
        # for overlapping trials)
        target_shape[tid] = sampleinfo[-1][1]

        # Allocate target memorymap
        target._filename = obj._gen_filename()
        target_dat = open_memmap(target._filename, mode="w+",
                                 dtype=obj.data.dtype, shape=target_shape)
        del target_dat

        # The crucial part here: `idx` is a "local" by-trial index of the
        # form `[:,:,2:10]` whereas `target_idx` has to keep track of the
        # global progression in `target_data`
        for sk, trl in enumerate(trials):
            source_trl = self._copy_trial(trialno,
                                            obj._filename,
                                            obj.trl,
                                            obj.hdr,
                                            obj.dimord,
                                            obj.segmentlabel)
            target_idx[tid] = slice(*sampleinfo[sk, :])
            target_dat = open_memmap(target._filename, mode="r+")[target_idx]
            target_dat[...] = source_trl[idx]
            del target_dat

        # FIXME: Clarify how we want to do this...
        target._dimlabels["sample"] = sampleinfo

        # Re-number samples if necessary
        

        # By-sample copy
        if trials is None:
            mem_size = np.prod(target_shape)*self.data.dtype*1024**(-2)
            if mem_size >= 100:
                spw_warning("Memory footprint of by-sample selection larger than 100MB",
                            caller="SyNCoPy core:select")
            target_dat[...] = self.data[idx]
            del target_dat
            self.clear()

        # By-trial copy
        else:
            del target_dat
            sid = self.dimord.index(self.segmentlabel)
            target_shape[sid] = sum([shp[sid] for shp in np.array(self.shapes)[trials]])
            target_idx = [slice(None)] * len(self.dimord)
            target_sid = 0
            for trialno in trials:
                source_trl = self._copy_trial(trialno,
                                                self._filename,
                                                self.trl,
                                                self.hdr,
                                                self.dimord,
                                                self.segmentlabel)
                trl_len = source_trl.shape[sid]
                target_idx[sid] = slice(target_sid, target_sid + trl_len)
                target_dat = open_memmap(target._filename, mode="r+")[target_idx]
                target_dat[...] = source_trl[idx]
                del target_dat
                target_sid += trl_len

    # Shallow copy: simply create a view of the source memmap
    # Cover the case: channel=3, all trials!
    else:
        target._data = open_memmap(self._filename, mode="r")[idx]

    return target


def _selectdata_discrete():
    pass


def _makeidx(obj, trials, deepcopy, exact_match, **kwargs):
    """
    Local input parser
    """
    
    # Make sure `obj` is a valid `BaseData`-like object
    try:
        spw_basedata_parser(obj, varname="obj", writable=None, empty=False)
    except Exception as exc:
        raise exc

    # Make sure the input dimensions make sense
    if not set(kwargs.keys()).issubset(self.dimord):
        raise SPYValueError(legal=self.dimord, actual=list(kwargs.keys()))

    # Process `trials`
    if trials is not None:
        if isinstance(trials, tuple):
            start = trials[0]
            if trials[1] is None:
                stop = self.trl.shape[0]
            else:
                stop = trials[1]
            trials = np.arange(start, stop)
        if not set(trials).issubset(range(self.trl.shape[0])):
            lgl = "trial selection between 0 and {}".format(str(self.trl.shape[0]))
            raise SPYValueError(legal=lgl, varname="trials")
        if isinstance(trials, int):
            trials = np.array([trials])
    else:
        trials = np.arange(self.trl.shape[0])

    # Time-based selectors work differently for continuous/discrete data,
    # handle those separately from other dimensional labels
    selectors = {}
    if "time" in kwargs.keys():
        selectors["time"] = kwargs.pop("time")

    # Calculate indices for each provided dimensional selector
    for lbl, selection in kwargs.items():
        ref = np.array(self.dimord[lbl])
        lgl = "component of `obj.{}`".format(lbl)

        # Value-range selection
        if isinstance(selection, tuple):
            if len(selection) != 2:
                raise SPYValueError(legal="two-element tuple",
                                    actual="tuple of length {}".format(str(len(selection))),
                                    varname=lbl)
            bounds = [None, None]
            for sk, sel in enumerate(selection):
                if isinstance(sel, str):
                    try:
                        bounds[sk] = list(ref).index(sel)
                    except:
                        raise SPYValueError(legal=lgl, actual=sel)
                elif isinstance(sel, numbers.Number):
                    if not exact_match:
                        bounds[sk] = ref[np.abs(ref - sel).argmin()]
                    else:
                        try:
                            bounds[sk] = list(ref).index(sel)
                        except:
                            raise SPYValueError(legal=lgl, actual=sel)
                elif sel is None:
                    if sk == 0:
                        bounds[sk] = ref[0]
                    if sk == 1:
                        bounds[sk] = ref[-1]
                else:
                    raise SPYTypeError(sel, varname=lbl, expected="string, number or None")
            bounds[1] += 1
            selectors[lbl] = slice(*bounds)

        # Index-range selection
        elif isinstance(selection, slice):
            if not len(range(*selection.indices(ref.size))):
                lgl = "non-empty selection"
                act = "empty selector"
                raise SPYValueError(legal=lgl, varname=lbl, actual=act)
            selectors[lbl] = slice(selection.start, selection.stop, selection.step)
            
        # Multi-index selection: try to convert contiguous lists to slices
        elif isinstance(selection, (list, np.ndarray)):
            if not set(selection).issubset(range(ref.size)):
                vname = "list-selector for `obj.{}`".format(lbl)
                raise SPYValueError(legal=lgl, varname=vname)
            if np.unique(np.diff(selection)).size == 1:
                selectors[lbl] = slice(selection[0], selection[-1] + 1)
            else:
                selectors[lbl] = list(selection)

        # Single-value selection
        elif isinstance(selection, float):
            if not exact_match:
                selectors[lbl] = ref[np.abs(ref - selection).argmin()]
            else:
                try:
                    selectors[lbl] = list(ref).index(selection)
                except:
                    raise SPYValueError(legal=lgl, actual=selection)

        # Single-index selection
        elif isinstance(selection, int):
            if selection not in range(ref.size):
                raise SPYValueError(legal=lgl, actual=selection)
            selectors[lbl] = selection

        # You had your chance...
        else:
            raise SPYTypeError(selection, varname=lbl,
                               expected="tuple, list-like, slice, float or int")
        
    return selectors, trials


def redefinetrial(obj, trialdefinition=None, pre=None, post=None, start=None,
                  trigger=None, stop=None):
    """
    Docstring coming soon(ish)

    Supports the following assignments:

    obj + None
    obj + trialdefinition array
    AnalogData + EventData w/sampleinfo/t0/trialinfo
    AnalogData + EventData + pre/post/start/stop/trigger keywords
    """

    # Start by vetting input object
    try:
        data_parser(obj, varname="obj", writable=None, empty=False)
    except Exception as exc:
        raise exc

    # Check array/object holding trial specifications
    if trialdefinition is not None:
        if trialdefinition.__class__.__name__ == "EventData":
            try:
                data_parser(trialdefinition, varname="trialdefinition",
                            writable=None, empty=False)
            except Exception as exc:
                raise exc
            evt = True
        else:
            try:
                array_parser(trialdefinition, varname="trialdefinition", dims=2)
            except Exception as exc:
                raise exc
            trl = trialdefinition
            ref = obj
            tgt = obj
            evt = False
    else:
        # Construct object-class-specific `trl` arrays treating data-set as single trial
        if any(["ContinuousData" in str(base) for base in obj.__class__.__mro__]):
            trl = np.array([[0, obj.data.shape[obj.dimord.index("time")], 0]])
        else:
            sidx = obj.dimord.index("sample")
            trl = np.array([[np.nanmin(obj.data[:,sidx]),
                             np.nanmax(obj.data[:,sidx]), 0]])
        ref = obj
        tgt = obj
        evt = False

    # AnalogData + EventData w/sampleinfo
    if obj.__class__.__name__ == "AnalogData" and evt and evt.sampleinfo is not None:
        trl = ...
        ref = trialdefinition
        tgt = obj

    # AnalogData + EventData w/keywords or just EventData w/keywords
    elif any([kw is not None for kw in [pre, post, start, trigger, stop]]):
        if obj.__class__.__name__ == "EventData" and evt is False: 
            ref = obj
            tgt = obj
        elif obj.__class__.__name__ == "AnalogData" and evt is True:
            ref = trialdefinition
            tgt = obj
        else:
            print('warning')

        # Get input dimensions
        nevent = ref.data.shape[0]

        # If both `pre` and `start` or `post` and `stop` are `None`, abort
        if (pre is None and start is None) or (post is None and stop is None):
            lgl = "`pre` or `start` and `post` or `stop` to be not `None`"
            act = "both `pre` and `start` and/or `post` and `stop` are simultaneously `None`"
            raise SPYValueError(legal=lgl, actual=act)

        # If provided, ensure keywords make sense, otherwise allocate defaults
        kwrds = {}
        vnames = ["pre", "post"]
        vdict = {"pre": {"var": pre, "hasnan": False, "ntype": None, "fillvalue": 0},
                 "post": {"var": post, "hasnan": False, "ntype": None, "fillvalue": 0},
                 "start": {"var": start, "hasnan": None, "ntype": "int_like", "fillvalue": np.nan}, 
                 "trigger": {"var": trigger, "hasnan": None, "ntype": "int_like", "fillvalue": np.nan},
                 "stop": {"var": stop, "hasnan": None, "ntype": "int_like", "fillvalue": np.nan}}
        for vname, opts in vdict.items():
            if opts["var"] is not None:
                if isinstance(opts["var"], numbers.Number):
                    try:
                        scalar_parser(opts["var"], varname=vname, ntype=opts["ntype"],
                                      lims=[-np.inf, np.inf])
                    except Exception as exc:
                        raise exc
                    var = np.full((nevent,), opts["var"])
                else:
                    try:
                        array_parser(opts["var"], varname=vname, hasinf=False,
                                     hasnan=opts["hasnan"], ntype=opts["ntype"],
                                     dims=(nevent,))
                    except Exception as exc:
                        raise exc
                kwrds[vnames[vk]] = opts["var"]
            else:
                kwrds[vnames[vk]] = np.full((nevent,), opts["fillvalue"])

        # Prepare `trl` array and convert event-codes + sample-numbers to lists
        trl = np.empty((nevent, 3))
        evtid = list(ref.data[ref.dimord.index("eventid")])
        evtsp = list(ref.data[ref.dimord.index("sample")])

        # Do this line-by-line: halt on error (if event-id is not found in `ref`)
        for trialno in range(nevent):

            # Allocate begin and end of trial
            begin = None
            end = None
            t0 = 0

            # First, try to assign `start`, then `t0`
            if not np.isnan(kwrds["start"][trialno]):
                try:
                    idx = evtid.index(kwrds["start"][trialno])
                except:
                    act = str(kwrds["start"][trialno])
                    lgl = "existing event-id"
                    raise SPYValueError(legal=lgl, varname="start", actual=act)
                begin = evtsp.pop(idx)
                evtid.pop(idx)
                
            if not np.isnan(kwrds["trigger"][trialno]):
                try:
                    idx = evtid.index(kwrds["trigger"][trialno])
                except:
                    act = str(kwrds["trigger"][trialno])
                    lgl = "existing event-id"
                    raise SPYValueError(legal=lgl, varname="trigger", actual=act)
                t0 = evtsp.pop(idx)
                evtid.pop(idx)

            # Trial-begin is either `trigger-pre` or `start-pre`
            if begin is not None:
                t0 -= begin
                begin -= pre
            else:
                begin = t0 - pre

            # Try to assign `stop`, if we got nothing, use `t0+post`
            if not np.isnan(kwrds["stop"][trialno]):
                try:
                    idx = evtid.index(kwrds["stop"][trialno])
                except:
                    act = str(kwrds["stop"][trialno])
                    lgl = "existing event-id"
                    raise SPYValueError(legal=lgl, varname="stop", actual=act)
                end = evtsp.pop(idx) + post
                evtid.pop(idx)
            else:
                end = t0 + post

            # Finally, write line of `trl`
            trl[trialno, :] = [begin, end, t0]
                
    # The triplet `sampleinfo`, `t0` and `trialinfo` works identically for
    # all data genres
    if trl.shape[1] < 3:
        raise SPYValueError("array of shape (no. of trials, 3+)",
                            varname="trialdefinition",
                            actual="shape = {shp:s}".format(shp=str(trl.shape)))
    for obt in [ref, tgt]:
        tgt.sampleinfo = trl[:,:2]
        tgt._t0 = np.array(trl[:,2], dtype=int)
        tgt.trialinfo = trl[:,3:]

        # In the discrete case, we have some additinal work to do
        if any(["DiscreteData" in str(base) for base in obt.__class__.__mro__]):

            # Compute trial-IDs by matching data samples with provided trial-bounds
            samples = obj.data[:, obj.dimord.index("sample")]
            starts = obj.sampleinfo[:, 0]
            sorted = starts.argsort()
            obj.trialid = np.searchsorted(starts, samples, side="right", sorter=sorted) - 1

    # Write log entry
    logmsg = "updated trial-definition with [" \
             + " x ".join([str(numel) for numel in trl.shape]) \
             + "] element array"
    if ref == tgt:
        ref.log = logmsg
    else:
        ref_log = ref._log.replace("\n\n", "\n\t")
        tgt.log = "trial-definition extracted from EventData object: "
        tgt._log += ref_log
        ref.log = logmsg
    
    return
