# -*- coding: utf-8 -*-
#
# Base functions for interacting with SyNCoPy data objects
# 
# Created: 2019-02-25 11:30:46
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-05-09 13:56:20>

# Builtin/3rd party package imports
import numbers
import sys
import numpy as np

# Local imports
from syncopy.shared import data_parser, array_parser, scalar_parser
from syncopy.shared.errors import SPYTypeError, SPYValueError

__all__ = ["selectdata", "definetrial"]


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


def definetrial(obj, trialdefinition=None, pre=None, post=None, start=None,
                  trigger=None, stop=None, clip_edges=False):
    """(Re-)define trials of a Syncopy data object
    
    Data can be structured into trials based on timestamps of a start, trigger
    and end events::

                    start    trigger    stop
        |---- pre ----|--------|---------|--- post----|


    Parameters
    ----------
        obj : Syncopy data object (:class:`BaseData`-like)
        trialdefinition : :class:`EventData` object or Mx3 array 
            [start, stop, trigger_offset] sample indices for `M` trials
        pre : float
            offset time (s) before start event
        post : float 
            offset time (s) after end event
        start : int
            event code (id) to be used for start of trial
        stop : int
            event code (id) to be used for end of trial
        trigger : 
            event code (id) to be used center (t=0) of trial        
        clip_edges : bool
            trim trials to actual data-boundaries. 


    Returns
    -------
        Syncopy data object (:class:`BaseData`-like))
    
    
    Notes
    -----
    :func:`definetrial` supports the following argument combinations:

    .. code-block:: python

        # define M trials based on [start, end, offset] indices
        definetrial(obj, trialdefinition=[M x 3] array) 
        
        # define trials based on event codes stored in <:class:`EventData` object>
        definetrial(obj, trialdefinition=<:class:`EventData` object>, 
                         pre=0, post=0, start=startCode, stop=stopCode, 
                         trigger=triggerCode)

        # apply same trial definition as defined in <:class:`EventData` object>        
        definetrial(<AnalogData object>, 
                    trialdefinition=<EventData object w/sampleinfo/t0/trialinfo>)

        # define whole recording as single trial    
        definetrial(obj, trialdefinition=None)
    
    

    
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
    if obj.__class__.__name__ == "AnalogData" and evt and trialdefinition.sampleinfo is not None:
        if obj.samplerate is None or trialdefinition.samplerate is None:
            lgl = "non-`None` value - make sure `samplerate` is set before defining trials"
            act = "None"
            raise SPYValueError(legal=lgl, varname="samplerate", actual=act)
        ref = trialdefinition
        tgt = obj
        trl = np.array(ref.trialinfo)
        t0 = np.array(ref.t0).reshape((ref.t0.size,1))
        trl = np.hstack([ref.sampleinfo, t0, trl])
        trl = np.round((trl/ref.samplerate) * tgt.samplerate).astype(int)

    # AnalogData + EventData w/keywords or just EventData w/keywords
    if any([kw is not None for kw in [pre, post, start, trigger, stop]]):

        # Make sure we actually have valid data objects to work with
        if obj.__class__.__name__ == "EventData" and evt is False: 
            ref = obj
            tgt = obj
        elif obj.__class__.__name__ == "AnalogData" and evt is True:
            ref = trialdefinition
            tgt = obj
        else:
            lgl = "AnalogData with associated EventData object"
            act = "{} and {}".format(obj.__class__.__name__,
                                     trialdefinition.__class__.__name__)
            raise SPYValueError(legal=lgl, actual=act, varname="input")

        # The only case we might actually need it: ensure `clip_edges` is valid
        if not isinstance(clip_edges, bool):
            raise SPYTypeError(clip_edges, varname="clip_edges", expected="Boolean")

        # Ensure that objects have their sampling-rates set, otherwise break
        if ref.samplerate is None or tgt.samplerate is None:
            lgl = "non-`None` value - make sure `samplerate` is set before defining trials"
            act = "None"
            raise SPYValueError(legal=lgl, varname="samplerate", actual=act)

        # Get input dimensions
        szin = []
        for var in [pre, post, start, trigger, stop]:
            if isinstance(var, (np.ndarray, list)):
                szin.append(len(var))
        if np.unique(szin).size > 1:
            lgl = "all trial-related arrays to have the same length"
            act = "arrays with sizes {}".format(str(np.unique(szin)).replace("[","").replace("]",""))
            raise SPYValueError(legal=lgl, varname="trial-keywords", actual=act)
        if len(szin):
            ntrials = szin[0]
            ninc = 1
        else:
            ntrials = 1
            ninc = 0

        # If both `pre` and `start` or `post` and `stop` are `None`, abort
        if (pre is None and start is None) or (post is None and stop is None):
            lgl = "`pre` or `start` and `post` or `stop` to be not `None`"
            act = "both `pre` and `start` and/or `post` and `stop` are simultaneously `None`"
            raise SPYValueError(legal=lgl, actual=act)
        if (trigger is None) and (pre is not None or post is not None):
            lgl = "non-None `trigger` with `pre`/`post` timing information"
            act = "`trigger` = `None`"
            raise SPYValueError(legal=lgl, actual=act)

        # If provided, ensure keywords make sense, otherwise allocate defaults
        kwrds = {}
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
                    opts["var"] = np.full((ntrials,), opts["var"])
                else:
                    try:
                        array_parser(opts["var"], varname=vname, hasinf=False,
                                     hasnan=opts["hasnan"], ntype=opts["ntype"],
                                     dims=(ntrials,))
                    except Exception as exc:
                        raise exc
                kwrds[vname] = opts["var"]
            else:
                kwrds[vname] = np.full((ntrials,), opts["fillvalue"])

        # Prepare `trl` and convert event-codes + sample-numbers to lists
        trl = []
        evtid = list(ref.data[:, ref.dimord.index("eventid")])
        evtsp = list(ref.data[:, ref.dimord.index("sample")])
        nevents = len(evtid)
        searching = True
        trialno = 0
        cnt = 0
        act = ""

        # Do this line-by-line: halt on error (if event-id is not found in `ref`)
        while searching:

            # Allocate begin and end of trial
            begin = None
            end = None
            t0 = 0
            idxl = []

            # First, try to assign `start`, then `t0`
            if not np.isnan(kwrds["start"][trialno]):
                try:
                    sidx = evtid.index(kwrds["start"][trialno])
                except:
                    act = str(kwrds["start"][trialno])
                    vname = "start"
                    break
                begin = evtsp[sidx]/ref.samplerate
                evtid[sidx] = -np.pi
                idxl.append(sidx)
                
            if not np.isnan(kwrds["trigger"][trialno]):
                try:
                    idx = evtid.index(kwrds["trigger"][trialno])
                except:
                    act = str(kwrds["trigger"][trialno])
                    vname = "trigger"
                    break
                t0 = evtsp[idx]/ref.samplerate
                evtid[idx] = -np.pi
                idxl.append(idx)

            # Trial-begin is either `trigger - pre` or `start - pre`
            if begin is not None:
                begin -= kwrds["pre"][trialno]
            else:
                begin = t0 - kwrds["pre"][trialno]

            # Try to assign `stop`, if we got nothing, use `t0 + post`
            if not np.isnan(kwrds["stop"][trialno]):
                evtid[:sidx] = [np.pi]*sidx
                try:
                    idx = evtid.index(kwrds["stop"][trialno])
                except:
                    act = str(kwrds["stop"][trialno])
                    vname = "stop"
                    break
                end = evtsp[idx]/ref.samplerate + kwrds["post"][trialno]
                evtid[idx] = -np.pi
                idxl.append(idx)
            else:
                end = t0 + kwrds["post"][trialno]

            # Off-set `t0`
            t0 -= begin

            # Make sure current trial setup makes (some) sense
            if begin >= end:
                lgl = "non-overlapping trial begin-/end-samples"
                act = "trial-begin at {}, trial-end at {}".format(str(begin), str(end))
                raise SPYValueError(legal=lgl, actual=act)
            
            # Finally, write line of `trl`
            trl.append([begin, end, t0])

            # Update counters and end this mess when we're done
            trialno += ninc
            cnt += 1
            evtsp = evtsp[max(idxl, default=-1) + 1:]
            evtid = evtid[max(idxl, default=-1) + 1:]
            if trialno == ntrials or cnt == nevents:
                searching = False

        # Abort if the above loop ran into troubles
        if len(trl) < ntrials:
            import ipdb; ipdb.set_trace()
            if len(act) > 0:
                raise SPYValueError(legal="existing event-id",
                                    varname=vname, actual=act)

        # Make `trl` a NumPy array
        trl = np.round(np.array(trl) * tgt.samplerate).astype(int)

    # If appropriate, clip `trl` to AnalogData object's bounds (if wanted)
    if clip_edges and evt:
        msk = trl[:, 0] < 0
        trl[msk, 0] = 0
        dmax = tgt.data.shape[tgt.dimord.index("time")]
        msk = trl[:, 1] > dmax
        trl[msk, 1] = dmax
        if np.any(trl[:, 0] >= trl[:, 1]):
            lgl = "non-overlapping trials"
            act = "some trials are overlapping after clipping to AnalogData object range"
            raise SPYValueError(legal=lgl, actual=act)
                
    # The triplet `sampleinfo`, `t0` and `trialinfo` works identically for
    # all data genres
    if trl.shape[1] < 3:
        raise SPYValueError("array of shape (no. of trials, 3+)",
                            varname="trialdefinition",
                            actual="shape = {shp:s}".format(shp=str(trl.shape)))

    # Finally: assign `sampleinfo`, `t0` and `trialinfo` (and potentially `trialid`)
    tgt.sampleinfo = trl[:,:2]
    tgt._t0 = np.array(trl[:,2], dtype=int)
    tgt.trialinfo = trl[:,3:]

    # In the discrete case, we have some additinal work to do
    if any(["DiscreteData" in str(base) for base in tgt.__class__.__mro__]):

        # Compute trial-IDs by matching data samples with provided trial-bounds
        samples = tgt.data[:, tgt.dimord.index("sample")]
        starts = tgt.sampleinfo[:, 0]
        ends = tgt.sampleinfo[:, 1]
        sorted = starts.argsort()
        startids = np.searchsorted(starts, samples, side="right", sorter=sorted)
        endids = np.searchsorted(ends, samples, side="left", sorter=sorted)
        mask = startids == endids
        startids -= 1
        startids[mask] = -1
        tgt.trialid = startids

    # Write log entry
    if ref == tgt:
        ref.log = "updated trial-definition with [" \
                  + " x ".join([str(numel) for numel in trl.shape]) \
                  + "] element array"
    else:
        ref_log = ref._log.replace("\n\n", "\n\t")
        tgt.log = "trial-definition extracted from EventData object: "
        tgt._log += ref_log
        tgt.cfg = {"method" : sys._getframe().f_code.co_name,
                   "EventData object": ref.cfg}
        ref.log = "updated trial-defnition of {} object".format(tgt.__class__.__name__)
    
    return
