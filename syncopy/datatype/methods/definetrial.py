# -*- coding: utf-8 -*-
#
# Set/update trial settings of Syncopy data objects
#

# Builtin/3rd party package imports
import sys
import numpy as np

# Local imports
from syncopy.shared.parsers import data_parser, array_parser, scalar_parser
from syncopy.shared.errors import SPYTypeError, SPYValueError

__all__ = ["definetrial"]


def definetrial(obj, trialdefinition=None, pre=None, post=None, start=None,
                trigger=None, stop=None, clip_edges=False):
    """(Re-)define trials of a Syncopy data object

    Data can be structured into trials based on timestamps of a start, trigger
    and end events::

                    start    trigger    stop
        |---- pre ----|--------|---------|--- post----|

    **Note**: To define a trial encompassing the whole dataset simply invoke this
    routine with no arguments, i.e., ``definetrial(obj)`` or equivalently
    ``obj.definetrial()``

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

    >>> # define M trials based on [start, end, offset] indices
    >>> definetrial(obj, trialdefinition=[M x 3] array)

    >>> # define trials based on event codes stored in <:class:`EventData` object>
    >>> definetrial(obj, trialdefinition=<EventData object>,
                    pre=0, post=0, start=startCode, stop=stopCode,
                    trigger=triggerCode)

    >>> # apply same trial definition as defined in <:class:`EventData` object>
    >>> definetrial(<AnalogData object>,
                    trialdefinition=<EventData object w/sampleinfo/t0/trialinfo>)

    >>> # define whole recording as single trial
    >>> definetrial(obj, trialdefinition=None)

    """

    # Start by vetting input object
    try:
        data_parser(obj, varname="obj")
    except Exception as exc:
        raise exc
    if obj.data is None:
        lgl = "non-empty Syncopy data object"
        act = "empty Syncopy data object"
        raise SPYValueError(legal=lgl, varname="obj", actual=act)

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
            array_parser(trialdefinition, varname="trialdefinition", dims=2)

            if any(["ContinuousData" in str(base) for base in obj.__class__.__mro__]):
                scount = obj.data.shape[obj.dimord.index("time")]
            else:
                scount = np.inf
            try:
                array_parser(trialdefinition[:, :2], varname="sampleinfo", dims=(None, 2), hasnan=False,
                         hasinf=False, ntype="int_like", lims=[0, scount])
            except Exception as exc:
                raise exc

            trl = np.array(trialdefinition, dtype="float")
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
        t0 = np.array(ref._t0).reshape((ref._t0.size,1))
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
                if np.issubdtype(type(opts["var"]), np.number):
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
    tgt._trialdefinition = trl

    # In the discrete case, we have some additinal work to do
    if any(["DiscreteData" in str(base) for base in tgt.__class__.__mro__]):

        # Compute trial-IDs by matching data samples with provided trial-bounds
        samples = tgt.data[:, tgt.dimord.index("sample")]
        idx = np.searchsorted(samples, tgt.sampleinfo.ravel())
        idx = idx.reshape(tgt.sampleinfo.shape)

        tgt._trialslice = [slice(st,end) for st,end in idx]
        tgt.trialid = np.full((samples.shape), -1, dtype=int)
        for itrl, itrl_slice in enumerate(tgt._trialslice):
            tgt.trialid[itrl_slice] = itrl

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
