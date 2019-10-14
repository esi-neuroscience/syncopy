# -*- coding: utf-8 -*-
# 
# Base functions for interacting with SyNCoPy data objects
# 
# Created: 2019-02-25 11:30:46
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-10-11 13:47:16>

# Builtin/3rd party package imports
import numbers
import sys
import numpy as np

# Local imports
from syncopy.shared.parsers import data_parser, array_parser, scalar_parser
from syncopy.shared.errors import SPYTypeError, SPYValueError

__all__ = ["selectdata", "definetrial", "padding"]


def selectdata(data, trials=None, channels=None, toi=None, toilim=None, foi=None,
               foilim=None, tapers=None, units=None, eventids=None):
    """
    Select a subset of data from a Syncopy object
    
    **##################### Hello Alpha users! #####################** 
    
    This function is not implemented yet - the version of 
    Syncopy you're using only supports in-place selection via a `select` dictionary. 
    The keys supported by the `select` dictionary are identical to the keyword
    arguments discussed below, e.g., the following code snippet works
    
    >>> select = {"toilim" : [-0.25, 0]}
    >>> cfg = spy.get_defaults(spy.freqanalysis)
    >>> cfg.select = select
    >>> spy.freqanalysis(cfg, data)

    **##############################################################** 

    
    **Usage summary**
    
    List of Syncopy data objects and respective valid data selectors:
    
    :class:`~syncopy.AnalogData` : trials, channels, toi/toilim
        Examples
        
        >>> spy.selectdata(data, trials=[0, 3, 5], channels=["channel01", "channel02"])
        >>> cfg = spy.StructDict() 
        >>> cfg.trials = [5, 3, 0]; cfg.toilim = [0.25, 0.5]
        >>> spy.selectdata(cfg, data)
        
    :class:`~syncopy.SpectralData` : trials, channels, toi/toilim, foi/foilim, tapers
        Examples
        
        >>> spy.selectdata(data, trials=[0, 3, 5], channels=["channel01", "channel02"])
        >>> cfg = spy.StructDict()
        >>> cfg.foi = [30, 40, 50]; cfg.tapers = slice(2, 4)
        >>> spy.selectdata(cfg, data)
        
    :class:`~syncopy.EventData` : trials, toi/toilim, eventids
        Examples
        
        >>> spy.selectdata(data, toilim=[-1, 2.5], eventids=[0, 1])
        >>> cfg = spy.StructDict()
        >>> cfg.trials = [0, 0, 1, 0]; cfg.eventids = slice(2, None)
        >>> spy.selectdata(cfg, data)
        
    :class:`~syncopy.SpikeData` : trials, toi/toilim, units
        Examples
        
        >>> spy.selectdata(data, toilim=[-1, 2.5], units=range(0, 10))
        >>> cfg = spy.StructDict()
        >>> cfg.toi = [1.25, 3.2]; cfg.trials = [0, 1, 2, 3]
        >>> spy.selectdata(cfg, data)
    
    **Note** Any property that is not specifically accessed via one of the provided
    selectors is taken as is, e.g., ``spy.selectdata(data, trials=[1, 2])``
    selects the entire contents of trials no. 2 and 3, while 
    ``spy.selectdata(data, channels=range(0, 50))`` selects the first 50 channels
    of `data` across all defined trials. Consequently, if no keywords are specified,
    the entire contents of `data` is selected. 
    
    Full documentation below. 
    
    Parameters
    ----------
    data : Syncopy data object
        A non-empty Syncopy data object. **Note** the type of `data` determines
        which keywords can be used.  Some keywords are only valid for certain 
        types of Syncopy objects, e.g., "freqs" is not a valid selector for an 
        :class:`~syncopy.AnalogData` object. 
    trials : list (integers) or None
        List of integers representing trial numbers to be selected; can include 
        repetitions and need not be sorted (e.g., ``trials = [0, 1, 0, 0, 2]`` 
        is valid) but must be finite and not NaN. If `trials` is `None`, all trials 
        are selected. 
    channels : list (integers or strings), slice, range or None
        Channel-selection; can be a list of channel names (``['channel3', 'channel1']``), 
        a list of channel indices (``[3, 5]``), a slice (``slice(3, 10)``) or 
        range (``range(3, 10)``). Note that following Python conventions, channels 
        are counted starting at zero, and range and slice selections are half-open 
        intervals of the form `[low, high)`, i.e., low is included , high is 
        excluded. Thus, ``channels = [0, 1, 2]`` or ``channels = slice(0, 3)`` 
        selects the first up to (and including) the third channel. Selections can 
        be unsorted and may include repetitions but must match exactly, be finite 
        and not NaN. If `channels` is `None`, all channels are selected. 
    toi : list (floats) or None
        Time-points to be selected (in seconds) in each trial. Timing is expected 
        to be on a by-trial basis (e.g., relative to trigger onsets). Selections 
        can be approximate, unsorted and may include repetitions but must be 
        finite and not NaN. Fuzzy matching is performed for approximate selections 
        (i.e., selected time-points are close but not identical to timing information 
        found in `data`) using a nearest-neighbor search for elements of `toi`. 
        If `toi` is `None`, the entire time-span in each trial is selected. 
    toilim : list (floats [tmin, tmax]) or None
        Time-window ``[tmin, tmax]`` (in seconds) to be extracted from each trial. 
        Window specifications must be sorted (e.g., ``[2.2, 1.1]`` is invalid) 
        and not NaN but may be unbounded (e.g., ``[1.1, np.inf]`` is valid). Edges 
        `tmin` and `tmax` are included in the selection. 
        If `toilim` is `None`, the entire time-span in each trial is selected. 
    foi : list (floats) or None
        Frequencies to be selected (in Hz). Selections can be approximate, unsorted 
        and may include repetitions but must be finite and not NaN. Fuzzy matching 
        is performed for approximate selections (i.e., selected frequencies are 
        close but not identical to frequencies found in `data`) using a nearest-
        neighbor search for elements of `foi` in `data.freq`. If `foi` is `None`, 
        all frequencies are selected. 
    foilim : list (floats [fmin, fmax]) or None
        Frequency-window ``[fmin, fmax]`` (in Hz) to be extracted. Window 
        specifications must be sorted (e.g., ``[90, 70]`` is invalid) and not NaN 
        but may be unbounded (e.g., ``[-np.inf, 60.5]`` is valid). Edges `fmin` 
        and `fmax` are included in the selection. If `foilim` is `None`, all 
        frequencies are selected. 
    tapers : list (integers or strings), slice, range or None
        Taper-selection; can be a list of taper names (``['dpss-win-1', 'dpss-win-3']``), 
        a list of taper indices (``[3, 5]``), a slice (``slice(3, 10)``) or range 
        (``range(3, 10)``). Note that following Python conventions, tapers are 
        counted starting at zero, and range and slice selections are half-open 
        intervals of the form `[low, high)`, i.e., low is included , high is 
        excluded. Thus, ``tapers = [0, 1, 2]`` or ``tapers = slice(0, 3)`` selects 
        the first up to (and including) the third taper. Selections can be unsorted 
        and may include repetitions but must match exactly, be finite and not NaN. 
        If `tapers` is `None`, all tapers are selected. 
    units : list (integers or strings), slice, range or None
        Unit-selection; can be a list of unit names (``['unit10', 'unit3']``), a 
        list of unit indices (``[3, 5]``), a slice (``slice(3, 10)``) or range 
        (``range(3, 10)``). Note that following Python conventions, units are 
        counted starting at zero, and range and slice selections are half-open 
        intervals of the form `[low, high)`, i.e., low is included , high is 
        excluded. Thus, ``units = [0, 1, 2]`` or ``units = slice(0, 3)`` selects 
        the first up to (and including) the third unit. Selections can be unsorted 
        and may include repetitions but must match exactly, be finite and not NaN.
        If `units` is `None`, all units are selected. 
    eventids : list (integers), slice, range or None
        Event-ID-selection; can be a list of event-id codes (``[2, 0, 1]``), slice 
        (``slice(0, 2)``) or range (``range(0, 2)``). Note that following Python 
        conventions, range and slice selections are half-open intervals of the 
        form `[low, high)`, i.e., low is included , high is excluded. Selections 
        can be unsorted and may include repetitions but must match exactly, be
        finite and not NaN. If `eventids` is `None`, all events are selected. 
        
    Returns
    -------
    dataselection : Syncopy data object
        Syncopy data object of the same type as `data` but containing only the 
        subset specified by provided selectors. 
        
    Notes
    -----
    This routine represents a convenience function for creating new Syncopy objects
    based on existing data entities. However, in many situations, the creation 
    of a new object (and thus the allocation of additional disk-space) might not 
    be necessary: all Syncopy compute kernels, such as :func:`~syncopy.freqanalysis`,
    support **in-place data selection**. 
    
    Consider the following example: assume `data` is an :class:`~syncopy.AnalogData` 
    object representing 220 trials of LFP recordings containing baseline (between 
    second -0.25 and 0) and stimulus-on data (on the interval [0.25, 0.5]). 
    To compute the baseline spectrum, data-selection does **not**
    have to be performed before calling :func:`~syncopy.freqanalysis` but instead
    can be done in-place:
    
    >>> import syncopy as spy
    >>> cfg = spy.get_defaults(spy.freqanalysis)
    >>> cfg.method = 'mtmfft'
    >>> cfg.taper = 'dpss'
    >>> cfg.output = 'pow'
    >>> cfg.tapsmofrq = 10
    >>> # define baseline/stimulus-on ranges
    >>> baseSelect = {"toilim": [-0.25, 0]}
    >>> stimSelect = {"toilim": [0.25, 0.5]}
    >>> # in-place selection of baseline interval performed by `freqanalysis`
    >>> cfg.select = baseSelect
    >>> baselineSpectrum = spy.freqanalysis(cfg, data)
    >>> # in-place selection of stimulus-on time-frame performed by `freqanalysis`
    >>> cfg.select = stimSelect
    >>> stimonSpectrum = spy.freqanalysis(cfg, data)
    
    Especially for large data-sets, in-place data selection performed by Syncopy's
    compute kernels does not only save disk-space but can significantly increase 
    performance.  
    
    Examples
    --------
    Use :func:`~syncopy.tests.misc.generate_artificial_data` to create a synthetic 
    :class:`syncopy.AnalogData` object. 
    
    >>> from syncopy.tests.misc import generate_artificial_data
    >>> adata = generate_artificial_data(nTrials=10, nChannels=32) 
    
    Assume a hypothetical trial onset at second 2.0 with the first second of each
    trial representing baseline recordings. To extract only the stimulus-on period
    from `adata`, one could use
    
    >>> stimon = spy.selectdata(adata, toilim=[2.0, np.inf])
    
    Note that this is equivalent to
    
    >>> stimon = adata.selectdata(toilim=[2.0, np.inf])
    
    See also
    --------
    :meth:`syncopy.AnalogData.selectdata` : corresponding class method
    :meth:`syncopy.SpectralData.selectdata` : corresponding class method
    :meth:`syncopy.EventData.selectdata` : corresponding class method
    :meth:`syncopy.SpikeData.selectdata` : corresponding class method
    """
    
    raise NotImplementedError("Coming soon!")


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
            try:
                array_parser(trialdefinition, varname="trialdefinition", dims=2)
            except Exception as exc:
                raise exc
            
            if any(["ContinuousData" in str(base) for base in obj.__class__.__mro__]):
                scount = obj.data.shape[obj.dimord.index("time")]
            else:
                scount = np.inf
            try:
                array_parser(trialdefinition[:, :2], varname="sampleinfo", dims=(None, 2), hasnan=False, 
                         hasinf=False, ntype="int_like", lims=[0, scount])
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


def padding(data, padtype, pad="absolute", padlength=None, prepadlength=None,
            postpadlength=None, unit="samples", create_new=True):
    """
    Perform data padding on Syncopy object or :class:`numpy.ndarray`
    
    **Usage summary**
    
    Depending on the value of `pad` the following padding length specifications
    are supported:
    
    +------------+----------------------+---------------+----------------------+----------------------+
    | `pad`      | `data`               | `padlength`   | `prepadlength`       | `postpadlength`      |
    +============+======================+===============+======================+======================+
    | 'absolute' | Syncopy object/array | number        | `None`/`bool`        | `None`/`bool`        |
    +------------+----------------------+---------------+----------------------+----------------------+
    | 'relative' | Syncopy object/array | number/`None` | number/`None`/`bool` | number/`None`/`bool` |
    +------------+----------------------+---------------+----------------------+----------------------+
    | 'maxlen'   | Syncopy object       | `None`/`bool` | `None`/`bool`        | `None`/`bool`        |
    +------------+----------------------+---------------+----------------------+----------------------+
    | 'nextpow2' | Syncopy object/array | `None`/`bool` | `None`/`bool`        | `None`/`bool`        |
    +------------+----------------------+---------------+----------------------+----------------------+
    
    * `data` can be either a Syncopy object containing multiple trials or a
      :class:`numpy.ndarray` representing a single trial
    * (pre/post)padlength: can be either `None`, `True`/`False` or a positive
      number: if `True` indicates where to pad, e.g., by using ``pad =
      'maxlen'`` and  ``prepadlength = True``, `data` is padded at the beginning
      of each trial. **Only** if `pad` is 'relative' are scalar values supported
      for `prepadlength` and `postpadlength`
    * ``pad = 'absolute'``: pad to desired absolute length, e.g., by using ``pad
      = 5`` and ``unit = 'time'`` all trials are (if necessary) padded to 5s
      length. Here, `padlength` **has** to be provided, `prepadlength` and
      `postpadlength` can be `None` or `True`/`False`
    * ``pad = 'relative'``: pad by provided `padlength`, e.g., by using
      ``padlength = 20`` and ``unit = 'samples'``, 20 samples are padded
      symmetrically around (before and after) each trial. Use ``padlength = 20``
      and ``prepadlength = True`` **or** directly ``prepadlength = 20`` to pad
      before each trial. Here, at least one of `padlength`, `prepadlength` or
      `postpadlength` **has** to be provided. 
    * ``pad = 'maxlen'``: (only valid for **Syncopy objects**) pad up to maximal
      trial length found in `data`. All lengths have to be either Boolean
      indicating padding location or `None` (if all are `None`, symmetric
      padding is performed)
    * ``pad = 'nextpow2'``: pad each trial up to closest power of two. All
      lengths have to be either Boolean indicating padding location or `None`
      (if all are `None`, symmetric padding is performed)
    
    Full documentation below. 
    
    Parameters 
    ----------
    data : Syncopy object or :class:`numpy.ndarray`
        Non-empty Syncopy data object or array representing numeric data to be
        padded. **NOTE**: if `data` is a :class:`numpy.ndarray`, it is assumed
        that it represents recordings from only a single trial, where its first
        axis corresponds to time. In other words, `data` is a
        'time'-by-'channel' array such that its rows reflect samples and its
        columns represent channels. If `data` is a Syncopy object, trial
        information and dimensional order are fetched from `data.trials` and
        `data.dimord`, respectively. 
    padtype : str
        Padding value(s) to be used. Available options are:

        * 'zero' : pad using zeros
        * 'nan' : pad using `np.nan`'s
        * 'mean' : pad with by-channel mean value across each trial
        * 'localmean' : pad with by-channel mean value using only `padlength` or
          `prepadlength`/`postpadlength` number of boundary-entries for averaging
        * 'edge' : pad with trial-boundary values
        * 'mirror' : pad with reflections of trial-boundary values
        
    pad : str
        Padding mode to be used. Available options are:
        
        * 'absolute' : pad each trial to achieve a desired absolute length such
          that all trials have identical length post padding. If `pad` is `absolute`
          a `padlength` **has** to be provided, `prepadlength` and `postpadlength`
          may be `True` or `False`, respectively (see Examples for details).
        * 'relative' : pad each trial by provided `padlength` such that all trials
          are extended by the same amount regardless of their original lengths.
          If `pad` is `relative`, `prepadlength` and `postpadlength` can either 
          be specified directly (using numerical values) or implicitly by only
          providing `padlength` and setting `prepadlength` and `postpadlength`
          to `True` or `False`, respectively (see Examples for details). If `pad`
          is `relative` at least one of `padlength`, `prepadlength` or `postpadlength`
          **has** to be provided. 
        * 'maxlen' : only usable if `data` is a Syncopy object. If `pad` is
          'maxlen' all trials are padded to achieve the length of the longest
          trial in `data`, i.e., post padding, all trials have the same length, 
          which equals the size of the longest trial pre-padding. For 
          ``pad = 'maxlen'``, `padlength`, `prepadlength` as well as `postpadlength` 
          have to be either Boolean or `None` indicating the preferred padding 
          location (pre-trial, post-trial or symmetrically pre- and post-trial). 
          If all are `None`, symmetric padding is performed (see Examples for 
          details). 
        * 'nextpow2' : pad each trial to achieve a length equals the closest power
          of two of its original length. For ``pad = 'nextpow2'``, `padlength`, 
          `prepadlength` as well as `postpadlength` have to be either Boolean
          or `None` indicating the preferred padding location (pre-trial, post-trial 
          or symmetrically pre- and post-trial). If all are `None`, symmetric 
          padding is performed (see Examples for details). 

    padlength : None, bool or positive scalar
        Length to be padded to `data` (if `padlength` is scalar-valued) or
        padding location (if `padlength` is Boolean). Depending on the value of
        `pad`, `padlength` can be used to pre-pend (if `padlength` is a positive
        number and `prepadlength` is `True`) or append trials (if `padlength` is
        a positive number and `postpadlength` is `True`). If neither
        `prepadlength` nor `postpadlength` are specified (i.e, both are `None`),
        symmetric pre- and post-trial padding is performed (i.e., ``0.5 * padlength``
        before and after each trial - note that odd sample counts are rounded downward
        to the nearest even integer). If ``unit = 'time'``, `padlength` is assumed 
        to be given in seconds, otherwise (``unit = 'samples'``), `padlength` is 
        interpreted as sample-count. Note that only ``pad = 'relative'`` and 
        ``pad = 'absolute'`` support numeric values of `padlength`. 
    prepadlength : None, bool or positive scalar
        Length to be pre-pended before each trial (if `prepadlength` is
        scalar-valued) or pre-padding flag (if `prepadlength` is `True`). If
        `prepadlength` is `True`, pre-padding length is either directly inferred
        from `padlength` or implicitly derived from chosen padding mode defined
        by `pad`. If ``unit = 'time'``, `prepadlength` is assumed to be given in
        seconds, otherwise (``unit = 'samples'``), `prepadlength` is interpreted
        as sample-count. Note that only ``pad = 'relative'`` supports numeric
        values of `prepadlength`. 
    postpadlength : None, bool or positive scalar
        Length to be appended after each trial (if `postpadlength` is
        scalar-valued) or post-padding flag (if `postpadlength` is `True`). If
        `postpadlength` is `True`, post-padding length is either directly inferred
        from `padlength` or implicitly derived from chosen padding mode defined
        by `pad`. If ``unit = 'time'``, `postpadlength` is assumed to be given in
        seconds, otherwise (``unit = 'samples'``), `postpadlength` is interpreted
        as sample-count. Note that only ``pad = 'relative'`` supports numeric
        values of `postpadlength`. 
    unit : str
        Unit of numerical values given by `padlength` and/or `prepadlength`
        and/or `postpadlength`. If ``unit = 'time'``, `padlength`,
        `prepadlength`, and `postpadlength` are assumed to be given in seconds,
        otherwise (``unit = 'samples'``), `padlength`, `prepadlength`, and
        `postpadlength` are interpreted as sample-counts. **Note** Providing
        padding lengths in seconds (i.e., ``unit = 'time'``) is only supported
        if `data` is a Syncopy object. 
    create_new : bool
        If `True`, a padded copy of the same type as `data` is returned (a
        :class:`numpy.ndarray` or Syncopy object). If `create_new` is `False`,
        either a single dictionary (if `data` is a :class:`numpy.ndarray`) or a
        ``len(data.trials)``-long list of dictionaries (if `data` is a Syncopy
        object) with all necessary options for performing the actual padding
        operation with :func:`numpy.pad` is returned.  
        
    Returns
    -------
    pad_dict : dict, if `data` is a :class:`numpy.ndarray` and ``create_new = False``
        Dictionary whose items contain all necessary parameters for calling
        :func:`numpy.pad` to perform the desired padding operation on `data`. 
    pad_dicts : list, if `data` is a Syncopy object and ``create_new = False``
        List of dictionaries for calling :func:`numpy.pad` to perform the
        desired padding operation on all trials found in `data`. 
    out : :class:`numpy.ndarray`, if `data` is a :class:`numpy.ndarray` and ``create_new = True``
        Padded version (deep copy) of `data`
    out : Syncopy object, if `data` is a Syncopy object and ``create_new = True``
        Padded version (deep copy) of `data`
        
    Notes
    -----
    This method emulates (and extends) FieldTrip's `ft_preproc_padding` by
    providing a convenience wrapper for NumPy's :func:`numpy.pad` that performs
    the actual heavy lifting. 
    
    Examples
    --------
    Consider the following small array representing a toy-problem-trial of `ns` 
    samples across `nc` channels:
    
    >>> nc = 7; ns = 30
    >>> trl = np.random.randn(ns, nc)
    
    We start by padding a total of 10 zeros symmetrically to `trl`
    
    >>> padded = spy.padding(trl, 'zero', pad='relative', padlength=10)
    >>> padded[:6, :]
    array([[ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
       [-1.0866,  2.3358,  0.8758,  0.5196,  0.8049, -0.659 , -0.9173]])
    >>> padded[-6:, :]
    array([[ 0.027 ,  1.8069,  1.5249, -0.7953, -0.8933,  1.0202, -0.6862],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ]])
    >>> padded.shape
    (40, 7)
    
    Note that the above call is equivalent to
    
    >>> padded_ident = spy.padding(trl, 'zero', pad='relative', padlength=10, prepadlength=True, postpadlength=True)
    >>> np.array_equal(padded_ident, padded)
    True
    >>> padded_ident = spy.padding(trl, 'zero', pad='relative', prepadlength=5, postpadlength=5)
    >>> np.array_equal(padded_ident, padded)
    True
    
    Similarly, 
    
    >>> prepad = spy.padding(trl, 'nan', pad='relative', prepadlength=10)
    
    is the same as
    
    >>> prepad_ident = spy.padding(trl, 'nan', pad='relative', padlength=10, prepadlength=True)
    >>> np.allclose(prepad, prepad_ident, equal_nan=True)
    True
    
    Define bogus trials on `trl` and create a dummy object with unit samplerate
    
    >>> tdf = np.vstack([np.arange(0, ns, 5),
                         np.arange(5, ns + 5, 5),
                         np.ones((int(ns / 5), )),
                         np.ones((int(ns / 5), )) * np.pi]).T
    >>> adata = spy.AnalogData(trl, trialdefinition=tdf, samplerate=1)

    Pad each trial to the closest power of two by appending by-trial channel 
    averages. However, do not perform actual padding, but only prepare dictionaries
    of parameters to be passed on to :func:`numpy.pad`
    
    >>> pad_dicts = spy.padding(adata, 'mean', pad='nextpow2', postpadlength=True, create_new=False)
    >>> len(pad_dicts) == len(adata.trials) 
    True
    >>> pad_dicts[0]
    {'pad_width': array([[0, 3],
        [0, 0]]), 'mode': 'mean'}
        
    Similarly, the following call generates a list of dictionaries preparing 
    absolute padding by prepending zeros with :func:`numpy.pad`
    
    >>> pad_dicts = spy.padding(adata, 'zero', pad='absolute', padlength=10, prepadlength=True, create_new=False)
    >>> pad_dicts[0]
    {'pad_width': array([[5, 0],
        [0, 0]]), 'mode': 'constant', 'constant_values': 0}
            
    See also
    --------
    numpy.pad : fast array padding in NumPy
    """

    # Detect whether input is data object or array-like
    if any(["BaseData" in str(base) for base in data.__class__.__mro__]):
        try:
            data_parser(data, varname="data", dataclass="AnalogData",
                        empty=False)
        except Exception as exc:
            raise exc
        timeAxis = data.dimord.index("time")
        spydata = True
    elif data.__class__.__name__ == "FauxTrial":
        if len(data.shape) != 2:
            lgl = "two-dimensional AnalogData trial segment"
            act = "{}-dimensional trial segment"
            raise SPYValueError(legal=lgl, varname="data", 
                                actual=act.format(len(data.shape)))
        timeAxis = data.dimord.index("time")
        spydata = False
    else:
        try:
            array_parser(data, varname="data", dims=2)
        except Exception as exc:
            raise exc
        timeAxis = 0
        spydata = False

    # FIXME: Creation of new spy-object currently not supported
    if not isinstance(create_new, bool):
        raise SPYTypeError(create_new, varname="create_new", expected="bool")
    if spydata and create_new:
        raise NotImplementedError("Creation of padded spy objects currently not supported. ")

    # Use FT-compatible options (sans FT option 'remove')
    if not isinstance(padtype, str):
        raise SPYTypeError(padtype, varname="padtype", expected="string")
    options = ["zero", "nan", "mean", "localmean", "edge", "mirror"]
    if padtype not in options:
        lgl = "'" + "or '".join(opt + "' " for opt in options)
        raise SPYValueError(legal=lgl, varname="padtype", actual=padtype)

    # Check `pad` and ensure we can actually perform the requested operation
    if not isinstance(pad, str):
        raise SPYTypeError(pad, varname="pad", expected="string")
    options = ["absolute", "relative", "maxlen", "nextpow2"]
    if pad not in options:
        lgl = "'" + "or '".join(opt + "' " for opt in options)
        raise SPYValueError(legal=lgl, varname="pad", actual=pad)
    if pad == "maxlen" and not spydata:
        lgl = "syncopy data object when using option 'maxlen'"
        raise SPYValueError(legal=lgl,
                            varname="pad", actual="maxlen")

    # Make sure a data object was provided if we're working with time values
    if not isinstance(unit, str):
        raise SPYTypeError(unit, varname="unit", expected="string")
    options = ["samples", "time"]
    if unit not in options:
        lgl = "'" + "or '".join(opt + "' " for opt in options)
        raise SPYValueError(legal=lgl, varname="unit", actual=unit)
    if unit == "time" and not spydata:
        raise SPYValueError(legal="syncopy data object when using option 'time'",
                            varname="unit", actual="time")

    # Set up dictionary for type-checking of provided padding lengths
    nt_dict = {"samples": "int_like", "time": None}

    # If we're padding up to an absolute bound or the max. length across
    # trials, compute lower bound for padding (in samples or seconds)
    if pad in ["absolute", "maxlen"]:
        if spydata:
            maxTrialLen = np.diff(data.sampleinfo).max()
        else:
            maxTrialLen = data.shape[timeAxis] # if `pad="absolute" and data is array
    else:
        maxTrialLen = np.inf
    if unit == "time":
        padlim = maxTrialLen/data.samplerate
    else:
        padlim = maxTrialLen

    # To ease option processing, collect padding length keywords in dict
    plengths = {"padlength": padlength, "prepadlength": prepadlength,
                "postpadlength": postpadlength}

    # In case of relative padding, we need at least one scalar value to proceed
    if pad == "relative":

        # If `padlength = None`, pre- or post- need to be set; if `padlength`
        # is set, both pre- and post- need to be `None` or `True`/`False`.
        # After this code block, pre- and post- are guaranteed to be numeric.
        if padlength is None:
            for key in ["prepadlength", "postpadlength"]:
                if plengths[key] is not None:
                    try:
                        scalar_parser(plengths[key], varname=key, ntype=nt_dict[unit],
                                      lims=[0, np.inf])
                    except Exception as exc:
                        raise exc
                else:
                    plengths[key] = 0
        else:
            try:
                scalar_parser(padlength, varname="padlength", ntype=nt_dict[unit],
                              lims=[0, np.inf])
            except Exception as exc:
                raise exc
            for key in ["prepadlength", "postpadlength"]:
                if not isinstance(plengths[key], (bool, type(None))):
                    raise SPYTypeError(plengths[key], varname=key, expected="bool or None")

            if prepadlength is None and postpadlength is None:
                prepadlength = True
                postpadlength = True
            else:
                prepadlength = prepadlength is not None
                postpadlength = postpadlength is not None
                
            if prepadlength and postpadlength:
                plengths["prepadlength"] = padlength/2
                plengths["postpadlength"] = padlength/2
            else:
                plengths["prepadlength"] = prepadlength*padlength
                plengths["postpadlength"] = postpadlength*padlength
                
        # Under-determined: abort if requested padding length is 0
        if all(value == 0 for value in plengths.values() if value is not None):
            lgl = "either non-zero value of `padlength` or `prepadlength` " + \
                  "and/or `postpadlength` to be set"
            raise SPYValueError(legal=lgl, varname="padlength", actual="0|None")

    else:

        # For absolute padding, the desired length has to be >= max. trial length
        if pad == "absolute":
            try:
                scalar_parser(padlength, varname="padlength", ntype=nt_dict[unit],
                              lims=[padlim, np.inf])
            except Exception as exc:
                raise exc
            for key in ["prepadlength", "postpadlength"]:
                if not isinstance(plengths[key], (bool, type(None))):
                    raise SPYTypeError(plengths[key], varname=key, expected="bool or None")

        # For `maxlen` or `nextpow2` we don't want any numeric entries at all
        else:
            for key, value in plengths.items():
                if not isinstance(value, (bool, type(None))):
                    raise SPYTypeError(value, varname=key, expected="bool or None")

            # Warn of potential conflicts
            if padlength and (prepadlength or postpadlength):
                print("WARNING: Found `padlength` and `prepadlength` and/or " +\
                      "`postpadlength`. Symmetric padding is performed. ")

        # If both pre-/post- are `None`, set them to `True` to use symmetric
        # padding, otherwise convert `None` entries to `False`
        if prepadlength is None and postpadlength is None:
            plengths["prepadlength"] = True
            plengths["postpadlength"] = True
        else:
            plengths["prepadlength"] = plengths["prepadlength"] is not None
            plengths["postpadlength"] = plengths["postpadlength"] is not None

    # Update pre-/post-padding and (if required) convert time to samples
    prepadlength = plengths["prepadlength"]
    postpadlength = plengths["postpadlength"]
    if unit == "time":
        if pad == "relative":
            prepadlength = int(prepadlength*data.samplerate)
            postpadlength = int(postpadlength*data.samplerate)
        elif pad == "absolute":
            padlength = int(padlength*data.samplerate)

    # Construct dict of keywords for ``np.pad`` depending on chosen `padtype`
    kws = {"zero": {"mode": "constant", "constant_values": 0},
           "nan": {"mode": "constant", "constant_values": np.nan},
           "localmean": {"mode": "mean", "stat_length": -1},
           "mean": {"mode": "mean"},
           "edge": {"mode": "edge"},
           "mirror": {"mode": "reflect"}}

    # If in put was syncopy data object, padding is done on a per-trial basis
    if spydata:

        # A list of input keywords for ``np.pad`` is constructed, no matter if
        # we actually want to build a new object or not
        pad_opts = []
        for trl in data.trials:
            nSamples = trl.shape[timeAxis]
            if pad == "absolute":
                padding = (padlength - nSamples)/(prepadlength + postpadlength)
            elif pad == "relative":
                padding = True
            elif pad == "maxlen":
                padding = (maxTrialLen - nSamples)/(prepadlength + postpadlength)
            elif pad == "nextpow2":
                padding = (_nextpow2(nSamples) - nSamples)/(prepadlength + postpadlength)
            pw = np.zeros((2, 2), dtype=int)
            pw[timeAxis, :] = [prepadlength * padding, postpadlength * padding]
            pad_opts.append(dict({"pad_width": pw}, **kws[padtype]))
            if padtype == "localmean":
                pad_opts[-1]["stat_length"] = pw[timeAxis, :]

        if create_new:
            pass
        else:
            return pad_opts

    # Input was a array/FauxTrial (i.e., single trial) - we have to do the padding just once
    else:

        nSamples = data.shape[timeAxis]
        if pad == "absolute":
            padding = (padlength - nSamples)/(prepadlength + postpadlength)
        elif pad == "relative":
            padding = True
        elif pad == "nextpow2":
            padding = (_nextpow2(nSamples) - nSamples)/(prepadlength + postpadlength)
        pw = np.zeros((2, 2), dtype=int)
        pw[timeAxis, :] = [prepadlength * padding, postpadlength * padding]
        pad_opts = dict({"pad_width": pw}, **kws[padtype])
        if padtype == "localmean":
            pad_opts["stat_length"] = pw[timeAxis, :]

        if create_new:
            if isinstance(data, np.ndarray):
                return np.pad(data, **pad_opts)
            else: # FIXME: currently only supports FauxTrial
                shp = list(data.shape)
                shp[timeAxis] += pw[timeAxis, :].sum()
                idx = list(data.idx)
                if isinstance(idx[timeAxis], slice):
                    idx[timeAxis] = slice(idx[timeAxis].start, 
                                          idx[timeAxis].start + shp[timeAxis])
                else:
                    idx[timeAxis] = pw[timeAxis, 0] * [idx[timeAxis][0]] + idx[timeAxis] \
                                    + pw[timeAxis, 1] * [idx[timeAxis][-1]]
                return data.__class__(shp, idx, data.dtype, data.dimord)
        else:
            return pad_opts

def _nextpow2(number):
    n = 1
    while n < number:
        n *= 2
    return n
