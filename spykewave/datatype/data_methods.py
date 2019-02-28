# data_methods.py - Base functions for interacting with SpykeWave data objects
# 
# Created: February 25 2019
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-02-27 16:00:14>

# Builtin/3rd party package imports
import numbers
import numpy as np

# Local imports
from spykewave.utils import SPWTypeError, SPWValueError, spy_data_parser

__all__ = ["selectdata"]

##########################################################################################
def selectdata(obj, segments=None, deepcopy=False, exact_match=False, **kwargs):
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
        return _selectdata_continuous(obj, segments, deepcopy, exact_match, **kwargs)
    elif any(["DiscreteData" in str(base) for base in obj.__class__.__bases__]):
        raise NotImplementedError("coming soon")
    else:
        raise SPWTypeError(obj, varname="obj", expected="SpkeWave data object")
    
##########################################################################################
def _selectdata_continuous(obj, segments, deepcopy, exact_match, **kwargs):

    # Make sure provided object is inherited from `ContinuousData`
    if not any(["ContinuousData" in str(base) for base in obj.__class__.__bases__]):
        raise SPWTypeError(obj, varname="obj", expected="SpkeWave ContinuousData object")
        
    # Convert provided selectors to array indices
    segments, selectors = _makeidx(obj, segments, deepcopy, exact_match, **kwargs)

    # Make sure our Boolean switches are actuall Boolean
    if not isinstance(deepcopy, bool):
        raise SPWTypeError(deepcopy, varname="deepcopy", expected="bool")
    if not isinstance(exact_match, bool):
        raise SPWTypeError(exact_match, varname="exact_match", expected="bool")

    # If time-based selection is requested, make some necessary preparations
    if "time" in selectors.keys():
        time_sel = selectors.pop("time")
        time_ref = np.array(obj.time[segments[0]])
        time_slice = [None, None]
        if isinstance(time_sel, tuple):
            if len(time_sel) != 2:
                raise SPWValueError(legal="two-element tuple",
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
                            raise SPWValueError(legal="exact time-point", actual=ts)
            time_slice = slice(*time_slice)
        elif isinstance(time_sel, slice):
            if not len(range(*time_sel.indices(time_ref.size))):
                lgl = "non-empty time-selection"
                act = "empty selector"
                raise SPWValueError(legal=lgl, varname=lbl, actual=act)
            time_slice = slice(time_sel.start, time_sel.stop, time_sel.step)
        elif isinstance(time_sel, (list, np.ndarray)):
            if not set(time_sel).issubset(range(time_ref.size))\
               or np.unique(np.diff(time_sel)).size != 1:
                vname = "contiguous list of time-points"
                raise SPWValueError(legal=lgl, varname=vname)
            time_slice = slice(time_sel[0], time_sel[-1] + 1)
        else:
            raise SPWTypeError(time_sel, varname="time-selection",
                               expected="tuple, slice or list-like")
    else:
        time_slice = slice(0, None)

        # SHALLOWCOPY
        sampleinfo = np.empty((segments.size, 2))
        for sk, seg in enumerate(segments):
            sinfo = range(*obj.sampleinfo[seg, :])[time_slice]
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

        # Re-number segments: offset correction + close gaps b/w segments
        sampleinfo = obj.sampleinfo[segments, :] - obj.sampleinfo[segments[0], 0]
        stop = 0
        for sk in range(sampleinfo.shape[0]):
            sinfo = range(*sampleinfo[sk, :])[time_slice]
            nom_len = sinfo.stop - sinfo.start
            start = min(sinfo.start, stop)
            real_len = min(nom_len, sinfo.stop - stop)
            sampleinfo[sk, :] = [start, start + nom_len]
            stop = start + real_len + 1
            
        # Based on requested segments, set shape of target array (accounting
        # for overlapping segments)
        target_shape[tid] = sampleinfo[-1][1]

        # Allocate target memorymap
        target._filename = obj._gen_filename()
        target_dat = open_memmap(target._filename, mode="w+",
                                 dtype=obj.data.dtype, shape=target_shape)
        del target_dat

        # The crucial part here: `idx` is a "local" by-segment index of the
        # form `[:,:,2:10]` whereas `target_idx` has to keep track of the
        # global progression in `target_data`
        for sk, seg in enumerate(segments):
            source_seg = self._copy_segment(segno,
                                            obj._filename,
                                            obj.seg,
                                            obj.hdr,
                                            obj.dimord,
                                            obj.segmentlabel)
            target_idx[tid] = slice(*sampleinfo[sk, :])
            target_dat = open_memmap(target._filename, mode="r+")[target_idx]
            target_dat[...] = source_seg[idx]
            del target_dat

        # FIXME: Clarify how we want to do this...
        target._dimlabels["sample"] = sampleinfo

        # Re-number samples if necessary
        

        # By-sample copy
        if segments is None:
            mem_size = np.prod(target_shape)*self.data.dtype*1024**(-2)
            if mem_size >= 100:
                spw_warning("Memory footprint of by-sample selection larger than 100MB",
                            caller="SpykeWave core:select")
            target_dat[...] = self.data[idx]
            del target_dat
            self.clear()

        # By-segment copy
        else:
            del target_dat
            sid = self.dimord.index(self.segmentlabel)
            target_shape[sid] = sum([shp[sid] for shp in np.array(self.shapes)[segments]])
            target_idx = [slice(None)] * len(self.dimord)
            target_sid = 0
            for segno in segments:
                source_seg = self._copy_segment(segno,
                                                self._filename,
                                                self.seg,
                                                self.hdr,
                                                self.dimord,
                                                self.segmentlabel)
                seg_len = source_seg.shape[sid]
                target_idx[sid] = slice(target_sid, target_sid + seg_len)
                target_dat = open_memmap(target._filename, mode="r+")[target_idx]
                target_dat[...] = source_seg[idx]
                del target_dat
                target_sid += seg_len

    # Shallow copy: simply create a view of the source memmap
    # Cover the case: channel=3, all segments!
    else:
        target._data = open_memmap(self._filename, mode="r")[idx]

    return target

##########################################################################################
def _makeidx(obj, segments, deepcopy, exact_match, **kwargs):
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
        raise SPWValueError(legal=self.dimord, actual=list(kwargs.keys()))

    # Process `segments`
    if segments is not None:
        if isinstance(segments, tuple):
            start = segments[0]
            if segments[1] is None:
                stop = self.seg.shape[0]
            else:
                stop = segments[1]
            segments = np.arange(start, stop)
        if not set(segments).issubset(range(self.seg.shape[0])):
            lgl = "segment selection between 0 and {}".format(str(self.seg.shape[0]))
            raise SPWValueError(legal=lgl, varname="segments")
        if isinstance(segments, int):
            segments = np.array([segments])
    else:
        segments = np.arange(self.seg.shape[0])

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
                raise SPWValueError(legal="two-element tuple",
                                    actual="tuple of length {}".format(str(len(selection))),
                                    varname=lbl)
            bounds = [None, None]
            for sk, sel in enumerate(selection):
                if isinstance(sel, str):
                    try:
                        bounds[sk] = list(ref).index(sel)
                    except:
                        raise SPWValueError(legal=lgl, actual=sel)
                elif isinstance(sel, numbers.Number):
                    if not exact_match:
                        bounds[sk] = ref[np.abs(ref - sel).argmin()]
                    else:
                        try:
                            bounds[sk] = list(ref).index(sel)
                        except:
                            raise SPWValueError(legal=lgl, actual=sel)
                elif sel is None:
                    if sk == 0:
                        bounds[sk] = ref[0]
                    if sk == 1:
                        bounds[sk] = ref[-1]
                else:
                    raise SPWTypeError(sel, varname=lbl, expected="string, number or None")
            bounds[1] += 1
            selectors[lbl] = slice(*bounds)

        # Index-range selection
        elif isinstance(selection, slice):
            if not len(range(*selection.indices(ref.size))):
                lgl = "non-empty selection"
                act = "empty selector"
                raise SPWValueError(legal=lgl, varname=lbl, actual=act)
            selectors[lbl] = slice(selection.start, selection.stop, selection.step)
            
        # Multi-index selection: try to convert contiguous lists to slices
        elif isinstance(selection, (list, np.ndarray)):
            if not set(selection).issubset(range(ref.size)):
                vname = "list-selector for `obj.{}`".format(lbl)
                raise SPWValueError(legal=lgl, varname=vname)
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
                    raise SPWValueError(legal=lgl, actual=selection)

        # Single-index selection
        elif isinstance(selection, int):
            if selection not in range(ref.size):
                raise SPWValueError(legal=lgl, actual=selection)
            selectors[lbl] = selection

        # You had your chance...
        else:
            raise SPWTypeError(selection, varname=lbl,
                               expected="tuple, list-like, slice, float or int")
        
    return selectors, segments
