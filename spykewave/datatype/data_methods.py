# data_methods.py - Base functions for interacting with SpykeWave data objects
# 
# Created: February 25 2019
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-02-25 17:46:01>

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

##########################################################################################
def _selectdata_continuous(obj, segments, deepcopy, exact_match, **kwargs):

    segments, selectors = _makeidx(obj, segments, deepcopy, exact_match, **kwargs)

    if not any(["ContinuousData" in str(base) for base in data.__class__.__bases__]):
        raise SPWTypeError(obj, varname="obj", expected="SpkeWave ContinuousData object")
        

    # Build multi-index for each provided dimensional selector
    idx = [slice(None)] * len(self.dimord)
    for lbl, selector in kwargs.items():
        id = self.dimord.index(lbl)

        if selector 

        

        if isinstance(selection, str):
            selection = self._dimlabels[lbl].index(selection)
        elif isinstance(selection, (list, np.ndarray)):
            if isinstance(selection[0], str):
                for k in range(len(selection)):
                    selection[k] = self._dimlabels[lbl].index(selection[k])
        idx[id] = selection
        if isinstance(selection, slice):
            target_shape[id] = len(range(*selection.indices(self.data.shape[id])))
        elif isinstance(selection, int):
            target_shape[id] = 1
        else:
            if not deepcopy:
                spw_warning("Shallow copy only possible for int or slice selectors",
                            caller="SpykeWave core:select")
                deepcopy = True
            target_shape[id] = len(selection)

    # Allocate shallow copy for target and construct shape of target data
    target_shape = list(self.data.shape)
    target = self.copy()

    # If we have to perform a deep-copy operation, things are little involved
    if deepcopy:

        # Allocate target memorymap
        target._filename = self._gen_filename()
        target_dat = open_memmap(target._filename, mode="w+",
                                 dtype=self.data.dtype, shape=target_shape)

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
                stop = len(self.seg.shape[0]) - 1
            else:
                stop = segments[1]
            segments = range(start, stop)
        if not set(segments).issubset(range(self.seg.shape[0])):
            lgl = "segment selection between 0 and {}".format(str(self.seg.shape[0]))
            raise SPWValueError(legal=lgl, varname="segments")
        if isinstance(segments, int):
            segments = [segments]
    else:
        segments = range(self.seg.shape[0])

    # Calculate indices for each provided dimensional selector
    selectors = {}
    for lbl, selection in kwargs.items():
        ref = np.array(self.dimord[lbl])

        # Value-range selection
        if isinstance(selection, tuple):
            bounds = [None, None]
            for sk, sel in enumerate(selection):
                if isinstance(sel, str):
                    try:
                        bounds[sk] = list(ref).index(sel)
                    except:
                        raise SPWValueError(...)
                elif isinstance(sel, numbers.Number):
                    if not exact_match:
                        bounds[sk] = ref[np.abs(ref - sel).argmin()]
                    else:
                        try:
                            bounds[sk] = list(ref).index(sel)
                        except:
                            raise SPWValueError(...)
                elif sel is None:
                    if sk == 0:
                        bounds[sk] = ref[0]
                    if sk == 1:
                        bounds[sk] = ref[-1]
                else:
                    raise SPWTypeError(...)
            bounds[1] += 1
            selectors[lbl] = slice(*bounds)

        # Index-range selection
        elif isinstance(selection, slice):
            if not len(range(*selection.indices(ref.size))):
                raise SPWValueError('zero-element selection')
            selectors[lbl] = slice(selection.start, selection.stop, selection.step)
            
        # Multi-index selection: try to convert contiguous lists to slices
        elif isinstance(selection, (list, np.ndarray)):
            if not set(selection).issubset(ref):
                raise SPWValueError(...)
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
                    raise SPWValueError(...)

        # Single-index selection
        elif isinstance(selection, int):
            if selection not in range(ref.size):
                raise SPWValueError(...)
            selectors[lbl] = selection

        # You had your chance...
        else:
            raise SPWTypeError(...)
        
    return selectors, segments
