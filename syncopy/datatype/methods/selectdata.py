# -*- coding: utf-8 -*-
#
# Syncopy data selection methods
#

# Builtin/3rd party package imports
import numpy as np

# Local imports
from syncopy.shared.parsers import data_parser
from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYInfo, SPYWarning
from syncopy.shared.kwarg_decorators import unwrap_cfg, unwrap_io, detect_parallel_client
from syncopy.shared.computational_routine import ComputationalRoutine

__all__ = ["selectdata"]


@unwrap_cfg
@detect_parallel_client
def selectdata(data,
               trials=None,
               channel=None,
               channel_i=None,
               channel_j=None,
               toi=None,
               toilim=None,
               foi=None,
               foilim=None,
               taper=None,
               unit=None,
               eventid=None,
               out=None,
               inplace=False,
               clear=False,
               **kwargs):
    """
    Create a new Syncopy object from a selection

    **Usage Notice**

    Syncopy offers two modes for selecting data:

    * **in-place** selections mark subsets of a Syncopy data object for processing
      via a ``select`` dictionary *without* creating a new object
    * **deep-copy** selections copy subsets of a Syncopy data object to keep and
      preserve in a new object created by :func:`~syncopy.selectdata`

    All Syncopy metafunctions, such as :func:`~syncopy.freqanalysis`, support
    **in-place** data selection via a ``select`` keyword, effectively avoiding
    potentially slow copy operations and saving disk space. The keys accepted
    by the `select` dictionary are identical to the keyword arguments discussed
    below. In addition, ``select = "all"`` can be used to select entire object
    contents. Examples

    >>> select = {"toilim" : [-0.25, 0]}
    >>> spy.freqanalysis(data, select=select)
    >>> # or equivalently
    >>> cfg = spy.get_defaults(spy.freqanalysis)
    >>> cfg.select = select
    >>> spy.freqanalysis(cfg, data)

    **Usage Summary**

    List of Syncopy data objects and respective valid data selectors:

    :class:`~syncopy.AnalogData` : trials, channel, toi/toilim
        Examples

        >>> spy.selectdata(data, trials=[0, 3, 5], channel=["channel01", "channel02"])
        >>> cfg = spy.StructDict()
        >>> cfg.trials = [5, 3, 0]; cfg.toilim = [0.25, 0.5]
        >>> spy.selectdata(cfg, data)

    :class:`~syncopy.SpectralData` : trials, channel, toi/toilim, foi/foilim, taper
        Examples

        >>> spy.selectdata(data, trials=[0, 3, 5], channel=["channel01", "channel02"])
        >>> cfg = spy.StructDict()
        >>> cfg.foi = [30, 40, 50]; cfg.taper = slice(2, 4)
        >>> spy.selectdata(cfg, data)

    :class:`~syncopy.EventData` : trials, toi/toilim, eventid
        Examples

        >>> spy.selectdata(data, toilim=[-1, 2.5], eventid=[0, 1])
        >>> cfg = spy.StructDict()
        >>> cfg.trials = [0, 0, 1, 0]; cfg.eventid = slice(2, None)
        >>> spy.selectdata(cfg, data)

    :class:`~syncopy.SpikeData` : trials, toi/toilim, unit, channel
        Examples

        >>> spy.selectdata(data, toilim=[-1, 2.5], unit=range(0, 10))
        >>> cfg = spy.StructDict()
        >>> cfg.toi = [1.25, 3.2]; cfg.trials = [0, 1, 2, 3]
        >>> spy.selectdata(cfg, data)

    **Note** Any property that is not specifically accessed via one of the provided
    selectors is taken as is, e.g., ``spy.selectdata(data, trials=[1, 2])``
    selects the entire contents of trials no. 2 and 3, while
    ``spy.selectdata(data, channel=range(0, 50))`` selects the first 50 channels
    of `data` across all defined trials. Consequently, if no keywords are specified,
    the entire contents of `data` is selected.

    **Full documentation below**

    Parameters
    ----------
    data : Syncopy data object
        A non-empty Syncopy data object. **Note** the type of `data` determines
        which keywords can be used.  Some keywords are only valid for certain
        types of Syncopy objects, e.g., "freqs" is not a valid selector for an
        :class:`~syncopy.AnalogData` object.
    trials : list (integers) or None or "all"
        List of integers representing trial numbers to be selected; can include
        repetitions and need not be sorted (e.g., ``trials = [0, 1, 0, 0, 2]``
        is valid) but must be finite and not NaN. If `trials` is `None`, or
        ``trials = "all"`` all trials are selected.
    channel : list (integers or strings), slice, range, str, int, None or "all"
        Channel-selection; can be a list of channel names (``['channel3', 'channel1']``),
        a list of channel indices (``[3, 5]``), a slice (``slice(3, 10)``) or
        range (``range(3, 10)``). Note that following Python conventions, channels
        are counted starting at zero, and range and slice selections are half-open
        intervals of the form `[low, high)`, i.e., low is included , high is
        excluded. Thus, ``channel = [0, 1, 2]`` or ``channel = slice(0, 3)``
        selects the first up to (and including) the third channel. Selections can
        be unsorted and may include repetitions but must match exactly, be finite
        and not NaN. If `channel` is `None`, or ``channel = "all"`` all channels
        are selected.
    toi : list (floats), float, None or "all"
        Time-points to be selected (in seconds) in each trial. Timing is expected
        to be on a by-trial basis (e.g., relative to trigger onsets). Selections
        can be approximate, unsorted and may include repetitions but must be
        finite and not NaN. Fuzzy matching is performed for approximate selections
        (i.e., selected time-points are close but not identical to timing information
        found in `data`) using a nearest-neighbor search for elements of `toi`.
        If `toi` is `None` or ``toi = "all"``, the entire time-span in each trial
        is selected.
    toilim : list (floats [tmin, tmax]) or None or "all"
        Time-window ``[tmin, tmax]`` (in seconds) to be extracted from each trial.
        Window specifications must be sorted (e.g., ``[2.2, 1.1]`` is invalid)
        and not NaN but may be unbounded (e.g., ``[1.1, np.inf]`` is valid). Edges
        `tmin` and `tmax` are included in the selection.
        If `toilim` is `None` or ``toilim = "all"``, the entire time-span in each
        trial is selected.
    foi : list (floats), float, None or "all"
        Frequencies to be selected (in Hz). Selections can be approximate, unsorted
        and may include repetitions but must be finite and not NaN. Fuzzy matching
        is performed for approximate selections (i.e., selected frequencies are
        close but not identical to frequencies found in `data`) using a nearest-
        neighbor search for elements of `foi` in `data.freq`. If `foi` is `None`
        or ``foi = "all"``, all frequencies are selected.
    foilim : list (floats [fmin, fmax]) or None or "all"
        Frequency-window ``[fmin, fmax]`` (in Hz) to be extracted. Window
        specifications must be sorted (e.g., ``[90, 70]`` is invalid) and not NaN
        but may be unbounded (e.g., ``[-np.inf, 60.5]`` is valid). Edges `fmin`
        and `fmax` are included in the selection. If `foilim` is `None` or
        ``foilim = "all"``, all frequencies are selected.
    taper : list (integers or strings), slice, range, str, int, None or "all"
        Taper-selection; can be a list of taper names (``['dpss-win-1', 'dpss-win-3']``),
        a list of taper indices (``[3, 5]``), a slice (``slice(3, 10)``) or range
        (``range(3, 10)``). Note that following Python conventions, tapers are
        counted starting at zero, and range and slice selections are half-open
        intervals of the form `[low, high)`, i.e., low is included , high is
        excluded. Thus, ``taper = [0, 1, 2]`` or ``taper = slice(0, 3)`` selects
        the first up to (and including) the third taper. Selections can be unsorted
        and may include repetitions but must match exactly, be finite and not NaN.
        If `taper` is `None` or ``taper = "all"``, all tapers are selected.
    unit : list (integers or strings), slice, range, str, int, None or "all"
        Unit-selection; can be a list of unit names (``['unit10', 'unit3']``), a
        list of unit indices (``[3, 5]``), a slice (``slice(3, 10)``) or range
        (``range(3, 10)``). Note that following Python conventions, units are
        counted starting at zero, and range and slice selections are half-open
        intervals of the form `[low, high)`, i.e., low is included , high is
        excluded. Thus, ``unit = [0, 1, 2]`` or ``unit = slice(0, 3)`` selects
        the first up to (and including) the third unit. Selections can be unsorted
        and may include repetitions but must match exactly, be finite and not NaN.
        If `unit` is `None` or ``unit = "all"``, all units are selected.
    eventid : list (integers), slice, range, int, None or "all"
        Event-ID-selection; can be a list of event-id codes (``[2, 0, 1]``), slice
        (``slice(0, 2)``) or range (``range(0, 2)``). Note that following Python
        conventions, range and slice selections are half-open intervals of the
        form `[low, high)`, i.e., low is included , high is excluded. Selections
        can be unsorted and may include repetitions but must match exactly, be
        finite and not NaN. If `eventid` is `None` or ``eventid = "all"``, all
        events are selected.
    inplace : bool
        If `inplace` is `True` **no** new object is created. Instead the provided
        selection is stored in the input object's `selection` attribute for later
        use. By default `inplace` is `False` and all calls to `selectdata` create
        a new Syncopy data object.
    clear : bool
        If `True` remove any active in-place selection. Note that in-place
        selections can also be removed manually by assinging `None` to the
        `selection` property, i.e., ``mydata.selection = None`` is equivalent
        to ``spy.selectdata(mydata, clear=True)`` or ``mydata.selectdata(clear=True)``

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
    be necessary: all Syncopy metafunctions, such as :func:`~syncopy.freqanalysis`,
    support **in-place** data selection.

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
    metafunctions does not only save disk-space but can significantly increase
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
    :func:`syncopy.show` : Show (subsets) of Syncopy objects
    """

    # Ensure our one mandatory input is usable
    try:
        data_parser(data, varname="data", empty=False)
    except Exception as exc:
        raise exc

    # Vet the only inputs not checked by `Selector`
    if not isinstance(inplace, bool):
        raise SPYTypeError(inplace, varname="inplace", expected="Boolean")
    if not isinstance(clear, bool):
        raise SPYTypeError(clear, varname="clear", expected="Boolean")

    # If provided, make sure output object is appropriate
    if not inplace:
        if out is not None:
            try:
                data_parser(out, varname="out", writable=True, empty=True,
                            dataclass=data.__class__.__name__,
                            dimord=data.dimord)
            except Exception as exc:
                raise exc
            new_out = False
        else:
            out = data.__class__(dimord=data.dimord)
            new_out = True
    else:
        if out is not None:
            lgl = "no output object for in-place selection"
            raise SPYValueError(lgl, varname="out", actual=out.__class__.__name__)

    # Collect provided selection keywords in dict
    selectDict = {"trials": trials,
                  "channel": channel,
                  "channel_i": channel_i,
                  "channel_j": channel_j,
                  "toi": toi,
                  "toilim": toilim,
                  "foi": foi,
                  "foilim": foilim,
                  "taper": taper,
                  "unit": unit,
                  "eventid": eventid}

    # The only valid anonymous kw is "parallel" (i.e., filter out typos like 'trails' etc.)
    if len(kwargs) > 0:
        if list(kwargs.keys()) != ["parallel"]:
            kwargs.pop("parallel", None)
            expected = list(selectDict.keys()) + ["out", "inplace", "clear", "parallel"]
            lgl = "dict with one or all of the following keys: '" +\
                  "'".join(opt + "', " for opt in expected)[:-2]
            act = "dict with keys '" +\
                  "'".join(key + "', " for key in kwargs.keys())[:-2]
            raise SPYValueError(legal=lgl, varname="kwargs", actual=act)

    # First simplest case: determine whether we just need to clear an existing selection
    if clear:
        if any(value is not None for value in selectDict.values()):
            lgl = "no data selectors if `clear = True`"
            raise SPYValueError(lgl, varname="select", actual=selectDict)
        if data.selection is None:
            SPYInfo("No in-place selection found. ")
        else:
            data.selection = None
            SPYInfo("In-place selection cleared")
        return

    # Pass provided selections on to `Selector` class which performs error checking
    data.selection = selectDict

    # If an in-place selection was requested we're done
    if inplace:
        return

    # Inform the user what's about to happen
    selectionSize = _get_selection_size(data)
    sUnit = "MB"
    if selectionSize > 1000:
        selectionSize /= 1024
        sUnit = "GB"
    msg = "Copying {dsize:3.2f} {dunit:s} of data based on selection " +\
        "to create new {objkind:s} object on disk"
    SPYInfo(msg.format(dsize=selectionSize, dunit=sUnit, objkind=data.__class__.__name__))

    # Create inventory of all available selectors and actually provided values
    # to create a bookkeeping dict for logging
    log_dct = {"inplace": inplace, "clear": clear}
    log_dct.update(selectDict)
    log_dct.update(**kwargs)

    # Fire up `ComputationalRoutine`-subclass to do the actual selecting/copying
    selectMethod = DataSelection()
    selectMethod.initialize(data, out._stackingDim, chan_per_worker=kwargs.get("chan_per_worker"))
    selectMethod.compute(data, out, parallel=kwargs.get("parallel"),
                         log_dict=log_dct)

    # Wipe data-selection slot to not alter input object
    data.selection = None

    # Either return newly created output object or simply quit
    return out if new_out else None


def _get_selection_size(data):
    """
    Local helper routine for computing the on-disk size of an active data-selection
    """
    fauxTrials = [data._preview_trial(trlno) for trlno in data.selection.trials]
    fauxSizes = [np.prod(ftrl.shape)*ftrl.dtype.itemsize for ftrl in fauxTrials]
    return sum(fauxSizes) / 1024**2


@unwrap_io
def _selectdata(trl, noCompute=False, chunkShape=None):
    if noCompute:
        return trl.shape, trl.dtype
    return trl


class DataSelection(ComputationalRoutine):

    computeFunction = staticmethod(_selectdata)

    def process_metadata(self, data, out):

        # Get/set timing-related selection modifiers
        out.trialdefinition = data.selection.trialdefinition
        # if data.selection._timeShuffle: # FIXME: should be implemented down the road
        #     out.time = data.selection.timepoints
        if data.selection._samplerate:
            out.samplerate = data.samplerate

        # Get/set dimensional attributes changed by selection
        for prop in data.selection._dimProps:
            selection = getattr(data.selection, prop)
            if selection is not None:
                if np.issubdtype(type(selection), np.number):
                    selection = [selection]
                setattr(out, prop, getattr(data, prop)[selection])
