# -*- coding: utf-8 -*-
# 
# Syncopy data selection methods
# 
# Created: 2019-10-14 12:46:54
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-10-17 15:48:40>

# Builtin/3rd party package imports
import inspect
# import numbers
# import sys
# import numpy as np

# Local imports
from syncopy.shared.parsers import get_defaults, data_parser, unwrap_cfg, unwrap_io
from syncopy.shared.errors import SPYValueError
from syncopy.shared.computational_routine import ComputationalRoutine

__all__ = ["selectdata"]


@unwrap_cfg
def selectdata(data, trials=None, channels=None, toi=None, toilim=None, foi=None,
               foilim=None, tapers=None, units=None, eventids=None):
    """
    Create a new Syncopy object from a selection

    **Usage Notice**    
    
    Syncopy offers two modes for selecting data: 
    
    * **in-place** selections mark subsets of a Syncopy data object for processing 
      via a ``select`` dictionary *without* creating a new object
    * **deep-copy** selections copy subsets of a Syncopy data object to keep and 
      preserve in a new object created by :func:`~syncopy.selectdata`
    
    All Syncopy compute kernels, such as :func:`~syncopy.freqanalysis`, support 
    **in-place** data selection via a ``select`` dictionary, effectively avoiding
    potentially slow copy operations and saving disk space. The keys accepted 
    by the `select` dictionary are identical to the keyword arguments discussed 
    below, e.g.,
    
    >>> select = {"toilim" : [-0.25, 0]}
    >>> spy.freqanalysis(data, select=select)
    >>> # or equivalently 
    >>> cfg = spy.get_defaults(spy.freqanalysis)
    >>> cfg.select = select
    >>> spy.freqanalysis(cfg, data)
    
    **Usage Summary**
    
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

    # Ensure our one mandatory input is usable
    try:
        data_parser(data, varname="data", empty=False)
    except Exception as exc:
        raise exc

    # For later reference: dynamically fetch name of current function
    funcName = "Syncopy <{}>".format(inspect.currentframe().f_code.co_name)

    # Create inventory of all available selectors and actually provided values 
    # and raise exception in case unavailable selectors were provided 
    inventory = get_defaults(globals()[inspect.currentframe().f_code.co_name])
    provided = locals()
    # available = get_defaults(data.selectdata)
    
    # actualSelection = {}
    # for key in available:
    #     actualSelection[key] = provided.pop(key)
        
    # for key, value in provided.items():
    #     if value != inventory[key]:
    #         lgl = "one or all of the following selectors: '" +\
    #               "'".join(opt + "', " for opt in available.keys())[:-2]
    #         raise SPYValueError(legal=lgl, varname=key)
        
    actualSelection = {}
    for key in inventory:
        actualSelection[key] = provided[key]
        
    data._selection = actualSelection
    selectMethod = DataSelection()
    selectMethod.initialize(data)
    selectMethod.compute(data, out)


    return data.selectdata(**actualSelection)


@unwrap_io
def _selectdata(trl, noCompute=False, chunkShape=None):
    if noCompute:
        return trl.shape, trl.dtype
    return trl

class DataSelection(ComputationalRoutine):

    computeFunction = staticmethod(_selectdata)

    def process_metadata(self, data, out):
        
        # Some index gymnastics to get trial begin/end "samples"
        if data._selection is not None:
            chanSec = data._selection.channel
            trl = data._selection.trialdefinition
            for row in range(trl.shape[0]):
                trl[row, :2] = [row, row + 1]
        else:
            chanSec = slice(None)
            time = np.arange(len(data.trials))
            time = time.reshape((time.size, 1))
            trl = np.hstack((time, time + 1, 
                             np.zeros((len(data.trials), 1)), 
                             np.array(data.trialinfo)))

