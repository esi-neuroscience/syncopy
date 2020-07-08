# -*- coding: utf-8 -*-
# 
# SyNCoPy spectral estimation methods
# 
# Created: 2019-01-22 09:07:47
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2020-07-08 14:11:35>

# Builtin/3rd party package imports
from numbers import Number
import numpy as np
import scipy.signal.windows as spwin

# Local imports
from syncopy.shared.parsers import data_parser, scalar_parser, array_parser 
from syncopy.shared.tools import get_defaults
from syncopy.datatype import SpectralData, padding
from syncopy.datatype.methods.padding import _nextpow2
import syncopy.specest.wavelets as spywave 
from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYWarning
from syncopy.shared.kwarg_decorators import (unwrap_cfg, unwrap_select, 
                                             detect_parallel_client)
from syncopy.shared.tools import best_match
from syncopy.specest.mtmfft import MultiTaperFFT
from syncopy.specest.mtmconvol import MultiTaperFFTConvol
from syncopy.specest.wavelet import _get_optimal_wavelet_scales, WaveletTransform

# Module-wide output specs
spectralDTypes = {"pow": np.float32,
                  "fourier": np.complex128,
                  "abs": np.float32}

#: output conversion of complex fourier coefficients
spectralConversions = {"pow": lambda x: (x * np.conj(x)).real.astype(np.float32),
                       "fourier": lambda x: x.astype(np.complex128),
                       "abs": lambda x: (np.absolute(x)).real.astype(np.float32)}

#: available outputs of :func:`~syncopy.freqanalysis`
availableOutputs = tuple(spectralConversions.keys())

#: available tapers of :func:`~syncopy.freqanalysis`
availableTapers = ("hann", "dpss")

#: available spectral estimation methods of :func:`~syncopy.freqanalysis`
availableMethods = ("mtmfft", "mtmconvol", "wavelet")

__all__ = ["freqanalysis"]


@unwrap_cfg
@unwrap_select
@detect_parallel_client
def freqanalysis(data, method='mtmfft', output='fourier',
                 keeptrials=True, foi=None, foilim=None, pad=None, padtype='zero',
                 padlength=None, prepadlength=None, postpadlength=None, 
                 polyremoval=None, 
                 taper="hann", tapsmofrq=None, keeptapers=False,
                 wav="Morlet", t_ftimwin=None, toi=None, width=6, 
                 out=None, **kwargs):
    """
    Perform (time-)frequency analysis of Syncopy :class:`~syncopy.AnalogData` objects
    
    **Usage Summary**
    
    Options available in all analysis methods:
    
    * **output** : one of :data:`~.availableOutputs`; return power spectra, complex 
      Fourier spectra or absolute values. 
    * **foi**/**foilim** : frequencies of interest; either array of frequencies or 
      frequency window (not both)
    * **keeptrials** : return individual trials or grand average
    * **polyremoval** : de-trending method to use (0 = mean, 1 = linear, 2 = quadratic, 
      3 = cubic, etc.)
            
    List of available analysis methods and respective distinct options:
    
    :func:`~syncopy.specest.mtmfft.mtmfft` : (Multi-)tapered Fourier transform
        Perform frequency analysis on time-series trial data using either a single 
        taper window (Hanning) or many tapers based on the discrete prolate 
        spheroidal sequence (DPSS) that maximize energy concentration in the main
        lobe. 
        
        * **taper** : one of :data:`~.availableTapers`
        * **tapsmofrq** : spectral smoothing box for tapers (in Hz)
        * **keeptapers** : return individual tapers or average
        * **pad** : padding method to use (`None`, `True`, `False`, `'absolute'`, 
          `'relative'`, `'maxlen'` or `'nextpow2'`). If `None`, then `'nextpow2'`
          is selected by default. 
        * **padtype** : values to pad data with (`'zero'`, `'nan'`, `'mean'`, `'localmean'`, 
          `'edge'` or `'mirror'`)
        * **padlength** : number of samples to pre-pend and/or append to each trial 
        * **prepadlength** : number of samples to pre-pend to each trial 
        * **postpadlength** : number of samples to append to each trial 

    :func:`~syncopy.specest.mtmconvol.mtmconvol` : (Multi-)tapered sliding window Fourier transform
        Perform time-frequency analysis on time-series trial data based on a sliding 
        window short-time Fourier transform using either a single Hanning taper or 
        multiple DPSS tapers. 
        
        * **taper** : one of :data:`~.availableTapers`
        * **tapsmofrq** : spectral smoothing box for tapers (in Hz)
        * **keeptapers** : return individual tapers or average
        * **pad** : flag indicating, whether or not to pad trials. If `None`, 
          trials are padded only if sliding window centroids are too close
          to trial boundaries for the entire window to cover available data-points. 
        * **toi** : time-points of interest; can be either an array representing 
          analysis window centroids (in sec), a scalar between 0 and 1 encoding 
          the percentage of overlap between adjacent windows or "all" to center 
          a window on every sample in the data. 
        * **t_ftimwin** : sliding window length (in sec)

    **Full documentation below** 
    
    Parameters
    ----------
    data : `~syncopy.AnalogData`
        A non-empty Syncopy :class:`~syncopy.datatype.AnalogData` object
    method : str
        Spectral estimation method, one of :data:`~.availableMethods` 
        (see below).
    output : str
        Output of spectral estimation. One of :data:`~.availableOutputs` (see below); 
        use `'pow'` for power spectrum (:obj:`numpy.float32`), `'fourier'` for complex 
        Fourier coefficients (:obj:`numpy.complex128`) or `'abs'` for absolute 
        values (:obj:`numpy.float32`).
    keeptrials : bool
        If `True` spectral estimates of individual trials are returned, otherwise
        results are averaged across trials. 
    foi : array-like or None
        Frequencies of interest (Hz) for output. If desired frequencies cannot be 
        matched exactly, the closest possible frequencies are used. If `foi` is `None`
        or ``foi = "all"``, all attainable frequencies (i.e., zero to Nyquist / 2) 
        are selected. 
    foilim : array-like (floats [fmin, fmax]) or None or "all"
        Frequency-window ``[fmin, fmax]`` (in Hz) of interest. Window 
        specifications must be sorted (e.g., ``[90, 70]`` is invalid) and not NaN 
        but may be unbounded (e.g., ``[-np.inf, 60.5]`` is valid). Edges `fmin` 
        and `fmax` are included in the selection. If `foilim` is `None` or 
        ``foilim = "all"``, all frequencies are selected. 
    pad : str or None or bool
        One of `None`, `True`, `False`, `'absolute'`, `'relative'`, `'maxlen'` or
        `'nextpow2'`. 
        If `pad` is `None` or ``pad = True``, then method-specific defaults are 
        chosen. Specifically, if `method` is `'mtmfft'` then `pad` is set to 
        `'nextpow2'` so that all trials in `data` are padded to the next power of 
        two higher than the sample-count of the longest (selected) trial in `data`. Conversely, 
        time-frequency analysis methods (`'mtmconvol'` and `'wavelet'`), only perform
        padding if necessary, i.e., if time-window centroids are chosen too close
        to trial boundaries for the entire window to cover available data-points. 
        If `pad` is `False`, then no padding is performed. Then in case of 
        ``method = 'mtmfft'`` all trials have to have approximately the same 
        length (up to the next even sample-count), if ``method = 'mtmconvol'`` or 
        ``method = 'wavelet'``, window-centroids have to keep sufficient
        distance from trial boundaries. For more details on the padding methods 
        `'absolute'`, `'relative'`, `'maxlen'` and `'nextpow2'` see :func:`syncopy.padding`. 
    padtype : str
        Values to be used for padding. Can be `'zero'`, `'nan'`, `'mean'`, 
        `'localmean'`, `'edge'` or `'mirror'`. See :func:`syncopy.padding` for 
        more information.
    padlength : None, bool or positive scalar
        Only valid if `method` is `'mtmfft'` and `pad` is `'absolute'` or `'relative'`. 
        Number of samples to pad data with. See :func:`syncopy.padding` for more 
        information.
    prepadlength : None or bool or int
        Only valid if `method` is `'mtmfft'` and `pad` is `'relative'`. Number of 
        samples to pre-pend to each trial. See :func:`syncopy.padding` for more 
        information.
    postpadlength : None or bool or int
        Only valid if `method` is `'mtmfft'` and `pad` is `'relative'`. Number of 
        samples to append to each trial. See :func:`syncopy.padding` for more 
        information.
    polyremoval : int or None
        **FIXME: Not implemented yet**
        Order of polynomial used for de-trending data in the time domain prior 
        to spectral analysis. A value of 0 corresponds to subtracting the mean 
        ("de-meaning"), ``polyremoval = 1`` removes linear trends (subtracting the 
        least squares fit of a linear polynomial), ``polyremoval = N`` for `N > 1` 
        subtracts a polynomial of order `N` (``N = 2`` quadratic, ``N = 3`` cubic 
        etc.). If `polyremoval` is `None`, no de-trending is performed. 
    taper : str
        Only valid if `method` is `'mtmfft'` or `'mtmconvol'`. Windowing function, 
        one of :data:`~.availableTapers` (see below).
    tapsmofrq : float
        Only valid if `method` is `'mtmfft'` or `'mtmconvol'`. The amount of spectral 
        smoothing through  multi-tapering (Hz). Note that smoothing frequency 
        specifications are one-sided, i.e., 4 Hz smoothing means plus-minus 4 Hz, 
        i.e., a 8 Hz smoothing box.
    keeptapers : bool
        Only valid if `method` is `'mtmfft'` or `'mtmconvol'`. If `True`, return 
        spectral estimates for each taper, otherwise results are averaged across
        tapers. 
    t_ftimwin : scalar
        Only valid if `method` is `'mtmconvol'`. Sliding window length (in seconds). 
    toi : scalar or array-like or "all"
        **Mandatory input** for time-frequency analysis methods (`method` is either 
        `"mtmconvol"` or `"wavelet"`). 
        If `toi` is scalar, it must be a value between 0 and 1 indicating the 
        percentage of overlap between time-windows specified by `t_ftimwin` (only
        valid if `method` is `'mtmconvol'`, invalid for `'wavelet'`). 
        If `toi` is an array it explicitly selects the centroids of analysis 
        windows (in seconds). If `toi` is `"all"`, analysis windows are centered
        on all samples in the data. 
    width : scalar
        Only valid if `method` is `'wavelet'`. Nondimensional frequency constant 
        of wavelet function. For a Morlet wavelet this number should be >= 6, which 
        corresponds to 6 cycles within the analysis window (FIXME: how many SDs of the Gaussian window?)
    out : None or :class:`SpectralData` object
        None if a new :class:`SpectralData` object is to be created, or an empty :class:`SpectralData` object
        

    Returns
    -------
    spec : :class:`~syncopy.SpectralData`
        (Time-)frequency spectrum of input data
        
    Notes
    -----
    Coming soon...
    
    Examples
    --------
    Coming soon...
        

    .. autodata:: syncopy.specest.freqanalysis.availableMethods

    .. autodata:: syncopy.specest.freqanalysis.availableOutputs

    .. autodata:: syncopy.specest.freqanalysis.availableTapers
    
    See also
    --------
    syncopy.specest.mtmfft.mtmfft : (multi-)tapered Fourier transform of multi-channel time series data
    syncopy.specest.mtmconvol.mtmconvol : time-frequency analysis of multi-channel time series data with a sliding window FFT
    numpy.fft.fft : NumPy's reference FFT implementation
    scipy.signal.stft : SciPy's Short Time Fourier Transform
    """
    
    # Make sure our one mandatory input object can be processed
    try:
        data_parser(data, varname="data", dataclass="AnalogData",
                    writable=None, empty=False)
    except Exception as exc:
        raise exc
    timeAxis = data.dimord.index("time")

    # Get everything of interest in local namespace
    defaults = get_defaults(freqanalysis)
    lcls = locals()

    # Ensure a valid computational method was selected    
    if method not in availableMethods:
        lgl = "'" + "or '".join(opt + "' " for opt in availableMethods)
        raise SPYValueError(legal=lgl, varname="method", actual=method)

    # Ensure a valid output format was selected    
    if output not in spectralConversions.keys():
        lgl = "'" + "or '".join(opt + "' " for opt in spectralConversions.keys())
        raise SPYValueError(legal=lgl, varname="output", actual=output)

    # Parse all Boolean keyword arguments
    for vname in ["keeptrials", "keeptapers"]:
        if not isinstance(lcls[vname], bool):
            raise SPYTypeError(lcls[vname], varname=vname, expected="Bool")
        
    # If only a subset of `data` is to be processed, make some necessary adjustments
    # and compute minimal sample-count across (selected) trials
    if data._selection is not None:
        trialList = data._selection.trials
        sinfo = np.zeros((len(trialList), 2))
        for tk, trlno in enumerate(trialList):
            trl = data._preview_trial(trlno)
            tsel = trl.idx[timeAxis]
            if isinstance(tsel, list):
                sinfo[tk, :] = [0, len(tsel)]
            else:
                sinfo[tk, :] = [trl.idx[timeAxis].start, trl.idx[timeAxis].stop]
    else:
        trialList = list(range(len(data.trials)))
        sinfo = data.sampleinfo
    lenTrials = np.diff(sinfo)
    numTrials = len(trialList)
    
    # Set default padding options: after this, `pad` is either `None`, `False` or `str`
    defaultPadding = {"mtmfft": "nextpow2", "mtmconvol": None, "wavelet": None}
    if pad is None or pad is True:
        pad = defaultPadding[method]

    # Sliding window FFT does not support "fancy" padding
    if method == "mtmconvol" and isinstance(pad, str): 
        msg = "method 'mtmconvol' only supports in-place padding for windows " +\
            "exceeding trial boundaries. Your choice of `pad = '{}'` will be ignored. "
        SPYWarning(msg.format(pad))
        pad = None
        
    # Ensure padding selection makes sense: do not pad on a by-trial basis but 
    # use the longest trial as reference and compute `padlength` from there
    # (only relevant for "global" padding options such as `maxlen` or `nextpow2`)
    if pad:
        if not isinstance(pad, str):
            raise SPYTypeError(pad, varname="pad", expected="str or None")
        if pad == "maxlen":
            padlength = lenTrials.max()
            prepadlength = True
            postpadlength = False
        elif pad == "nextpow2":
            padlength = 0
            for ltrl in lenTrials:
                padlength = max(padlength, _nextpow2(ltrl))
            pad = "absolute"
            prepadlength = True
            postpadlength = False
        padding(data._preview_trial(trialList[0]), padtype, pad=pad, padlength=padlength,
                prepadlength=prepadlength, postpadlength=postpadlength)
    
        # Compute `minSampleNum` accounting for padding
        minSamplePos = lenTrials.argmin()
        minSampleNum = padding(data._preview_trial(trialList[minSamplePos]), padtype, pad=pad,
                               padlength=padlength, prepadlength=True).shape[timeAxis]
    else:
        pad = None
        if method == "mtmfft" and np.unique((np.floor(lenTrials / 2))).size > 1:
            lgl = "trials of approximately equal length for method 'mtmfft'"
            act = "trials of unequal length"
            raise SPYValueError(legal=lgl, varname="data", actual=act)
        minSampleNum = lenTrials.min()
        
    # Compute length (in samples) of shortest trial
    minTrialLength = minSampleNum/data.samplerate
    
    # Basic sanitization of frequency specifications
    if foi is not None:
        if isinstance(foi, str):
            if foi == "all":
                foi = None
            else:
                raise SPYValueError(legal="'all' or `None` or list/array", 
                                    varname="foi", actual=foi)
        else:
            try:
                array_parser(foi, varname="foi", hasinf=False, hasnan=False,
                             lims=[0, data.samplerate/2], dims=(None,))
            except Exception as exc:
                raise exc
    if foilim is not None:
        if isinstance(foilim, str):
            if foilim == "all":
                foilim = None
            else:
                raise SPYValueError(legal="'all' or `None` or `[fmin, fmax]`", 
                                    varname="foilim", actual=foilim)
        else:
            try:
                array_parser(foilim, varname="foilim", hasinf=False, hasnan=False,
                            lims=[0, data.samplerate/2], dims=(2,))
            except Exception as exc:
                raise exc
    if foi is not None and foilim is not None:
        lgl = "either `foi` or `foilim` specification"
        act = "both"
        raise SPYValueError(legal=lgl, varname="foi/foilim", actual=act)
        
    # FIXME: implement detrending
    # see also https://docs.obspy.org/_modules/obspy/signal/detrend.html#polynomial
    if polyremoval is not None:
        raise NotImplementedError("Detrending has not been implemented yet.")
        try:
            scalar_parser(polyremoval, varname="polyremoval", lims=[0, 8], ntype="int_like")
        except Exception as exc:
            raise exc

    # Prepare keyword dict for logging (use `lcls` to get actually provided 
    # keyword values, not defaults set above)
    log_dct = {"method": method,
               "output": output,
               "keeptapers": keeptapers,
               "keeptrials": keeptrials,
               "polyremoval": polyremoval,
               "pad": lcls["pad"],
               "padtype": lcls["padtype"],
               "padlength": lcls["padlength"],
               "foi": lcls["foi"]}
    
    # 1st: Check time-frequency inputs to prepare/sanitize `toi`
    if method in ["mtmconvol", "wavelet"]:
        
        # Get start/end timing info respecting potential in-place selection
        if toi is None:
            raise SPYTypeError(toi, varname="toi", expected="scalar or array-like or 'all'")
        if data._selection is not None:
            tStart = data._selection.trialdefinition[:, 2] / data.samplerate
        else:
            tStart = data._t0 / data.samplerate
        tEnd = tStart + np.diff(sinfo).squeeze() / data.samplerate
        
        # Process `toi`: we have to account for three scenarios: (1) center sliding
        # windows on all samples in (selected) trials (2) `toi` was provided as 
        # percentage indicating the degree of overlap b/w time-windows and (3) a set
        # of discrete time points was provided. These three cases are encoded in 
        # `overlap, i.e., ``overlap > 1` => all, `0 < overlap < 1` => percentage, 
        # `overlap < 0` => discrete `toi`
        if isinstance(toi, str):
            if toi != "all":
                lgl = "`toi = 'all'` to center analysis windows on all time-points"
                raise SPYValueError(legal=lgl, varname="toi", actual=toi)
            overlap = 1.1
            toi = None
            equidistant = True
        elif isinstance(toi, Number):
            if method == "wavelet":
                lgl = "array of time-points wavelets are to be centered on"
                act = "scalar value"
                raise SPYValueError(legal=lgl, varname="toi", actual=act)
            try:
                scalar_parser(toi, varname="toi", lims=[0, 1])
            except Exception as exc:
                raise exc
            overlap = toi
            equidistant = True
        else:
            overlap = -1
            try:
                array_parser(toi, varname="toi", hasinf=False, hasnan=False,
                             lims=[tStart.min(), tEnd.max()], dims=(None,))
            except Exception as exc:
                raise exc
            toi = np.array(toi)
            tSteps = np.diff(toi)
            if (tSteps < 0).any():
                lgl = "ordered list/array of time-points"
                act = "unsorted list/array"
                raise SPYValueError(legal=lgl, varname="toi", actual=act)
            # This is imho a bug in NumPy - even `arange` and `linspace` may produce 
            # arrays that are numerically not exactly equidistant - `unique` will
            # show several entries here - use `allclose` to identify "even" spacings
            equidistant = np.allclose(tSteps, [tSteps[0]] * tSteps.size)

        # If `toi` was 'all' or a percentage, use entire time interval of (selected)
        # trials and check if those trials have *approximately* equal length 
        if toi is None:
            if not np.allclose(lenTrials, [minSampleNum] * lenTrials.size):
                msg = "processing trials of different lengths (min = {}; max = {} samples)" +\
                    " with `toi = 'all'`"
                SPYWarning(msg.format(int(minSampleNum), int(lenTrials.max())))
            if pad is False:
                lgl = "`pad` to be `None` or `True` to permit zero-padding " +\
                    "at trial boundaries to accommodate windows if `0 < toi < 1` " +\
                    "or if `toi` is 'all'"
                act = "False"
                raise SPYValueError(legal=lgl, actual=act, varname="pad")

        # Code recycling: `overlap`, `equidistant` etc. are really only relevant 
        # for `mtmconvol`, but we use padding calc below for `wavelet` as well
        if method == "mtmconvol":
            try:
                scalar_parser(t_ftimwin, varname="t_ftimwin", lims=[1/data.samplerate, minTrialLength])
            except Exception as exc:
                raise exc
        else:        
            t_ftimwin = 0
        nperseg = int(t_ftimwin * data.samplerate)
        minSampleNum = nperseg
        halfWin = int(nperseg / 2)

        # `mtmconvol`: compute no. of samples overlapping across adjacent windows        
        if overlap < 0:         # `toi` is equidistant range or disjoint points
            noverlap = nperseg - int(tSteps[0] * data.samplerate)
        elif 0 <= overlap <= 1: # `toi` is percentage
            noverlap = int(overlap * nperseg)
        else:                   # `toi` is "all"
            noverlap = nperseg - 1
        
        # `toi` is array
        if overlap < 0:
            
            # Compute necessary padding at begin/end of trials to fit sliding windows
            offStart = ((toi[0] - tStart) * data.samplerate).astype(np.intp)
            padBegin = halfWin - offStart
            padBegin = ((padBegin > 0) * padBegin).astype(np.intp)
            
            offEnd = ((tEnd - toi[-1]) * data.samplerate).astype(np.intp)
            padEnd = halfWin - offEnd
            padEnd = ((padEnd > 0) * padEnd).astype(np.intp)
            
            # Abort if padding was explicitly forbidden
            if pad is False and (np.any(padBegin) or np.any(padBegin)):
                lgl = "windows within trial bounds"
                act = "windows exceeding trials no. " +\
                    "".join(str(trlno) + ", "\
                        for trlno in np.array(trialList)[(padBegin + padEnd) > 0])[:-2]
                raise SPYValueError(legal=lgl, varname="pad", actual=act)

            # Compute sample-indices (one slice/list per trial) from time-selections
            soi = []            
            if not equidistant:
                for tk in range(numTrials):
                    starts = (data.samplerate * (toi - tStart[tk]) - halfWin).astype(np.intp)
                    starts += padBegin[tk]
                    stops = (data.samplerate * (toi - tStart[tk]) + halfWin + 1).astype(np.intp)
                    stops += padBegin[tk]
                    stops = np.maximum(stops, stops - starts, dtype=np.intp)
                    # starts = ((starts > 0) * starts).astype(np.intp)
                    soi.append([slice(start, stop) for start, stop in zip(starts, stops)])
            else:
                for tk in range(numTrials):
                    start = int(data.samplerate * (toi[0] - tStart[tk]) - halfWin)
                    stop = int(data.samplerate * (toi[-1] - tStart[tk]) + halfWin + 1)
                    soi.append(slice(max(0, start), max(stop, stop - start)))

        # `toi` is percentage or "all"                    
        else:
            
            padBegin = np.zeros((numTrials,))
            padEnd = np.zeros((numTrials,))
            soi = [slice(None)] * numTrials
                
        import ipdb; ipdb.set_trace()
                    
        # Update `log_dct` w/method-specific options (use `lcls` to get actually
        # provided keyword values, not defaults set in here)
        log_dct["toi"] = lcls["toi"]
        
    # Check options specific to mtm*-methods (particularly tapers and foi/freqs alignment)
    if "mtm" in method:

        # See if taper choice is supported
        if taper not in availableTapers:
            lgl = "'" + "or '".join(opt + "' " for opt in availableTapers)
            raise SPYValueError(legal=lgl, varname="taper", actual=taper)
        taper = getattr(spwin, taper)

        # Advanced usage: see if `taperopt` was provided - if not, leave it empty
        taperopt = kwargs.get("taperopt", {})
        if not isinstance(taperopt, dict):
            raise SPYTypeError(taperopt, varname="taperopt", expected="dictionary")

        # Construct array of maximally attainable frequencies
        nFreq = int(np.floor(minSampleNum / 2) + 1)
        freqs = np.linspace(0, data.samplerate / 2, nFreq)
        
        # Match desired frequencies as close as possible to actually attainable freqs
        if foi is not None:
            foi, _ = best_match(freqs, foi, squash_duplicates=True)
        elif foilim is not None:
            foi, _ = best_match(freqs, foilim, span=True, squash_duplicates=True)
        else:
            foi = freqs
            
        # Abort if desired frequency selection is empty
        if foi.size == 0:
            lgl = "non-empty frequency specification"
            act = "empty frequency selection"
            raise SPYValueError(legal=lgl, varname="foi/foilim", actual=act)
        
        # Set/get `tapsmofrq` if we're working w/Slepian tapers
        if taper.__name__ == "dpss":

            # Try to derive "sane" settings by using 3/4 octave smoothing of highest `foi`
            # following Hill et al. "Oscillatory Synchronization in Large-Scale
            # Cortical Networks Predicts Perception", Neuron, 2011
            if tapsmofrq is None:
                foimax = foi.max()
                tapsmofrq = (foimax * 2**(3/4/2) - foimax * 2**(-3/4/2)) / 2
            else:
                try:
                    scalar_parser(tapsmofrq, varname="tapsmofrq", lims=[1, np.inf])
                except Exception as exc:
                    raise exc
            
            # Get/compute number of tapers to use (at least 1 and max. 50)
            nTaper = taperopt.get("Kmax", 1)
            if not taperopt:
                nTaper = int(max(2, min(50, np.floor(tapsmofrq * minSampleNum * 1 / data.samplerate))))
                taperopt = {"NW": tapsmofrq, "Kmax": nTaper}
                
        else:
            nTaper = 1

        # Warn the user in case `tapsmofrq` has no effect
        if tapsmofrq is not None and taper.__name__ != "dpss":
            msg = "`tapsmofrq` is only used if `taper` is `dpss`!"
            SPYWarning(msg)
            
        # Update `log_dct` w/method-specific options (use `lcls` to get actually
        # provided keyword values, not defaults set in here)
        log_dct["taper"] = lcls["taper"]
        log_dct["tapsmofrq"] = lcls["tapsmofrq"]
        log_dct["nTaper"] = nTaper
        
        # Check for non-default values of options not supported by chosen method
        kwdict = {"wav": wav, "width": width}
        for name, kwarg in kwdict.items():
            if kwarg is not lcls[name]:
                msg = "option `{}` has no effect in methods `mtmfft` and `mtmconvol`!"
                SPYWarning(msg.format(name))
            
    # Now, prepare explicit compute-classes for chosen method
    if method == "mtmfft":
        
        # Check for non-default values of options not supported by chosen method
        kwdict = {"t_ftimwin": t_ftimwin, "toi": toi}
        for name, kwarg in kwdict.items():
            if kwarg is not lcls[name]:
                msg = "option `{}` has no effect in method `mtmfft`!"
                SPYWarning(msg.format(name))
                
        # Set up compute-class
        specestMethod = MultiTaperFFT(
            dt=1/data.samplerate,
            nTaper=nTaper, 
            timeAxis=timeAxis, 
            taper=taper, 
            taperopt=taperopt,
            tapsmofrq=tapsmofrq,
            pad=pad,
            padtype=padtype,
            padlength=padlength,
            foi=foi,
            keeptapers=keeptapers,
            polyremoval=polyremoval,
            output_fmt=output)
        
    elif method == "mtmconvol":

        # Set up compute-class
        specestMethod = MultiTaperFFTConvol(
            soi,
            list(padBegin),
            list(padEnd),
            samplerate=data.samplerate,
            noverlap=noverlap,
            nperseg=nperseg,
            equidistant=equidistant,
            toi=toi,
            foi=foi,
            nTaper=nTaper, 
            timeAxis=timeAxis, 
            taper=taper, 
            taperopt=taperopt,
            pad=pad,
            padtype=padtype,
            padlength=padlength,
            prepadlength=prepadlength,
            postpadlength=postpadlength,
            keeptapers=keeptapers,
            polyremoval=polyremoval,
            output_fmt=output)

    elif method == "wavelet":
        pass

        # check if taper, tapsmofrq, keeptapers, t_ftimwin (set to 0 above) is defined 
        
        # check for consistency of width, wav
        
        options = ["Morlet", "Paul", "DOG", "Ricker", "Marr", "Mexican_hat"]
        if wav not in options:
            lgl = "'" + "or '".join(opt + "' " for opt in options)
            raise SPYValueError(legal=lgl, varname="wav", actual=wav)
        wav = getattr(spywave, wav)

        if isinstance(toi, Number):
            try:
                scalar_parser(toi, varname="toi", lims=[0, 1])
            except Exception as exc:
                raise exc
        else:
            try:
                array_parser(toi, varname="toi", hasinf=False, hasnan=False,
                             lims=[timing.min(), timing.max()], dims=(None,))
            except Exception as exc:
                raise exc
            toi = np.array(toi)
            toi.sort()

        if foi is None:
            foi = 1 / _get_optimal_wavelet_scales(minTrialLength,
                                                  1/data.samplerate,
                                                  dj=0.25)

        # FIXME: width setting depends on chosen wavelet
        if width is not None:
            try:
                scalar_parser(width, varname="width", lims=[1, np.inf])
            except Exception as exc:
                raise exc

        # Update `log_dct` w/method-specific options (use `lcls` to get actually
        # provided keyword values, not defaults set in here)
        log_dct["wav"] = lcls["wav"]
        log_dct["toi"] = lcls["toi"]
        log_dct["width"] = lcls["width"]

        # Set up compute-class
        specestMethod = WaveletTransform(1/data.samplerate, 
                                         timeAxis,
                                         foi,
                                         toi=toi,
                                         polyremoval=polyremoval,
                                         wav=wav,
                                         width=width,
                                         output_fmt=output)
        
    # If provided, make sure output object is appropriate
    if out is not None:
        try:
            data_parser(out, varname="out", writable=True, empty=True,
                        dataclass="SpectralData",
                        dimord=SpectralData().dimord)
        except Exception as exc:
            raise exc
        new_out = False
    else:
        out = SpectralData(dimord=SpectralData._defaultDimord)        
        new_out = True

    # Perform actual computation
    specestMethod.initialize(data, 
                             chan_per_worker=kwargs.get("chan_per_worker"),
                             keeptrials=keeptrials)
    specestMethod.compute(data, out, parallel=kwargs.get("parallel"), log_dict=log_dct)

    # Either return newly created output container or simply quit
    return out if new_out else None
