# -*- coding: utf-8 -*-
#
# Syncopy spectral estimation methods
#

# Builtin/3rd party package imports
from numbers import Number
import numpy as np

# Syncopy imports
from syncopy.shared.parsers import data_parser, scalar_parser, array_parser 
from syncopy.shared.tools import get_defaults
from syncopy.datatype import SpectralData, padding
from syncopy.datatype.methods.padding import _nextpow2
from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYWarning, SPYInfo
from syncopy.shared.kwarg_decorators import (unwrap_cfg, unwrap_select,
                                             detect_parallel_client)
from syncopy.shared.tools import best_match

# method specific imports
import syncopy.specest.wavelets as spywave
import syncopy.specest.superlet as superlet
from .wavelet import get_optimal_wavelet_scales

# Local imports
from .const_def import (
    spectralConversions,
    availableTapers,
    availableWavelets,
    availableMethods,
    generalParameters
)

from .compRoutines import (
    SuperletTransform,
    WaveletTransform,
    MultiTaperFFT,
    MultiTaperFFTConvol
)

__all__ = ["freqanalysis"]


# @unwrap_cfg
# @unwrap_select
# @detect_parallel_client
def freqanalysis(data, method='mtmfft', output='fourier',
                 keeptrials=True, foi=None, foilim=None, pad=None, padtype='zero',
                 padlength=None, prepadlength=None, postpadlength=None,
                 polyremoval=None,
                 taper="hann", tapsmofrq=None, nTaper=None, keeptapers=False,
                 toi="all", t_ftimwin=None, wavelet="Morlet", width=6, order=None,
                 order_max=None, order_min=1, c_1=3, adaptive=False,
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
        * **tapsmofrq** : spectral smoothing box for slepian tapers (in Hz)
        * **nTaper** : number of orthogonal tapers for slepian tapers
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
        * **tapsmofrq** : spectral smoothing box for slepian tapers (in Hz)
        * **nTaper** : number of orthogonal tapers for slepian tapers
        * **keeptapers** : return individual tapers or average
        * **pad** : flag indicating, whether or not to pad trials. If `None`,
          trials are padded only if sliding window centroids are too close
          to trial boundaries for the entire window to cover available data-points.
        * **toi** : time-points of interest; can be either an array representing
          analysis window centroids (in sec), a scalar between 0 and 1 encoding
          the percentage of overlap between adjacent windows or "all" to center
          a window on every sample in the data.
        * **t_ftimwin** : sliding window length (in sec)

    :func:`~syncopy.specest.wavelet.wavelet` : (Continuous non-orthogonal) wavelet transform
        Perform time-frequency analysis on time-series trial data using a non-orthogonal
        continuous wavelet transform.

        * **wav** : one of :data:`~.availableWavelets`
        * **toi** : time-points of interest; can be either an array representing
          time points (in sec) to center wavelets on or "all" to center a wavelet
          on every sample in the data. #FIXME: not correct: toi only affects pre-trimming 
          and subsampling of results!
        * **width** : Nondimensional frequency constant of Morlet wavelet function (>= 6)
        * **order** : Order of Paul wavelet function (>= 4) or derivative order
          of real-valued DOG wavelets (2 = mexican hat)

    :func:`~syncopy.specest.superlet.superlet` : Superlet transform
        Perform time-frequency analysis on time-series trial data using
        the super-resolution superlet transform (SLT) from [Moca2021]_.

        * **order_max** : Maximal order of the superlet

        * **order_min** : Minimal order of the superlet

        * **c_1** : Number of cycles of the base Morlet wavelet

        * **adaptive** : If set to `True` perform fractional adaptive SLT,
          otherwise perform multiplicative SLT

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
        the sliding window time-frequency analysis method (`'mtmconvol'`), only performs
        padding if necessary, i.e., if time-window centroids are chosen too close
        to trial boundaries for the entire window to cover available data-points.
        If `pad` is `False`, then no padding is performed. Then in case of
        ``method = 'mtmfft'`` all trials have to have approximately the same
        length (up to the next even sample-count), if ``method = 'mtmconvol'``, 
        window-centroids have to keep sufficient
        distance from trial boundaries. For more details on the padding methods
        `'absolute'`, `'relative'`, `'maxlen'` and `'nextpow2'` see :func:`syncopy.padding`.
    padtype : str
        Values to be used for padding. Can be `'zero'`, `'nan'`, `'mean'`,
        `'localmean'`, `'edge'` or `'mirror'`. See :func:`syncopy.padding` for
        more information. Only valid for method `'mtmfft'`.
    padlength : None, bool or positive int
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
        Only valid if `method` is `'mtmfft'` or `'mtmconvol'` and `taper` is `'dpss'`. 
        The amount of spectral smoothing through  multi-tapering (Hz). 
        Note that smoothing frequency specifications are one-sided, 
        i.e., 4 Hz smoothing means plus-minus 4 Hz, i.e., a 8 Hz smoothing box.
    nTaper : int
        Only valid if `method` is `'mtmfft'` or `'mtmconvol'` and `taper='dpss'`. 
        Number of orthogonal tapers to use.    
    keeptapers : bool
        Only valid if `method` is `'mtmfft'` or `'mtmconvol'`. 
        If `True`, return spectral estimates for each taper. 
        Otherwise power spectrum is averaged across tapers, 
        if and only if `output` is `pow`.
    toi : float or array-like or "all"
        **Mandatory input** for time-frequency analysis methods (`method` is either
        `"mtmconvol"` or `"wavelet"` or `"superlet"`).
        If `toi` is scalar, it must be a value between 0 and 1 indicating the
        percentage of overlap between time-windows specified by `t_ftimwin` (only
        valid if `method` is `'mtmconvol'`).
        If `toi` is an array it explicitly selects the centroids of analysis
        windows (in seconds), if `toi` is `"all"`, analysis windows are centered
        on all samples in the data for `method="mtmconvol"`. For wavelet based
        methods (`"wavelet"` or `"superlet"`) toi needs to be either an 
        equidistant array of time points or "all".
    t_ftimwin : positive float
        Only valid if `method` is `'mtmconvol'`. Sliding window length (in seconds).
    wav : str
        Only valid if `method` is `'wavelet'`. Wavelet function to use, one of
        :data:`~.availableWavelets` (see below).
    width : positive float
        Only valid if `method` is `'wavelet'` and `wav` is `'Morlet'`. Nondimensional
        frequency constant of Morlet wavelet function. This number should be >= 6,
        which corresponds to 6 cycles within the analysis window to ensure sufficient
        spectral sampling.
    order : positive int
        Only valid if `method` is `'wavelet'` and `wav` is `'Paul'` or `'DOG'`. Order
        of the wavelet function. If `wav` is `'Paul'`, `order` should be chosen
        >= 4 to ensure that the analysis window contains at least a single oscillation.
        At an order of 40, the Paul wavelet  exhibits about the same number of cycles
        as the Morlet wavelet with a `width` of 6.
        All other supported wavelets functions are *real-valued* derivatives of
        Gaussians (DOGs). Hence, if `wav` is `'DOG'`, `order` represents the derivative order.
        The special case of a second order DOG yields a function known as "Mexican Hat",
        "Marr" or "Ricker" wavelet, which can be selected alternatively by setting
        `wav` to `'Mexican_hat'`, `'Marr'` or `'Ricker'`. **Note**: A real-valued
        wavelet function encodes *only* information about peaks and discontinuities
        in the signal and does *not* provide any information about amplitude or phase.
    order_max : int
        Only valid if `method` is `'superlet'`.
        Maximal order of the superlet set. Controls the maximum
        number of cycles within a SL together
        with the `c_1` parameter: c_max = c_1 * order_max
    order_min : int
        Only valid if `method` is `'superlet'`.
        Minimal order of the superlet set. Controls
        the minimal number of cycles within a SL together
        with the `c_1` parameter: c_min = c_1 * order_min
        Note that for admissability reasons c_min should be at least 3!
    c_1 : int
        Only valid if `method` is `'superlet'`.
        Number of cycles of the base Morlet wavelet. If set to lower
        than 3 increase `order_min` as to never have less than 3 cycles
        in a wavelet!
    adaptive : bool
        Only valid if `method` is `'superlet'`.
        Wether to perform multiplicative SLT or fractional adaptive SLT.
        If set to True, the order of the wavelet set will increase
        linearly with the frequencies of interest from `order_min`
        to `order_max`. If set to False the same SL will be used for
        all frequencies.
    out : None or :class:`SpectralData` object
        None if a new :class:`SpectralData` object is to be created, or an empty :class:`SpectralData` object


    Returns
    -------
    spec : :class:`~syncopy.SpectralData`
        (Time-)frequency spectrum of input data

    Notes
    -----
    .. [Moca2021] Moca, Vasile V., et al. "Time-frequency super-resolution with superlets."
       Nature communications 12.1 (2021): 1-18.


    Examples
    --------
    Coming soon...


    .. autodata:: syncopy.specest.freqanalysis.availableMethods

    .. autodata:: syncopy.specest.freqanalysis.availableOutputs

    .. autodata:: syncopy.specest.freqanalysis.availableTapers

    .. autodata:: syncopy.specest.freqanalysis.availableWavelets

    See also
    --------
    syncopy.specest.mtmfft.mtmfft : (multi-)tapered Fourier transform of multi-channel time series data
    syncopy.specest.mtmconvol.mtmconvol : time-frequency analysis of multi-channel time series data with a sliding window FFT
    syncopy.specest.wavelet.wavelet : time-frequency analysis of multi-channel time series data using a wavelet transform
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
    lenTrials = np.diff(sinfo).squeeze()
    numTrials = len(trialList)

    # Set default padding options: after this, `pad` is either `None`, `False` or `str`
    defaultPadding = {"mtmfft": "nextpow2",
                      "mtmconvol": None,
                      "wavelet": None,
                      "superlet" : None}
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
    # (only relevant for "global" padding options such as `maxlen` or
    # `nextpow2`)for mtmfft method
    if method == 'mtmfft':
        # pad was None or True or a string...
        if pad:
            if not isinstance(pad, str):
                raise SPYTypeError(pad, varname="pad", expected="str or None")            
            if pad == "maxlen": # FIXME: this is not working, bug in padding()?!
                if padlength:
                    msg = f'option `padlength` has no effect for pad set to {pad}'
                padlength = lenTrials.max()
            elif pad == "nextpow2":
                if padlength:
                    msg = f'option `padlength` has no effect for pad set to {pad}'       
                padlength = 0
                for ltrl in lenTrials:
                    padlength = max(padlength, _nextpow2(ltrl))
                pad = "absolute"
            padding(data._preview_trial(trialList[0]), padtype, pad=pad, padlength=padlength,
                    prepadlength=prepadlength, postpadlength=postpadlength)

            # Compute `minSampleNum` accounting for padding
            minSamplePos = lenTrials.argmin()
            minSampleNum = padding(data._preview_trial(trialList[minSamplePos]), padtype, pad=pad,
                                   padlength=padlength, prepadlength=True).shape[timeAxis]
            
        elif np.unique((np.floor(lenTrials / 2))).size > 1:
            lgl = "trials of approximately equal length for method 'mtmfft' or set pad to True"
            act = "trials of unequal length"
            raise SPYValueError(legal=lgl, varname="data", actual=act)
        else:
            minSampleNum = int(lenTrials.min())
            
    # no manual padding for other methods atm
    else:
        minSampleNum = lenTrials.min()

    # Compute length (in samples) of shortest trial
    minTrialLength = minSampleNum / data.samplerate

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
            foi = np.array(foi, dtype="float")
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
            # foilim is of shape (2,)
            if foilim[0] > foilim[1]:
                msg = "Sorting foilim low to high.."
                SPYInfo(msg)
                foilim = np.sort(foilim)
            
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

    # --------------------------------
    # 1st: Check time-frequency inputs
    # to prepare/sanitize `toi` 
    # --------------------------------
    
    if method in ["mtmconvol", "wavelet", "superlet"]:

        # Get start/end timing info respecting potential in-place selection
        if toi is None:
            raise SPYTypeError(toi, varname="toi", expected="scalar or array-like or 'all'")
        if data._selection is not None:
            tStart = data._selection.trialdefinition[:, 2] / data.samplerate
        else:
            tStart = data._t0 / data.samplerate
        tEnd = tStart + lenTrials / data.samplerate

        
    # for these methods only 'all' or an equidistant array
    # of time points (sub-sampling, trimming) are valid
    if method in ["wavelet", "superlet"]:

        valid = True
        if isinstance(toi, Number):
            valid = False

        elif isinstance(toi, str):
            if toi != "all":
                valid = False
            else:
                # take everything
                preSelect = [slice(None)] * numTrials
                postSelect = [slice(None)] * numTrials
                                
        elif not iter(toi):
            valid = False

        # this is the sequence type            
        else:
            toi = np.array(toi)
            # catch non-numeric sequence
            if not np.issubdtype(toi.dtype, np.number):
                valid = False
            # check for equidistancy
            elif not np.allclose(np.diff(toi, 2), np.zeros(len(toi) - 2)):
                valid = False
            # trim (preSelect) and subsample output (postSelect)
            else:
                preSelect = []
                postSelect = []                
                # get sample intervals and relative indices from toi
                for tk in range(numTrials):
                    start = int(data.samplerate * (toi[0] - tStart[tk]))
                    stop = int(data.samplerate * (toi[-1] - tStart[tk]) + 1)
                    preSelect.append(slice(max(0, start), max(stop, stop - start)))
                    smpIdx = np.minimum(lenTrials[tk] - 1,
                                        data.samplerate * (toi - tStart[tk]) - start)
                    postSelect.append(smpIdx.astype(np.intp))
                    
        # get out if sth wasn't right
        if not valid:
            lgl = "array of equidistant time-points or 'all' for wavelet based methods"
            raise SPYValueError(legal=lgl, varname="toi", actual=toi)

    # Process `toi` for sliding window multi taper fft,
    # we have to account for three scenarios: (1) center sliding
    # windows on all samples in (selected) trials (2) `toi` was provided as
    # percentage indicating the degree of overlap b/w time-windows and (3) a set
    # of discrete time points was provided. These three cases are encoded in
    # `overlap, i.e., ``overlap > 1` => all, `0 < overlap < 1` => percentage,
    # `overlap < 0` => discrete `toi`

    elif method == "mtmconvol":

        # overlap = None
        if isinstance(toi, str):
            if toi != "all":
                lgl = "`toi = 'all'` to center analysis windows on all time-points"
                raise SPYValueError(legal=lgl, varname="toi", actual=toi)
            equidistant = True
            
        elif isinstance(toi, Number):
            try:
                scalar_parser(toi, varname="toi", lims=[0, 1])
            except Exception as exc:
                raise exc
            overlap = toi
            equidistant = True
        # this captures all other cases, e.i. toi is of sequence type
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

        # get the sliding window size
        try:
            scalar_parser(t_ftimwin, varname="t_ftimwin", lims=[1 / data.samplerate, minTrialLength])
        except Exception as exc:
            raise exc
        
        nperseg = int(t_ftimwin * data.samplerate)
        minSampleNum = nperseg
        halfWin = int(nperseg / 2)

        # `mtmconvol`: compute no. of samples overlapping across adjacent windows
        if overlap < 0:         # `toi` is equidistant range or disjoint points
            noverlap = nperseg - max(1, int(tSteps[0] * data.samplerate))
        elif 0 <= overlap <= 1: # `toi` is percentage
            noverlap = min(nperseg - 1, int(overlap * nperseg))
        else:                   # `toi` is "all"
            noverlap = nperseg - 1

        # `toi` is array
        if overlap < 0:
            print('AAA' * 5)
            # Compute necessary padding at begin/end of trials to fit sliding windows
            offStart = ((toi[0] - tStart) * data.samplerate).astype(np.intp)
            print(offStart, halfWin)
            padBegin = halfWin - offStart
            print('A', padBegin)
            padBegin = ((padBegin > 0) * padBegin).astype(np.intp)
            print('B', padBegin)
            offEnd = ((tEnd - toi[-1]) * data.samplerate).astype(np.intp)
            padEnd = halfWin - offEnd
            print('A', padEnd)            
            padEnd = ((padEnd > 0) * padEnd).astype(np.intp)
            print('B', padEnd)
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
                    soi.append([slice(start, stop) for start, stop in zip(starts, stops)])
                print('Soi:', soi)

            else:
                for tk in range(numTrials):
                    start = int(data.samplerate * (toi[0] - tStart[tk]) - halfWin)
                    stop = int(data.samplerate * (toi[-1] - tStart[tk]) + halfWin + 1)
                    soi.append(slice(max(0, start), max(stop, stop - start)))
                print('Soi:', soi)
        # `toi` is percentage or "all"
        else:

            padBegin = np.zeros((numTrials,))
            padEnd = np.zeros((numTrials,))
            soi = [slice(None)] * numTrials

        # Update `log_dct` w/method-specific options (use `lcls` to get actually
        # provided keyword values, not defaults set in here)
        log_dct["toi"] = lcls["toi"]
    
    # --------------------------------------------
    # Check options specific to mtm*-methods
    # (particularly tapers and foi/freqs alignment)
    # --------------------------------------------
    
    if "mtm" in method:
        
        # Construct array of maximally attainable frequencies
        freqs = np.fft.rfftfreq(minSampleNum, 1 / data.samplerate)

        # Match desired frequencies as close as possible to
        # actually attainable freqs
        # What happens if padding is applied later?!
        if foi is not None:
            foi, _ = best_match(freqs, foi, squash_duplicates=True)
        elif foilim is not None:
            foi, _ = best_match(freqs, foilim, span=True, squash_duplicates=True)
        else:
            msg = (f"Automatic FFT frequency selection from {freqs[0]:.1f}Hz to " 
                   f"{freqs[-1]:.1f}Hz")
            SPYInfo(msg)
            foi = freqs

        # Abort if desired frequency selection is empty
        if foi.size == 0:
            lgl = "non-empty frequency specification"
            act = "empty frequency selection"
            raise SPYValueError(legal=lgl, varname="foi/foilim", actual=act)

        # See if taper choice is supported
        if taper not in availableTapers:
            lgl = "'" + "or '".join(opt + "' " for opt in availableTapers)
            raise SPYValueError(legal=lgl, varname="taper", actual=taper)

        # Warn user about DPSS only settings
        if taper != "dpss":     
            if tapsmofrq is not None:
                msg = "`tapsmofrq` is only used if `taper` is `dpss`!"
                SPYWarning(msg)
            if nTaper is not None:
                msg = "`nTaper` is only used if `taper` is `dpss`!"
                SPYWarning(msg)
            if keeptapers:
                msg = "`keeptapers` is only used if `taper` is `dpss`!"
                SPYWarning(msg)            
                    
        # Set/get `tapsmofrq` if we're working w/Slepian tapers
        if taper == "dpss":

            # direct mtm estimate (averaging) only valid for spectral power
            if not keeptapers and output != "pow":
                lgl = "'pow', the only option for taper averaging"
                raise SPYValueError(legal=lgl, varname="output", actual=output)
            
            # Try to derive "sane" settings by using 3/4 octave
            # smoothing of highest `foi`
            # following Hill et al. "Oscillatory Synchronization in Large-Scale
            # Cortical Networks Predicts Perception", Neuron, 2011
            if tapsmofrq is None:
                foimax = foi.max()
                tapsmofrq = (foimax * 2**(3/4/2) - foimax * 2**(-3/4/2)) / 2
                msg = f'Automatic setting of `tapsmofreq` to {tapsmofrq:.2f}'
                SPYInfo(msg)
                
            else:
                try:
                    scalar_parser(tapsmofrq, varname="tapsmofrq", lims=[1, np.inf])
                except Exception as exc:
                    raise exc
            
            # Get/compute number of tapers to use (at least 1 and max. 50)
            if not nTaper:
                nTaper = int(max(2, min(50, np.floor(tapsmofrq * minSampleNum * 1 / data.samplerate))))
                msg = f'Automatic setting of `nTaper` to {nTaper}'
                SPYInfo(msg)
            else:
                try:
                    scalar_parser(nTaper,
                                  varname="nTaper",
                                  ntype="int_like", lims=[1, np.inf])
                except Exception as exc:
                    raise exc
                
        # only taper with frontend supported options is DPSS                            
        else:
            nTaper = 1
                                    
        # Update `log_dct` w/method-specific options (use `lcls` to get actually
        # provided keyword values, not defaults set in here)
        log_dct["taper"] = lcls["taper"]
        log_dct["tapsmofrq"] = lcls["tapsmofrq"]

    # -------------------------------------------------------
    # Now, prepare explicit compute-classes for chosen method
    # -------------------------------------------------------

    if method == "mtmfft":

        _check_effective_parameters(MultiTaperFFT, defaults, lcls)

        # method specific parameters
        method_kwargs = {
            'samplerate' : data.samplerate,
            'taper' : taper
        }
        if taper == 'dpss':
            method_kwargs['nTaper'] = nTaper
            method_kwargs['tapsmofrq'] = tapsmofrq
                    
        # Set up compute-class
        specestMethod = MultiTaperFFT(
            samplerate=data.samplerate,
            foi=foi,
            timeAxis=timeAxis,
            pad=pad,
            padtype=padtype,
            padlength=padlength,
            keeptapers=keeptapers,
            polyremoval=polyremoval,
            output_fmt=output,
            method_kwargs=method_kwargs)

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

        _check_effective_parameters(WaveletTransform, defaults, lcls)
        
        # Check wavelet selection
        if wavelet not in availableWavelets:
            lgl = "'" + "or '".join(opt + "' " for opt in availableWavelets)
            raise SPYValueError(legal=lgl, varname="wavelet", actual=wavelet)
        if wavelet not in ["Morlet", "Paul"]:
            msg = "the chosen wavelet '{}' is real-valued and does not provide " +\
                "any information about amplitude or phase of the data. This wavelet function " +\
                "may be used to isolate peaks or discontinuities in the signal. "
            SPYWarning(msg.format(wavelet))

        # Check for consistency of `width`, `order` and `wavelet`
        if wavelet == "Morlet":
            try:
                scalar_parser(width, varname="width", lims=[1, np.inf])
            except Exception as exc:
                raise exc
            wfun = getattr(spywave, wavelet)(w0=width)
        else:
            if width != lcls["width"]:
                msg = "option `width` has no effect for wavelet '{}'"
                SPYWarning(msg.format(wavelet))

        if wavelet == "Paul":
            try:
                scalar_parser(order, varname="order", lims=[4, np.inf], ntype="int_like")
            except Exception as exc:
                raise exc
            wfun = getattr(spywave, wavelet)(m=order)
        elif wavelet == "DOG":
            try:
                scalar_parser(order, varname="order", lims=[1, np.inf], ntype="int_like")
            except Exception as exc:
                raise exc
            wfun = getattr(spywave, wavelet)(m=order)
        else:
            if order is not None:
                msg = "option `order` has no effect for wavelet '{}'"
                SPYWarning(msg.format(wavelet))
            wfun = getattr(spywave, wavelet)()

        # Process frequency selection (`toi` was taken care of above): `foilim`
        # selections are wrapped into `foi` thus the seemingly weird if construct
        # first argument
        if foi is None:
            scales = get_optimal_wavelet_scales(
                wfun.scale_from_period, # all availableWavelets sport one!
                int(minTrialLength * data.samplerate),
                1 / data.samplerate)
        if foilim is not None:
            foi = np.arange(foilim[0], foilim[1] + 1)
        if foi is not None:
            foi[foi < 0.01] = 0.01
            scales = wfun.scale_from_period(1 / foi)

        # Update `log_dct` w/method-specific options (use `lcls` to get actually
        # provided keyword values, not defaults set in here)
        log_dct["wavelet"] = lcls["wavelet"]
        log_dct["width"] = lcls["width"]
        log_dct["order"] = lcls["order"]
        
        # method specific parameters
        method_kwargs = {
            'samplerate' : data.samplerate,
            'scales' : scales,
            'wavelet' : wfun
        }

        # Set up compute-class
        specestMethod = WaveletTransform(
            preSelect,
            postSelect,
            # list(padBegin),
            # list(padEnd),
            toi=toi,
            timeAxis=timeAxis,
            polyremoval=polyremoval,
            output_fmt=output,
            method_kwargs=method_kwargs)

    elif method == "superlet":
        
        _check_effective_parameters(SuperletTransform, defaults, lcls)
        
        # check and parse superlet specific arguments
        if order_max is None:
            lgl = "Positive integer needed for order_max"
            raise SPYValueError(legal=lgl, varname="order_max",
                                actual=None)
        else:
            scalar_parser(
                order_max,
                varname="order_max",
                lims=[1, np.inf],
                ntype="int_like"
            )

        scalar_parser(
            order_min, varname="order_min",
            lims=[1, order_max],
            ntype="int_like"
        )
        scalar_parser(c_1, varname="c_1", lims=[1, np.inf], ntype="int_like")

        # if no frequencies are user selected, take a sensitive default
        if foi is None and foilim is None:
            scales = get_optimal_wavelet_scales(
                superlet.scale_from_period, 
                int(minTrialLength * data.samplerate), 
                1 / data.samplerate)

        if foi is not None:
            scales = superlet.scale_from_period(1 / foi)

        # frequency range in 1Hz steps
        elif foilim is not None:
            foi = np.arange(foilim[0], foilim[1] + 1)
            scales = superlet.scale_from_period(1 / foi)

        # FASLT needs ordered frequencies low - high
        # meaning the scales have to go high - low
        if adaptive:
            if len(scales) < 2:
                lgl = "A range of frequencies"
                act = "Single frequency"
                raise SPYValueError(legal=lgl, varname="foi", actual=act)
            if np.any(np.diff(scales) > 0):
                msg = "Sorting frequencies low to high for adaptive SLT.." 
                SPYWarning(msg)
                scales = np.sort(scales)[::-1]
            
        log_dct["c_1"] = lcls["c_1"]
        log_dct["order_max"] = lcls["order_max"]
        log_dct["order_min"] = lcls["order_min"]

        # method specific parameters
        method_kwargs = {
            'samplerate' : data.samplerate,
            'scales' : scales,
            'order_max' : order_max,
            'order_min' : order_min,
            'c_1' : c_1,
            'adaptive' : adaptive
        }

        # Set up compute-class
        specestMethod = SuperletTransform(
            preSelect,
            postSelect,
            # list(padBegin),
            # list(padEnd),
            toi=toi,
            timeAxis=timeAxis,
            output_fmt=output,
            method_kwargs=method_kwargs)

    # -------------------------------------------------
    # Sanitize output and call the ComputationalRoutine
    # -------------------------------------------------

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

    # Either return newly created output object or simply quit
    return out if new_out else None


def _check_effective_parameters(CR, defaults, lcls):

    '''
    For a given ComputationalRoutine, compare set parameters
    (*lcls*) with the accepted parameters and the *defaults*
    to warn if any ineffective parameters are set.
    
    #FIXME: If general structure of this function proofs 
    useful for all CRs/syncopy in general,
    probably best to move this to syncopy.shared.tools

    Parameters
    ----------

    CR : :class:`~syncopy.shared.computational_routine.ComputationalRoutine
    defaults : dict
        Result of :func:`~syncopy.shared.tools.get_defaults`, the function
        parameter names plus values with default values
    lcls : dict
        Result of `locals()`, all names and values of the local name space
    '''
    # list of possible parameter names of the CR
    expected = CR.method_keys + CR.cF_keys
    relevant = [name for name in defaults if name not in generalParameters]
    for name in relevant:                
        if name not in expected and (lcls[name] != defaults[name]):  
            msg = f"option `{name}` has no effect in method `{CR.method}`!"
            SPYWarning(msg, caller=__name__.split('.')[-1])
