# -*- coding: utf-8 -*-
#
# Syncopy spectral estimation methods
#

# Builtin/3rd party package imports
import numpy as np

# Syncopy imports
from syncopy.shared.parsers import data_parser, scalar_parser, array_parser
from syncopy.shared.tools import get_defaults, get_frontend_cfg
from syncopy.datatype import SpectralData
from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYWarning, SPYInfo
from syncopy.shared.kwarg_decorators import (unwrap_cfg, unwrap_select,
                                             detect_parallel_client)
from syncopy.shared.tools import best_match
from syncopy.shared.const_def import spectralConversions
import syncopy as spy

from syncopy.shared.input_processors import (
    process_taper,
    process_foi,
    process_padding,
    check_effective_parameters,
    check_passed_kwargs
)

# method specific imports - they should go!
from syncopy.specest.fooofspy import default_fooof_opt
import syncopy.specest.wavelets as spywave
import syncopy.specest.superlet as superlet
from .wavelet import get_optimal_wavelet_scales

# Local imports

from .compRoutines import (
    SuperletTransform,
    WaveletTransform,
    MultiTaperFFT,
    MultiTaperFFTConvol,
    FooofSpy
)

availableFooofOutputs = ['fooof', 'fooof_aperiodic', 'fooof_peaks']
availableOutputs = tuple(spectralConversions.keys())
availableWavelets = ("Morlet", "Paul", "DOG", "Ricker", "Marr", "Mexican_hat")
availableMethods = ("mtmfft", "mtmconvol", "wavelet", "superlet", "welch")


@unwrap_cfg
@unwrap_select
@detect_parallel_client
def freqanalysis(data, method='mtmfft', output='pow',
                 keeptrials=True, foi=None, foilim=None,
                 pad='maxperlen', polyremoval=0, taper="hann", demean_taper=False,
                 taper_opt=None, tapsmofrq=None, nTaper=None, keeptapers=False,
                 toi="all", t_ftimwin=None, wavelet="Morlet", width=6, order=None,
                 order_max=None, order_min=1, c_1=3, adaptive=False,
                 out=None, fooof_opt=None, **kwargs):
    """
    Perform (time-)frequency analysis of Syncopy :class:`~syncopy.AnalogData` objects

    **Usage Summary**

    Options available in all analysis methods:

    * **output** : one of :data:`~syncopy.specest.freqanalysis.availableOutputs`;
      return power spectra, complex Fourier spectra or absolute values.
    * **foi**/**foilim** : frequencies of interest; either array of frequencies or
      frequency window (not both)
    * **keeptrials** : return individual trials or grand average
    * **polyremoval** : de-trending method to use (0 = mean, 1 = linear or `None`)

    List of available analysis methods and respective distinct options:

    "mtmfft" : (Multi-)tapered Fourier transform
        Perform frequency analysis on time-series trial data using either a single
        taper window (Hanning) or many tapers based on the discrete prolate
        spheroidal sequence (DPSS) that maximize energy concentration in the main
        lobe.

        * **taper** : one of :data:`~syncopy.shared.const_def.availableTapers`
        * **tapsmofrq** : spectral smoothing box for slepian tapers (in Hz)
        * **nTaper** : number of orthogonal tapers for slepian tapers
        * **keeptapers** : return individual tapers or average
        * **pad**: either pad to an absolute length or set to `'nextpow2'`

        Post-processing of the resulting spectra with FOOOF is available
        via setting `output` to one of `'fooof'`, `'fooof_aperiodic'` or
        `'fooof_peaks'`, see below for details. The returned spectrum represents
        the full fooofed spectrum for `'fooof'`, the aperiodic
        fit for `'fooof_aperiodic'`, and the peaks (Gaussians fit to them) for
        `'fooof_peaks'`. Returned data is in linear scale. Noisy input
        data will most likely lead to fitting issues with fooof, always inspect
        your results!

    "mtmconvol" : (Multi-)tapered sliding window Fourier transform
        Perform time-frequency analysis on time-series trial data based on a sliding
        window short-time Fourier transform using either a single Hanning taper or
        multiple DPSS tapers.

        * **taper** : one of :data:`~syncopy.shared.const_def.availableTapers`
        * **tapsmofrq** : spectral smoothing box for slepian tapers (in Hz)
        * **nTaper** : number of orthogonal tapers for slepian tapers
        * **keeptapers** : return individual tapers or average
        * **toi** : time-points of interest; can be either an array representing
          analysis window centroids (in sec), a scalar between 0 and 1 encoding
          the percentage of overlap between adjacent windows or "all" to center
          a window on every sample in the data.
        * **t_ftimwin** : sliding window length (in sec)

    "welch" : Welch's method for the estimation of power spectra based on
        time-averaging over short, modified periodograms. Here, *modified* means that
        a taper is applied.
        See [Welch1967]_ for details.

        * **toi** : time-points of interest; a scalar between 0 and 1 encoding
          the percentage of overlap between adjacent windows.
        * **t_ftimwin** : sliding window length (in sec)
        * **taper** : one of :data:`~syncopy.shared.const_def.availableTapers`
        * **tapsmofrq** : spectral smoothing box for slepian tapers (in Hz)
        * **nTaper** : number of orthogonal tapers for slepian tapers
        * **keeptapers** : must be `False` with Welch. For multi-tapering,
          taper averaging happens as part of the modified periodogram computation,
           i.e., before the window averaging performed by Welch.

    "wavelet" : (Continuous non-orthogonal) wavelet transform
        Perform time-frequency analysis on time-series trial data using a non-orthogonal
        continuous wavelet transform.

        * **wavelet** : one of :data:`~syncopy.specest.freqanalysis.availableWavelets`
        * **toi** : time-points of interest; can be either an array representing
          time points (in sec) or "all"(pre-trimming and subsampling of results)
        * **width** : Nondimensional frequency constant of Morlet wavelet function (>= 6)
        * **order** : Order of Paul wavelet function (>= 4) or derivative order
          of real-valued DOG wavelets (2 = mexican hat)

    "superlet" : Superlet transform
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
        Spectral estimation method, one of :data:`~syncopy.specest.freqanalysis.availableMethods`
        (see below).
    output : str
        Output of spectral estimation. One of :data:`~syncopy.specest.freqanalysis.availableOutputs` (see below);
        use `'pow'` for power spectrum (:obj:`numpy.float32`), `'fourier'` for complex
        Fourier coefficients (:obj:`numpy.complex64`) or `'abs'` for absolute
        values (:obj:`numpy.float32`). Use one of `'fooof'`, `'fooof_aperiodic'` or
        `'fooof_peaks'` to request post-processing of the results with FOOOF, also see
        the `'fooof_opt'` parameter description.
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
    pad : 'maxperlen', float or 'nextpow2'
        For the default `maxperlen`, no padding is performed in case of equal
        length trials, while trials of varying lengths are padded to match the
        longest trial. If `pad` is a number all trials are padded so that `pad` indicates
        the absolute length of all trials after padding (in seconds). For instance
        ``pad = 2`` pads all trials to an absolute length of 2000 samples, if and
        only if the longest trial contains at maximum 2000 samples and the
        samplerate is 1kHz. If `pad` is `'nextpow2'` all trials are padded to the
        nearest power of two (in samples) of the longest trial.
    polyremoval : int or None
        Order of polynomial used for de-trending data in the time domain prior
        to spectral analysis. A value of 0 corresponds to subtracting the mean
        ("de-meaning"), ``polyremoval = 1`` removes linear trends (subtracting the
        least squares fit of a linear polynomial).
        If `polyremoval` is `None`, no de-trending is performed. Note that
        for spectral estimation de-meaning is very advisable and hence also the
        default.
    tapsmofrq : float or None
        Only valid if `method` is `'mtmfft'` or `'mtmconvol'`
        Enables multi-tapering and sets the amount of one-sided spectral
        smoothing with slepian tapers in Hz.
    nTaper : int or None
        Only valid if `method` is `'mtmfft'` or `'mtmconvol'` and `tapsmofrq` is set.
        Number of orthogonal tapers to use for multi-tapering. It is not recommended to set the number
        of tapers manually! Leave at `None` for the optimal number to be set automatically.
    taper : str or None, optional
        Only valid if ``method`` is ``'mtmfft'`` or ``'mtmconvol'``. Windowing function,
        one of :data:`~syncopy.shared.const_def.availableTapers`
        For multi-tapering with slepian tapers use `tapsmofrq` directly.
    demean_taper : bool
        Set to `True` to perform de-meaning after tapering. Recommended for later Granger
        analysis with :func:`~syncopy.connectivityanalysis`. Only valid for ``method='mtmfft'``.
    taper_opt : dict or None
        Dictionary with keys for additional taper parameters.
        For example :func:`~scipy.signal.windows.kaiser` has
        the additional parameter 'beta'. For multi-tapering use `tapsmofrq` directly.
    keeptapers : bool
        Only valid if `method` is `'mtmfft'` or `'mtmconvol'` and multi-tapering enabled
        via  setting `tapsmofrq`.
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
        Only valid if `method` is `'mtmconvol'` or `'welch'`. Sliding window length (in seconds).
    wavelet : str
        Only valid if `method` is `'wavelet'`. Wavelet function to use, one of
        :data:`~syncopy.specest.freqanalysis.availableWavelets` (see below).
    width : positive float
        Only valid if `method` is `'wavelet'` and `wavelet` is `'Morlet'`. Nondimensional
        frequency constant of Morlet wavelet function. This number should be >= 6,
        which corresponds to 6 cycles within the analysis window to ensure sufficient
        spectral sampling.
    order : positive int
        Only valid if `method` is `'wavelet'` and `wavelet` is `'Paul'` or `'DOG'`. Order
        of the wavelet function. If `wavelet` is `'Paul'`, `order` should be chosen
        >= 4 to ensure that the analysis window contains at least a single oscillation.
        At an order of 40, the Paul wavelet  exhibits about the same number of cycles
        as the Morlet wavelet with a `width` of 6.
        All other supported wavelets functions are *real-valued* derivatives of
        Gaussians (DOGs). Hence, if `wavelet` is `'DOG'`, `order` represents the derivative order.
        The special case of a second order DOG yields a function known as "Mexican Hat",
        "Marr" or "Ricker" wavelet, which can be selected alternatively by setting
        `wavelet` to `'Mexican_hat'`, `'Marr'` or `'Ricker'`. **Note**: A real-valued
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
    fooof_opt : dict or None
        Only valid if `method` is `'mtmfft'` and `output` is  one of
        `'fooof'`, `'fooof_aperiodic'`, or `'fooof_peaks'`.
        Additional keyword arguments passed to the `FOOOF` constructor. Available
        arguments include ``'peak_width_limits'``, ``'max_n_peaks'``, ``'min_peak_height'``,
        ``'peak_threshold'``, and ``'aperiodic_mode'``.
        Please refer to the
        `FOOOF docs <https://fooof-tools.github.io/fooof/generated/fooof.FOOOF.html#fooof.FOOOF>`_
        for the meanings and the defaults.
        See the FOOOF reference [Donoghue2020]_ for details.

    Returns
    -------
    spec : :class:`~syncopy.SpectralData`
        (Time-)frequency spectrum of input data. The `spec` may contain additional metadata,
        based on the `method` used to compute it:

        * For `method='mtmfft'` when `output` is one of
          `'fooof'`, `'fooof_aperiodic'`, or `'fooof_peaks'`, the `spec.metadata` property contains
          the keys listed and explained in :data:`~syncopy.specest.compRoutines.FooofSpy.metadata_keys`.


    Notes
    -----
    .. [Moca2021] Moca, Vasile V., et al. "Time-frequency super-resolution with superlets."
       Nature communications 12.1 (2021): 1-18.
    .. [Donoghue2020] Donoghue et al. 2020, DOI 10.1038/s41593-020-00744-x.
    .. [Welch1967] Welch. "The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms.", 1976, DOI 10.1109/TAU.1967.1161901

    **Options**

    .. autodata:: syncopy.specest.freqanalysis.availableMethods

    .. autodata:: syncopy.specest.freqanalysis.availableOutputs

    .. autodata:: syncopy.shared.const_def.availableTapers

    .. autodata:: syncopy.specest.freqanalysis.availableWavelets

    Examples
    --------
    Coming soon...



    See also
    --------
    syncopy.specest.mtmfft.mtmfft : (multi-)tapered Fourier transform of multi-channel time series data
    syncopy.specest.mtmconvol.mtmconvol : time-frequency analysis of multi-channel time series data with a sliding window FFT
    syncopy.specest.wavelet.wavelet : time-frequency analysis of multi-channel time series data using a wavelet transform
    syncopy.specest.fooofspy.fooofspy : parameterization of neural power spectra with the 'fitting oscillations & one over f' method
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
    # check for ineffective additional kwargs
    check_passed_kwargs(lcls, defaults, frontend_name="freqanalysis")

    new_cfg = get_frontend_cfg(defaults, lcls, kwargs)

    is_fooof = False
    if method == "mtmfft" and output.startswith("fooof"):
        is_fooof = True
        output_fooof = output
        output = "pow"  # We need to change this as the mtmfft running first will complain otherwise.

    if is_fooof:
        if output_fooof not in availableFooofOutputs:
            lgl = "'" + "or '".join(opt + "' " for opt in availableFooofOutputs)
            raise SPYValueError(legal=lgl, varname="output_fooof", actual=output_fooof)

    # Ensure a valid computational method was selected
    if method not in availableMethods:
        lgl = "'" + "or '".join(opt + "' " for opt in availableMethods)
        raise SPYValueError(legal=lgl, varname="method", actual=method)

    # Ensure a valid output format was selected
    valid_outputs = spectralConversions.keys()
    if output not in valid_outputs:
        lgl = "'" + "or '".join(opt + "' " for opt in valid_outputs)
        raise SPYValueError(legal=lgl, varname="output", actual=output)

    # Parse all Boolean keyword arguments
    for vname in ["keeptrials", "keeptapers", "demean_taper"]:
        if not isinstance(lcls[vname], bool):
            raise SPYTypeError(lcls[vname], varname=vname, expected="Bool")

    # If only a subset of `data` is to be processed, make some necessary adjustments
    # of the sampleinfo and trial lengths
    if data.selection is not None:
        # Refuse to go ahead with active time selection and provided `toi` on top`
        if any(tsel != slice(None) for tsel in data.selection.time) and isinstance(toi, (np.ndarray, list)):
            lgl = "no `toi` specification due to active in-place time-selection in input dataset"
            raise SPYValueError(legal=lgl, varname="toi", actual=toi)
        sinfo = data.selection.trialdefinition[:, :2]
        trialList = data.selection.trial_ids
    else:
        trialList = list(range(len(data.trials)))
        sinfo = data.sampleinfo
    lenTrials = np.diff(sinfo).squeeze()
    if not lenTrials.shape:
        lenTrials = lenTrials[None]
    numTrials = len(trialList)

    # check polyremoval
    if polyremoval is not None:
        scalar_parser(polyremoval, varname="polyremoval", ntype="int_like", lims=[0, 1])

    # --- Padding ---

    # Sliding window FFT does not support "fancy" padding
    if method in ["mtmconvol", "welch"] and isinstance(pad, str) and pad != defaults['pad']:
        msg = "methods 'mtmconvol' and 'welch' only support in-place padding for windows " +\
            "exceeding trial boundaries. Your choice of `pad = '{}'` will be ignored. "
        SPYWarning(msg.format(pad))

    if method == 'mtmfft':
        # the actual number of samples in case of later padding
        minSampleNum = process_padding(pad, lenTrials, data.samplerate)
    else:
        minSampleNum = lenTrials.min()

    # Compute length (in seconds) of shortest trial
    minTrialLength = minSampleNum / data.samplerate

    # Shortcut to data sampling interval
    dt = 1 / data.samplerate

    foi, foilim = process_foi(foi, foilim, data.samplerate)

    # see also https://docs.obspy.org/_modules/obspy/signal/detrend.html#polynomial
    if polyremoval is not None:
        try:
            scalar_parser(polyremoval, varname="polyremoval", lims=[0, 1], ntype="int_like")
        except Exception as exc:
            raise exc

    # Prepare keyword dict for logging (use `lcls` to get actually provided
    # keyword values, not defaults set above)
    log_dct = {"method": method,
               "output": output_fooof if is_fooof else output,
               "keeptapers": keeptapers,
               "keeptrials": keeptrials,
               "polyremoval": polyremoval,
               "pad": pad}

    # --------------------------------
    # 1st: Check time-frequency inputs
    # to prepare/sanitize `toi`
    # --------------------------------

    if method in ["mtmconvol", "wavelet", "superlet", "welch"]:

        # Get start/end timing info respecting potential in-place selection
        if toi is None:
            raise SPYTypeError(toi, varname="toi", expected="scalar or array-like or 'all'")
        if data.selection is not None:
            tStart = data.selection.trialdefinition[:, 2] / data.samplerate
        else:
            tStart = data._t0 / data.samplerate
        tEnd = tStart + lenTrials / data.samplerate

    # for these methods only 'all' or an equidistant array
    # of time points (sub-sampling, trimming) are valid
    if method in ["wavelet", "superlet"]:

        valid = True
        if np.issubdtype(type(toi), np.number):
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

        # this is the sequence type - can only be an interval!
        else:
            try:
                array_parser(toi, varname="toi", hasinf=False, hasnan=False,
                             lims=[tStart.min(), tEnd.max()], dims=(None,))
            except Exception as exc:
                raise exc
            toi = np.array(toi)
            # check for equidistancy
            if not np.allclose(np.diff(toi, 2), np.zeros(len(toi) - 2)):
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

        # Update `log_dct` w/method-specific options (use `lcls` to get actually
        # provided keyword values, not defaults set in here)
        log_dct["toi"] = lcls["toi"]

    # --------------------------------------------
    # Check options specific to mtm*-methods
    # (particularly tapers and foi/freqs alignment)
    # --------------------------------------------

    if "mtm" in method or method == "welch":

        if method in ["mtmconvol", "welch"]:
            # get the sliding window size
            try:
                scalar_parser(t_ftimwin, varname="t_ftimwin",
                              lims=[dt, minTrialLength])
            except Exception as exc:
                SPYInfo("Please specify 't_ftimwin' parameter.. exiting!")
                raise exc

            # this is the effective sliding window FFT sample size
            minSampleNum = int(t_ftimwin * data.samplerate)

            if method == "welch":
                if keeptapers:
                    raise SPYValueError(legal="keeptapers='False' with method='welch'", varname="keeptapers", actual=keeptapers)

                if output != "pow":
                    raise SPYValueError(legal="output='pow' with method='welch'", varname="output", actual=output)


        # Construct array of maximally attainable frequencies
        freqs = np.fft.rfftfreq(minSampleNum, dt)

        # Match desired frequencies as close as possible to
        # actually attainable freqs
        # these are the frequencies attached to the SpectralData by the CR!
        if foi is not None:
            foi, _ = best_match(freqs, foi, squash_duplicates=True)
        elif foilim is not None:
            foi, _ = best_match(freqs, foilim, span=True, squash_duplicates=True)
        else:
            msg = (f"Automatic FFT frequency selection from {freqs[0]:.1f}Hz to "
                   f"{freqs[-1]:.1f}Hz")
            SPYInfo(msg)
            foi = freqs
        log_dct["foi"] = foi

        # Abort if desired frequency selection is empty
        if foi.size == 0:
            lgl = "non-empty frequency specification"
            act = "empty frequency selection"
            raise SPYValueError(legal=lgl, varname="foi/foilim", actual=act)

        # sanitize taper selection and/or retrieve dpss settings
        taper, taper_opt = process_taper(taper,
                                         taper_opt,
                                         tapsmofrq,
                                         nTaper,
                                         keeptapers,
                                         foimax=foi.max(),
                                         samplerate=data.samplerate,
                                         nSamples=minSampleNum,
                                         output=output)

        # Update `log_dct` w/method-specific options
        log_dct["taper"] = taper
        if taper_opt and taper == 'dpss':
            log_dct["nTaper"] = taper_opt["Kmax"]
            log_dct["tapsmofrq"] = tapsmofrq
        elif taper_opt:
            log_dct["taper_opt"] = taper_opt

    # -------------------------------------------------------
    # Now, prepare explicit compute-classes for chosen method
    # -------------------------------------------------------

    if method == "mtmfft":

        check_effective_parameters(MultiTaperFFT, defaults, lcls)

        # method specific parameters
        method_kwargs = {
            'samplerate': data.samplerate,
            'taper': taper,
            'taper_opt': taper_opt,
            'nSamples': minSampleNum,
            'demean_taper': demean_taper
        }

        # Set up compute-class
        specestMethod = MultiTaperFFT(
            foi=foi,
            timeAxis=timeAxis,
            keeptapers=keeptapers,
            polyremoval=polyremoval,
            output=output,
            method_kwargs=method_kwargs)

    elif method in ["mtmconvol", "welch"]:

        check_effective_parameters(MultiTaperFFTConvol, defaults, lcls)

        # Process `toi` for sliding window multi taper fft,
        # we have to account for three scenarios: (1) center sliding
        # windows on all samples in (selected) trials (2) `toi` was provided as
        # percentage indicating the degree of overlap b/w time-windows and (3) a set
        # of discrete time points was provided. These three cases are encoded in
        # `overlap, i.e., ``overlap > 1` => all, `0 < overlap < 1` => percentage,
        # `overlap < 0` => discrete `toi`

        # overlap = None
        if isinstance(toi, str):
            if method == "welch":
                lgl = "toi to be a float in range [0, 1] for method='welch'"
                raise SPYValueError(legal=lgl, varname="toi", actual=toi)
            if toi != "all":
                lgl = "`toi = 'all'` to center analysis windows on all time-points"
                raise SPYValueError(legal=lgl, varname="toi", actual=toi)
            equidistant = True
            overlap = np.inf

        elif np.issubdtype(type(toi), np.number):
            scalar_parser(toi, varname="toi", lims=[0, 1])
            overlap = toi
            equidistant = True
        # this captures all other cases, e.i. toi is of sequence type
        else:
            if method == "welch":
                lgl = "toi to be a float in range [0, 1] for method='welch'"
                raise SPYValueError(legal=lgl, varname="toi", actual=toi)

            overlap = -1
            array_parser(toi, varname="toi", hasinf=False, hasnan=False,
                             lims=[tStart.min(), tEnd.max()], dims=(None,))
            toi = np.array(toi)
            tSteps = np.diff(toi)
            if (tSteps < 0).any():
                lgl = "ordered list/array of time-points"
                act = "unsorted list/array"
                raise SPYValueError(legal=lgl, varname="toi", actual=act)
            # Account for round-off errors: if toi spacing is almost at sample interval
            # manually correct it
            if np.isclose(tSteps.min(), dt):
                tSteps[np.isclose(tSteps, dt)] = dt
            if tSteps.min() < dt:
                msg = f"`toi` selection too fine, max. time resolution is {dt}s"
                SPYWarning(msg)
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

        # number of samples per window
        nperseg = int(t_ftimwin * data.samplerate)
        halfWin = int(nperseg / 2)
        postSelect = slice(None)  # select all is the default

        if 0 <= overlap <= 1:  # `toi` is percentage
            noverlap = min(nperseg - 1, int(overlap * nperseg))
        # windows get shifted exactly 1 sample
        # to get a spectral estimate at each sample
        else:
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

            # Treat equi-distant `toi` arrays with spacing large enough that windows
            # do not overlap as if they were not equidistant
            if tSteps.max() * data.samplerate > halfWin and equidistant:
                equidistant = False

            # Compute sample-indices (one slice/list per trial) from time-selections
            soi = []
            if equidistant:
                # soi just trims the input data to the [toi[0], toi[-1]] interval
                # postSelect then subsamples the spectral esimate to the user given toi
                postSelect = []
                for tk in range(numTrials):
                    start = max(0, int(round(data.samplerate * (toi[0] - tStart[tk]) - halfWin)))
                    stop = int(round(data.samplerate * (toi[-1] - tStart[tk]) + halfWin + 1))
                    soi.append(slice(start, max(stop, stop - start)))

                # chosen toi subsampling interval in sample units, min. is 1;
                # compute `delta_idx` s.t. stop - start / delta_idx == toi.size
                delta_idx = int(round((soi[0].stop - soi[0].start) / toi.size))
                delta_idx = delta_idx if delta_idx > 1 else 1
                postSelect = slice(None, None, delta_idx)

            else:
                for tk in range(numTrials):
                    starts = (data.samplerate * (toi - tStart[tk]) - halfWin).astype(np.intp)
                    starts += padBegin[tk]
                    stops = (data.samplerate * (toi - tStart[tk]) + halfWin + 1).astype(np.intp)
                    stops += padBegin[tk]
                    stops = np.maximum(stops, stops - starts, dtype=np.intp)
                    soi.append([slice(start, stop) for start, stop in zip(starts, stops)])
                    # postSelect here remains slice(None), as resulting spectrum
                    # has exactly one entry for each soi

        # `toi` is percentage or "all"
        else:
            soi = [slice(None)] * numTrials

        # Collect keyword args for `mtmconvol` in dictionary
        method_kwargs = {"samplerate": data.samplerate,
                         "nperseg": nperseg,
                         "noverlap": noverlap,
                         "taper": taper,
                         "taper_opt": taper_opt}

        # Set up compute-class
        specestMethod = MultiTaperFFTConvol(
            soi,
            postSelect,
            equidistant=equidistant,
            toi=toi,
            foi=foi,
            timeAxis=timeAxis,
            keeptapers=keeptapers,
            polyremoval=polyremoval,
            output=output,
            method_kwargs=method_kwargs)

    elif method == "wavelet":

        check_effective_parameters(WaveletTransform, defaults, lcls)

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

        # automatic frequency selection
        if foi is None and foilim is None:
            scales = get_optimal_wavelet_scales(
                wfun.scale_from_period,  # all availableWavelets sport one!
                int(minTrialLength * data.samplerate),
                dt)
            foi = 1 / wfun.fourier_period(scales)
            msg = (f"Setting frequencies of interest to {foi[0]:.1f}-"
                   f"{foi[-1]:.1f}Hz")
            SPYInfo(msg)
        else:
            if foilim is not None:
                foi = np.arange(foilim[0], foilim[1] + 1, dtype=float)
            # 0 frequency is not valid
            foi[foi < 0.01] = 0.01
            scales = wfun.scale_from_period(1 / foi)

        # Update `log_dct` w/method-specific options (use `lcls` to get actually
        # provided keyword values, not defaults set in here)
        log_dct["foi"] = foi
        log_dct["wavelet"] = lcls["wavelet"]
        log_dct["width"] = lcls["width"]
        log_dct["order"] = lcls["order"]

        # method specific parameters
        method_kwargs = {
            'samplerate': data.samplerate,
            'scales': scales,
            'wavelet': wfun
        }

        # Set up compute-class
        specestMethod = WaveletTransform(
            preSelect,
            postSelect,
            toi=toi,
            timeAxis=timeAxis,
            polyremoval=polyremoval,
            output=output,
            method_kwargs=method_kwargs)

    elif method == "superlet":

        check_effective_parameters(SuperletTransform, defaults, lcls)

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
                dt)
            foi = 1 / superlet.fourier_period(scales)
            msg = (f"Setting frequencies of interest to {foi[0]:.1f}-"
                   f"{foi[-1]:.1f}Hz")
            SPYInfo(msg)
        else:
            if foilim is not None:
                # frequency range in 1Hz steps
                foi = np.arange(foilim[0], foilim[1] + 1, dtype=float)
            # 0 frequency is not valid
            foi[foi < 0.01] = 0.01
            scales = superlet.scale_from_period(1. / foi)

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

        log_dct["foi"] = foi
        log_dct["c_1"] = lcls["c_1"]
        log_dct["order_max"] = lcls["order_max"]
        log_dct["order_min"] = lcls["order_min"]

        # method specific parameters
        method_kwargs = {
            'samplerate': data.samplerate,
            'scales': scales,
            'order_max': order_max,
            'order_min': order_min,
            'c_1': c_1,
            'adaptive': adaptive
        }

        # Set up compute-class
        specestMethod = SuperletTransform(
            preSelect,
            postSelect,
            toi=toi,
            timeAxis=timeAxis,
            polyremoval=polyremoval,
            output=output,
            method_kwargs=method_kwargs)

    # -------------------------------------------------
    # Sanitize output and call the ComputationalRoutine
    # -------------------------------------------------

    out = SpectralData(dimord=SpectralData._defaultDimord)

    # Perform actual computation
    specestMethod.initialize(data,
                             out._stackingDim,
                             chan_per_worker=kwargs.get("chan_per_worker"),
                             keeptrials=keeptrials)
    specestMethod.compute(data, out, parallel=kwargs.get("parallel"), log_dict=log_dct)

    # FOOOF is a post-processing method of MTMFFT output, so we handle it here, once
    # the MTMFFT has finished.
    if is_fooof:

        # Use the output of the MTMFFMT method as the new data and create new output data.
        fooof_data = out
        fooof_out = SpectralData(dimord=SpectralData._defaultDimord)

        # method specific parameters
        if fooof_opt is None:
            fooof_opt = default_fooof_opt

        # These go into the FOOOF constructor, so we keep them separate from the fooof_settings below.
        fooof_kwargs = {**default_fooof_opt, **fooof_opt}  # Join the ones from fooof_opt (the user) into the default fooof_kwargs.

        # Settings used during the FOOOF analysis (that are NOT passed to FOOOF constructor).
        # The user cannot influence these: in_freqs is derived from mtmfft output, freq_range is always None (=full mtmfft output spectrum).
        # We still define them here, and they are passed through to the backend and actually used there.
        fooof_settings = {
            'in_freqs': fooof_data.freq,
            'freq_range': None  # or something like [2, 40] to limit frequency range (post processing). Currently not exposed to user.
        }

        if fooof_data.freq[0] == 0:
            # FOOOF does not work with input frequency zero in the data.
            raise SPYValueError(legal="a frequency range that does not include zero. Use 'foi' or 'foilim' to restrict.", varname="foi/foilim", actual="Frequency range from {} to {}.".format(min(fooof_data.freq), max(fooof_data.freq)))

        # Set up compute-class
        #  - the output must be one of 'fooof', 'fooof_aperiodic',
        #    or 'fooof_peaks'.
        #  - everything passed as method_kwargs is passed as arguments
        #    to the fooof.FOOOF() constructor or functions, the other args are
        #    used elsewhere.
        fooofMethod = FooofSpy(output=output_fooof, fooof_settings=fooof_settings, method_kwargs=fooof_kwargs)

        # Update `log_dct` w/method-specific options
        log_dct["fooof_method"] = output_fooof
        log_dct["fooof_opt"] = fooof_kwargs

        # Perform actual computation
        fooofMethod.initialize(fooof_data,
                               fooof_out._stackingDim,
                               chan_per_worker=kwargs.get("chan_per_worker"),
                               keeptrials=keeptrials)
        fooofMethod.compute(fooof_data, fooof_out, parallel=kwargs.get("parallel"), log_dict=log_dct)
        out = fooof_out

    # Perform mtmconvolv post-processing for `method='welch'`.
    if method == "welch":
        welch_data = out
        out = spy.mean(welch_data, dim='time')

    # Attach potential older cfg's from the input
    # to support chained frontend calls.
    out.cfg.update(data.cfg)

    # Attach frontend parameters for replay.
    out.cfg.update({'freqanalysis': new_cfg})
    return out
