# -*- coding: utf-8 -*-
#
# Syncopy connectivity analysis methods
#

# Builtin/3rd party package imports
import numpy as np

# Syncopy imports
from syncopy.shared.parsers import data_parser, scalar_parser
from syncopy.shared.tools import get_defaults, best_match, get_frontend_cfg
from syncopy.datatype import CrossSpectralData, AnalogData, SpectralData
from syncopy.shared.errors import (
    SPYValueError,
    SPYWarning,
    SPYInfo)
from syncopy.shared.kwarg_decorators import (unwrap_cfg, unwrap_select,
                                             detect_parallel_client)
from syncopy.shared.input_processors import (
    process_taper,
    process_foi,
    process_padding,
    check_effective_parameters,
    check_passed_kwargs
)

from syncopy.nwanalysis.ST_compRoutines import CrossSpectra, CrossCovariance, SpectralDyadicProduct
from syncopy.nwanalysis.AV_compRoutines import NormalizeCrossSpectra, NormalizeCrossCov, GrangerCausality


availableMethods = ("coh", "corr", "granger")
coh_outputs = {"abs", "pow", "complex", "fourier", "angle", "real", "imag"}


@unwrap_cfg
@unwrap_select
@detect_parallel_client
def connectivityanalysis(data, method="coh", keeptrials=False, output="abs",
                         foi=None, foilim=None, pad='maxperlen',
                         polyremoval=0, tapsmofrq=None, nTaper=None,
                         taper="hann", taper_opt=None, **kwargs):

    """
    Perform connectivity analysis of Syncopy :class:`~syncopy.SpectralData` OR directly
    :class:`~syncopy.AnalogData` objects

    In case the input is an :class:`~syncopy.AnalogData` object, a (multi-)tapered Fourier
    analysis is performed implicitly to arrive at the cross spectral densities needed for
    coherence and Granger causality estimates.
    Relevant parameters are the same as for :func:`~syncopy.freqanalysis` with ``method='mtmfft'``:

        ('foi', 'foilim', 'pad', 'tapsmofrq', 'nTaper', 'taper', 'taper_opt', 'polyremoval')

    If the input is already in the spectral domain, so ``data`` is of class :class:`~syncopy.SpectralData`,
    no additional modification of the spectra is performed, and all parameters above to control
    the spectral estimation have no effect.

    **Usage Summary**

    List of available analysis methods and respective distinct options:

    "coh" : (Multi-) tapered coherency estimate
        Compute the normalized cross spectral densities
        between all channel combinations

        * **output** : one of ('abs', 'pow', 'complex', 'angle', 'imag' or 'real')

        **Spectral analysis** (input is :class:`~syncopy.AnalogData`):

        * **taper** : one of :data:`~syncopy.shared.const_def.availableTapers`
        * **tapsmofrq** : spectral smoothing box for slepian tapers (in Hz)
        * **nTaper** : (optional) number of orthogonal tapers for slepian tapers
        * **pad**: either pad to an absolute length in seconds or set to `'nextpow2'`

    "corr" : Cross-correlations
        Computes the one sided (positive lags) cross-correlations
        between all channel combinations of :class:`~syncopy.AnalogData`.
        The maximal lag is half the trial lengths.

        * **keeptrials** : set to `True` for single trial cross-correlations

    "granger" : Spectral Granger-Geweke causality
        Computes linear causality estimates between
        all channel combinations.

        WARNING: When inputting :class:`~syncopy.SpectralData` directly,
        it is very important that the previous `spy.freqanalysis` was
        done without foi/foilim specification as Granger causality needs all
        attainable frequencies (0, f_Nyquist)!

        **Spectral analysis** (input is :class:`~syncopy.AnalogData`):

        * **taper** : one of :data:`~syncopy.shared.const_def.availableTapers`
        * **tapsmofrq** : spectral smoothing box for slepian tapers (in Hz)
        * **nTaper** : (optional, not recommended) number of slepian tapers
        * **pad**: either pad to an absolute length in seconds or set to `'nextpow2'`

        After the computation, information about the convergence and potential
        regularization of the cross-spectral densities can be obtained
        by inspecting the ``.info`` property of the resulting :class:~`syncopy.CrossSpectralData`
        object. Keys of that info-dict are:
        ```{'converged', 'max rel. err', 'reg. factor', 'initial cond. num'}```.

    Parameters
    ----------
    data : `~syncopy.AnalogData`
        A non-empty Syncopy :class:`~syncopy.SpectralData` or
        :class:`~syncopy.AnalogData` object
    method : str
        Connectivity estimation method, one of ``'coh'`, 'corr', 'granger'``
    output : str
        Relevant for cross-spectral density estimation (``method='coh'``)
        Use ``'pow'`` for absolute squared coherence, ``'abs'`` for absolute value of coherence
        , ``'complex'`` for the complex valued coherency or ``'angle'``, ``'imag'`` or ``'real'``
        to extract the phase difference, imaginary or real part of the coherency respectively.
    keeptrials : bool
        Relevant for cross-correlations (``method='corr'``).
        If `True` single-trial cross-correlations are returned.
    foi : array-like or None
        Frequencies of interest (Hz) for output. If desired frequencies cannot be
        matched exactly, the closest possible frequencies are used. If ``foi`` is ``None``
        or ``foi = "all"``, all attainable frequencies (i.e., zero to Nyquist / 2)
        are selected.
    foilim : array-like [fmin, fmax] or None or "all"
        Frequency-window ``[fmin, fmax]`` (in Hz) of interest. The
        `foi` array will be constructed in 1Hz steps from `fmin` to
        `fmax` (inclusive).
    pad : 'maxperlen', float or 'nextpow2' -
        For the default ``'maxperlen'``, no padding is performed in case of equal
        length trials, while trials of varying lengths are padded to match the
        longest trial. If ``pad`` is a number all trials are padded so that ``pad`` indicates
        the absolute length of all trials after padding (in seconds). For instance
        ``pad = 2`` pads all trials to an absolute length of 2000 samples, if and
        only if the longest trial contains at maximum 2000 samples and the
        samplerate is 1kHz. If ``pad`` is ``'nextpow2'`` all trials are padded to the
        nearest power of two (in samples) of the longest trial.
    tapsmofrq : float or None
        Only valid if ``method`` is ``'coh'`` or ``'granger'``.
        Enables multi-tapering and sets the amount of spectral
        smoothing with slepian tapers in Hz.
    nTaper : int or None
        Only valid if ``method`` is ``'coh'`` or ``'granger'`` and ``tapsmofrq`` is set.
        Number of orthogonal tapers to use for multi-tapering. It is not recommended to set the number
        of tapers manually! Leave at ``None`` for the optimal number to be set automatically.
    taper : str or None, optional
        Only valid if ``method`` is ``'coh'`` or ``'granger'``. Windowing function,
        one of :data:`~syncopy.specest.const_def.availableTapers`
        For multi-tapering with slepian tapers use `tapsmofrq` directly.
    taper_opt : dict or None
        Dictionary with keys for additional taper parameters.
        For example :func:`~scipy.signal.windows.kaiser` has
        the additional parameter 'beta'. For multi-tapering set ``tapsmofrq`` directly.

    Returns
    -------
    out : `~syncopy.CrossSpectralData`
        The analyis result with dims ['time', 'freq', 'channel_i', channel_j']. The `out` may contain additional metadata,
        based on the `method` used to compute it:

        * For `method='granger'`, the `out.info` property contains
          the keys listed and explained in :data:`~syncopy.nwanalysis.AV_compRoutines.GrangerCausality.metadata_keys`.

    Examples
    --------
    In the following `adata` is an instance of :class:`~syncopy.AnalogData`

    Calculate the coherence between all channels with 2Hz spectral smoothing,
    and plot the results for two combinations between 30Hz and 90Hz:

    >>> coh = spy.connectivityanalysis(adata, method='coh', tapsmofrq=2)
    >>> coh.singlepanelplot(channel_i=0, channel_j=1, foilim=[30,90])
    >>> coh.singlepanelplot(channel_i=1, channel_j=2, foilim=[30,90])

    Compute the cross-correlation between channel 8 and 12 and
    plot the results for the first 200ms:

    >>> cfg = spy.StructDict()
    >>> cfg.method = 'corr'
    >>> cfg.select = {'channel': ['channel8', 'channel12']}
    >>> corr = spy.connectivityanalysis(adata, cfg)
    >>> corr.singlepanelplot(channel_i='channel8', channel_j='channel12', toilim=[0, 0.2])

    Estimate Granger causality between the same channels (re-using the cfg from above):

    >>> cfg.method = 'granger'
    >>> granger = spy.connectivityanalysis(adata, cfg)

    Plot the results between 15Hz and 30Hz:

    >>> granger.singlepanelplot(channel_i='channel8', channel_j='channel12', foilim=[15, 25])
    """

    # Make sure our one mandatory input object can be processed
    try:
        data_parser(data, varname="data", writable=None, empty=False)
    except Exception as exc:
        raise exc

    if not isinstance(data, (AnalogData, SpectralData)):
        lgl = "either AnalogData or SpectralData as input"
        act = f"{data.__class__.__name__}"
        raise SPYValueError(lgl, 'data', act)
    timeAxis = data.dimord.index("time")

    # Get everything of interest in local namespace
    defaults = get_defaults(connectivityanalysis)
    lcls = locals()
    # check for ineffective additional kwargs
    check_passed_kwargs(lcls, defaults, frontend_name="connectivity")

    new_cfg = get_frontend_cfg(defaults, lcls, kwargs)

    # Ensure a valid computational method was selected
    if method not in availableMethods:
        lgl = "'" + "or '".join(opt + "' " for opt in availableMethods)
        raise SPYValueError(legal=lgl, varname="method", actual=method)

    # output selection is only valid for coherence
    if method != 'coh' and output != defaults['output']:
        msg = f"Setting `output` for method {method} has not effect!"
        SPYWarning(msg)

    # if a subset selection is present
    # get sampleinfo and check for equidistancy
    if data.selection is not None:
        sinfo = data.selection.trialdefinition[:, :2]
        # user picked discrete set of time points
        if isinstance(data.selection.time[0], list):
            lgl = "equidistant time points"
            actual = "non-equidistant set of time points"
            raise SPYValueError(legal=lgl, varname="select", actual=actual)
    else:
        sinfo = data.sampleinfo
    lenTrials = np.diff(sinfo).squeeze()

    # check padding

    if method == "corr" and pad != 'maxperlen':
        lgl = "'maxperlen', no padding needed/allowed for cross-correlations"
        actual = f"{pad}"
        raise SPYValueError(legal=lgl, varname="pad", actual=actual)

    # check polyremoval
    if polyremoval is not None:
        scalar_parser(polyremoval, varname="polyremoval", ntype="int_like", lims=[0, 1])

    # Prepare keyword dict for logging (use `lcls` to get actually provided
    # keyword values, not defaults set above)
    log_dict = {"method": method,
                "keeptrials": keeptrials,
                "polyremoval": polyremoval,
                "pad": pad}

    # --- method specific processing ---

    if method == 'corr':

        if not isinstance(data, AnalogData):
            lgl = f"AnalogData instance as input for method {method}"
            actual = f"{data.__class__.__name__}"
            raise SPYValueError(lgl, 'data', actual)

        if lcls['foi'] is not None:
            msg = 'Parameter `foi` has no effect for method `corr`'
            SPYWarning(msg)

        check_effective_parameters(CrossCovariance, defaults, lcls)

        # single trial cross-correlations
        if keeptrials:
            av_compRoutine = None   # no trial average
            norm = True   # normalize individual trials within the ST CR
        else:
            av_compRoutine = NormalizeCrossCov()
            norm = False

        # parallel computation over trials
        st_compRoutine = CrossCovariance(samplerate=data.samplerate,
                                         polyremoval=polyremoval,
                                         timeAxis=timeAxis,
                                         norm=norm)
        # hard coded as class attribute
        st_dimord = CrossCovariance.dimord

    elif method in ['coh', 'granger']:

        nTrials = len(data.trials)
        if nTrials == 1:
            lgl = "multi-trial input data, spectral connectivity measures critically depend on trial averaging!"
            act = "only one trial"
            raise SPYValueError(lgl, 'data', act)

        if keeptrials is not False:
            lgl = "False, trial averaging needed for 'coh' and 'granger'!"
            act = keeptrials
            raise SPYValueError(lgl, varname="keeptrials", actual=act)

        # AnalogData - we have to setup implicit spectral analysis (mtmfft)
        if isinstance(data, AnalogData):
            # the actual number of samples in case of later padding
            nSamples = process_padding(pad, lenTrials, data.samplerate)

            check_effective_parameters(CrossSpectra, defaults, lcls)

            st_compRoutine, st_dimord = cross_spectra(data, method, nSamples,
                                                      foi, foilim, tapsmofrq,
                                                      nTaper, taper, taper_opt,
                                                      polyremoval, log_dict, timeAxis)
        # SpectralData input
        elif isinstance(data, SpectralData):

            # cross-spectra need complex input spectra
            if data.data.dtype != np.complex64 and data.data.dtype != np.complex128:
                lgl = "complex valued spectra, set `output='fourier` in spy.freqanalysis!"
                act = "real valued spectral data"
                raise SPYValueError(lgl, 'data', act)

            if method == 'granger':
                # check that for SpectralData input, we have empty time axes
                # no time-resolved Granger supported atm
                if isinstance(data, SpectralData):
                    if data.data.shape[data.dimord.index('time')] != len(data.trials):
                        raise NotImplementedError("Time resolved Granger causality from tf-spectra not available atm")

            # by constraining to output='fourier', detrimental taper averaging
            # gets already catched by freqanalysis!

            check_effective_parameters(SpectralDyadicProduct, defaults, lcls)
            # there are no free parameters here,
            # everything had to be setup during freqanalysis!
            st_compRoutine = SpectralDyadicProduct()
            st_dimord = SpectralDyadicProduct.dimord

    # --- Set up of computation of trial-averaged CSDs is complete ---

    if method == 'coh':

        if output not in coh_outputs:
            lgl = f"one of {coh_outputs}"
            raise SPYValueError(lgl, varname="output", actual=output)
        log_dict['output'] = output

        # final normalization after trial averaging
        av_compRoutine = NormalizeCrossSpectra(output=output)

    if method == 'granger':

        # spectral analysis only possible with AnalogData
        besides = ['tapsmofrq'] if isinstance(data, AnalogData) else None
        check_effective_parameters(GrangerCausality, defaults, lcls, besides=besides)

        # after trial averaging
        # hardcoded numerical parameters
        av_compRoutine = GrangerCausality(rtol=5e-6,
                                          nIter=100,
                                          cond_max=1e4
                                          )


    # -------------------------------------------------
    # Call the chosen single trial ComputationalRoutine
    # -------------------------------------------------

    # the single trial results need a new DataSet
    st_out = CrossSpectralData(dimord=st_dimord)

    # Perform the trial-parallelized computation of the matrix quantity
    st_compRoutine.initialize(data,
                              st_out._stackingDim,
                              chan_per_worker=None,   # no parallelisation over channels possible
                              keeptrials=keeptrials)
    st_compRoutine.compute(data, st_out, parallel=kwargs.get("parallel"), log_dict=log_dict)

    # for single trial cross-corr results <-> keeptrials is True
    if av_compRoutine is None:
        return st_out

    # ----------------------------------------------------------------------------------
    # Sanitize output and call the chosen ComputationalRoutine on the averaged ST output
    # ----------------------------------------------------------------------------------

    out = CrossSpectralData(dimord=st_dimord)

    # now take the trial average from the single trial CR as input
    av_compRoutine.initialize(st_out, out._stackingDim, chan_per_worker=None)
    av_compRoutine.pre_check()   # make sure we got a trial_average
    av_compRoutine.compute(st_out, out, parallel=kwargs.get("parallel"),
                           log_dict=log_dict)

    # attach potential older cfg's from the input
    # to support chained frontend calls..
    out.cfg.update(data.cfg)
    # attach frontend parameters for replay
    out.cfg.update({'connectivityanalysis': new_cfg})
    return out


def cross_spectra(data, method, nSamples,
                  foi, foilim, tapsmofrq,
                  nTaper, taper, taper_opt,
                  polyremoval, log_dict, timeAxis):

    '''
    Calculates the single trial cross-spectral densities from AnalogData
    '''


    # --- Basic foi sanitization ---

    foi, foilim = process_foi(foi, foilim, data.samplerate)

    # --- Setting up specific Methods ---
    if method == 'granger':

        if foi is not None or foilim is not None:
            lgl = "no foi specification for Granger analysis"
            actual = "foi or foilim specification"
            raise SPYValueError(lgl, 'foi/foilim', actual)

        nChannels = len(data.channel)
        nTrials = len(data.trials)
        # warn user if this ratio is not small
        if nChannels / nTrials > 0.1:
            msg = "Multi-channel Granger analysis can be numerically unstable, it is recommended to have at least 10 times the number of trials compared to the number of channels. Try calculating in sub-groups of fewer channels!"
            SPYWarning(msg)

    if method in ['coh', 'granger']:

        # --- set up computation of the single trial CSDs ---

        # Construct array of maximally attainable frequencies
        freqs = np.fft.rfftfreq(nSamples, 1 / data.samplerate)

        # Match desired frequencies as close as possible to
        # actually attainable freqs
        # these are the frequencies attached to the CrossSpectralData by the CR!
        if foi is not None:
            foi, _ = best_match(freqs, foi, squash_duplicates=True)
        elif foilim is not None:
            foi, _ = best_match(freqs, foilim, span=True, squash_duplicates=True)
        elif foi is None and foilim is None:
            # Construct array of maximally attainable frequencies
            msg = (f"Setting frequencies of interest to {freqs[0]:.1f}-"
                   f"{freqs[-1]:.1f}Hz")
            SPYInfo(msg)
            foi = freqs

        # sanitize taper selection and retrieve dpss settings
        taper, taper_opt = process_taper(taper,
                                         taper_opt,
                                         tapsmofrq,
                                         nTaper,
                                         keeptapers=False,   # ST_CSD's always average tapers
                                         foimax=foi.max(),
                                         samplerate=data.samplerate,
                                         nSamples=nSamples,
                                         output="pow")   # ST_CSD's always have this unit/norm

        log_dict["foi"] = foi
        log_dict["taper"] = taper
        if taper_opt and taper == 'dpss':
            log_dict["nTaper"] = taper_opt["Kmax"]
            log_dict["tapsmofrq"] = tapsmofrq
        elif taper_opt:
            log_dict["taper_opt"] = taper_opt

        # parallel computation over trials
        st_compRoutine = CrossSpectra(samplerate=data.samplerate,
                                      nSamples=nSamples,
                                      taper=taper,
                                      taper_opt=taper_opt,
                                      demean_taper=method == 'granger',
                                      polyremoval=polyremoval,
                                      timeAxis=timeAxis,
                                      foi=foi)
        # hard coded as class attribute
        st_dimord = CrossSpectra.dimord

        return st_compRoutine, st_dimord
