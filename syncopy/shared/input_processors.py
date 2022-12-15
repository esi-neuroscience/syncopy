# -*- coding: utf-8 -*-
#
# Processing of user submitted frontend arguments like foi, taper, etc.
# The processors return values needed directly for the
# downstream method calls.
# Input args are the parameters to check for validity + auxiliary parameters
# needed for the checks. Processors raise exceptions in case of invalid input.
#

# Builtin/3rd party package imports
import numpy as np
import numbers
from inspect import signature
from scipy.signal import windows

from syncopy.specest.mtmfft import _get_dpss_pars
from syncopy.shared.errors import SPYValueError, SPYWarning, SPYInfo
from syncopy.shared.parsers import scalar_parser, array_parser
from syncopy.shared.const_def import availableTapers, generalParameters, availablePaddingOpt
from syncopy.datatype.methods.padding import _nextpow2


def process_padding(pad, lenTrials, samplerate):

    """
    Simplified padding interface, for all taper based methods
    padding has to be done **after** tapering!

    This function returns a number indicating the total
    length in samples of all trials after padding. When
    inputted into fft related methods, the actual padding
    is then performed there.

    Parameters
    ----------
    pad : 'maxperlen', float or 'nextpow2'
        For the frontend default `maxperlen`, no padding is to
        be performed in case of equal length trials but unequal lengths
        trials get padded to the max. trial length.
        A float indicates the absolute length of
        all trials after padding in seconds. `'nextpow2'` pads all trials
        to the nearest power of two.
    lenTrials : sequence of int_like
        Sequence holding all individual trial lengths
    samplerate : float
        The sampling rate in Hz

    Returns
    -------
    abs_pad : int
        Absolute length of all trials after padding (in samples)

    """
    # supported padding options
    not_valid = False
    if not isinstance(pad, (numbers.Number, str)):
        not_valid = True
    elif isinstance(pad, str) and pad not in availablePaddingOpt:
        not_valid = True
        # bool is an int subclass, have to check for it separately...
    if isinstance(pad, bool):
        not_valid = True
    if not_valid:
        lgl = "'maxperlen', 'nextpow2' or a float number"
        actual = f"{pad}"
        raise SPYValueError(legal=lgl, varname="pad", actual=actual)

    # zero padding of ALL trials the same way
    if isinstance(pad, numbers.Number):

        scalar_parser(pad,
                      varname='pad',
                      lims=[lenTrials.max() / samplerate, np.inf])
        abs_pad = int(pad * samplerate)

    # or pad to optimal FFT lengths
    elif pad == 'nextpow2':
        abs_pad = _nextpow2(int(lenTrials.max()))

    # no padding in case of equal length trials
    elif pad == 'maxperlen':
        abs_pad = int(lenTrials.max())
        if lenTrials.min() != lenTrials.max():
            msg = f"Unequal trial lengths present, padding all trials to {abs_pad} samples"
            SPYInfo(msg)

    # `abs_pad` is now the (soon to be padded) signal length in samples

    return abs_pad


def process_foi(foi, foilim, samplerate):

    """
    Parameters
    ----------
    foi : 'all' or array like or None
        frequencies of interest
    foilim : 2-element sequence or None
        foi limits

    Other Parameters
    ----------------
    samplerate : float
        the samplerate in Hz

    Returns
    -------
    foi, foilim : tuple
        Either both are `None` or the
        user submitted one is parsed and returned

    Notes
    -----
    Setting both `foi` and `foilim` to `None` is valid, the
    subsequent analysis methods should all have a default way to
    select a standard set of frequencies (e.g. np.fft.fftfreq).
    """

    if foi is not None and foilim is not None:
        lgl = "either `foi` or `foilim` specification"
        act = "both"
        raise SPYValueError(legal=lgl, varname="foi/foilim", actual=act)

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
                             lims=[0, samplerate / 2], dims=(None,))
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
            array_parser(foilim, varname="foilim", hasinf=False, hasnan=False,
                         lims=[0, samplerate / 2], dims=(2,))

            # QUICKFIX for #392
            foilim = [float(f) for f in foilim]

            # foilim is of shape (2,)
            if foilim[0] > foilim[1]:
                msg = "Sorting foilim low to high.."
                SPYInfo(msg)
                foilim = np.sort(foilim)

    return foi, foilim


def process_taper(taper,
                  taper_opt,
                  tapsmofrq,
                  nTaper,
                  keeptapers,
                  foimax,
                  samplerate,
                  nSamples,
                  output):

    """
    General taper validation and Slepian/dpss input sanitization.

    For multi-tapering with slepian tapers the default is to max out
    `nTaper` to achieve the desired frequency smoothing bandwidth.
    For details about the Slepian settings see

    "The Effective Bandwidth of a Multitaper Spectral Estimator,
    A. T. Walden, E. J. McCoy and D. B. Percival"

    Parameters
    ----------
    taper : str
        Windowing function, one of :data:`~syncopy.shared.const_def.availableTapers`
    taper_opt : dict or None
        Dictionary holding additional keywords for tapers which have additional
        parameters like for example :func:`~scipy.signal.windows.kaiser`
    tapsmofrq : float or None
        Taper smoothing bandwidth for multi-tapering with implicit 'dpss' window
    nTaper : int_like or None
        Number of tapers to use for multi-tapering (not recommended)

    Other Parameters
    ----------------
    keeptapers : bool
    foimax : float
        Maximum frequency for the analysis
    samplerate : float
        the samplerate in Hz
    nSamples : int
        Number of samples
    output : str, one of {'abs', 'pow', 'fourier'}
        Fourier transformation output type

    Returns
    -------
    taper : str or None
        The user supplied taper
    taper_opt : dict
        For multi-tapering contains the
        keys `NW` and `Kmax` for `scipy.signal.windows.dpss`.
        For other tapers these are the additional parameters or
        an empty dictionary in case selected taper has no further args.
    """

    if taper == 'dpss':
        lgl = "set `tapsmofrq` parameter directly for multi-tapering"
        raise SPYValueError(legal=lgl, varname='taper', actual=taper)

    # no tapering at all
    if taper is None and tapsmofrq is None:
        return None, {}

    # See if taper choice is supported
    if taper not in availableTapers:
        lgl = "'" + "or '".join(opt + "' " for opt in availableTapers)
        raise SPYValueError(legal=lgl, varname="taper", actual=taper)

    if not isinstance(taper_opt, (dict, type(None))):
        lgl = "dict or None"
        actual = type(taper_opt)
        raise SPYValueError(lgl, "taper_opt", actual)

    # -- no multi-tapering --
    if tapsmofrq is None:
        if nTaper is not None:
            msg = "`nTaper` is only used for multi-tapering!"
            SPYWarning(msg)
        if keeptapers:
            msg = "`keeptapers` is only used for multi-tapering!"
            SPYWarning(msg)

        # availableTapers are given by windows.__all__
        parameters = signature(getattr(windows, taper)).parameters
        supported_kws = list(parameters.keys())
        # 'M' is the kw for the window length
        # for all of scipy's windows
        supported_kws.remove('M')
        supported_kws.remove('sym')

        if taper_opt is not None:

            if len(supported_kws) == 0:
                lgl = f"`None`, taper '{taper}' has no additional parameters"
                raise SPYValueError(lgl, varname='taper_opt', actual=taper_opt)

            for key in taper_opt:
                if key not in supported_kws:
                    lgl = f"one of {supported_kws} for `taper='{taper}'`"
                    raise SPYValueError(lgl, "taper_opt key", key)
            for key in supported_kws:
                if key not in taper_opt:
                    lgl = f"additional parameter '{key}' for `taper='{taper}'`"
                    raise SPYValueError(lgl, "taper_opt", None)
            # all supplied keys are fine
            return taper, taper_opt

        elif len(supported_kws) > 0:
            lgl = f"additional parameters for taper '{taper}': {supported_kws}"
            raise SPYValueError(lgl, varname='taper_opt', actual=taper_opt)
        else:
            # taper_opt was None and taper needs no additional parameters
            return taper, {}

    # -- multi-tapering --
    else:
        if taper != 'hann':
            lgl = "`None` for multi-tapering, just set `tapsmofrq`"
            raise SPYValueError(lgl, varname='taper', actual=taper)

        if taper_opt is not None:
            msg = "For multi-tapering use `tapsmofrq` and `nTaper` to control frequency smoothing, `taper_opt` has no effect"
            SPYWarning(msg)

        # direct mtm estimate (averaging) only valid for spectral power
        if not keeptapers and output != "pow":
            lgl = (f"'pow'|False or '{output}'|True, set either keeptapers=True "
                   "or `output='pow'`!")
            raise SPYValueError(legal=lgl, varname="output|keeptapers", actual=f"'{output}'|{keeptapers}")

        # --- minimal smoothing bandwidth ---
        # --- such that Kmax/nTaper is at least 1
        minBw = 2 * samplerate / nSamples
        # -----------------------------------

        # --- maximal smoothing bandwidth ---
        # --- such that Kmax < nSamples and NW < nSamples / 2
        maxBw = np.min([samplerate / 2 - 1 / nSamples,
                        samplerate * (nSamples + 1) / (2 * nSamples)])
        # -----------------------------------

        try:
            scalar_parser(tapsmofrq, varname="tapsmofrq", lims=[0, np.inf])
        except Exception:
            lgl = "smoothing bandwidth in Hz, typical values are in the range 1-10Hz"
            raise SPYValueError(legal=lgl, varname="tapsmofrq", actual=tapsmofrq)

        if tapsmofrq < minBw:
            msg = f'Setting tapsmofrq to the minimal attainable bandwidth of {minBw:.2f}Hz'
            SPYInfo(msg)
            tapsmofrq = minBw

        if tapsmofrq > maxBw:
            msg = f'Setting tapsmofrq to the maximal attainable bandwidth of {maxBw:.2f}Hz'
            SPYInfo(msg)
            tapsmofrq = maxBw

        # --------------------------------------------------------------
        # set parameters for scipy.signal.windows.dpss
        NW, Kmax = _get_dpss_pars(tapsmofrq, nSamples, samplerate)
        # --------------------------------------------------------------

        # tapsmofrq too large
        # if Kmax > nSamples or NW > nSamples / 2:

        # the recommended way:
        # set nTaper automatically to achieve exact effective smoothing bandwidth
        if nTaper is None:
            msg = f'Using {Kmax} taper(s) for multi-tapering'
            SPYInfo(msg)
            dpss_opt = {'NW': NW, 'Kmax': Kmax}
            return 'dpss', dpss_opt

        elif nTaper is not None:

            scalar_parser(nTaper,
                          varname="nTaper",
                          ntype="int_like", lims=[1, np.inf])

            if nTaper != Kmax:
                msg = f'''
                Manually setting the number of tapers is not recommended
                and may (strongly) distort the effective smoothing bandwidth!\n
                The optimal number of tapers is {Kmax}, you have chosen to use {nTaper}.
                '''
                SPYWarning(msg)

            dpss_opt = {'NW': NW, 'Kmax': nTaper}
            return 'dpss', dpss_opt


def check_effective_parameters(CR, defaults, lcls, besides=None):

    """
    For a given ComputationalRoutine, compare set parameters
    (*lcls*) with the accepted parameters and the frontend
    meta function *defaults* to warn if any ineffective parameters are set.

    Parameters
    ----------
    CR : :class:`~syncopy.shared.computational_routine.ComputationalRoutine
        Needs to have a `valid_kws` attribute
    defaults : dict
        Result of :func:`~syncopy.shared.tools.get_defaults`, the frontend
        parameter names plus values with default values
    lcls : dict
        Result of `locals()`, all names and values of the local (frontend-)name space
    besides : list or None
        List of kws which don't get checked
    """
    # list of possible parameter names of the CR
    expected = CR.valid_kws + ["parallel", "select"]
    if besides is not None:
        expected += besides

    relevant = [name for name in defaults if name not in generalParameters]

    for name in relevant:
        if name not in expected and (lcls[name] != defaults[name]):
            msg = f"option `{name}` has no effect for `{CR.__name__}`!"
            SPYWarning(msg, caller=__name__.split('.')[-1])


def check_passed_kwargs(lcls, defaults, frontend_name):
    """
    Catch additional kwargs passed to the frontends
    which have no effect
    """

    # unpack **kwargs of frontend call which
    # might contain arbitrary kws passed from the user
    kw_dict = lcls.get("kwargs")

    # nothing to do..
    if not kw_dict:
        return

    relevant = list(kw_dict.keys())
    expected = [name for name in defaults] + ['chan_per_worker']

    for name in relevant:
        if name not in expected:
            msg = f"option `{name}` has no effect in `{frontend_name}`!"
            SPYWarning(msg, caller=__name__.split('.')[-1])
