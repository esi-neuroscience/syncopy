# -*- coding: utf-8 -*-
#
# Validators for user submitted frontend arguments like foi, taper, etc.
# Input args are the parameters to check for validity + auxiliary parameters
# needed for the checks.
#

# Builtin/3rd party package imports
import numpy as np
from numbers import Number

from syncopy.shared.errors import SPYValueError, SPYWarning, SPYInfo
from syncopy.shared.parsers import scalar_parser, array_parser
from syncopy.shared.const_def import availableTapers, generalParameters, availablePaddingOpt
from syncopy.datatype.methods.padding import _nextpow2


def validate_padding(pad_to_length, lenTrials):
    """
    Simplified padding
    """
    # supported padding options
    not_valid = False
    if not isinstance(pad_to_length, (Number, str, type(None))):
        not_valid = True
    elif isinstance(pad_to_length, str) and pad_to_length not in availablePaddingOpt:
        not_valid = True
    if isinstance(pad_to_length, bool): # bool is an int subclass, check for it separately...
        not_valid = True
    if not_valid:
        lgl = "`None`, 'nextpow2' or an integer like number"
        actual = f"{pad_to_length}"
        raise SPYValueError(legal=lgl, varname="pad_to_length", actual=actual)

    # here we check for equal lengths trials in case of no user specified absolute padding length
    # we do a rough 'maxlen' padding, nextpow2 will be overruled in this case
    if lenTrials.min() != lenTrials.max() and not isinstance(pad_to_length, Number):
        abs_pad = int(lenTrials.max())
        msg = f"Unequal trial lengths present, automatic padding to {abs_pad} samples"
        SPYWarning(msg)

    # zero padding of ALL trials the same way
    if isinstance(pad_to_length, Number):

        scalar_parser(pad_to_length,
                      varname='pad_to_length',
                      ntype='int_like',
                      lims=[lenTrials.max(), np.inf])
        abs_pad = pad_to_length

    # or pad to optimal FFT lengths
    # (not possible for unequal lengths trials)
    elif pad_to_length == 'nextpow2':
        # after padding
        abs_pad = _nextpow2(int(lenTrials.min()))
    # no padding, equal lengths trials
    elif pad_to_length is None:
        abs_pad = int(lenTrials.max())

    # `abs_pad` is now the (soon to be padded) signal length in samples

    return abs_pad


def validate_foi(foi, foilim, samplerate):

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
                             lims=[0, samplerate/2], dims=(None,))
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
                             lims=[0, samplerate/2], dims=(2,))
            except Exception as exc:
                raise exc
            # foilim is of shape (2,)
            if foilim[0] > foilim[1]:
                msg = "Sorting foilim low to high.."
                SPYInfo(msg)
                foilim = np.sort(foilim)

    return foi, foilim


def validate_taper(taper,
                   tapsmofrq,
                   nTaper,
                   keeptapers,
                   foimax,
                   samplerate,
                   nSamples,
                   output):

    """
    General taper validation and Slepian/dpss input sanitization.
    The default is to max out `nTaper` to achieve the desired frequency
    smoothing bandwidth. For details about the Slepion settings see

    "The Effective Bandwidth of a Multitaper Spectral Estimator,
    A. T. Walden, E. J. McCoy and D. B. Percival"

    Parameters
    ----------
    taper : str
        Windowing function, one of :data:`~syncopy.shared.const_def.availableTapers`
    tapsmofrq : float or None
        Taper smoothing bandwidth for `taper='dpss'`
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
    dpss_opt : dict
        For multi-tapering (`taper='dpss'`) contains the
        parameters `NW` and `Kmax` for `scipy.signal.windows.dpss`.
        For all other tapers this is an empty dictionary.
    """

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

        # empty dpss_opt, only Slepians have options
        return {}

    # direct mtm estimate (averaging) only valid for spectral power
    if taper == "dpss" and not keeptapers and output != "pow":
        lgl = "'pow', the only valid option for taper averaging"
        raise SPYValueError(legal=lgl, varname="output", actual=output)

    # Set/get `tapsmofrq` if we're working w/Slepian tapers
    elif taper == "dpss":

        # --- minimal smoothing bandwidth ---
        # --- such that Kmax/nTaper is at least 1
        minBw = 2 * samplerate / nSamples
        # -----------------------------------

        # user set tapsmofrq directly
        if tapsmofrq is not None:
            try:
                scalar_parser(tapsmofrq, varname="tapsmofrq", lims=[0, np.inf])
            except Exception as exc:
                raise exc

            if tapsmofrq < minBw:
                msg = f'Setting tapsmofrq to the minimal attainable bandwidth of {minBw:.2f}Hz'
                SPYInfo(msg)
                tapsmofrq = minBw

        # we now enforce a user submitted smoothing bw
        else:
            lgl = "smoothing bandwidth in Hz, typical values are in the range 1-10Hz"
            raise SPYValueError(legal=lgl, varname="tapsmofrq", actual=tapsmofrq)

            # Try to derive "sane" settings by using 3/4 octave
            # smoothing of highest `foi`
            # following Hill et al. "Oscillatory Synchronization in Large-Scale
            # Cortical Networks Predicts Perception", Neuron, 2011
            # FIX ME: This "sane setting" seems quite excessive (huuuge bwidths)

            # tapsmofrq = (foimax * 2**(3 / 4 / 2) - foimax * 2**(-3 / 4 / 2)) / 2
            # msg = f'Automatic setting of `tapsmofrq` to {tapsmofrq:.2f}'
            # SPYInfo(msg)

        # --------------------------------------------
        # set parameters for scipy.signal.windows.dpss
        NW = tapsmofrq * nSamples / (2 * samplerate)
        # from the minBw setting NW always is at least 1
        Kmax = int(2 * NW - 1) # optimal number of tapers
        # --------------------------------------------

        # the recommended way:
        # set nTaper automatically to achieve exact effective smoothing bandwidth
        if nTaper is None:
            msg = f'Using {Kmax} taper(s) for multi-tapering'
            SPYInfo(msg)
            dpss_opt = {'NW' : NW, 'Kmax' : Kmax}
            return dpss_opt

        elif nTaper is not None:
            try:
                scalar_parser(nTaper,
                              varname="nTaper",
                              ntype="int_like", lims=[1, np.inf])
            except Exception as exc:
                raise exc

            if nTaper != Kmax:
                msg = f'''
                Manually setting the number of tapers is not recommended
                and may (strongly) distort the effective smoothing bandwidth!\n
                The optimal number of tapers is {Kmax}, you have chosen to use {nTaper}.
                '''
                SPYWarning(msg)

            dpss_opt = {'NW' : NW, 'Kmax' : nTaper}
            return dpss_opt


def check_effective_parameters(CR, defaults, lcls):

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
    """
    # list of possible parameter names of the CR
    expected = CR.valid_kws + ["parallel", "select"]
    relevant = [name for name in defaults if name not in generalParameters]
    for name in relevant:
        if name not in expected and (lcls[name] != defaults[name]):
            msg = f"option `{name}` has no effect in method `{CR.__name__}`!"
            SPYWarning(msg, caller=__name__.split('.')[-1])


def check_passed_kwargs(lcls, defaults, frontend_name):

    '''
    Catch additional kwargs passed to the frontends
    which have no effect
    '''

    # unpack **kwargs of frontend call which
    # might contain arbitrary kws passed from the user
    kw_dict = lcls.get("kwargs")

    # nothing to do..
    if not kw_dict:
        return

    relevant = list(kw_dict.keys())
    expected = [name for name in defaults]

    for name in relevant:
        if name not in expected:
            msg = f"option `{name}` has no effect in `{frontend_name}`!"
            SPYWarning(msg, caller=__name__.split('.')[-1])

