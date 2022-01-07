# -*- coding: utf-8 -*-
#
# Syncopy connectivity analysis methods
#

# Builtin/3rd party package imports
import numpy as np
from numbers import Number

# Syncopy imports
from syncopy.shared.parsers import data_parser, scalar_parser
from syncopy.shared.tools import get_defaults
from syncopy.datatype import CrossSpectralData
from syncopy.datatype.methods.padding import _nextpow2
from syncopy.shared.errors import (
    SPYValueError,
    SPYWarning,
    SPYInfo)
from syncopy.shared.kwarg_decorators import (unwrap_cfg, unwrap_select,
                                             detect_parallel_client)
from syncopy.shared.input_validators import (
    validate_taper,
    validate_foi,
    check_effective_parameters,
    check_passed_kwargs
)

from .ST_compRoutines import ST_CrossSpectra, ST_CrossCovariance
from .AV_compRoutines import NormalizeCrossSpectra, NormalizeCrossCov, GrangerCausality

__all__ = ["connectivity"]
availableMethods = ("coh", "corr", "granger")


@unwrap_cfg
@unwrap_select
@detect_parallel_client
def connectivity(data, method="coh", keeptrials=False, output="abs",
                 foi=None, foilim=None, pad_to_length=None,
                 polyremoval=None, taper="hann", tapsmofrq=None,
                 nTaper=None, out=None, **kwargs):

    """
    Perform connectivity analysis of Syncopy :class:`~syncopy.AnalogData` objects

    **Usage Summary**

    Options available in all analysis methods:

    * **foi**/**foilim** : frequencies of interest; either array of frequencies or
      frequency window (not both)
    * **polyremoval** : de-trending method to use (0 = mean, 1 = linear)

    List of available analysis methods and respective distinct options:
    
    "coh" : (Multi-) tapered coherency estimate
        Compute the normalized cross spectral densities
        between all channel combinations

        * **output** : one of ('abs', 'pow', 'fourier')
        * **taper** : one of :data:`~syncopy.shared.const_def.availableTapers`
        * **tapsmofrq** : spectral smoothing box for slepian tapers (in Hz)
        * **nTaper** : (optional) number of orthogonal tapers for slepian tapers
        * **pad_to_length**: either pad to an absolute length or set to `'nextpow2'`

    "corr" : Cross-correlations 
        Computes the one sided (positive lags) cross-correlations
        between all channel combinations. The maximal lag is half
        the trial lengths.

        * **keeptrials** : set to `True` for single trial cross-correlations

    "granger" : Spectral Granger-Geweke causality
        Computes linear causality estimates between
        all channel combinations. The intermediate cross-spectral
        densities can be computed via multi-tapering.

        * **taper** : one of :data:`~syncopy.shared.const_def.availableTapers`
        * **tapsmofrq** : spectral smoothing box for slepian tapers (in Hz)
        * **nTaper** : (optional, not recommended) number of slepian tapers
        * **pad_to_length**: either pad to an absolute length or set to `'nextpow2'`
    
    Parameters
    ----------
    data : `~syncopy.AnalogData`
        A non-empty Syncopy :class:`~syncopy.datatype.AnalogData` object
    method : str
        Connectivity estimation method, one of 'coh', 'corr', 'granger'
    output : str
        Relevant for cross-spectral density estimation (`method='coh'`)
        Use `'pow'` for absolute squared coherence, `'abs'` for absolute value of coherence 
        and`'fourier'` for the complex valued coherency.
    keeptrials : bool
        Relevant for cross-correlations (`method='corr'`)
        If `True` single-trial cross-correlations are returned.
    foi : array-like or None
        Frequencies of interest (Hz) for output. If desired frequencies cannot be
        matched exactly, the closest possible frequencies are used. If `foi` is `None`
        or ``foi = "all"``, all attainable frequencies (i.e., zero to Nyquist / 2)
        are selected.
    foilim : array-like (floats [fmin, fmax]) or None or "all"
        Frequency-window ``[fmin, fmax]`` (in Hz) of interest. The 
        `foi` array will be constructed in 1Hz steps from `fmin` to
        `fmax` (inclusive).
    pad_to_length : int, None or 'nextpow2' 
        Padding of the input data, if set to a number pads all trials
        to this absolute length. E.g. `pad_to_length=2000` pads all
        trials to 2000 samples, if and only if the longest trial is
        at maximum 2000 samples.
        Alternatively if all trials have the same initial lengths
        setting `pad_to_length='nextpow2'` pads all trials to 
        the next power of two.
        If `None` and trials have unequal lengths all trials get padded
        such that all have the absolute lengths of the longest trial.
    taper : str
        Only valid if `method` is `'coh'` or `'granger'`. Windowing function,
        one of :data:`~syncopy.specest.const_def.availableTapers` 
    tapsmofrq : float
        Only valid if `method` is `'coh'` or `'granger'` and `taper` is `'dpss'`.
        The amount of spectral smoothing through  multi-tapering (Hz).
        Note that smoothing frequency specifications are one-sided,
        i.e., 4 Hz smoothing means plus-minus 4 Hz, i.e., a 8 Hz smoothing box.
    nTaper : int or None
        Only valid if `method` is `'coh'` or `'granger'` and `taper='dpss'`.
        Number of orthogonal tapers to use. It is not recommended to set the number
        of tapers manually! Leave at `None` for the optimal number to be set automatically.

    Examples
    --------

    ...
    """

    # Make sure our one mandatory input object can be processed
    try:
        data_parser(data, varname="data", dataclass="AnalogData",
                    writable=None, empty=False)
    except Exception as exc:
        raise exc
    timeAxis = data.dimord.index("time")

    # Get everything of interest in local namespace
    defaults = get_defaults(connectivity)
    lcls = locals()
    # check for ineffective additional kwargs
    check_passed_kwargs(lcls, defaults, frontend_name="connectivity")
    
    # Ensure a valid computational method was selected
    if method not in availableMethods:
        lgl = "'" + "or '".join(opt + "' " for opt in availableMethods)
        raise SPYValueError(legal=lgl, varname="method", actual=method)

    # if a subset selection is present
    # get sampleinfo and check for equidistancy
    if data._selection is not None:
        sinfo = data._selection.trialdefinition[:, :2]
        trialList = data._selection.trials
        # user picked discrete set of time points
        if isinstance(data._selection.time[0], list):
            lgl = "equidistant time points (toi) or time slice (toilim)"
            actual = "non-equidistant set of time points"
            raise SPYValueError(legal=lgl, varname="select", actual=actual)
    else:
        trialList = list(range(len(data.trials)))
        sinfo = data.sampleinfo
    lenTrials = np.diff(sinfo).squeeze()
    numTrials = len(trialList)

    # check polyremoval
    if polyremoval is not None:
        scalar_parser(polyremoval, varname="polyremoval", ntype="int_like", lims=[0, 1])    
    
    # --- Padding ---

    if method == "corr" and pad_to_length:
        lgl = "`None`, no padding needed/allowed for cross-correlations"
        actual = f"{pad_to_length}"
        raise SPYValueError(legal=lgl, varname="pad_to_length", actual=actual)

    # here we check for equal lengths trials as is required for
    # trial averaging, in case of no user specified absolute padding length
    # we do a rough 'maxlen' padding, nextpow2 will be overruled in this case
    if lenTrials.min() != lenTrials.max() and not isinstance(pad_to_length, Number):
        pad_to_length = int(lenTrials.max()) 
        msg = f"Unequal trial lengths present, automatic padding to {pad_to_length} samples"
        SPYWarning(msg)
        
    # symmetric zero padding of ALL trials the same way
    if isinstance(pad_to_length, Number):

        scalar_parser(pad_to_length,
                      varname='pad_to_length',
                      ntype='int_like',
                      lims=[lenTrials.max(), np.inf])
        padding_opt = {
            'padtype' : 'zero',
            'pad' : 'absolute',
            'padlength' : pad_to_length
        }
        # after padding!
        nSamples = pad_to_length
        
    # or pad to optimal FFT lengths
    # (not possible for unequal lengths trials)
    elif pad_to_length == 'nextpow2':
        padding_opt = {
            'padtype' : 'zero',
            'pad' : 'nextpow2'
        }
        # after padding
        nSamples = _nextpow2(int(lenTrials.min()))
    # no padding
    else:
        padding_opt = None
        nSamples = int(lenTrials.min())

    # --- Basic foi sanitization ---

    foi, foilim = validate_foi(foi, foilim, data.samplerate)

    # only now set foi array for foilim in 1Hz steps
    if foilim is not None:
        foi = np.arange(foilim[0], foilim[1] + 1)

    # Prepare keyword dict for logging (use `lcls` to get actually provided
    # keyword values, not defaults set above)
    log_dict = {"method": method,
                "output": output,
                "keeptrials": keeptrials,
                "polyremoval": polyremoval,
                "pad_to_length": pad_to_length}
        
    # --- Setting up specific Methods ---

    if method in ['coh', 'granger']:

        # --- set up computation of the single trial CSDs ---
        
        if keeptrials is not False:
            lgl = "False, trial averaging needed!"
            act = keeptrials
            raise SPYValueError(lgl, varname="keeptrials", actual=act)
        
        if foi is None and foilim is None:
            # Construct array of maximally attainable frequencies
            freqs = np.fft.rfftfreq(nSamples, 1 / data.samplerate)
            msg = (f"Automatic FFT frequency selection from {freqs[0]:.1f}Hz to "
                   f"{freqs[-1]:.1f}Hz")
            SPYInfo(msg)
            foi = freqs

        # sanitize taper selection and retrieve dpss settings
        taper_opt = validate_taper(taper,
                                   tapsmofrq,
                                   nTaper,
                                   keeptapers=False, # ST_CSD's always average tapers
                                   foimax=foi.max(),
                                   samplerate=data.samplerate,
                                   nSamples=nSamples,
                                   output="pow") # ST_CSD's always have this unit/norm
        
        log_dict["foi"] = foi
        log_dict["taper"] = taper
        # only dpss returns non-empty taper_opt dict
        if taper_opt:
            log_dict["nTaper"] = taper_opt["Kmax"]
            log_dict["tapsmofrq"] = tapsmofrq

        check_effective_parameters(ST_CrossSpectra, defaults, lcls)
        # parallel computation over trials
        st_compRoutine = ST_CrossSpectra(samplerate=data.samplerate,
                                         padding_opt=padding_opt,
                                         taper=taper,
                                         taper_opt=taper_opt,
                                         polyremoval=polyremoval,
                                         timeAxis=timeAxis,
                                         foi=foi)
        # hard coded as class attribute
        st_dimord = ST_CrossSpectra.dimord

    if method == 'coh':    
        # final normalization after trial averaging
        av_compRoutine = NormalizeCrossSpectra(output=output)

    if method == 'granger':
        # after trial averaging
        # hardcoded numerical parameters
        av_compRoutine = GrangerCausality(rtol=1e-8,
                                          nIter=100,
                                          cond_max=1e5
                                          )
        
    if method == 'corr':
        check_effective_parameters(ST_CrossCovariance, defaults, lcls)

        # single trial cross-correlations
        if keeptrials:
            av_compRoutine = None # no trial average
            norm = True # normalize individual trials within the ST CR
        else:
            av_compRoutine = NormalizeCrossCov()
            norm = False
        
        # parallel computation over trials
        st_compRoutine = ST_CrossCovariance(samplerate=data.samplerate,
                                            padding_opt=padding_opt,
                                            polyremoval=polyremoval,
                                            timeAxis=timeAxis,
                                            norm=norm)
        # hard coded as class attribute
        st_dimord = ST_CrossCovariance.dimord

    # -------------------------------------------------
    # Call the chosen single trial ComputationalRoutine
    # -------------------------------------------------

    # the single trial results need a new DataSet
    st_out = CrossSpectralData(dimord=st_dimord)

    # Perform the trial-parallelized computation of the matrix quantity
    st_compRoutine.initialize(data,
                              st_out._stackingDim,
                              chan_per_worker=None, # no parallelisation over channels possible
                              keeptrials=keeptrials) # we most likely need trial averaging!
    st_compRoutine.compute(data, st_out, parallel=kwargs.get("parallel"), log_dict=log_dict)

    # if ever needed..
    # for single trial cross-corr results <-> keeptrials is True
    if keeptrials and av_compRoutine is None:
        if out is not None:
            msg = "Single trial processing does not support `out` argument but directly returns the results"
            SPYWarning(msg)
        return st_out 
    
    # ----------------------------------------------------------------------------------
    # Sanitize output and call the chosen ComputationalRoutine on the averaged ST output
    # ----------------------------------------------------------------------------------

    # If provided, make sure output object is appropriate
    if out is not None:
        try:
            data_parser(out, varname="out", writable=True, empty=True,
                        dataclass="CrossSpectralData",
                        dimord=st_dimord)
        except Exception as exc:
            raise exc
        new_out = False
    else:
        out = CrossSpectralData(dimord=st_dimord)
        new_out = True
        
    # now take the trial average from the single trial CR as input
    av_compRoutine.initialize(st_out, out._stackingDim, chan_per_worker=None)
    av_compRoutine.pre_check() # make sure we got a trial_average
    av_compRoutine.compute(st_out, out, parallel=False, log_dict=log_dict)

    # Either return newly created output object or simply quit
    return out if new_out else None
