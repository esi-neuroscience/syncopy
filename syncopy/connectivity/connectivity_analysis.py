# -*- coding: utf-8 -*-
#
# Syncopy connectivity analysis methods
#

# Builtin/3rd party package imports
import numpy as np
from numbers import Number

# Syncopy imports
from syncopy.shared.parsers import data_parser, scalar_parser, array_parser
from syncopy.shared.tools import get_defaults
from syncopy.datatype import CrossSpectralData, padding
from syncopy.datatype.methods.padding import _nextpow2
from syncopy.shared.tools import best_match
from syncopy.shared.errors import (
    SPYValueError,
    SPYTypeError,
    SPYWarning,
    SPYInfo)

from syncopy.shared.kwarg_decorators import (unwrap_cfg, unwrap_select,
                                             detect_parallel_client)

# Local imports
from .const_def import (
    availableTapers,
    availableMethods,
    generalParameters,
    nextpow2
)

# CRs still missing, CFs are already there
from .ST_compRoutines import (
    ST_CrossSpectra
)
from .AV_compRoutines import (
    Normalize_CrossMeasure
)
__all__ = ["connectivityanalysis"]


@unwrap_cfg
@unwrap_select
@detect_parallel_client
def connectivityanalysis(data, method="csd", keeptrials=False, output="abs",
                         foi=None, foilim=None, pad_to_length=None,
                         polyremoval=None, taper="hann", tapsmofrq=None,
                         nTaper=None, toi="all", out=None, 
                         **kwargs):

    """
    coming soon..
    """

    # Make sure our one mandatory input object can be processed
    try:
        data_parser(data, varname="data", dataclass="AnalogData",
                    writable=None, empty=False)
    except Exception as exc:
        raise exc
    timeAxis = data.dimord.index("time")

    # Get everything of interest in local namespace
    defaults = get_defaults(connectivityanalysis)
    lcls = locals()

    # Ensure a valid computational method was selected
    if method not in availableMethods:
        lgl = "'" + "or '".join(opt + "' " for opt in availableMethods)
        raise SPYValueError(legal=lgl, varname="method", actual=method)

    # If only a subset of `data` is to be processed,
    # make some necessary adjustments
    # and compute minimal sample-count across (selected) trials
    if data._selection is not None:
        trialList = data._selection.trials
        sinfo = np.zeros((len(trialList), 2))
        for tk, trlno in enumerate(trialList):
            trl = data._preview_trial(trlno)
            tsel = trl.idx[timeAxis]

            # user picked discrete set of time points
            if isinstance(tsel, list):
                lgl = "equidistant time points (toi) or time slice (toilim)"
                actual = "non-equidistant set of time points"
                raise SPYValueError(legal=lgl, varname="select", actual=actual)

            sinfo[tk, :] = [trl.idx[timeAxis].start, trl.idx[timeAxis].stop]
    else:
        trialList = list(range(len(data.trials)))
        sinfo = data.sampleinfo
    lenTrials = np.diff(sinfo).squeeze()

    # here we enforce equal lengths trials as is required for
    # sensical trial averaging - user is responsible for trial
    # specific padding and time axis alignments
    # OR we do a brute force 'maxlen' padding if there is unequal lengths?!
    if not lenTrials.min() == lenTrials.max():
        lgl = "trials of same lengths"
        actual = "trials of different lengths - please pre-pad!"
        raise SPYValueError(legal=lgl, varname="lenTrials", actual=actual)

    numTrials = len(trialList)

    # --- Padding ---

    # manual symmetric zero padding of ALL trials the same way
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
    elif pad_to_length == 'nextpow2':
        padding_opt = {
            'padtype' : 'zero',
            'pad' : 'nextpow2'
        }
        # after padding
        nSamples = nextpow2(int(lenTrials.min()))
    # no padding
    else:
        padding_opt = None
        nSamples = int(lenTrials.min())

    # --- foi sanitization ---

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

    # only now set foi array for foilim in 1Hz steps
    if foilim:
        foi = np.arange(foilim[0], foilim[1] + 1)

    if method ==  'csd':

        if foi is None and foilim is None:
            # Construct array of maximally attainable frequencies
            freqs = np.fft.rfftfreq(nSamples, 1 / data.samplerate)
            msg = (f"Automatic FFT frequency selection from {freqs[0]:.1f}Hz to "
                   f"{freqs[-1]:.1f}Hz")
            SPYInfo(msg)
            foi = freqs

        # Warn user about DPSS only settings
        if taper != "dpss":
            if tapsmofrq is not None:
                msg = "`tapsmofrq` is only used if `taper` is `dpss`!"
                SPYWarning(msg)
        
            
        # parallel computation over trials
        st_compRoutine = ST_CrossSpectra(samplerate=data.samplerate,
                                         padding_opt=padding_opt,
                                         taper=taper,
                                         taperopt={},
                                         polyremoval=polyremoval,
                                         timeAxis=timeAxis,
                                         foi=foi)
        
        # hard coded as class attribute
        st_dimord = ST_CrossSpectra.dimord
        
        # final normalization after trial averaging
        av_compRoutine = Normalize_CrossMeasure(output=output)

    # -------------------------------------------------
    # Call the chosen single trial ComputationalRoutine
    # -------------------------------------------------

    # the single trial results need a new DataSet
    st_out = CrossSpectralData(dimord=st_dimord)


    # Perform the trial-parallelized computation of the matrix quantity
    st_compRoutine.initialize(data,
                              chan_per_worker=None, # no parallelisation over channel possible
                              keeptrials=False) # we need trial averaging!    
    st_compRoutine.compute(data, st_out, parallel=kwargs.get("parallel"), log_dict={})

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
    av_compRoutine.initialize(st_out, chan_per_worker=None)
    av_compRoutine.check_input() # make sure we got a trial_average
    av_compRoutine.compute(st_out, out, parallel=False)
    
    # Either return newly created output object or simply quit
    return out if new_out else None

