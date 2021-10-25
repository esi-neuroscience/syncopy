# -*- coding: utf-8 -*-
#
# Syncopy connectivity analysis methods
#

# Builtin/3rd party package imports
import numpy as np

# Syncopy imports
from syncopy.shared.parsers import data_parser, scalar_parser, array_parser 
from syncopy.shared.tools import get_defaults
from syncopy.datatype import SpectralData, padding
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
    generalParameters
)

# CRs still missing, CFs are already there
from .single_trial_compRoutines import (
    cross_spectra_cF,
    cross_covariance_cF
)

__all__ = ["connectivityanalysis"]


@unwrap_cfg
@unwrap_select
@detect_parallel_client
def connectivityanalysis(data, method='csd', 
                         foi=None, foilim=None, pad_to_length=None,
                         polyremoval=None, taper="hann", tapsmofrq=None,
                         nTaper=None, toi="all",
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
            if isinstance(tsel, list):
                sinfo[tk, :] = [0, len(tsel)]
            else:
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
    
    print(lenTrials)

    # manual symmetric zero padding of ALL trials the same way
    
    if pad_to_length is not None:

        scalar_parser(pad_to_length,
                      varname='pad_to_length',
                      ntype='int_like',
                      lims=[lenTrials.max(), np.inf])
        
        padding_opt = {
            'padtype' : 'zero',
            'pad' : 'absolute',
            'padlength' : pad_to_length
        }

    else:
        padding_opt = None

    if method ==  'csd':
        # for now manually select a trial
        single_trial = data.trials[data._selection.trials]
        res, freqs = cross_spectra_cF(single_trial, padding_opt=padding_opt)
        print(res.shape)
