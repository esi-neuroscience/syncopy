# -*- coding: utf-8 -*-
#
# Syncopy PSTH frontend
#

import numpy as np

# Syncopy imports
from syncopy.shared.parsers import data_parser, scalar_parser, array_parser
from syncopy.shared.tools import get_defaults
from syncopy.datatype import SpikeData

from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYWarning, SPYInfo
from syncopy.shared.kwarg_decorators import (unwrap_cfg, unwrap_select,
                                             detect_parallel_client)


# method specific imports - they should go when
# we have multiple returns
# from .psth import Rice_rule

# Local imports
# from .compRoutines import PSTH

available_binsizes = ['rice', 'sqrt']
available_outputs = ['rate', 'spikecount', 'proportion']
available_latencies = ['maxperiod', 'minperiod', 'prestim', 'poststim']


@unwrap_cfg
@unwrap_select
@detect_parallel_client
def spike_psth(data,
               binsize='rice',
               output='rate',
               latency='maxperiod',
               keeptrials=True,
               **kwargs):

    """
    Peristimulus time histogram

    Parameters
    ----------
    data : :class:`~syncopy.SpikeData`
        A non-empty Syncopy :class:`~syncopy.datatype.SpikeData` object
    binsize : float or one of {'rice', 'sqrt'}, optional
        Binsize in seconds or get optimal bin width via
        Rice rule (`'rice'`) or square root of number of observations (`'sqrt'`)
    output : {'rate', 'spikecount', 'proportion'}, optional
        Set to `'rate'` to convert the output to firing rates (spikes/sec),
        'spikecount' to count the number spikes per trial or
        'proportion' to normalize the area under the PSTH to 1.
    latency : array_like or {'maxperiod', 'minperiod', 'prestim', 'poststim'}
        Either set desired time window (`[begin, end]`) for spike counting in
        seconds, 'maxperiod' (default) for the maximum period
        available or `'minperiod' for minimal time-window all trials share,
        or `'prestim'` (all t < 0) or `'poststim'` (all t > 0)
    keeptrials : bool, optional
        If `True` the psth's of individual trials are returned, otherwise
        results are averaged across trials.

    """

    # Make sure our one mandatory input object can be processed
    try:
        data_parser(
            data, varname="data", dataclass="SpikeData", writable=None, empty=False
        )
    except Exception as exc:
        raise exc

    # --- parse and digest `latency` (time window of analysis) ---
    
    if isinstance(latency, str):
        if latency not in available_latencies:
            lgl = f"one of {available_latencies}"
            act = latency
            raise SPYValueError(lgl, varname='latency', actual=act)

        if latency == 'minperiod':
            # relative start and end of trials
            beg_ends = (data.sampleinfo - (
                data.sampleinfo[:, 0] + data.trialdefinition[:, 2])[:, None]
                        ) / data.samplerate
        
    else:
        array_parser(latency, lims=[0, np.inf])

    
    
    pass
