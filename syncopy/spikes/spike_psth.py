# -*- coding: utf-8 -*-
#
# Syncopy PSTH frontend
#

import numpy as np

# Syncopy imports
from syncopy.shared.parsers import data_parser, scalar_parser, array_parser
from syncopy.shared.tools import get_defaults
from syncopy.datatype import SpikeData, TimeLockData

from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYWarning, SPYInfo
from syncopy.shared.kwarg_decorators import (unwrap_cfg, unwrap_select,
                                             detect_parallel_client)


# method specific imports - they should go when
# we have multiple returns
from syncopy.spikes.psth import Rice_rule, sqrt_rule, get_chan_unit_combs

# Local imports
from .compRoutines import PSTH

available_binsizes = {'rice': Rice_rule, 'sqrt': sqrt_rule}
available_outputs = ['rate', 'spikecount', 'proportion']
available_latencies = ['maxperiod', 'minperiod', 'prestim', 'poststim']

# ===DEV SNIPPET===
from syncopy.tests import synth_data as sd
spd = sd.poisson_noise(10, nUnits=7, nChannels=3, nSpikes=1000)
# =================


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
    vartriallen : bool, optional
        `True` (default): accept variable trial lengths and use all
        available trials and the samples in every trial.
        Missing values (empty bins) will be ignored in the
        computation and results stored as NaNs
        `False` : only select those trials that fully cover the
        window as specified by `latency` and discard
        those trials that do not.
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
            data, varname="data", dataclass="SpikeData",
            writable=None, empty=False, dimord=['sample', 'channel', 'unit']
        )
    except Exception as exc:
        raise exc

    # --- parse and digest `latency` (time window of analysis) ---

    if isinstance(latency, str):
        if latency not in available_latencies:
            lgl = f"one of {available_latencies}"
            act = latency
            raise SPYValueError(lgl, varname='latency', actual=act)

        # beginnings and ends of all trials in relative time
        beg_ends = (data.sampleinfo - (
            data.sampleinfo[:, 0] + data.trialdefinition[:, 2])[:, None]
                ) / data.samplerate
        trl_starts = beg_ends[:, 0]
        trl_ends = beg_ends[:, 1]

        # find overlapping interval for all trials
        if latency == 'minperiod':
            # latest start and earliest finish
            interval = [np.max(trl_starts), np.min(trl_ends)]
            if interval[0] > interval[1]:
                lgl = 'overlapping trials'
                act = f"{latency} - no common time window for all trials"
                raise SPYValueError(lgl, 'latency', act)

        # cover maximal time window where
        # there is still some data in at least 1 trial
        elif latency == 'maxperiod':
            interval = [np.min(trl_starts), np.max(trl_ends)]

        elif latency == 'prestim':
            if not np.any(trl_starts < 0):
                lgl = "pre-stimulus recordings"
                act = "no pre-stimulus (t < 0) events"
                raise SPYValueError(lgl, 'latency', act)
            interval = [np.min(trl_starts), 0]

        elif latency == 'poststim':
            if not np.any(trl_ends > 0):
                lgl = "post-stimulus recordings"
                act = "no post-stimulus (t > 0) events"
                raise SPYValueError(lgl, 'latency', act)
            interval = [0, np.max(trl_ends)]
    # explicit time window in seconds
    else:
        array_parser(latency, lims=[0, np.inf], dims=(2,))
        interval = latency

    # --- determine overall (all trials) histogram shape ---

    # get average trial size for auto-binning
    av_trl_size = data.data.shape[0] / len(data.trials)

    # TODO: respect time window (latency)
    if binsize in available_binsizes:
        nBins = available_binsizes[binsize](av_trl_size)
        bins = np.linspace(*interval, nBins)
    else:
        # include rightmost bin edge
        bins = np.arange(interval[0], interval[1] + binsize, binsize)
        nBins = len(bins)
    # print(interval, bins)

    # it's a sequential loop
    combs = get_chan_unit_combs(data.trials)

    # right away create the output labels for the channel axis
    chan_labels = [f'channel{i}_unit{j}' for i, j in combs]

    # now we have our global (single-trial, avg, std,..) histogram shape
    h_shape = (nBins, len(combs))

    # --- populate the log

    log_dict = {'bins': bins,
                'latency': latency
                }
    
    # --- set up CR ---

    method_kwargs = {
        'chan_unit_combs': combs,
        'tbins': bins,
        'samplerate': data.samplerate
    }

    # no named `trl_start` and `onset` for distributing positional args
    psth_cR = PSTH(data.trialdefinition[:, 0],
                   data.trialdefinition[:, 2],
                   method_kwargs=method_kwargs
                   )

    # only available dimord=['time', 'channel'])
    psth_results = TimeLockData()
    psth_cR.initialize(data, chan_per_worker=None,
                       out_stackingdim=psth_results._stackingDim)

    psth_cR.compute(data,
                    psth_results,
                    parallel=kwargs.get("parallel"),
                    log_dict=log_dict)

    return combs
