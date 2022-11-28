# -*- coding: utf-8 -*-
#
# Syncopy PSTH frontend
#

import numpy as np

# Syncopy imports
import syncopy as spy
from syncopy.shared.parsers import data_parser, scalar_parser, array_parser
from syncopy.shared.tools import get_defaults, get_frontend_cfg
from syncopy.datatype import TimeLockData
from syncopy.datatype.base_data import Indexer

from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYInfo
from syncopy.shared.kwarg_decorators import (
    unwrap_cfg,
    unwrap_select,
    detect_parallel_client
)
from syncopy.shared.input_processors import check_passed_kwargs
from syncopy.shared.latency import get_analysis_window, create_trial_selection

# Local imports
from syncopy.statistics.compRoutines import PSTH
from syncopy.statistics.psth import Rice_rule, sqrt_rule, get_chan_unit_combs

available_binsizes = {'rice': Rice_rule, 'sqrt': sqrt_rule}
available_outputs = ['rate', 'spikecount', 'proportion']


@unwrap_cfg
@unwrap_select
@detect_parallel_client
def spike_psth(data,
               binsize='rice',
               output='rate',
               latency='maxperiod',
               vartriallen=True,
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
        'proportion' to normalize the area under the PSTH to 1
        Defaults to `'rate'`
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

    if not isinstance(vartriallen, bool):
        raise SPYTypeError(vartriallen, varname='vartriallen', expected='Bool')

    defaults = get_defaults(spike_psth)
    lcls = locals()
    # check for ineffective additional kwargs
    check_passed_kwargs(lcls, defaults, frontend_name="spike_psth")
    # save frontend call in cfg
    new_cfg = get_frontend_cfg(defaults, lcls, kwargs)

    # digest selections
    if data.selection is not None:
        trl_def = data.selection.trialdefinition
        sinfo = data.selection.trialdefinition[:, :2]
        trials = data.selection.trials

    else:
        trl_def = data.trialdefinition
        sinfo = data.sampleinfo
        trials = data.trials
        trl_starts, trl_ends = data.trialintervals[:, 0], data.trialintervals[:, 1]

    # validate output parameter
    if output not in available_outputs:
        lgl = f"one of {available_outputs}"
        act = output
        raise SPYValueError(lgl, 'output', act)

    if isinstance(binsize, str):
        if binsize not in available_binsizes:
            lgl = f"one of {available_binsizes}"
            act = output
            raise SPYValueError(lgl, 'output', act)

    # --- parse and digest `latency` (time window of analysis) ---

    window = get_analysis_window(data, latency)

    # to restore later
    select_backup = None if data.selection is None else data.selection.select.copy()

    if not vartriallen:

        # this will create/ammend the selection, respecting the latency window
        select, numDiscard = create_trial_selection(data, window)

        msg = f"Discarded {numDiscard} trials which did not fit into latency window"
        SPYInfo(msg)

        # apply the updated selection
        data.selectdata(select, inplace=True)
        
        # now redefine local variables
        trl_def = data.selection.trialdefinition
        sinfo = data.selection.trialdefinition[:, :2]
        trials = data.selection.trials
    else:
        numDiscard = 0

    # --- determine overall (all selected trials) histogram shape ---

    # get average trial size for auto-binning
    av_trl_size = np.diff(sinfo).sum() / len(trials)

    if binsize in available_binsizes:
        nBins = available_binsizes[binsize](av_trl_size)
        bins = np.linspace(*window, nBins)
    else:
        # make sure we have at least 2 bins
        scalar_parser(binsize, varname='binsize', lims=[0, np.diff(window).squeeze()])
        # include rightmost bin edge
        bins = np.arange(window[0], window[1] + binsize, binsize)
        nBins = len(bins)

    # it's a sequential loop to get an array of [chan, unit] indices
    combs = get_chan_unit_combs(trials)

    # --- populate the log

    log_dict = {'bins': bins,
                'binsize': binsize,
                'latency': latency,
                'output': output,
                'vartriallen': vartriallen,
                'numDiscard': numDiscard
                }

    # --- set up CR ---

    # trl_start` and `onset` for distributing positional args to psth_cF
    trl_starts = trl_def[:, 0]
    trl_ends = trl_def[:, 1]
    trigger_onsets = trl_def[:, 2]
    psth_cR = PSTH(trl_starts,
                   trigger_onsets,
                   trl_ends,
                   chan_unit_combs=combs,
                   tbins=bins,
                   output=output,
                   samplerate=data.samplerate
                   )

    # only available dimord labels ['time', 'channel'])
    psth_results = TimeLockData()
    psth_cR.initialize(data, chan_per_worker=None,
                       out_stackingdim=psth_results._stackingDim,
                       keeptrials=keeptrials)

    psth_cR.compute(data,
                    psth_results,
                    parallel=kwargs.get("parallel"),
                    log_dict=log_dict)


    # calculate trial average and variance
    avg = spy.mean(psth_results, dim='trials', parallel=False)
    var = spy.var(psth_results, dim='trials', parallel=False)

    # attach data to TimeLockData
    psth_results._update_dataset('avg', avg.data)
    psth_results._update_dataset('var', var.data)

    # unregister datasets to detach from objects
    avg._unregister_dataset("data", del_from_file=False)
    var._unregister_dataset("data", del_from_file=False)

    # scramble filenames and delete unneeded objects
    avg.filename, var.filename = '', ''
    del avg, var

    # -- propagate old cfg and attach this one --
    psth_results.cfg.update(data.cfg)
    psth_results.cfg.update({'spike_psth': new_cfg})

    # finally revert possible in-place selections
    if select_backup is None:
        data.selection = None
    else:
        data.selectdata(select_backup, inplace=True)

    return psth_results
