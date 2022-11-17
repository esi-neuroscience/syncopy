# -*- coding: utf-8 -*-
# 
# Syncopy timelock-analysis methods
# 

import os
import numpy as np

# Syncopy imports

import syncopy as spy
from syncopy.shared.parsers import data_parser
from syncopy.shared.kwarg_decorators import (unwrap_cfg, unwrap_select,
                                             detect_parallel_client)
from syncopy.shared.input_processors import (
    check_effective_parameters,
    check_passed_kwargs
)
from syncopy.shared.tools import get_defaults, get_frontend_cfg
from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYInfo, SPYWarning

# local imports
from syncopy.statistics.misc import get_analysis_window, discard_trials_via_selection

__all__ = ["timelockanalysis"]


@unwrap_cfg
@unwrap_select
@detect_parallel_client
def timelockanalysis(data, latency='maxperiod', covariance=False, trials='all', **kwargs):
    """
    Average, variance and covariance for :class:`~syncopy.AnalogData` objects across trials

    Parameters
    ----------
    data : Syncopy :class:`~syncopy.AnalogData` object
        Syncopy :class:`~syncopy.AnalogData` object to be averaged across trials
    latency : array_like or {'maxperiod', 'minperiod', 'prestim', 'poststim'}
        Either set desired time window (`[begin, end]`) in
        seconds, 'maxperiod' (default) for the maximum period
        available or `'minperiod' for minimal time-window all trials share,
        or `'prestim'` (all t < 0) or `'poststim'` (all t > 0)
        FieldTrip note: this also sets `covarianceWindow`
    covariance : bool
        Set to ``True`` to also compute covariance over channels
    trials : 'all' or sequence
        Trial selection for FieldTrip compatibilty, alternatively use 
        standard Syncopy ``select`` dictionary which  also allows additional 
        selections over channels.

    Returns
    -------
    out : :class:`~syncopy.TimeLockData`
        Time locked data object, with additional datasets:
        "avg", "var" and "cov" if ``convariance`` was set to ``True``

    """

    try:
        data_parser(data, varname="data", empty=False,
                    dataclass="AnalogData")
    except Exception as exc:
        raise exc

    if not isinstance(covariance, bool):
        raise SPYTypeError(covariance, varname='covariance', expected='Bool')

    # -- standard block to check and store provided kwargs/cfg --

    defaults = get_defaults(timelockanalysis)
    lcls = locals()
    # check for ineffective additional kwargs
    check_passed_kwargs(lcls, defaults, frontend_name="timelockanalysis")
    # save frontend call in cfg
    new_cfg = get_frontend_cfg(defaults, lcls, kwargs)

    # digest selections
    if data.selection is not None:
        trl_def = data.selection.trialdefinition
        sinfo = data.selection.trialdefinition[:, :2]
        trials = data.selection.trials
        trl_ivl = data.selection.trialintervals
        # beginnings and ends of all (selected) trials in trigger-relative time
        trl_starts, trl_ends = trl_ivl[:, 0], trl_ivl[:, 1]

    else:
        trl_def = data.trialdefinition
        sinfo = data.sampleinfo
        trials = data.trials
        # beginnings and ends of all (selected) trials in trigger-relative time
        trl_starts, trl_ends = data.trialintervals[:, 0], data.trialintervals[:, 1]

    # --- parse and digest `latency` (time window of analysis) ---

    # parses str and sequence arguments and returns window as toilim
    window = get_analysis_window(data, latency)

    # to restore later
    select_backup = None if data.selection is None else data.selection.select.copy()

    # this will add/ammend the selection, respecting the latency window
    numDiscard = discard_trials_via_selection(data, window)

    if numDiscard > 0:
        msg = f"Discarded {numDiscard} trial(s) which did not fit into latency window"
        SPYWarning(msg)
    
    # apply latency window and create TimeLockData
    # via dummy AnalogData
    if data.selection is not None:
        select = data.selection.select.copy()
        select['toilim'] = window
        dummy = data.selectdata(select)
    else:
        dummy = data.selectdata(toilim=window)

    # no copy here
    tld = spy.TimeLockData(data=dummy.data,
                           samplerate=data.samplerate,
                           trialdefinition=dummy.trialdefinition)
    # del dummy
    dummy.data = None  # this is a trick to keep the hdf5 dataset alive

    # now calculate via standard statistics
    avg = spy.mean(tld, dim='trials')
    var = spy.var(tld, dim='trials')

    # and attach to TimeLockData
    tld._update_seq_dataset('avg', avg.data)
    tld._update_seq_dataset('var', var.data)

    # delete unneded objects
    avg.data = None
    var.data = None
    del avg, var

    # restore initial selection
    if select_backup:
        data.selectdata(select_backup, inplace=True)

    return tld
