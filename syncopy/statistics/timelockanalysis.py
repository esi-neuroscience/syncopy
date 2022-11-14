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
from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYInfo

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
                    dataclass=spy.AnalogData)
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
        
