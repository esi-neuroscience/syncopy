# -*- coding: utf-8 -*-
#
# Syncopy statistics common helpers
#

# 3rd party/built in imports
import numpy as np

# Syncopy imports
from syncopy.shared.parsers import array_parser
from syncopy.shared.errors import SPYValueError

available_latencies = ['maxperiod', 'minperiod', 'prestim', 'poststim']


def get_analysis_window(data, latency):
    """
    Given the frontend `latency` parameter determine the
    analysis time window [start, end] in seconds

    Parameters
    ----------
    data : Syncopy data object
    latency : {'maxperiod', 'minperiod', 'prestim', 'poststim'} or array like

    Returns
    -------
    window : list
        [start, end] in seconds
    """

    # beginnings and ends of all (selected) trials in trigger-relative time in seconds
    if data.selection is not None:
        trl_starts, trl_ends = data.selection.trialintervals[:, 0], data.selection.trialintervals[:, 1]
    else:
        trl_starts, trl_ends = data.trialintervals[:, 0], data.trialintervals[:, 1]

    if isinstance(latency, str):
        if latency not in available_latencies:
            lgl = f"one of {available_latencies}"
            act = latency
            raise SPYValueError(lgl, varname='latency', actual=act)

        # find overlapping window (timelocked time axis borders) for all trials
        if latency == 'minperiod':
            # latest start and earliest finish
            window = [np.max(trl_starts), np.min(trl_ends)]
            if window[0] > window[1]:
                lgl = 'overlapping trials'
                act = f"{latency} - no common time window for all trials"
                raise SPYValueError(lgl, 'latency', act)

        # cover maximal time window where
        # there is still some data in at least 1 trial
        elif latency == 'maxperiod':
            window = [np.min(trl_starts), np.max(trl_ends)]

        elif latency == 'prestim':
            if not np.any(trl_starts < 0):
                lgl = "pre-stimulus recordings"
                act = "no pre-stimulus (t < 0) events"
                raise SPYValueError(lgl, 'latency', act)
            window = [np.min(trl_starts), 0]

        elif latency == 'poststim':
            if not np.any(trl_ends > 0):
                lgl = "post-stimulus recordings"
                act = "no post-stimulus (t > 0) events"
                raise SPYValueError(lgl, 'latency', act)
            window = [0, np.max(trl_ends)]

    # explicit time window in seconds
    else:
        array_parser(latency, lims=[-np.inf, np.inf], dims=(2,))
        # check that at least some events are covered
        if latency[0] > trl_ends.max():
            lgl = "start of latency window before at least one trial ends"
            act = latency[0]
            raise SPYValueError(lgl, 'latency[0]', act)

        if latency[1] < trl_starts.min():
            lgl = "end of latency window after at least one trial starts"
            act = latency[1]
            raise SPYValueError(lgl, 'latency[1]', act)

        if latency[0] > latency[1]:
            lgl = "start < end latency window"
            act = f"start={latency[0]}, end={latency[1]}"
            raise SPYValueError(lgl, "latency", act)
        window = list(latency)

    return window


def discard_trials_via_selection(data, window):
    """
    Determine which trials fit into the desired
    analysis time ``window``, make (or ammend) a selection
    to discard those for the analysis
    """

    # beginnings and ends of all (selected) trials in trigger-relative time in seconds
    if data.selection is not None:
        trl_starts, trl_ends = data.selection.trialintervals[:, 0], data.selection.trialintervals[:, 1]
        trl_idx = np.arange(len(data.selection.trials))
    else:
        trl_starts, trl_ends = data.trialintervals[:, 0], data.trialintervals[:, 1]
        trl_idx = np.arange(len(data.trials))

    # trial idx for whole dataset
    bmask = (trl_starts <= window[0]) & (trl_ends >= window[1])

    # trials which fit completely into window
    fit_trl_idx = trl_idx[bmask]
    if fit_trl_idx.size == 0:
        lgl = 'at least one trial covering the latency window'
        act = 'no trial that completely covers the latency window'
        raise SPYValueError(lgl, varname='latency/vartriallen', actual=act)

    # the easy part, no selection so we make one
    if data.selection is None:
        data.selectdata(trials=fit_trl_idx, inplace=True)
        # redefinition needed
        numDiscard = len(trl_idx) - len(fit_trl_idx)
    else:
        # match fitting trials with selected ones
        fit_trl_idx = np.intersect1d(data.selection.trial_ids, fit_trl_idx)
        numDiscard = len(data.selection.trial_ids) - len(fit_trl_idx)

        if fit_trl_idx.size == 0:
            lgl = 'at least one trial covering the latency window'
            act = 'no trial that completely covers the latency window'
            raise SPYValueError(lgl, varname='latency/vartriallen', actual=act)

        # now modify and re-apply selection
        select = data.selection.select.copy()
        select['trials'] = fit_trl_idx
        data.selectdata(select, inplace=True)

    return numDiscard
