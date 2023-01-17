# -*- coding: utf-8 -*-
#
# Syncopy latency processing
#

# 3rd party/built in imports
import numpy as np
from copy import deepcopy

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
    latency : {'maxperiod', 'minperiod', 'prestim', 'poststim'} or array like [start, end]

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
            lgl = f"start of latency window < {trl_ends.max()}s"
            act = latency[0]
            raise SPYValueError(lgl, 'latency[0]', act)

        if latency[1] < trl_starts.min():
            lgl = f"end of latency window > {trl_starts.min()}s"
            act = latency[1]
            raise SPYValueError(lgl, 'latency[1]', act)

        if latency[0] > latency[1]:
            lgl = "start < end latency window"
            act = f"start={latency[0]}, end={latency[1]}"
            raise SPYValueError(lgl, "latency", act)
        window = list(latency)

    return window


def create_trial_selection(data, window):
    """
    Determine which trials fit into the desired
    analysis time ``window``. Creates or ammends a ``select`` dictionary
    for discarding not fitting trials via :func:`syncopy.selectdata`

    Note that no toi/toilim selection is done here, hence to
    arrive at a timelocked dataset applying ``selectdata(toilim=window)``
    is still needed.

    Parameters
    ----------
    data : Syncopy data object
    window : sequence [start, end]
        The time window in seconds

    Returns
    -------
    select : dict
        A new or ammended kwarg dictionary for :func:`~syncopy.selectdata`
        selecting only trials which completely cover the analysis ``window``
    numDiscard : int
        Number of to be discarded trials
    """

    # beginnings and ends of all (selected) trials in trigger-relative time in seconds
    if data.selection is not None:
        trl_starts, trl_ends = data.selection.trialintervals[:, 0], data.selection.trialintervals[:, 1]
        trl_idx = np.array(data.selection.trial_ids)
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
        select = {'trials': fit_trl_idx}
        numDiscard = len(trl_idx) - len(fit_trl_idx)
    else:
        sel_ids = np.array(data.selection.trial_ids)[bmask]
        # match fitting trials with selected ones
        intersection = np.intersect1d(data.selection.trial_ids, sel_ids)
        # intersect result is sorted, restore original selection order
        fit_trl_idx = np.array(
            [trl_id for trl_id in data.selection.trial_ids if trl_id in intersection])
        numDiscard = len(data.selection.trial_ids) - len(fit_trl_idx)

        if fit_trl_idx.size == 0:
            lgl = 'at least one trial covering the latency window'
            act = 'no trial that completely covers the latency window'
            raise SPYValueError(lgl, varname='latency/vartriallen', actual=act)

        # now modify 
        select = deepcopy(data.selection.select)
        select['trials'] = fit_trl_idx

    return select, numDiscard
