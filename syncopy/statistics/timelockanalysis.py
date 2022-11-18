# -*- coding: utf-8 -*-
#
# Syncopy timelock-analysis methods
#

import os
import numpy as np
import h5py

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
from syncopy.statistics.compRoutines import Covariance

__all__ = ["timelockanalysis"]


@unwrap_cfg
@unwrap_select
@detect_parallel_client
def timelockanalysis(data,
                     latency='maxperiod',
                     covariance=False,
                     ddof=None,
                     trials='all',
                     **kwargs):
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
    ddof : int or None
        Degrees of freedom for covariance estimation, defaults to ``N - 1``
    trials : 'all' or sequence
        Trial selection for FieldTrip compatibility, alternatively use
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

    log_dict = {'latency': latency,
                'covariance': covariance,
                'ddof': ddof,
                'trials': trials
                }

    # to restore later
    select_backup = None if data.selection is None else data.selection.select.copy()

    if data.selection is not None:
        if trials != 'all' and data.selection.select['trials'] is not None:
            lgl = "either `trials != 'all'` or selection"
            act = "trial keyword and trial selection"
            raise SPYValueError(lgl, 'trials', act)
        # evaluate legacy `trials` keyword value as selection
        elif trials != 'all':
            select = data.selection.select
            select['trials'] = trials
            data.selectdata(select, inplace=True)
    elif trials != 'all':
        # error handling done here
        data.selectdata(trials=trials, inplace=True)

    # digest selections
    if data.selection is not None:
        # select trials either via selection of keyword:        
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

    # this will add/ammend the selection, respecting the latency window
    numDiscard = discard_trials_via_selection(data, window)

    if numDiscard > 0:
        msg = f"Discarded {numDiscard} trial(s) which did not fit into latency window"
        SPYWarning(msg)

    # apply latency window and create TimeLockData
    if data.selection is not None:
        select = data.selection.select.copy()
        select['toilim'] = window
        data.selectdata(select, inplace=True)
    else:
        data.selectdata(toilim=window, inplace=True)

    # start empty
    tld = spy.TimeLockData(samplerate=data.samplerate)

    # stream cut/selected trials into new dataset
    dset = _dataset_from_trials(data,
                                dset_name='data',
                                filename=tld._gen_filename())

    # no copy here
    tld.data = dset
    tld.trialdefinition = data.selection.trialdefinition

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

    # -- set up covariance CR --

    covCR = Covariance(ddof=ddof, statAxis=data.dimord.index('time'))
    out = spy.CrossSpectralData(dimord=spy.CrossSpectralData._defaultDimord)
    covCR.initialize(
        data,
        out._stackingDim,
        keeptrials=False,
    )
    # and compute
    covCR.compute(data, out, parallel=kwargs.get("parallel"), log_dict=log_dict)

    # attach computed cov as array
    tld._update_seq_dataset('cov', out.data[0, 0, ...])

    # restore initial selection
    if select_backup:
        data.selectdata(select_backup, inplace=True)

    return tld


def _dataset_from_trials(spy_data, dset_name='new_data', filename=None):
    """
    Helper to construct a new dataset from
    a trial Indexer, respecting selections
    """

    stackDim = spy_data._stackingDim
    # re-initialize the Indexer
    def trials():
        if spy_data.selection is None:
            return spy_data.trials
        else:
            return spy_data.selection.trials

    # shapes have to match except for stacking dim
    # which is guaranteed by the source trials Indexer
    stackingDimSize = sum([trl.shape[stackDim] for trl in trials()])

    new_shape = list(trials()[0].shape)
    # plug in stacking dimension
    new_shape[stackDim] = stackingDimSize

    if filename is None:
        # generates new name with same extension
        filename = spy_data._gen_filename()

    # create new hdf5 File and dataset
    with h5py.File(filename, mode='w') as h5f:
        new_ds = h5f.create_dataset(dset_name, shape=new_shape)

        # all-to-all indexer
        idx = [slice(None) for _ in range(len(new_shape))]
        # stacking dim chunk size counter
        stacking = 0
        # now stream the trials into the new dataset
        for trl in trials():
            # length along stacking dimension
            trl_len = trl.shape[stackDim]
            # define the chunk and increment stacking dim indexer
            idx[stackDim] = slice(stacking, trl_len)
            stacking += trl_len
            # insert the trial
            new_ds[tuple(idx)] = trl

    # open again for reading and return dataset directly
    return h5py.File(filename, mode='r+')[dset_name]
