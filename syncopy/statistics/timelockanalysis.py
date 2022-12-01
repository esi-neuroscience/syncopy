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
from syncopy.shared.latency import get_analysis_window, create_trial_selection

# local imports
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
                     keeptrials=False,
                     **kwargs):
    """
    Average, variance and covariance for :class:`~syncopy.AnalogData` objects across trials
    If input ``data`` is not timelocked already, toilim and trial selections will be
    applied according to the ``latency`` setting.

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
    keeptrials : bool
        Set to ``False`` to get single trial covariances in ``out.cov``

    Returns
    -------
    out : :class:`~syncopy.TimeLockData`
        Time locked data object, with additional datasets:
        "avg", "var" and "cov" if ``convariance`` was set to ``True``

    """

    # -- check user input --

    data_parser(data, varname="data", empty=False,
                dataclass="AnalogData")

    if ddof is not None:
        if not isinstance(ddof, int) or ddof < 0:
            lgl = "positive integer value"
            act = ddof
            raise SPYValueError(lgl, 'ddof', act)

    if not isinstance(covariance, bool):
        raise SPYTypeError(covariance, varname='covariance', expected='bool')

    if not isinstance(keeptrials, bool):
        raise SPYTypeError(covariance, varname='keeptrials', expected='bool')

    # latency gets checked within selectdata(latency=...)

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

    # -- create outtput object --

    # start empty, data gets added later
    tld = spy.TimeLockData(samplerate=data.samplerate)

    # -- propagate old cfg and attach this one --
    tld.cfg.update(data.cfg)
    tld.cfg.update({'timelockanalysis': new_cfg})

    # to restore later as we apply selection inside here
    select_backup = None if data.selection is None else data.selection.select.copy()

    if data.selection is not None:
        if trials != 'all' and data.selection.select.get('trials') is not None:
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

    # --- apply `latency` (time window of analysis) ---
    if data.selection is not None:
        select = data.selection.select.copy()
        # update selection
        data.selectdata(select, latency=latency, inplace=True)
    else:
        # create new selection
        data.selectdata(latency=latency, inplace=True)

    # stream copy cut/selected trials/time window into new dataset
    # by exploiting the in place selection
    dset = _dataset_from_trials(data,
                                dset_name='data',
                                filename=tld._gen_filename())

    # no copy here
    tld.data = dset
    tld.trialdefinition = data.selection.trialdefinition

    # now calculate via standard statistics
    avg = spy.mean(tld, dim='trials', parallel=False)
    var = spy.var(tld, dim='trials', parallel=False)

    # attach data to TimeLockData
    tld._update_dataset('avg', avg.data)
    tld._update_dataset('var', var.data)

    # unregister datasets to detach from objects
    avg._unregister_dataset("data", del_from_file=False)
    var._unregister_dataset("data", del_from_file=False)

    # scramble filenames and delete unneeded objects
    avg.filename, var.filename = '', ''
    del avg, var

    # -- set up covariance CR --

    if covariance:
        check_effective_parameters(Covariance, defaults, lcls, besides=['covariance', 'trials', 'latency'])
        covCR = Covariance(ddof=ddof, statAxis=data.dimord.index('time'))
        # dimord is time x freq x channel x channel
        out = spy.CrossSpectralData(dimord=spy.CrossSpectralData._defaultDimord)

        covCR.initialize(
            data,
            out._stackingDim,
            keeptrials=keeptrials,
        )
        # and compute
        covCR.compute(data, out, parallel=kwargs.get("parallel"), log_dict=log_dict)

        # attach computed cov as array
        tld._update_dataset('cov', out.data[:, 0, ...].squeeze())

    # -- restore initial selection or wipe --

    if select_backup:
        # this rewrites the cfg
        data.selectdata(select_backup, inplace=True)
    else:
        data.selection = None
        # erase local selection entry
        data.cfg.pop('selectdata')

    return tld


def _dataset_from_trials(spy_data, dset_name='new_data', filename=None):
    """
    Helper to construct a new dataset from
    a trial Indexer, respecting selections

    This function is only needed if a dataset is to be tranferred
    between two different Syncopy data classes, for the
    same data class a standard ``new = old.selectdata(..., inplace=False)``
    does the trick.
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
            idx[stackDim] = slice(stacking, stacking + trl_len)
            stacking += trl_len
            # insert the trial
            new_ds[tuple(idx)] = trl

    # open again for reading and return dataset directly
    return h5py.File(filename, mode='r+')[dset_name]
