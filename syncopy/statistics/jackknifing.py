# -*- coding: utf-8 -*-
#
# General, CR agnostic, JackKnife implementation for trial statistics
#

import numpy as np
import h5py
from copy import deepcopy

# Syncopy imports
import syncopy as spy
from syncopy.shared.computational_routine import propagate_properties
from syncopy.shared.parsers import data_parser, scalar_parser, array_parser

from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYError
from syncopy.shared.kwarg_decorators import unwrap_select

from syncopy.statistics.compRoutines import NumpyStatDim

# create test data on the fly
nTrials = 4
ad = spy.AnalogData(data=[i * np.ones((10, 4)) for i in range(nTrials)], samplerate=1)
spec = spy.freqanalysis(ad)
dim = 'time'
axis = ad.dimord.index(dim)
# create CR to jackknife
CR = NumpyStatDim(operation='mean', axis=axis)
# use same CR for direct estimate
mean_est = spy.mean(ad, dim=dim, keeptrials=False)

@unwrap_select
def trial_replicates(spy_data, CR, **kwargs):
    """
    General meta-function to compute the jackknife replicates
    along trials of a ComputationalRoutine `CR` by creating
    the full set of leave-one-out (loo) trial selections.

    The CR must compute a statistic over trials, meaning its
    individual results can be represented as a single trial.
    Examples are connectivity measures like coherence or
    any trial averaged quantity.

    The resulting data object has the same number of trials as the input,
    with each `trial` holding one trial averaged loo result, i.e. the
    jackknife replicates.

    Parameters
    ----------
    spy_data : Syncopy data object, e.g. :class:`~syncopy.AnalogData`

    CR : A derived :class:`~syncopy.shared.computational_routine.ComputationalRoutine` instance
        The computational routine computing the desired statistic to be jackknifed

    Returns
    ------
    jack_out : Syncopy data object, e.g. :class:`~syncopy.TimeLockData`
        The datatype will be determined by the supplied `CR`,
        yet instead of single-trial results each trial represents
        one trial-averaged jackknife replicate
    """

    if spy_data.selection is not None:
        # create a back up
        select_backup = deepcopy(spy_data.selection.select)
        selection_cleanup = False
    else:
        # create all-to-all selection
        # for easier property propagation
        spy_data.selectdata(inplace=True)
        selection_cleanup = True

    # now we have definitely a selection
    all_trials = spy_data.selection.trial_ids

    # create the leave-one-out (loo) trial selections
    loo_trial_selections = []
    for trl_id in all_trials:
        # shallow copy is sufficient here
        loo = all_trials.copy()
        loo.remove(trl_id)
        loo_trial_selections.append(loo)

    # --- CR computations --

    log_dict = {}

    # manipulate existing selection
    select = spy_data.selection.select
    # create loo selections and run CR
    # to compute and collect jackknife replicates
    loo_outs = []
    for trl_idx, loo in enumerate(loo_trial_selections):
        select['trials'] = loo
        spy_data.selectdata(select, inplace=True)

        # initialize transient output object of same datatype
        out = spy_data.__class__(dimord=spy_data.dimord)

        # (re-)initialize supplied CR and compute a trial average
        CR.initialize(spy_data, spy_data._stackingDim,
                      keeptrials=False, chan_per_worker=kwargs.get('chan_per_worker'))
        CR.compute(spy_data, out, parallel=kwargs.get("parallel"), log_dict=log_dict)

        # keep the loo output alive for now
        loo_outs.append(out)

    # reference to determine shapes and stacking
    loo1 = loo_outs[0]
    stack_dim = spy_data._stackingDim

    # prepare virtual dataset to collect results from
    # individual loo CR outputs w/o copying

    # each of the same shaped(!) loo replicates fills one single trial
    # slot of the final jackknifing result
    jack_shape = list(loo1.data.shape)
    jack_shape[stack_dim] = len(loo_outs) * jack_shape[stack_dim]
    layout = h5py.VirtualLayout(shape=tuple(jack_shape), dtype=loo1.data.dtype)

    # all loo results have the same shape, determine stacking step from 1st loo result
    stack_step = int(np.diff(loo1.sampleinfo[0])[0])

    # stacking index template
    stack_idx = [np.s_[:] for _ in range(loo1.data.ndim)]

    # now collect all loo datasets into the virtual dataset
    # to construct the jacknife result
    for loo_idx, out in enumerate(loo_outs):
        # stack along stacking dim
        stack_idx[stack_dim] = np.s_[loo_idx * stack_step:(loo_idx + 1) * stack_step]
        layout[tuple(stack_idx)] = h5py.VirtualSource(out.data)

        # to keep actual data alive even
        # when loo replicates go out of scope
        out._persistent_hdf5 = True

    # initialize jackknife output object of
    # same datatype as the loo replicates
    jack_out = loo1.__class__(dimord=spy_data.dimord)

    # finally create the virtual dataset
    with h5py.File(jack_out._filename, mode='w') as h5file:
        h5file.create_virtual_dataset('data', layout)
        # bind to syncopy object
        jack_out.data = h5file['data']

    # reopen dataset after I/O operation above
    jack_out._reopen()

    # attach properties like channel labels etc.
    propagate_properties(loo1, jack_out)

    # create proper trialdefinition
    # FIXME: not clear how to handle offsets (3rd column), set to 0 for now
    trl_def = np.column_stack([np.arange(len(loo_outs)) * stack_step,
                               np.arange(len(loo_outs)) * stack_step + stack_step,
                               np.zeros(len(loo_outs))])
    jack_out.trialdefinition = trl_def

    # revert selection state of the input
    if selection_cleanup:
        spy_data.selection = None
    else:
        spy_data.selectdata(select_backup)

    return jack_out


def bias_var(replicates, estimate):
    """
    Implements the general jackknife recipe to
    compute the bias and variance of a statiscial parameter
    over trials from an ensemble of leave-one-out replicates
    and the original estimate.

    Parameters
    ----------
    replicates : Syncopy data object, e.g. :class:`~syncopy.SpectralData`
        Each trial represents one jackknife replicate
    estimate : Syncopy data object, e.g. :class:`~syncopy.SpectralData`
        Must have exactly one trial representing the direct trial statistic
        to be jackknifed
    """

    if len(estimate.trials) != 1:
        lgl = "original trial statistic with one remaining trial"
        act = f"{len(estimate.trials)} trials"
        raise SPYValueError(lgl, 'estimate', act)

    if len(replicates.trials) <= 1:
        lgl = "jackknife replicates with at least 2 trials"
        act = f"{len(replicates.trials)} trials"
        raise SPYValueError(lgl, 'replicates', act)

    # 1st average the replicates which
    # gives the jackknife estimate
    jack_est = spy.mean(replicates, dim='trials')

    # compute the bias, shapes should match as both
    # quantities come from the same data and
    # got computed by the same CR
    if jack_est.data.shape != estimate.data.shape:
        msg = ("Got mismatching shapes for jackknife bias computation:\n"
               f"jack: {jack_est.data.shape}, original estimate: {estimate.data.shape}"
               )
        raise SPYError(msg)

    nTrials = len(replicates.trials)
    bias = (nTrials - 1) * (jack_est - estimate)
    bias_corrected = estimate - bias

    # Variance calculation, we have to construct a new
    # data object for this and compute trial-by-trial (each replicate)

    return bias, bias_corrected
