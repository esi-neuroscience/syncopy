# -*- coding: utf-8 -*-
#
# General, CR agnostic, JackKnife implementation for trial statistics
#
import h5py
import numpy as np
from copy import deepcopy

# Syncopy imports
import syncopy as spy
from syncopy.shared.computational_routine import propagate_properties
from syncopy.shared.parsers import data_parser, scalar_parser, array_parser
from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYError
from syncopy.shared.kwarg_decorators import unwrap_select


def trial_replicates(spy_data, raw_estimate, CR, **kwargs):
    """
    General meta-function to compute the jackknife replicates
    along trials of a ComputationalRoutine `CR` by creating
    the full set of leave-one-out (loo) trial selections.

    The CR must compute a statistic over trials, meaning its
    result is represented as a single trial. Examples are
    connectivity measures like coherence or any trial averaged quantity.

    The resulting data object has the same number of trials as the input,
    with each `trial` holding one trial averaged loo result, i.e. the
    jackknife replicates.

    Parameters
    ----------
    spy_data : syncopy data object, e.g. :class:`~syncopy.AnalogData`

    raw_estimate : syncopy data object, e.g. :class:`~syncopy.SpectralData`
        Must have exactly one trial representing the direct trial statistic
        to be jackknifed

    CR : A derived :class:`~syncopy.shared.computational_routine.ComputationalRoutine` instance
        The computational routine computing the desired statistic to be jackknifed

    Returns
    ------
    jack_out : syncopy data object, e.g. :class:`~syncopy.TimeLockData`
        The datatype will be ``out_class`` where each trial
        represents one (trial-averaged) jackknife replicate
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

        # initialize transient output object for loo CR result
        out = raw_estimate.__class__(dimord=raw_estimate.dimord,
                                     samplerate=raw_estimate.samplerate)

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
    jack_out = loo1.__class__(dimord=raw_estimate.dimord)

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


def bias_var(raw_estimate, replicates):
    """
    Implements the general jackknife recipe to
    compute the bias and variance of a statiscial parameter
    over trials from an ensemble of leave-one-out replicates
    and the original raw estimate.

    Note that the jackknife bias-corrected estimate then simply is:

        jack_estimate = raw_estimate - bias

    Parameters
    ----------
    raw_estimate : syncopy data object, e.g. :class:`~syncopy.SpectralData`
        Must have exactly one trial representing the direct trial statistic
        to be jackknifed
    replicates : syncopy data object, e.g. :class:`~syncopy.SpectralData`
        Each trial represents one jackknife replicate

    Returns
    -------
    bias : syncopy data object, e.g. :class:`~syncopy.SpectralData`
        The bias of the original estimator
    variance : syncopy data object, e.g. :class:`~syncopy.SpectralData`
        The sample variance of the jackknife replicates, e.g. the standard
        error of the mean
    """

    if len(raw_estimate.trials) != 1:
        lgl = "original trial statistic with one remaining trial"
        act = f"{len(raw_estimate.trials)} trials"
        raise SPYValueError(lgl, 'raw_estimate', act)

    if len(replicates.trials) <= 1:
        lgl = "jackknife replicates with at least 2 trials"
        act = f"{len(replicates.trials)} trials"
        raise SPYValueError(lgl, 'replicates', act)

    # 1st average the replicates which
    # gives the single trial jackknife estimate
    jack_avg = spy.mean(replicates, dim='trials')

    # compute the bias, shapes should match as both
    # quantities come from the same data and
    # got computed by the same CR
    if jack_avg.data.shape != raw_estimate.data.shape:
        msg = ("Got mismatching shapes for jackknife bias computation:\n"
               f"jack: {jack_avg.data.shape}, original estimate: {raw_estimate.data.shape}"
               )
        raise SPYError(msg)

    nTrials = len(replicates.trials)
    bias = (nTrials - 1) * (jack_avg - raw_estimate)

    # Variance calculation
    # compute sequentially into accumulator array
    var = np.zeros(raw_estimate.data.shape)
    for loo in replicates.trials:
        var += (jack_avg.trials[0] - loo)**2
    # normalize
    var *= (nTrials - 1)

    # create the syncopy data object for the variance
    variance = raw_estimate.__class__(samplerate=raw_estimate.samplerate,
                                      dimord=raw_estimate.dimord)

    # bind to syncopy object -> creates the hdf5 dataset
    variance.data = var
    propagate_properties(raw_estimate, variance)

    return bias, variance


@unwrap_select
def do_jk(spy_data, raw_estimate, CR):
    """
    Convenience function to demonstrate
    how to interface the jackknife recipe

    Parameters
    ----------
    spy_data : syncopy data object, e.g. :class:`~syncopy.AnalogData`
        The input which was used to generate the ``raw_esimate`` from the ``CR``
    raw_estimate : syncopy data object, e.g. :class:`~syncopy.SpectralData`
        Must have exactly one trial representing the direct trial statistic
        to be jackknifed
    CR : A derived :class:`~syncopy.shared.computational_routine.ComputationalRoutine` instance
        The computational routine which computed the ``raw_estimate`` statistic to be jackknifed

    Returns
    -------
    jack_estimate : syncopy data object
        The bias-corrected jackknife estimate
    bias : syncopy data object
        The bias of the ``raw estimate`` determined by jackknifing
    variance : syncopy data object
        The variance of the ``raw estimate`` determined by jackknifing
    """

    replicates = trial_replicates(spy_data, raw_estimate, CR)
    bias, variance = bias_var(raw_estimate, replicates)

    jack_estimate = raw_estimate - bias
    return jack_estimate, bias, variance
