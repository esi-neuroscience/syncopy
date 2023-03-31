# -*- coding: utf-8 -*-
#
# General, CR agnostic, JackKnife implementation for trial statistics
#
import h5py
import numpy as np

# Syncopy imports
import syncopy as spy
from syncopy.shared.computational_routine import propagate_properties
from syncopy.shared.errors import SPYValueError, SPYError


def trial_avg_replicates(trl_ensemble):
    """
    Compute the jackknife replicates of the trial average
    for the full set of leave-one-out (loo) trial selections.

    The resulting data object has the same number of trials as the input,
    with each `trial` holding one loo average, i.e. the
    trivial jackknife replicates of the average. These can then be
    further used as input for CRs which operate on trial averages
    to compute the non-trivial jackknife replicates of the desired statistic,
    i.e. coherence.


    Parameters
    ----------
    trl_ensemble : syncopy data object, e.g. :class:`~syncopy.SpectralData`
        Single trial data from where the loo replicates will be created

    Returns
    ------
    replicates : syncopy data object
        The datatype will be the same as the input ``trl_ensemble``,
        where each trial represents one jackknife replicate (trial average)
    """

    if trl_ensemble.selection is None:
        # create all-to-all selection
        # for easier property propagation
        trl_ensemble.selectdata(inplace=True)
        selection_cleanup = True

    # now we definitely have a selection
    all_trials = trl_ensemble.selection.trial_ids
    nTrials = len(all_trials)

    # -- set up output object --

    # each of the loo averages fills one single trial
    # slot of the `replicates` dataset, hence it has the same shape
    # as the input
    replicates = trl_ensemble.__class__(samplerate=trl_ensemble.samplerate,
                                        dimord=trl_ensemble.dimord)

    with h5py.File(replicates._filename, mode='w') as h5file:
        dset = h5file.create_dataset('data', shape=trl_ensemble.data.shape,
                                     dtype=trl_ensemble.data.dtype)
        replicates.data = dset

    # we still need to write into it
    replicates._reopen()

    # -- replicate computations --

    # first calculate the standard trial average
    # this will also catch non-equal trials in the input
    trl_avg = spy.mean(trl_ensemble, dim='trials')

    # all loo replicates have the same shape as
    # the original single trial results, so the stepping
    # along the stacking dim is fixed
    stack_step = int(np.diff(trl_ensemble.sampleinfo[0])[0])
    stack_dim = trl_ensemble._stackingDim
    # stacking index template
    stack_idx = [np.s_[:] for _ in range(trl_ensemble.data.ndim)]

    # for each loo replicate we just have to subtract
    # the specific trial from the average
    for loo_idx in all_trials:
        # trial average is 'single trial', so this is memory safe
        # this is the simple loo average - the 'replicate'
        loo_avg = nTrials * trl_avg.data[()] - trl_ensemble.trials[loo_idx]
        # normalize
        loo_avg /= nTrials - 1

        # stack along stacking dim
        stack_idx[stack_dim] = np.s_[loo_idx * stack_step:(loo_idx + 1) * stack_step]
        replicates.data[tuple(stack_idx)] = loo_avg

    # attach properties like channel labels etc.
    propagate_properties(trl_ensemble, replicates)

    # create proper trialdefinition
    # FIXME: not clear how to handle offsets (3rd column), set to 0 for now
    trl_def = np.column_stack([np.arange(len(all_trials)) * stack_step,
                               np.arange(len(all_trials)) * stack_step + stack_step,
                               np.zeros(len(all_trials))])
    replicates.trialdefinition = trl_def

    # revert selection state of the input
    if selection_cleanup:
        trl_ensemble.selection = None

    return replicates


def bias_var(direct_estimate, replicates):
    """
    Implements the general jackknife recipe to
    compute the bias and variance of a statistical parameter
    over trials from an ensemble of leave-one-out replicates
    and the original raw estimate.

    Note that the jackknife bias-corrected estimate then simply is:

        jack_estimate = direct_estimate - bias

    Parameters
    ----------
    direct_estimate : syncopy data object, e.g. :class:`~syncopy.SpectralData`
        Must have exactly one trial representing the direct trial statistic
        to be jackknifed
    replicates : syncopy data object, e.g. :class:`~syncopy.SpectralData`
        Each trial represents one jackknife replicate

    Returns
    -------
    bias : syncopy data object, e.g. :class:`~syncopy.SpectralData`
        The bias of the original estimator
    variance : syncopy data object, e.g. :class:`~syncopy.SpectralData`
        The sample variance of the jackknife replicates
    """

    if len(direct_estimate.trials) != 1:
        lgl = "original trial statistic with one remaining trial"
        act = f"{len(direct_estimate.trials)} trials"
        raise SPYValueError(lgl, 'direct_estimate', act)

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
    if jack_avg.data.shape != direct_estimate.data.shape:
        msg = ("Got mismatching shapes for jackknife bias computation:\n"
               f"jack: {jack_avg.data.shape}, original estimate: {direct_estimate.data.shape}"
               )
        raise SPYError(msg)

    nTrials = len(replicates.trials)
    prefac = nTrials - 1
    # to avoid different type real/complex warning..
    prefac = prefac + 0j if np.issubdtype(direct_estimate.data.dtype, complex) else prefac
    bias = prefac * (jack_avg - direct_estimate)

    # Variance calculation, it is always real (as opposed to pseudo-variance)
    # compute sequentially into accumulator array
    var = np.zeros(direct_estimate.data.shape, dtype=np.float32)
    for loo in replicates.trials:
        # need abs for complex variance
        var += (np.abs(jack_avg.trials[0] - loo))**2
    # normalize
    var *= (nTrials - 1)

    # create the syncopy data object for the variance
    variance = direct_estimate.__class__(samplerate=direct_estimate.samplerate,
                                         dimord=direct_estimate.dimord)

    # bind to syncopy object -> creates the hdf5 dataset
    variance.data = var
    propagate_properties(direct_estimate, variance)

    return bias, variance
