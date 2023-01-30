# -*- coding: utf-8 -*-
#
# General, CR agnostic, JackKnife implementation
#

import numpy as np
import h5py
from copy import deepcopy

# Syncopy imports
import syncopy as spy
from syncopy.shared.parsers import data_parser, scalar_parser, array_parser

from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYInfo
from syncopy.shared.kwarg_decorators import (
    unwrap_cfg,
    unwrap_select
)

from syncopy.statistics.compRoutines import NumpyStatDim
from syncopy.statistics.psth import Rice_rule, sqrt_rule, get_chan_unit_combs


# create test data on the fly
ad = spy.AnalogData(data=[i * np.ones((10, 4)) for i in range(10)], samplerate=1)
spec = spy.freqanalysis(ad)
axis = ad.dimord.index('time')
CR = NumpyStatDim(operation='mean', axis=axis)

@unwrap_select
def jacknife_cr(spy_data, CR, **kwargs):
    """
    General meta-function to compute the jackknife estimates
    of an arbitrary ComputationalRoutine by creating
    the full set of leave-one-out (loo) trial selections.

    The resulting dataset has the same shape as the input,
    with each `trial` holding one trial averaged loo result.
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
        stack_idx[stack_dim] = np.s_[trl_idx * stack_step:(trl_idx + 1) * stack_step]
        layout[tuple(stack_idx)] = h5py.VirtualSource(out.data)

    # initialize jackknife output object of same datatype
    jack_out = loo1.__class__(dimord=spy_data.dimord)

    # finally create the virtual dataset
    with h5py.File(jack_out._filename, mode='w') as h5file:
        h5file.create_virtual_dataset('data', layout)
        # bind to syncopy object
        jack_out.data = h5file['data']

    if selection_cleanup:
        spy_data.selection = None
    else:
        spy_data.selectdata(select_backup)

    # reopen dataset to get a
    # healthy state of the returned object
    jack_out._reopen()
    return jack_out
