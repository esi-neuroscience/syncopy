# -*- coding: utf-8 -*-
#
# General, CR agnostic, JackKnife implementation
#

import numpy as np
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
ad = spy.AnalogData(data=[np.ones((10,4)) for _ in range(10)], samplerate=1)
axis = ad.dimord.index('time')
CR = NumpyStatDim(operation='mean', axis=axis)

@unwrap_select
def jacknife_cr(spy_data, CR, **kwargs):
    """
    General meta-function to compute the trial jackknife estimates
    of an arbitrary ComputationalRoutine by creating
    the full set of leave-one-out (loo) selections.
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

    # collect the leave-one-out (loo) trial selections
    loo_trial_selections = []
    for trl_id in all_trials:
        # shallow copy is sufficient here
        loo = all_trials.copy()
        loo.remove(trl_id)
        loo_trial_selections.append(loo)

    # --- CR computations --

    log_dict = {}

    # initialize jackknife output object of same datatype
    jack_out = spy_data.__class__(dimord=spy_data.dimord)

    # manipulate existing selection
    select = spy_data.selection.select
    # create loo selections and run CR
    # to compute jackknife replicates
    for loo in loo_trial_selections:
        select['trials'] = loo
        spy_data.selectdata(select, inplace=True)

        # initialize transient output object of same datatype
        out = spy_data.__class__(dimord=spy_data.dimord)

        # (re-)initialize supplied CR and compute a trial average
        CR.initialize(spy_data, spy_data._stackingDim,
                      keeptrials=False, chan_per_worker=kwargs.get('chan_per_worker'))
        CR.compute(spy_data, out, parallel=kwargs.get("parallel"), log_dict=log_dict)

        # each replicate gets trial averaged, grab that trial and write
        # into jackknife output
        out.trials[0]

    if selection_cleanup:
        spy_data.selection = None
    else:
        spy_data.selectdata(select_backup)

    return loo_trial_selections
