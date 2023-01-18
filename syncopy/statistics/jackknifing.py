# -*- coding: utf-8 -*-
#
# General, CR agnostic, JackKnife implementation
#

import numpy as np
from copy import deepcopy

# Syncopy imports
import syncopy as spy
from syncopy.shared.parsers import data_parser, scalar_parser, array_parser
from syncopy.shared.tools import get_defaults, get_frontend_cfg
from syncopy.datatype import TimeLockData

from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYInfo
from syncopy.shared.kwarg_decorators import (
    unwrap_cfg,
    unwrap_select,
    detect_parallel_client
)
from syncopy.shared.input_processors import check_passed_kwargs
from syncopy.shared.latency import get_analysis_window, create_trial_selection

# Local imports
from syncopy.statistics.compRoutines import PSTH
from syncopy.statistics.psth import Rice_rule, sqrt_rule, get_chan_unit_combs

available_binsizes = {'rice': Rice_rule, 'sqrt': sqrt_rule}
available_outputs = ['rate', 'spikecount', 'proportion']

# create test data on the fly

@unwrap_select
def jacknife_cr(data, CR, CR_kwargs, **kwargs):
    """
    General meta-function to compute the jackknife estimate
    of an arbitrary ComputationalRoutine by creating
    leave-one-out selections
    """
    print(data.selection, 's')

    if data.selection is not None:
        # create a back up
        selection_backup = deepcopy(data.selection.select)
        selection_cleanup = False
    else:
        # create all-to-all selection
        # for easier property propagation
        data.selectdata(inplace=True)
        selection_cleanup = True

    # now we have definitely a selection
    all_trials = data.selection.trial_ids

    # collect the leave-one-out (loo) trial selections
    loo_trial_selections = []
    for trl_id in all_trials:
        # shallow copy is sufficient here
        loo = all_trials.copy()
        loo.remove(trl_id)
        loo_trial_selections.append(loo)

    # strip of trial selection at first
    select = data.selection.select
    select.pop('trials')
    # create loo selections and run CR
    # to compute jackknife replicates
    for loo in loo_trial_selections:
        select['trials'] = loo
        data.selectdata(select, inplace=True)
        print(data.selection)

    if selection_cleanup:
        data.selection = None

    return loo_trial_selections
