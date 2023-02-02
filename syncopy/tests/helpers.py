# -*- coding: utf-8 -*-
#
# Helper functions for frontend test design
#
# The `run_` function signatures take a callable,
# the `method_call`, as 1st argument
#

# 3rd party imports
import itertools
import numpy as np
import matplotlib.pyplot as plt
from syncopy.shared.errors import SPYValueError, SPYTypeError

# fix random generators
test_seed = 42


def run_padding_test(method_call, pad_length):
    """
    The callable should test a solution and support
    a single keyword argument `pad`
    """

    pad_options = [pad_length, 'nextpow2', 'maxperlen']
    for pad in pad_options:
        method_call(pad=pad)

    # test invalid pads
    try:
        method_call(pad=-0.1) # trials should be longer than 0.1 seconds
    except SPYValueError as err:
        assert 'pad' in str(err)
        assert 'expected value to be greater' in str(err)

    try:
        method_call(pad='IamNoPad')
    except SPYValueError as err:
        assert 'Invalid value of `pad`' in str(err)
        assert 'nextpow2' in str(err)

    try:
        method_call(pad=np.array([1000]))
    except SPYValueError as err:
        assert 'Invalid value of `pad`' in str(err)
        assert 'nextpow2' in str(err)


def run_polyremoval_test(method_call):
    """
    The callable should test a solution and support
    a single keyword argument `polyremoval`
    """

    poly_options = [0, 1]
    for poly in poly_options:
        method_call(polyremoval=poly)

    # test invalid polyremoval options
    try:
        method_call(polyremoval=2)
    except SPYValueError as err:
        assert 'polyremoval' in str(err)
        assert 'expected value to be greater' in str(err)

    try:
        method_call(polyremoval='IamNoPad')
    except SPYTypeError as err:
        assert 'Wrong type of `polyremoval`' in str(err)

    try:
        method_call(polyremoval=np.array([1000]))
    except SPYTypeError as err:
        assert 'Wrong type of `polyremoval`' in str(err)


def mk_selection_dicts(nTrials, nChannels, toi_min, toi_max, min_len=0.25):

    """
    Takes 5 numbers, the last three descibing a `latency` time-interval
    and creates cartesian product like `select` keyword
    arguments. One random selection is enough!

    Returns
    -------
    selections : list
        The list of dicts holding the keys and values for
        Syncopy selections.
    """
    # at least 10 trials
    assert nTrials > 9
    # at least 2 channels
    assert nChannels > 1
    # at least 250ms
    assert (toi_max - toi_min) > 0.25

    # create 1 random trial and channel selections
    trials, channels = [], []
    for _ in range(1):

        sizeTr = np.random.randint(10, nTrials + 1)
        trials.append(list(np.random.choice(
            nTrials, size=sizeTr
        )
        ))

        sizeCh = np.random.randint(2, nChannels + 1)
        channels.append(['channel' + str(i + 1)
                         for i in
                         np.random.choice(
                             nChannels, size=sizeCh, replace=False)])

    # 1 random toilim
    toilims = []
    while len(toilims) < 1:

        toil = np.sort(np.random.rand(2)) * (toi_max - toi_min) + toi_min
        # at least min_len (250ms)
        if np.diff(toil) < min_len:
            continue
        else:
            toilims.append(toil)

    # combinatorics of all selection options
    # order matters to assign the selection dict keys!
    toilim_combinations = itertools.product(trials,
                                            channels,
                                            toilims)

    selections = []
    for comb in toilim_combinations:

        sel_dct = {}
        sel_dct['trials'] = comb[0]
        sel_dct['channel'] = comb[1]
        sel_dct['latency'] = comb[2]
        selections.append(sel_dct)

    return selections

def teardown():
    """Cleanup to run at the end of a set of tests, typically at the end of a Test class."""
    # Close matplotlib plot windows:
    plt.close('all')
