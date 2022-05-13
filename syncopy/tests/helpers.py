# -*- coding: utf-8 -*-
#
# Helper functions for frontend test design
#
# The runner signatures take a callable,
# the `method_call`, as 1st argument
#

# 3rd party imports
import itertools
import numpy as np

from syncopy.shared.errors import SPYValueError, SPYTypeError

# fix random generators
np.random.seed(40203)


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


def run_foi_test(method_call, foilim, positivity=True):

    # only positive frequencies
    assert np.min(foilim) >= 0
    assert np.max(foilim) <= 500

    # fois
    foi1 = np.arange(foilim[0], foilim[1])  # 1Hz steps
    foi2 = np.arange(foilim[0], foilim[1], 0.25)  # 0.5Hz steps
    foi3 = 'all'
    fois = [foi1, foi2, foi3, None]

    for foi in fois:
        result = method_call(foi=foi, foilim=None)
        # check here just for finiteness and positivity
        assert np.all(np.isfinite(result.data))
        if positivity:
            assert np.all(result.data[0, ...] >= -1e-10)

    # 2 foilims
    foilims = [[2, 60], [7.65, 45.1234], None]
    for foil in foilims:
        result = method_call(foilim=foil, foi=None)
        # check here just for finiteness and positivity
        assert np.all(np.isfinite(result.data))
        if positivity:
            assert np.all(result.data[0, ...] >= -1e-10)

    # make sure specification of both foi and foilim triggers a
    # Syncopy ValueError
    try:
        result = method_call(foi=foi, foilim=foil)
    except SPYValueError as err:
        assert 'foi/foilim' in str(err)

    # make sure out-of-range foi selections are detected
    try:
        result = method_call(foilim=[-1, 70], foi=None)
    except SPYValueError as err:
        assert 'foilim' in str(err)
        assert 'bounded by' in str(err)

    try:
        result = method_call(foi=np.arange(550, 700), foilim=None)
    except SPYValueError as err:
        assert 'foi' in str(err)
        assert 'bounded by' in str(err)


def mk_selection_dicts(nTrials, nChannels, toi_min, toi_max, min_len=0.25):

    """
    Takes 4 numbers, the last two descibing a time-interval

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

    # create 3 random trial and channel selections
    trials, channels = [], []
    for _ in range(3):

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

    # create toi selections, signal length is toi_max
    # with -1s as offset (from synthetic data instantiation)
    # subsampling does NOT WORK due to precision issues :/
    # toi1 = np.linspace(-.4, 2, 100)
    tois = [None, 'all']
    toi_combinations = itertools.product(trials,
                                         channels,
                                         tois)

    # 2 random toilims
    toilims = []
    while len(toilims) < 2:

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
    # digest generators to create all selection dictionaries
    for comb in toi_combinations:

        sel_dct = {}
        sel_dct['trials'] = comb[0]
        sel_dct['channel'] = comb[1]
        sel_dct['toi'] = comb[2]
        selections.append(sel_dct)

    for comb in toilim_combinations:

        sel_dct = {}
        sel_dct['trials'] = comb[0]
        sel_dct['channel'] = comb[1]
        sel_dct['toilim'] = comb[2]
        selections.append(sel_dct)

    return selections
