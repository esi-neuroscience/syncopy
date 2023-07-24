# -*- coding: utf-8 -*-
#
# Utilities for syncopy's synthetic data generators
#

# Builtin/3rd party package imports
from inspect import signature
import numpy as np
import functools

from syncopy import AnalogData
from syncopy.shared.parsers import scalar_parser
from syncopy.shared.kwarg_decorators import (
    unwrap_cfg,
    _append_docstring,
    _append_signature,
)


def collect_trials(trial_func):
    """
    Decorator to wrap around a single trial (nSamples x nChannels shaped np.ndarray)
    synthetic data function. Creates a generator expression to arrive
    memory safely at a multi-trial :class:``~syncopy.AnalogData`` object.


    All single trial producing functions (the ``trial_func``) should
    accept `nChannels` and `nSamples` as keyword arguments, OR provide
    other means to define those numbers, e.g.
    `AdjMat` for :func:`~syncopy.synth_data.ar2_network`

    If the single trial function also accepts a `samplerate` parameter, forward it directly.

    If the underlying trial generating function also accepts
    a `seed`, forward this directly. One can set `seed_per_trial=False` to use
    the same seed for all trials, or leave `seed_per_trial=True` (the default),
    to have this function internally generate a list
    of seeds with len equal to `nTrials` from the given seed, with one seed per trial.

    One can set the `seed` to `None`, which will select a random seed each time,
    (and it will differ between trials).

    The default `nTrials=None` is the identity wrapper and
    just returns the output of the trial generating function
    directly, so a single trial :class:`numpy.ndarray`.
    """

    @unwrap_cfg
    @functools.wraps(trial_func)
    def wrapper_synth(*args, nTrials=100, samplerate=1000, seed=None, seed_per_trial=True, **tf_kwargs):
        seed_array = None  # One seed per trial.
        # Use the single seed to create one seed per trial.
        if nTrials is not None and seed is not None and seed_per_trial:
            rng = np.random.default_rng(seed)
            seed_array = rng.integers(1_000_000, size=nTrials)

        # append samplerate parameter if also needed by the generator
        if "samplerate" in signature(trial_func).parameters.keys():
            tf_kwargs["samplerate"] = samplerate

        # bypass: directly return a single trial (may pass on the scalar seed if the function supports it)
        if nTrials is None:
            if "seed" in signature(trial_func).parameters.keys():
                tf_kwargs["seed"] = seed
            return trial_func(**tf_kwargs)

        # collect trials
        else:
            scalar_parser(nTrials, "nTrials", ntype="int_like", lims=[1, np.inf])

            # create the trial generator
            def mk_trl_generator():

                for trial_idx in range(nTrials):
                    if "seed" in signature(trial_func).parameters.keys():
                        if seed_array is not None:
                            tf_kwargs["seed"] = seed_array[trial_idx]
                        else:
                            tf_kwargs["seed"] = seed
                    yield trial_func(*args, **tf_kwargs)

            trl_generator = mk_trl_generator()

            data = AnalogData(trl_generator, samplerate=samplerate)

        return data

    # Append `nTrials` and `seed` keyword entry to wrapped function's docstring and signature
    nTrialsDocEntry = (
        "    nTrials : int or None\n"
        "        Number of trials for the returned :class:`~syncopy.AnalogData` object.\n"
        "        When set to `None` a single-trial :class:`~numpy.ndarray`\n"
        "        is returned."
    )

    wrapper_synth.__doc__ = _append_docstring(trial_func, nTrialsDocEntry)
    wrapper_synth.__signature__ = _append_signature(trial_func, "nTrials", kwdefault=100)

    return wrapper_synth
