# -*- coding: utf-8 -*-
#
# Synthetic data generators for testing and tutorials
#

# Builtin/3rd party package imports
from inspect import signature
import numpy as np
import functools

from syncopy import AnalogData, SpikeData


_2pi = np.pi * 2


def collect_trials(trial_generator):
    """
    Decorator to wrap around a (nSamples x nChannels) shaped np.array producing
    synthetic data routine, and returning an :class:`~syncopy.AnalogData`
    object.

    All backend single trial generating functions (`trial_generator`) should
    accept `nChannels` and `nSamples` as keyword arguments, OR provide
    other means to define those numbers, e.g.
    `AdjMat` for :func:`~syncopy.synth_data.AR2_network`

    If the underlying trial generating function also accepts
    a `samplerate`, forward this directly.

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

    @functools.wraps(trial_generator)
    def wrapper_synth(nTrials=None, samplerate=1000, seed=None, seed_per_trial=True, **tg_kwargs):

        seed_array = None  # One seed per trial.
        if nTrials is not None and seed is not None and seed_per_trial:  # Use the single seed to create one seed per trial.
            rng = np.random.default_rng(seed)
            seed_array = rng.integers(1000000, size=nTrials)

        # append samplerate parameter if also needed by the generator
        if 'samplerate' in signature(trial_generator).parameters.keys():
            tg_kwargs['samplerate'] = samplerate

        # do nothing (may pass on the scalar seed if the function supports it)
        if nTrials is None:
            if 'seed' in signature(trial_generator).parameters.keys():
                tg_kwargs['seed'] = seed
            return trial_generator(**tg_kwargs)
        # collect trials
        else:
            trl_list = []

            for trial_idx in range(nTrials):
                if 'seed' in signature(trial_generator).parameters.keys():
                    if seed_array is not None:
                        tg_kwargs['seed'] = seed_array[trial_idx]
                    else:
                        tg_kwargs['seed'] = seed
                trl_arr = trial_generator(**tg_kwargs)
                trl_list.append(trl_arr)

            data = AnalogData(trl_list, samplerate=samplerate)
        return data

    return wrapper_synth


# ---- Synthetic AnalogData ----


@collect_trials
def white_noise(nSamples=1000, nChannels=2, seed=42):
    """
    Plain white noise with unity standard deviation.

    Pass an extra `nTrials` `int` Parameter to generate multi-trial data using the `@collect_trials` decorator.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(size=(nSamples, nChannels))


@collect_trials
def linear_trend(y_max, nSamples=1000, nChannels=2):
    """
    A linear trend  on all channels from 0 to `y_max` in `nSamples`.

    Pass an extra `nTrials` `int` Parameter to generate multi-trial data using the `@collect_trials` decorator.
    """
    trend = np.linspace(0, y_max, nSamples)
    return np.column_stack([trend for _ in range(nChannels)])


@collect_trials
def harmonic(freq, samplerate, nSamples=1000, nChannels=2):
    """
    A harmonic with frequency `freq`.

    Pass an extra `nTrials` `int` Parameter to generate multi-trial data using the `@collect_trials` decorator.
    """
    # the sampling times vector needed for construction
    tvec = np.arange(nSamples) * 1 / samplerate
    # the  harmonic
    harm = np.cos(2 * np.pi * freq * tvec)
    return np.column_stack([harm for _ in range(nChannels)])


# noisy phase evolution <-> phase diffusion
@collect_trials
def phase_diffusion(freq,
                    eps=.1,
                    samplerate=1000,
                    nChannels=2,
                    nSamples=1000,
                    rand_ini=False,
                    return_phase=False,
                    seed=None):

    """
    Linear (harmonic) phase evolution + a Brownian noise term
    inducing phase diffusion around the deterministic phase drift with
    slope ``2pi * freq`` (angular frequency).

    The linear phase increments are given by ``dPhase = 2pi * freq/samplerate``,
    the Brownian increments are scaled with `eps` relative to these
    phase increments.

    Parameters
    ----------
    freq : float
        Harmonic frequency in Hz
    eps : float
        Scaled Brownian increments
        `1` means the single Wiener step
        has on average the size of the
        harmonic increments
    samplerate : float
        Sampling rate in Hz
    nChannels : int
        Number of channels
    nSamples : int
        Number of samples in time
    rand_ini : bool, optional
        If set to ``True`` initial phases are randomized
    return_phase : bool, optional
        If set to true returns the phases in radians
    seed: None or int, passed on to `np.random.default_rng`.
          Set to an `int` to get reproducible results, or `None` for random ones.
    nTrials: int, number of trials to generate using the `@collect_trials` decorator.

    Returns
    -------
    phases : numpy.ndarray
        Synthetic `nSamples` x `nChannels` data array simulating noisy phase
        evolution/diffusion
    """

    # white noise
    wn = white_noise(nSamples=nSamples, nChannels=nChannels, seed=seed)

    tvec = np.linspace(0, nSamples / samplerate, nSamples)
    omega0 = 2 * np.pi * freq
    lin_phase = np.tile(omega0 * tvec, (nChannels, 1)).T

    # randomize initial phase
    if rand_ini:
        rng = np.random.default_rng(seed)
        ps0 = 2 * np.pi * rng.uniform(size=nChannels)
        lin_phase += ps0

    # relative Brownian increments
    rel_eps = np.sqrt(omega0 / samplerate * eps)
    brown_incr = rel_eps * wn

    # combine harmonic and diffusive dyncamics
    phases = lin_phase + np.cumsum(brown_incr, axis=0)
    if not return_phase:
        return np.cos(phases)
    else:
        return phases


@collect_trials
def AR2_network(AdjMat=None, nSamples=1000, alphas=[0.55, -0.8], seed=None):

    """
    Simulation of a network of coupled AR(2) processes

    With the default parameters the individual processes
    (as in Dhamala 2008) have a spectral peak at 40Hz
    with a sampling frequency of 200Hz.

    NOTE: There is no check for stability: setting the
          `alphas` ad libitum and/or defining large
          and dense (many connections) systems will
          almost surely lead to an unstable system

    NOTE: One can set the number of channels via the shape
          of the supplied `AdjMat`. Defaults to 2.

    Parameters
    ----------
    AdjMat : np.ndarray or None
        `nChannel` x `nChannel` adjacency matrix where
        entry ``(i,j)`` is the coupling strength
        from channel ``i -> j``.
        If left at `None`, the default 2 Channel system
        with unidirectional ``2 -> 1`` coupling is generated.
        See also `mk_RandomAdjMat`.
    nSamples : int, optional
        Number of samples in time
    alphas : 2-element sequence, optional
        The AR(2) parameters for lag1 and lag2
    seed : None or int.
        Random seed to init random number generator, passed on to `np.random.default_rng` function.
        When using this function with an `nTrials` argument (`@collect_trials` wrapper), and you *do*
        want the data of all trials to be identical (and reproducible),
        pass a single scalar seed and set 'seed_per_trial=False'.
    nTrials: int, number of trials to generate using the `@collect_trials` decorator.

    Returns
    -------
    sol : numpy.ndarray
        The `nSamples` x `nChannel`
        solution of the network dynamics
    """

    # default system layout as in Dhamala 2008:
    # unidirectional (2->1) coupling
    if AdjMat is None:
        AdjMat = np.zeros((2, 2), dtype=np.float32)
        AdjMat[1, 0] = .25
    else:
        # cast to our standard type
        AdjMat = AdjMat.astype(np.float32)

    nChannels = AdjMat.shape[0]
    alpha1, alpha2 = alphas
    # diagonal 'self-interaction' with lag 1
    DiagMat = np.diag(nChannels * [alpha1])

    sol = np.zeros((nSamples, nChannels), dtype=np.float32)
    # pick the 1st values at random
    rng = np.random.default_rng(seed)
    sol[:2, :] = rng.normal(size=(2, nChannels))

    for i in range(2, nSamples):
        sol[i, :] = (DiagMat + AdjMat.T) @ sol[i - 1, :] + alpha2 * sol[i - 2, :]
        sol[i, :] += rng.normal(size=(nChannels))

    return sol


def AR2_peak_freq(a1, a2, samplerate=1):
    """
    Helper function to tune spectral peak of AR(2) process
    """
    if np.any((a1**2 + 4 * a2) > 0):
        raise ValueError("No complex roots!")

    return np.arccos(a1 * (a2 - 1) / (4 * a2)) * 1 / _2pi * samplerate


def mk_RandomAdjMat(nChannels=3, conn_thresh=0.25, max_coupling=0.25, seed=None):
    """
    Create a random adjacency matrix
    for the network of AR(2) processes
    where entry ``(i,j)`` is the coupling
    strength from channel ``i -> j``

    Parameters
    ---------
    nChannels : int
        Number of channels (network nodes)
    conn_thresh : float
        Connectivity threshold for the Bernoulli
        sampling of the network connections. Setting
        ``conn_thresh = 1`` yields a fully connected network
        (not recommended).
    max_coupling : float < 0.5, optional
        Total input into single channel
        normalized by number of couplings
        (for stability).
    seed: None or int, passed on to `np.random.default_rng`.
          Set to an int to get reproducible results.

    Returns
    -------
    AdjMat : numpy.ndarray
        `nChannels` x `nChannels` adjacency matrix where
    """

    # random numbers in [0,1)
    rng = np.random.default_rng(seed)
    AdjMat = rng.random((nChannels, nChannels))

    # all smaller than threshold elements get set to 1 (coupled)
    AdjMat = (AdjMat < conn_thresh).astype(float)

    # set diagonal to 0 to easier identify coupling
    np.fill_diagonal(AdjMat, 0)

    # normalize such that total input
    # does not exceed max. coupling
    norm = AdjMat.sum(axis=0)
    norm[norm == 0] = 1
    AdjMat = AdjMat / norm[None, :] * max_coupling

    return AdjMat


# ---- Synthetic SpikeData ----


def poisson_noise(nTrials=10,
                  nSpikes=10000,
                  nChannels=3,
                  nUnits=10,
                  intensity=.1,
                  samplerate=10000,
                  seed=None):

    """
    Poisson (Shot-)noise generator

    The expected trial length in samples is given by:

        ``nSpikes`` / (``intensity`` * ``nTrials``)

    Dividing again by the ``samplerate` gives the
    expected trial length in seconds.

    Individual trial lengths get randomly
    shortened by up to 10% of this expected length.

    The trigger offsets are also
    randomized between 5% and 20% of the shortest
    trial length.

    Lastly, the distribution of the Poisson ``intensity`` along channels and units
    has uniformly randomized weights, meaning that typically
    you get very active channels/units and some which are almost quiet.

    Parameters
    ----------
    nTrials : int
        Number of trials
    nSpikes : int
        The total number of spikes over all trials to generate
    nChannels : int
        Number of channels
    nUnits : int
        Number of units
    intensity : int
        Expected number of spikes per sampling interval
    samplerate : float
        Sampling rate in Hz
    seed: None or int, passed on to `np.random.default_rng`.
          Set to an int to get reproducible results.

    Returns
    -------
    sdata : :class:`~syncopy.SpikeData`
        The generated spike data

    Notes
    -----
    Originally conceived by `Alejandro Tlaie Boria https://github.com/atlaie_`

    Examples
    --------
    With `nSpikes=20_000`, `samplerate=10_000`, `nTrials=10` and the default `intensity=0.1`
    we can expect a trial length of about 2 seconds:

    >>> spike_data = poisson_noise(nTrials=10, nSpikes=20_000, samplerate=10_000)

    Example output of the 1st trial [start, end] in seconds:

    >>> spike_data.trialintervals[0]
    >>> array([-0.3004, 1.6459])

    Which is close to 2 seconds.

    """

    # uniform random weights
    def get_rdm_weights(size, seed=seed):
        rng = np.random.default_rng(seed)
        pvec = rng.uniform(size=size)
        return pvec / pvec.sum()

    # total length of all trials combined
    rng = np.random.default_rng(seed)
    T_max = int(nSpikes / intensity)

    spike_samples = np.sort(rng.choice(range(T_max), size=nSpikes, replace=False))
    channels = rng.choice(
        np.arange(nChannels), p=get_rdm_weights(nChannels),
        size=nSpikes, replace=True
    )

    uvec = np.arange(nUnits)
    pvec = get_rdm_weights(nUnits)
    units = rng.choice(uvec, p=pvec, size=nSpikes, replace=True)

    # originally fixed trial size
    step = T_max // nTrials
    trl_intervals = np.arange(T_max + 1, step=step)

    # 1st trial
    idx_start = trl_intervals[:-1]
    idx_end = trl_intervals[1:] - 1

    # now randomize trial length a bit, max 10% size difference
    idx_end = idx_end - np.r_[rng.integers(step // 10, size=nTrials - 1), 0]

    shortest_trial = np.min(idx_end - idx_start)
    idx_offset = -rng.choice(
        np.arange(0.05 * shortest_trial, 0.2 * shortest_trial, dtype=int), size=nTrials, replace=True
    )

    trldef = np.vstack([idx_start, idx_end, idx_offset]).T
    data = np.vstack([spike_samples, channels, units]).T
    sdata = SpikeData(
        data=data,
        trialdefinition=trldef,
        dimord=["sample", "channel", "unit"],
        samplerate=samplerate,
    )

    return sdata
