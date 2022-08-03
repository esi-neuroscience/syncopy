# -*- coding: utf-8 -*-
#
# Synthetic data generators for testing and tutorials
#

# Builtin/3rd party package imports
from inspect import signature
import numpy as np
import functools
import random

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

    The default `nTrials=None` is the identity wrapper and
    just returns the output of the trial generating function
    directly, so a single trial :class:`numpy.ndarray`.
    """

    @functools.wraps(trial_generator)
    def wrapper_synth(nTrials=None, samplerate=1000, **tg_kwargs):

        # append samplerate parameter if also needed by the generator
        if 'samplerate' in signature(trial_generator).parameters.keys():
            tg_kwargs['samplerate'] = samplerate

        # do nothing
        if nTrials is None:
            return trial_generator(**tg_kwargs)
        # collect trials
        else:
            trl_list = []

            for _ in range(nTrials):
                trl_arr = trial_generator(**tg_kwargs)
                trl_list.append(trl_arr)

            data = AnalogData(trl_list, samplerate=samplerate)
        return data

    return wrapper_synth


# ---- Synthetic AnalogData ----


@collect_trials
def white_noise(nSamples=1000, nChannels=2):

    """
    Plain white noise with unity standard deviation
    """
    return np.random.randn(nSamples, nChannels)


@collect_trials
def linear_trend(y_max, nSamples=1000, nChannels=2):

    """
    A linear trend  on all channels from 0 to `y_max` in `nSamples`
    """

    trend = np.linspace(0, y_max, nSamples)
    return np.column_stack([trend for _ in range(nChannels)])


@collect_trials
def harmonic(freq, samplerate, nSamples=1000, nChannels=2):

    """
    A harmonic with frequency `freq`
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
                    fs=1000,
                    nChannels=2,
                    nSamples=1000,
                    return_phase=False):

    """
    Linear (harmonic) phase evolution + a Brownian noise term
    inducing phase diffusion around the deterministic phase drift with
    slope ``2pi * freq`` (angular frequency).

    The linear phase increments are given by ``dPhase = 2pi * freq/fs``,
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
    fs : float
        Sampling rate in Hz
    nChannels : int
        Number of channels
    nSamples : int
        Number of samples in time
    return_phase : bool, optional
        If set to true returns the phases in radians

    Returns
    -------
    phases : numpy.ndarray
        Synthetic `nSamples` x `nChannels` data array simulating noisy phase
        evolution/diffusion
    """

    # white noise
    wn = np.random.randn(nSamples, nChannels)

    delta_ts = np.ones(nSamples) * 1 / fs
    omega0 = 2 * np.pi * freq
    lin_incr = np.tile(omega0 * delta_ts, (nChannels, 1)).T

    # relative Brownian increments
    rel_eps = np.sqrt(omega0 / fs * eps)
    brown_incr = rel_eps * wn
    phases = np.cumsum(lin_incr + brown_incr, axis=0)
    if not return_phase:
        return np.cos(phases)
    else:
        return phases


@collect_trials
def AR2_network(AdjMat=None, nSamples=1000, alphas=[0.55, -0.8]):

    """
    Simulation of a network of coupled AR(2) processes

    With the default parameters the individual processes
    (as in Dhamala 2008) have a spectral peak at 40Hz
    with a sampling frequency of 200Hz.

    NOTE: There is no check for stability: setting the
          `alphas` ad libitum and/or defining large
          and dense (many connections) systems will
          almost surely lead to an unstable system

    Parameters
    ----------
    AdjMat : np.ndarray or None
        `nChannel` x `nChannel` adjacency matrix where
        entry ``(i,j)`` is the coupling strength
        from channel ``i -> j``.
        If left at `None`, the default 2 Channel system
        with unidirectional ``2 -> 1`` coupling is generated.
    nSamples : int, optional
        Number of samples in time
    alphas : 2-element sequence, optional
        The AR(2) parameters for lag1 and lag2

    Returns
    -------
    sol : numpy.ndarray
        The `nSamples` x `nChannel`
        solution of the network dynamics
    """


    # default system layout as in Dhamala 2008:
    # unidirectional (2->1) coupling
    if AdjMat is None:
        AdjMat = np.zeros((2, 2))
        AdjMat[1, 0] = .25

    nChannels = AdjMat.shape[0]
    alpha1, alpha2 = alphas
    # diagonal 'self-interaction' with lag 1
    DiagMat = np.diag(nChannels * [alpha1])

    sol = np.zeros((nSamples, nChannels))
    # pick the 1st values at random
    sol[:2, :] = np.random.randn(2, nChannels)

    for i in range(2, nSamples):

        sol[i, :] = (DiagMat + AdjMat.T) @ sol[i - 1, :] + alpha2 * sol[i - 2, :]
        sol[i, :] += np.random.randn(nChannels)

    return sol


def AR2_peak_freq(a1, a2, fs=1):

    """
    Helper function to tune spectral peak of AR(2) process
    """

    if np.any((a1**2 + 4 * a2) > 0):
        raise ValueError("No complex roots!")

    return np.arccos(a1 * (a2 - 1) / (4 * a2)) * 1 / _2pi * fs


def mk_RandomAdjMat(nChannels=3, conn_thresh=0.25, max_coupling=0.25):
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

    Returns
    -------
    AdjMat : numpy.ndarray
        `nChannels` x `nChannels` adjacency matrix where
    """


    # random numbers in [0,1)
    AdjMat = np.random.random_sample((nChannels, nChannels))

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
                  samplerate=30000
                  ):

    """
    Poisson (Shot-) noise generator


    Parameters
    ----------
    nTrials : int
        Number of trials
    nSpikes : int
        The total number of spikes to generate
    nChannels : int
        Number of channels
    nUnits : int
        Number of units
    intensity : int
        Average number of spikes per sampling interval
    samplerate : float
        Sampling rate in Hz

    Returns
    -------
    sdata : :class:`~syncopy.SpikeData`
        The generated spike data

    Notes
    -----
    Originally conceived by `Alejandro Tlaie Boria https://github.com/atlaie_`

    """

    def get_rdm_weights(size):
        pvec = np.random.uniform(size=size)
        return pvec / pvec.sum()

    T_max = int(1 / intensity * nSpikes)
    spike_times = np.sort(random.sample(range(T_max), nSpikes))
    channels = np.random.choice(
        np.arange(nChannels), p=get_rdm_weights(nChannels),
        size=nSpikes, replace=True
    )

    uvec = np.arange(nUnits)
    pvec = get_rdm_weights(nUnits)
    units = np.random.choice(uvec, p=pvec, size=nSpikes, replace=True)
    # units = np.r_[units, np.random.choice(uvec, size=nSpikes // 2, replace=True)]
    # if nSpikes % 2 == 1:
    #     units = np.r_[units, [np.random.choice(uvec)]]

    trl_intervals = np.sort(random.sample(range(T_max), nTrials + 1))
    # 1st trial
    idx_start = trl_intervals[:-1]
    idx_end = trl_intervals[1:] - 1

    idx_offset = -np.random.choice(
        np.arange(1, np.min(idx_end - idx_start)), size=nTrials, replace=True
    )

    trldef = np.vstack([idx_start, idx_end, idx_offset]).T
    data = np.vstack([spike_times, channels, units]).T
    sdata = SpikeData(
        data=data,
        trialdefinition=trldef,
        dimord=["sample", "channel", "unit"],
        samplerate=samplerate,
    )

    return sdata
