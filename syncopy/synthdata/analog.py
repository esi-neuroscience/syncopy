# -*- coding: utf-8 -*-
#
# Synthetic analog data generators for testing and tutorials
#

# Builtin/3rd party package imports
import numpy as np
# syncopy imports
from .utils import collect_trials


_2pi = np.pi * 2


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
def AR2_network(AdjMat=None, nSamples=1000, alphas=(0.55, -0.8), seed=None):

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


