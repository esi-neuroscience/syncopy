# -*- coding: utf-8 -*-
#
# Synthetic spike data generators for testing and tutorials
#

# Builtin/3rd party package imports
import numpy as np

# syncopy imports
from syncopy import SpikeData
from syncopy.shared.kwarg_decorators import unwrap_cfg

# ---- Synthetic SpikeData ----


@unwrap_cfg
def poisson_noise(
    nTrials=10,
    nSpikes=10000,
    nChannels=3,
    nUnits=10,
    intensity=0.1,
    samplerate=10000,
    seed=None,
):

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
    channels = rng.choice(np.arange(nChannels), p=get_rdm_weights(nChannels), size=nSpikes, replace=True)

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
        np.arange(0.05 * shortest_trial, 0.2 * shortest_trial, dtype=int),
        size=nTrials,
        replace=True,
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
