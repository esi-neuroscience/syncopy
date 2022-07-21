import numpy as np
from scipy.stats import iqr


def psth(trl_dat,
         cu_combos,
         trl_start=0,
         onset=0,
         bins=None,
         samplerate=1000):

    """
    Peristimulus time histogram

    The single trial input `trl_dat` needs 3 columns:
        - 1st: the sample numbers of the spikes
        - 2nd: channel number of each spike
        - 3rd: unit number of each spike (set to constant if not available)

    Parameters
    ----------
    trl_dat : :class:`~np.ndarray`
        Single trial spike data with shape (nEvents x 3)
    cu_combos : :class:`~np.ndarray`
        All (sorted) numeric channel-unit combinations to bin for
        arangend in a (N, 2) shaped array, where each row is
        one unique combination (say [4, 1] for channel4 - unit1)
    trl_start : int
        Start of the trial in sample units
    onset : int
        Trigger onset in samples units
    bins: :class:`~numpy.array` or None
        An array of monotonically increasing PSTH bin edges
        in seconds including the rightmost edge
        Defaults with `None` to the Rice rule

    Returns
    -------
    counts : :class:`~np.ndarray`
        Spike counts for each available channelX_unitY
        combination with shape (nBins, nChannelUnits)

    See Also
    --------
    `Rice Rule <https://en.wikipedia.org/wiki/Histogram#Rice_Rule>`_ on Wikipedia
    """


    # for readability
    samples = trl_dat[:, 0]
    channels = trl_dat[:, 1]
    units = trl_dat[:, 2]

    # get relative spike times, no time-sorting needed for histograms!
    times = _calc_time(samples, trl_start, onset, samplerate)

    # Auto-select bin widths
    if bins is None:
        nBins = Rice_rule(len(times))
        bins = np.linspace(times.min(), times.max(), nBins + 1)
    else:
        nBins = len(bins) - 1

    # this could mean here [chan1, chan5, chan10]!
    chan_vec = np.unique(channels)
    # now the ith channel bin maps to the ith available channel
    # (0, 1, 2) -> (1, 5, 10)
    bins = [bins, np.arange(chan_vec.size + 1)]

    # this is the global(!) output shape - some columns may be filled with 0s
    # -> no firing for that chan-unit combo in this specific trial
    counts = np.zeros((nBins, len(cu_combos)))

    # available units in this trial
    unique_units = np.unique(units)

    # create boolean mapping of all trial specific combinations
    # into global output shape
    map_cu = lambda c, u: np.all(cu_combos == [c, u], axis=1)
    # now map with respect to unit for all single trial channels (-bins)
    map_unit = {u: np.logical_or.reduce([map_cu(c, u) for c in chan_vec]) for u in unique_units}

    for i, iunit in enumerate(unique_units):
        unit_idx = (units == iunit)

        if np.sum(unit_idx):
            # over all channels, so this counts different units actually
            unit_counts = np.histogram2d(times[unit_idx],
                                         channels[unit_idx],
                                         bins=bins)[0]
            # get indices to inject the results
            # at the right position
            cu_idx = map_unit[iunit]
            counts[:, cu_idx] = unit_counts

    return counts, bins


def _calc_time(samples, trl_start, onset, samplerate):

    """
    Calculates the event times relative to the trigger
    from sample numbers of individual events.
    """

    times = (samples - trl_start + onset) / samplerate

    return times


# --- Bin selection rules ---

def sqrt_rule(nSamples):

    """
    Get number of bins via square root of number of samples
    """

    return int(np.ceil(np.sqrt(nSamples)))


def Freedman_Diaconis_rule(samples):

    """
    Get number of bins from number and min/max of samples,
    probably too low for 'typical' spike data.
    """

    h = 2 * iqr(samples) / pow(samples.size, 1 / 3)
    Nbins = int((samples.max() - samples.min()) / h)
    return Nbins


def Rice_rule(nSamples):

    """
    Get number of bins from number of samples
    """

    Nbins = int(2 * pow(nSamples, 1 / 3))
    return Nbins
