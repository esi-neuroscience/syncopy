import numpy as np
from scipy.stats import iqr


def psth(trl_dat,
         cu_combs,
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
    cu_combs : :class:`~np.ndarray`
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

    # sort according to channels
    # to sort into correct bins
    idx = np.argsort(trl_dat[:, 1])

    # for readability
    samples = trl_dat[idx, 0]
    channels = trl_dat[idx, 1]
    units = trl_dat[idx, 2]
    nChannels = np.unique(channels).size

    # get relative spike times, no time-sorting needed for histograms!
    times = _calc_time(samples, trl_start, onset, samplerate)

    # Auto-select bin widths
    if bins is None:
        nBins = Rice_rule(len(times))
        bins = np.linspace(times.min(), times.max(), nBins + 1)
    else:
        nBins = len(bins) - 1

    # globally available units
    unique_units = np.unique(cu_combs[:, 1])
    # globally available channels
    unique_chans = np.unique(cu_combs[:, 0])
    nChannels = len(unique_chans)

    # attach 2nd bin dimension along channels
    bins = [bins, np.arange(nChannels + 1)]

    # counts = np.zeros((nBins, len(cu_combs) + 1))
    counts = np.zeros((nBins, nChannels, len(unique_units)))

    # for i, comb in enumerate(cu_combs):
    for i, iunit in enumerate(unique_units):
        unit_idx = (units == iunit)
        # boolean index for specific chan-unit combination
        # bidx = np.all(trl_dat[:, 1:] == comb, axis=1)

        if np.sum(unit_idx):
            counts[..., i] = np.histogram2d(times[unit_idx],
                                            channels[unit_idx],
                                            bins=bins)[0]
        # no spikes in this unit
        else:
            counts[..., i] = np.zeros(nBins, nChannels)

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
