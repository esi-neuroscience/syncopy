import numpy as np
from scipy.stats import iqr


def psth(trl_dat,
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
    trl_start : int
        Start of the trial in sample units
    onset : int
        Trigger onset in samples
    bins: :class:`~numpy.array` or None
        An array of monotonically increasing PSTH bin edges
        in seconds including the rightmost edge

    Returns
    -------
    counts : :class:`~np.ndarray`
        Spike counts for each channel and unit
        with shape (nUnits, nBins, nChannels)
    """

    # for readability
    samples = trl_dat[:, 0]
    channels = trl_dat[:, 1]
    units = trl_dat[:, 2]
    nChannels = np.unique(channels).size

    # get relative spike times
    times = _calc_time(samples, trl_start, onset, samplerate)

    # Auto-select bin widths
    if bins is None:
        nBins = Rice_rule(times)
        bins = np.linspace(times[0], times[-1], nBins + 1)
    else:
        nBins = len(bins) - 1

    # attach 2nd bin dimension along channels
    bins = [bins, np.arange(nChannels + 1)]

    unique_units = np.unique(units)
    counts = np.zeros((len(unique_units), nBins, nChannels))

    for i, iunit in enumerate(unique_units):
        unit_idx = (units == iunit)
        if np.sum(unit_idx):
            counts[i, :, :] = np.histogram2d(times[unit_idx],
                                             channels[unit_idx],
                                             bins=bins)[0]
        # no spikes in this unit
        else:
            counts[i, :, :] = np.zeros(nBins, nChannels)

    return counts, bins


def _calc_time(samples, trl_start, onset, samplerate):

    """
    Calculates the event times relative to the trigger
    from sample numbers of individual events.
    """

    times = (samples - trl_start + onset) / samplerate

    return times


# --- Bin selection rules ---

def Freedman_Diaconis_rule(samples):

    """
    Get optimal number of bins from samples,
    probably too low for 'typical' spike data.
    """

    h = 2 * iqr(samples) / pow(samples.size, 1 / 3)
    Nbins = int((samples.max() - samples.min()) / h)
    return Nbins


def Rice_rule(samples):

    """
    Get optimal number of bins from samples
    """

    Nbins = int(2 * pow(len(samples), 1 / 3))
    return Nbins
