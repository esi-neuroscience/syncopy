import numpy as np
import logging
import platform
from scipy.stats import iqr


def psth(trl_dat,
         trl_start,
         onset,
         trl_end,
         chan_unit_combs=None,
         tbins=None,
         output='rate',
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
        Trigger onset in samples units
    trl_end : int
        End of the trial in sample units
    chan_unit_combs : :class:`~np.ndarray`
        All (sorted) numeric channel-unit combinations to bin for
        arangend in a (N, 2) shaped array, where each row is
        one unique combination (say [4, 1] for channel4 - unit1)
        If `None` will infer from the supplied SpikeData array.
    tbins: :class:`~numpy.array` or None
        An array of monotonically increasing PSTH bin edges
        in seconds including the rightmost edge
        Defaults with `None` to the Rice rule
    output : {'rate', 'spikecount', 'proportion'}, optional
        Set to `'rate'` to convert the output to firing rates (spikes/sec),
        'spikecount' to count the number spikes per trial or
        'proportion' to normalize the area under the PSTH to 1
        Defaults to `'rate'`
    samplerate : float
        Samplerate in Hz

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

    logger = logging.getLogger("syncopy_" + platform.node())
    logger.debug(f"Computing peristimulus time histogram (PSTH) on data with {samples.size} samples, {channels.size} channels, {units.size} units and samplerate {samplerate}.")

    # get relative spike times for all events in trial
    times = _calc_time(samples, trl_start, onset, samplerate)

    # Auto-select bin widths as backend fallback
    if tbins is None:
        nBins = Rice_rule(len(times))
        tbins = np.linspace(times.min(), times.max(), nBins + 1)
    else:
        nBins = len(tbins) - 1

    # this could mean here [chan1, chan5, chan10]!
    chan_vec = np.unique(channels)
    # now the ith channel bin maps to the ith available channel
    # (0, 1, 2) -> (1, 5, 10)
    bins = [tbins, np.arange(chan_vec.size + 1)]

    # inference here from a single trial is just a fallback
    # for testing etc.
    if chan_unit_combs is None:
        chan_unit_combs = get_chan_unit_combs([trl_dat])

    # this is the global(!) output shape - some columns may be filled with 0s
    # -> no firing for that chan-unit combo in this specific trial
    counts = np.zeros((nBins, len(chan_unit_combs)))

    # available units in this trial
    unique_units = np.unique(units)

    # create boolean mapping of all trial specific combinations
    # into global output shape
    map_cu = lambda c, u: np.all(chan_unit_combs == [c, u], axis=1)

    # now map with respect to unit for all single trial channels (-bins)
    map_unit = {u: np.logical_or.reduce([map_cu(c, u) for c in chan_vec]) for u in unique_units}

    # map into histogram time x channel dimensions
    map_unit_hist = {u: [np.any(map_cu(c, u)) for c in chan_vec] for u in unique_units}

    # configure output
    if output in ['rate', 'spikecount']:
        density = False
    elif output == 'proportion':
        density = True

    for i, iunit in enumerate(unique_units):
        unit_idx = (units == iunit)

        if np.sum(unit_idx):
            # over all channels, so this counts different units actually
            unit_counts = np.histogram2d(times[unit_idx],
                                         channels[unit_idx],
                                         bins=bins, density=density)[0]

            # get indices to inject the results
            # at the right position
            cu_idx = map_unit[iunit]

            # de-selects non-existent combinations in histogram
            chan_hist_idx = map_unit_hist[iunit]
            counts[:, cu_idx] = unit_counts[:, chan_hist_idx].astype(np.float32)

    # --- mask time bins which are outside of this trial ---
    # in trigger relative (timelocked) time
    trl_start_reltime = onset / samplerate
    trl_end_reltime = (trl_end - trl_start + onset) / samplerate

    # mask indices along the time bin axis
    # which are completely outside of latency window
    # mask all
    if np.all(tbins < trl_start_reltime):
        min_idx = len(counts)
    else:
        min_idx = np.argmin(tbins < trl_start_reltime)
    # mask all
    if np.all(tbins > trl_end_reltime):
        min_idx = len(counts)
        max_idx = 0
    else:
        max_idx = np.argmin(tbins <= trl_end_reltime)

    if min_idx != 0:
        counts[:min_idx] = np.nan
    if max_idx != 0:
        counts[max_idx:] = np.nan

    # normalize to counts per second
    if output == 'rate':
        tbin_width = np.diff(tbins)[0]
        counts *= 1 / tbin_width

    # normalize only along time axis for 1d time-histograms
    # `density=True` normalized the full 2d-histogram
    elif output == 'proportion':
        norm = np.nansum(counts, axis=0)[None, :]
        # deal with potential 0's
        norm[norm == 0] = 1
        counts /= norm

    return counts, bins


def _calc_time(samples, trl_start, onset, samplerate):

    """
    Calculates the event times relative to the trigger
    from sample numbers of individual events (the `samples`).
    """
    times = (samples - trl_start + onset) / samplerate

    return times


def get_chan_unit_combs(trials):

    """
    Get all channelX-unitY indice combinations with at least one event
    by checking every single trial array sequentially in `trials`.
    """
    combs = []

    # the straightforward way would be: np.unique(data.data[:, 1:], axis=0)
    # however this loads 66% the size of the total data into memory
    for trial in trials:
        combs.append(np.unique(trial[:, 1:], axis=0))

    combs = np.unique(np.concatenate(combs), axis=0)
    return combs


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
