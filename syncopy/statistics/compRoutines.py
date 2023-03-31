# -*- coding: utf-8 -*-
#
# Computational Routines for statstical methods
#

# Builtin/3rd party package imports
from inspect import signature
import numpy as np
from numpy.lib import stride_tricks

# backend method imports
from .psth import psth

# syncopy imports
from syncopy.shared.computational_routine import ComputationalRoutine, propagate_properties
from syncopy.shared.kwarg_decorators import process_io


@process_io
def npstats_cF(trl_dat, operation='mean', axis=0, noCompute=False, chunkShape=None):

    """
    Numpy summary statistics on single-trial arrays along indicated `axis`.

    Parameters
    ----------
    trl_dat : :class:`numpy.ndarray`
        Single trial data of arbitrary dimension
    operation : {'mean', 'std', 'var', 'median'}
        The statistical operation to perform
    axis : int
        The axis over which to calulate the average

    Returns
    -------
    res : :class:`numpy.ndarray`
        Result of the average

    Notes
    -----
    This method is intended to be used as :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
    inside a :class:`~syncopy.shared.computational_routine.ComputationalRoutine`.
    Thus, input parameters are presumed to be forwarded from a parent metafunction.
    Consequently, this function does **not** perform any error checking and operates
    under the assumption that all inputs have been externally validated and cross-checked.
    """

    if noCompute:
        # initialize result array
        out_shape = list(trl_dat.shape)
        out_shape[axis] = 1   # this axis will be summed over

        return out_shape, trl_dat.dtype

    return NumpyStatDim.methods[operation](trl_dat, axis=axis, keepdims=True)


class NumpyStatDim(ComputationalRoutine):

    """
    Simple compute class which applies basic numpy statistical funtions
    along  a ``dim`` of a Syncopy data object. The resulting Syncopy object
    is of the same type as the input object, having one of its
    dimensions reduced to a singleton due to the summary statistical operation.

    Notes
    -----
    If ``keeptrials`` is set to False, a trial average is additionally computed
    as in any other CR. For a standalone trial average use the _trial_statistics function.

    See also
    --------
    _trial_statistics: Sequential computation of statistics over trials
    """

    methods = {'mean': np.nanmean,
               'std': np.nanstd,
               'var': np.nanvar,
               'median': np.nanmedian
               }

    computeFunction = staticmethod(npstats_cF)

    def process_metadata(self, in_data, out_data):

        # the dimension over which the statistic got computed
        dim = in_data.dimord[self.cfg['axis']]

        out_data.samplerate = in_data.samplerate

        # Get/set timing-related selection modifiers
        # We've set a fallback all-to-all selection in any case

        # time axis really gone, only one trial and time got averaged out
        if dim == 'time' and not self.keeptrials:
            trldef = np.array([[0, 1, 0]])

        # trial average, needs equal trial lengths.. just copy from 1st
        elif dim != 'time' and not self.keeptrials:
            trldef = in_data.selection.trialdefinition[0, :][None, :]

        # each trial has empty time axis, so we attach trivial trialdefinition:
        # 1 sample per trial for stacking
        elif dim == 'time' and self.keeptrials:
            nTrials = len(in_data.selection.trials)
            stacking_time = np.arange(nTrials)[:, None]
            trldef = np.hstack((stacking_time, stacking_time + 1,
                               np.zeros((nTrials, 1))))

        # nothing happened on the time axis
        else:
            trldef = in_data.selection.trialdefinition

        out_data.trialdefinition = trldef

        # Get/set dimensional attributes changed by selection
        for prop in in_data.selection._dimProps:
            selection = getattr(in_data.selection, prop)
            # due to fallback all-to-all selection this captures
            # all existing dimensions
            if selection is not None:
                # propagate without change
                if dim not in prop:
                    if np.issubdtype(type(selection), np.number):
                        selection = [selection]
                    setattr(out_data, prop, getattr(in_data, prop)[selection])
                # set to singleton or None
                else:
                    # numerical freq axis is gone after averaging
                    if dim == 'freq':
                        out_data.freq = None
                    else:
                        setattr(out_data, prop, [self.cfg['operation']])

@process_io
def cov_cF(trl_dat,
           ddof=None,
           statAxis=0,
           noCompute=False,
           chunkShape=None):

    """
    Covariance between channels via ``np.cov``

    Parameters
    ----------
    trl_dat : :class:`numpy.ndarray`
        Single trial multi-channel data
    ddof : int, optional
        Degrees of freedom
    statAxis : int, optional
        Index of axis holding the observations in `trl_dat` (0 or 1)
    """

    # our variables are put in columns (rowvar=False)
    if statAxis != 0:
        dat = trl_dat.T
    else:
        dat = trl_dat

    nChannels = dat.shape[1]

    # mockup CrossSpectralData shape
    outShape = (1, 1, nChannels, nChannels)

    # For initialization of computational routine,
    # just return output shape and dtype
    if noCompute:
        return outShape, np.float32

    cov = np.cov(trl_dat, ddof=ddof, rowvar=False)

    # attach dummy time and freq axes
    return cov[None, None, ...]


class Covariance(ComputationalRoutine):

    """
    Compute class that calculates covariance of :class:`~syncopy.AnalogData` objects

    Sub-class of :class:`~syncopy.shared.computational_routine.ComputationalRoutine`,
    see :doc:`/developer/compute_kernels` for technical details on Syncopy's compute
    classes and metafunctions.

    Notes
    -----
    Outputs a :class:`~syncopy.CrossSpectralData` object with singleton time and freq
    axis. The backing hdf5 dataset then gets stripped of the empty axes and attached
    as additional ``.cov`` dataset to a :class:`~syncopy.TimeLockData` object in 
    the respective frontend.

    See also
    --------
    syncopy.timelockanalysis : parent metafunction
    """

    computeFunction = staticmethod(cov_cF)

    # 1st argument,the data, gets omitted
    valid_kws = list(signature(cov_cF).parameters.keys())[1:-1]

    def process_metadata(self, data, out):

        if data.selection is not None:
            chanSec = data.selection.channel
            trldef = data.selection.trialdefinition
            for row in range(trldef.shape[0]):
                trldef[row, :2] = [row, row + 1]
        else:
            chanSec = slice(None)
            time = np.arange(len(data.trials))
            time = time.reshape((time.size, 1))
            trldef = np.hstack((time, time + 1,
                                np.zeros((len(data.trials), 1)),
                                np.array(data.trialinfo)))

        # Attach constructed trialdef-array, time axis is gone 
        if self.keeptrials:
            out.trialdefinition = trldef
        else:
            out.trialdefinition = np.array([[0, 1, 0]])

        # Attach remaining meta-data
        out.samplerate = data.samplerate
        out.channel_i = np.array(data.channel[chanSec])
        out.channel_j = np.array(data.channel[chanSec])


@process_io
def psth_cF(trl_dat,
            trl_start,
            onset,
            trl_end,
            chan_unit_combs=None,
            tbins=None,
            output='rate',
            samplerate=1000,
            noCompute=False,
            chunkShape=None):

    """
    Peristimulus time histogram

    Backend :func:`~syncopy.spikes.psth.psth` `method_kwargs`:

        {'trl_start', 'onset', 'bins', 'samplerate'}

    Parameters
    ----------
    trl_dat : 2D :class:`numpy.ndarray`
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
    noCompute : bool
        Preprocessing flag. If `True`, do not perform actual calculation but
        instead return expected shape and :class:`numpy.dtype` of output
        array.
    chunkShape : None or tuple
        If not `None`, represents shape of output `tl_data`

    Returns
    -------
    tl_data : 2D :class:`numpy.ndarray`
        Spike counts for each unit
        with shape (nBins, nUnits)


    Notes
    -----
    This method is intended to be used as
    :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
    inside a :class:`~syncopy.shared.computational_routine.ComputationalRoutine`.
    Thus, input parameters are presumed to be forwarded from a parent metafunction.
    Consequently, this function does **not** perform any error checking and operates
    under the assumption that all inputs have been externally validated and cross-checked.

    See also
    --------
    syncopy.spike_psth : parent metafunction
    backend method : :func:`~syncopy.spikes.psth.psth`
    PSTH : :class:`~syncopy.shared.computational_routine.ComputationalRoutine` instance
                     that calls this method as :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`

    """

    nChanUnit = len(chan_unit_combs)
    nBins = len(tbins) - 1

    # For initialization of computational routine,
    # just return output shape and dtype
    if noCompute:
        outShape = (nBins, nChanUnit)
        return outShape, np.float32

    # call backend method
    counts, bins = psth(trl_dat, trl_start, onset, trl_end,
                        chan_unit_combs=chan_unit_combs,
                        tbins=tbins, samplerate=samplerate, output=output)

    return counts


class PSTH(ComputationalRoutine):
    """
    Compute class that performs psth analysis of :class:`~syncopy.SpikeData` objects

    Sub-class of :class:`~syncopy.shared.computational_routine.ComputationalRoutine`,
    see :doc:`/developer/compute_kernels` for technical details on Syncopy's compute
    classes and metafunctions.

    See also
    --------
    syncopy.spike_psth : parent metafunction
    """

    computeFunction = staticmethod(psth_cF)

    # 1st argument,the data, gets omitted
    valid_kws = list(signature(psth).parameters.keys())[1:]
    valid_kws += list(signature(psth_cF).parameters.keys())[1:-1]

    def process_metadata(self, data, out):

        tbins = self.cfg['tbins']
        # compute new time axis / samplerate
        bin_midpoints = stride_tricks.sliding_window_view(tbins, (2,)).mean(axis=1)
        srate = 1 / np.diff(bin_midpoints).mean()

        # each trial has the same length
        # for "timelocked" (same bins) psth data
        trl_len = len(tbins) - 1
        if data.selection is not None:
            nTrials = len(data.selection.trial_ids)
        else:
            nTrials = len(data.trials)

        # create trialdefinition, offsets are all equal
        # for timelocked data
        trl = np.zeros((nTrials, 3))
        sample_idx = np.arange(0, nTrials * trl_len + 1, trl_len)
        trl[:, :2] = stride_tricks.sliding_window_view(sample_idx, (2,))
        # negative relative time is pre-stimulus!
        # note that bin edges are set on the input data (high-res) time axis
        # we can only approximate atm with the new 1/srate time steps
        offsets = np.rint(bin_midpoints[0] * srate)
        trl[:, 2] = offsets

        # Attach meta-data
        if self.keeptrials:
            out.trialdefinition = trl
        else:
            out.trialdefinition = trl[[0], :]

        out.samplerate = srate
        # join labels for final unitX_channelY channel labels
        chan_str = "channel{}_unit{}"
        out.channel = [chan_str.format(c, u) for c, u in self.cfg['chan_unit_combs']]

        if not self.keeptrials:
            # the ad-hoc averaging does not work well here because of NaNs
            # so we rather delete the data to 'not keep the trials'
            out.data = None
            # TODO: add real average operator here
