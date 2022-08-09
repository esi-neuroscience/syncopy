# -*- coding: utf-8 -*-
#
# Computational Routines for the spike analysis methods
#

# Builtin/3rd party package imports
from inspect import signature
import numpy as np
from numpy.lib import stride_tricks

# backend method imports
from .psth import psth, Rice_rule

# syncopy imports
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.shared.kwarg_decorators import unwrap_io


@unwrap_io
def psth_cF(trl_dat,
            noCompute=False,
            chunkShape=None,
            method_kwargs=None):

    """
    Peristimulus time histogram

    Backend :func:`~syncopy.spikes.psth.psth` `method_kwargs`:

        {'trl_start', 'onset', 'bins', 'samplerate'}

    Parameters
    ----------
    trl_dat : 2D :class:`numpy.ndarray`
        Single trial spike data with shape (nEvents x 3)
    noCompute : bool
        Preprocessing flag. If `True`, do not perform actual calculation but
        instead return expected shape and :class:`numpy.dtype` of output
        array.
    chunkShape : None or tuple
        If not `None`, represents shape of output `tl_data`
    method_kwargs : dict
        Keyword arguments passed to :func:`~syncopy.spikes.psth.psth`
        controlling the actual spike analysis method

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

    nUnits = np.unique(trl_dat)[2]
    nChannels = np.unique(trl_dat)[2]
    nBins = len(method_kwargs['bins']) - 1

    # For initialization of computational routine,
    # just return output shape and dtype
    if noCompute:
        outShape = (nBins, nUnits * nChannels)
        return outShape, np.int32

    # call backend method
    # counts has shape (nBins, nUnits, nChannels)
    counts, bins = psth(trl_dat, **method_kwargs)

    # split out channel counts along the units axis
    counts.shape = (nBins, nUnits * nChannels)

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

        # Get trialdef array + channels from source
        if data.selection is not None:
            chanSec = data.selection.channel
        else:
            chanSec = slice(None)

        # Get trialdef array + units from source
        if data.selection is not None:
            unitSec = data.selection.unit
        else:
            unitSec = slice(None)

        # compute new time axis / samplerate
        bin_midpoints = stride_tricks.sliding_window_view(self.cfg['bins'], (2,)).mean(axis=1)
        srate = 1 / np.diff(bin_midpoints).mean()

        # each trial has the same length
        # for "timelocked" (same bins) psth data
        trl_len = len(self.cfg['bins']) - 1
        nTrials = len(self.data.trials)

        # create trialdefinition, offsets are all 0
        # for timelocked data
        trl = np.zeros((nTrials, 3))
        sample_idx = np.arange(0, nTrials * trl_len + 1, trl_len)
        trl[:, :2] = stride_tricks.sliding_window_view(sample_idx, (2,))

        # Attach meta-data
        out.trialdefinition = trl
        out.samplerate = srate
        # join labels for final unitX_channelY channel labels
        out.channel = [u + '_' + c for u in data.unit[unitSec] for c in data.channel[chanSec]]
