# -*- coding: utf-8 -*-
#
# Backend methods/functions for 
# rational (p/q) resampling
#

# Builtin/3rd party package imports
import fractions
import numpy as np
import scipy.signal as sci_sig


def resample(data, orig_fs, new_fs, window=('kaiser', 0.5)):

    """
    Uses SciPy's polyphase method for the implementation
    of the standard resampling procedure: 
        upsampling : FIR filtering : downsampling

    Parameters
    ----------
    data  : (N, K) :class:`numpy.ndarray`
        Uniformly sampled multi-channel time-series data
        The 1st dimension is interpreted as the time axis,
        columns represent individual channels.
    orig_fs : float
        The original sampling rate
    new_fs : float
        The target sampling rate after resampling
    window : string, tuple, or array_like, optional
        Either a window (+parameters) for the FIR filter 
        to be implicitly designed/used, or the 1D 
        FIR filter array directly. Supported windows
        are :data:`~syncopy.shared.const_def.availableTapers`
        Defaults to a Kaiser window with beta=0.5.

    Returns
    -------
    resampled : (N, K) :class:`~numpy.ndarray`
        The resampled signals
    
    See also
    --------
    `SciPy's resample implementation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html>`_
    syncopy.preproc.compRoutines.downsample_cF : Straightforward and cheap downsampling 
    """

    # get up/down sampling factors
    up, down = _get_pq(orig_fs, new_fs)
    resampled = sci_sig.resample_poly(data, up, down, axis=0, window=window)

    return resampled

def _get_pq(orig_fs, new_fs):

    """
    Get the up/down (p/q) sampling
    factors from the original and target
    sampling rate.

    NOTE: Can return very large factors for 
          "almost irrational" sampling rate ratios!
    """
    
    frac = fractions.Fraction.from_float(new_fs / orig_fs)
    # trim down 
    frac = frac.limit_denominator()

    return frac.numerator, frac.denominator


