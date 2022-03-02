# -*- coding: utf-8 -*-
#
# computeFunctions and -Routines for parallel calculation
# of common preprocessing steps like IIR filtering
#

# Builtin/3rd party package imports
import numpy as np
import scipy.signal as sci
from inspect import signature

# syncopy imports
from syncopy.shared.tools import best_match
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.shared.kwarg_decorators import unwrap_io


@unwrap_io
def but_filtering_cF(dat,
                     samplerate=1,
                     filter_type='lp',
                     freq=None,
                     order=None,
                     direction='twopass',
                     polyremoval=None,
                     timeAxis=0,
                     noCompute=False
                     ):
    """
    Provides basic filtering of signals with IIR (Butterworth)
    filters. Supported are low-pass, high-pass,
    band-pass and band-stop (Notch) filtering.

    dat : (N, K) :class:`numpy.ndarray`
        Uniformly sampled multi-channel time-series data
        The 1st dimension is interpreted as the time axis,
        columns represent individual channels.
        Dimensions can be transposed to `(K, N)` with the `timeAxis` parameter
    filter_type : {'lp', 'hp', 'bp, 'bs'}, optional
        Select type of filter, either low-pass `'lp'`,
        high-pass `'hp'`, band-pass `'bp'` or band-stop (Notch) `'bs'`.
    freq : float or array_like
        Cut-off frequency for low- and high-pass filters or sequence
        of two frequencies for band-stop and band-pass filter.
    order : int
        Order of the filter. Higher orders yield a sharper transition width
        or 'roll off' of the filter.
    direction : {'twopass', 'onepass'}
       Filter direction:
       `'twopass'` - zero-phase forward and reverse filter
       `'onepass'` - forward filter, introduces group delays
    polyremoval : int or None
        Order of polynomial used for de-trending data in the time domain prior
        to filtering. A value of 0 corresponds to subtracting the mean
        ("de-meaning"), ``polyremoval = 1`` removes linear trends (subtracting the
        least squares fit of a linear polynomial).
    timeAxis : int, optional
        Index of running time axis in `dat` (0 or 1)
    noCompute : bool
        Preprocessing flag. If `True`, do not perform actual calculation but
        instead return expected shape and :class:`numpy.dtype` of output
        array.

    Returns
    -------
    filtered : (N, K) :class:`~numpy.ndarray`
        The filtered signals

    Notes
    -----
    This method is intended to be used as
    :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
    inside a :class:`~syncopy.shared.computational_routine.ComputationalRoutine`.
    Thus, input parameters are presumed to be forwarded from a parent metafunction.
    Consequently, this function does **not** perform any error checking and operates
    under the assumption that all inputs have been externally validated and cross-checked.

    """

    # attach dummy channel axis in case only a
    # single signal/channel is the input
    if dat.ndim < 2:
        dat = dat[:, np.newaxis]

    # Re-arrange array if necessary and get dimensional information
    if timeAxis != 0:
        dat = dat.T       # does not copy but creates view of `dat`
    else:
        dat = dat

    # filtering does not change the shape
    outShape = dat.shape
    if noCompute:
        return outShape, np.float32

    # detrend
    if polyremoval == 0:
        # SciPy's overwrite_data not working for type='constant' :/
        dat = sci.detrend(dat, type='constant', axis=0, overwrite_data=True)
    elif polyremoval == 1:
        dat = sci.detrend(dat, type='linear', axis=0, overwrite_data=True)

    # design the butterworth filter with "second-order-sections" output
    sos = sci.butter(order, freq, filter_type, fs=samplerate, output='sos')

    # do the filtering
    if direction == 'twopass':
        filtered = sci.sosfiltfilt(sos, dat, axis=0)
        return filtered

    elif direction == 'onepass':
        filtered = sci.sosfilt(sos, dat, axis=0)
        return filtered
