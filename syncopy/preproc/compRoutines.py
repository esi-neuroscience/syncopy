# -*- coding: utf-8 -*-
#
# computeFunctions and -Routines for parallel calculation
# of FIR and IIR Filter operations
#

# Builtin/3rd party package imports
import numpy as np
import scipy.signal as sci
from inspect import signature

# syncopy imports
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.shared.kwarg_decorators import unwrap_io

# backend imports
from .firws import design_wsinc, apply_fir, minphaserceps


@unwrap_io
def sinc_filtering_cF(dat,
                      samplerate=1,
                      filter_type='lp',
                      freq=None,
                      order=None,
                      window="hamming",
                      direction='onepass',
                      polyremoval=None,
                      timeAxis=0,
                      noCompute=False,
                      chunkShape=None
                      ):
    """
    Provides basic filtering of signals with FIR (windowed sinc)
    filters. Supported are low-pass, high-pass,
    band-pass and band-stop (Notch) filtering.

    dat : (N, K) :class:`numpy.ndarray`
        Uniformly sampled multi-channel time-series data
        The 1st dimension is interpreted as the time axis,
        columns represent individual channels.
        Dimensions can be transposed to `(K, N)` with the `timeAxis` parameter
    samplerate : float
        Sampling frequency in Hz
    filter_type : {'lp', 'hp', 'bp, 'bs'}, optional
        Select type of filter, either low-pass `'lp'`,
        high-pass `'hp'`, band-pass `'bp'` or band-stop (Notch) `'bs'`.
    freq : float or array_like
        Cut-off frequency for low- and high-pass filters or sequence
        of two frequencies for band-stop and band-pass filter.
    order : int, optional
        Order of the filter, or length of the windowed sinc. The default
        `None` will create a filter of maximal order which is the number of
        samples in the trial.
        Higher orders yield a sharper transition width
        or less 'roll off' of the filter, but are more computationally expensive.
    window : {"hamming", "hann", "blackmann", "kaiser"}
        The type of taper to use for the sinc function
    direction : {'twopass', 'onepass', 'onepass-minphase'}
        Filter direction:
       `'twopass'` - zero-phase forward and reverse filter, IIR and FIR
       `'onepass'` - forward filter, introduces group delays for IIR, zerophase for FIR
       `'onepass-minphase' - forward causal/minumum phase filter, FIR only
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

    # max order is signal length
    if order is None:
        order = dat.shape[0]

    # construct the filter
    fkernel = design_wsinc(window, order, freq / samplerate, filter_type)

    # filtering by convolution
    if direction == 'onepass':
        filtered = apply_fir(dat, fkernel)

    # for symmetric filters actual
    # filter direction does NOT matter
    elif direction == 'twopass':
        filtered = apply_fir(dat, fkernel)
        filtered = apply_fir(filtered, fkernel)

    elif direction == 'onepass-minphase':
        # 0-phase transform
        fkernel = minphaserceps(fkernel)
        filtered = apply_fir(dat, fkernel)

    return filtered


class Sinc_Filtering(ComputationalRoutine):

    """
    Compute class that performs filtering with windowed sinc filters
    of :class:`~syncopy.AnalogData` objects

    Sub-class of :class:`~syncopy.shared.computational_routine.ComputationalRoutine`,
    see :doc:`/developer/compute_kernels` for technical details on Syncopy's compute
    classes and metafunctions.

    See also
    --------
    syncopy.preprocessing : parent metafunction
    """

    computeFunction = staticmethod(sinc_filtering_cF)

    # 1st argument,the data, gets omitted
    valid_kws = list(signature(sinc_filtering_cF).parameters.keys())[1:]

    def process_metadata(self, data, out):

        # Some index gymnastics to get trial begin/end "samples"
        if data._selection is not None:
            chanSec = data._selection.channel
            trl = data._selection.trialdefinition
            for row in range(trl.shape[0]):
                trl[row, :2] = [row, row + 1]
        else:
            chanSec = slice(None)
            trl = data.trialdefinition

        out.trialdefinition = trl

        out.samplerate = data.samplerate
        out.channel = np.array(data.channel[chanSec])


@unwrap_io
def but_filtering_cF(dat,
                     samplerate=1,
                     filter_type='lp',
                     freq=None,
                     order=6,
                     direction='twopass',
                     polyremoval=None,
                     timeAxis=0,
                     noCompute=False,
                     chunkShape=None
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
    order : int, optional
        Order of the filter, default is 6.
        Higher orders yield a sharper transition width
        or less 'roll off' of the filter, but are more computationally expensive.
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

    See also
    --------
    `Scipy butterworth documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html>`_

    """

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


class But_Filtering(ComputationalRoutine):

    """
    Compute class that performs filtering with butterworth filters 
    of :class:`~syncopy.AnalogData` objects

    Sub-class of :class:`~syncopy.shared.computational_routine.ComputationalRoutine`,
    see :doc:`/developer/compute_kernels` for technical details on Syncopy's compute
    classes and metafunctions.

    See also
    --------
    syncopy.preprocessing : parent metafunction
    """

    computeFunction = staticmethod(but_filtering_cF)

    # 1st argument,the data, gets omitted
    valid_kws = list(signature(but_filtering_cF).parameters.keys())[1:]

    def process_metadata(self, data, out):

        # Some index gymnastics to get trial begin/end "samples"
        if data._selection is not None:
            chanSec = data._selection.channel
            trl = data._selection.trialdefinition
            for row in range(trl.shape[0]):
                trl[row, :2] = [row, row + 1]
        else:
            chanSec = slice(None)
            trl = data.trialdefinition

        out.trialdefinition = trl

        out.samplerate = data.samplerate
        out.channel = np.array(data.channel[chanSec])
