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
from syncopy.shared.computational_routine import ComputationalRoutine, propagate_properties
from syncopy.shared.const_def import spectralConversions, spectralDTypes
from syncopy.shared.kwarg_decorators import process_io

# backend imports
from .firws import design_wsinc, apply_fir, minphaserceps
from .resampling import downsample, resample


@process_io
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
    window : {"hamming", "hann", "blackman", "kaiser"}
        The type of taper to use for the sinc function
    direction : {'twopass', 'onepass', 'onepass-minphase'}
        Filter direction:
       `'twopass'` - zero-phase forward and reverse filter, IIR and FIR
       `'onepass'` - forward filter, introduces group delays for IIR, zerophase for FIR
       `'onepass-minphase' - forward causal/minimum phase filter, FIR only
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


class SincFiltering(ComputationalRoutine):

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

        propagate_properties(data, out)


@process_io
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


class ButFiltering(ComputationalRoutine):

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

        propagate_properties(data, out)


@process_io
def rectify_cF(dat, noCompute=False, chunkShape=None):

    """
    Provides straightforward rectification via `np.abs`.

    dat : (N, K) :class:`numpy.ndarray`
        Uniformly sampled multi-channel time-series data
    noCompute : bool
        If `True`, do not perform actual calculation but
        instead return expected shape and :class:`numpy.dtype` of output
        array.

    Returns
    -------
    rectified : (N, K) :class:`~numpy.ndarray`
        The rectified signals

    Notes
    -----
    This method is intended to be used as
    :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
    inside a :class:`~syncopy.shared.computational_routine.ComputationalRoutine`.
    Thus, input parameters are presumed to be forwarded from a parent metafunction.
    Consequently, this function does **not** perform any error checking and operates
    under the assumption that all inputs have been externally validated and cross-checked.

    """

    # operation does not change the shape
    outShape = dat.shape
    if noCompute:
        return outShape, np.float32

    return np.abs(dat)


class Rectify(ComputationalRoutine):

    """
    Compute class that performs rectification
    of :class:`~syncopy.AnalogData` objects

    Sub-class of :class:`~syncopy.shared.computational_routine.ComputationalRoutine`,
    see :doc:`/developer/compute_kernels` for technical details on Syncopy's compute
    classes and metafunctions.

    See also
    --------
    syncopy.preprocessing : parent metafunction
    """

    computeFunction = staticmethod(rectify_cF)

    # 1st argument,the data, gets omitted
    valid_kws = list(signature(rectify_cF).parameters.keys())[1:]

    def process_metadata(self, data, out):

        propagate_properties(data, out)


@process_io
def hilbert_cF(dat, output='abs', timeAxis=0, noCompute=False, chunkShape=None):

    """
    Provides Hilbert transformation with various outputs, band-pass filtering
    beforehand highly recommended.

    dat : (N, K) :class:`numpy.ndarray`
        Uniformly sampled multi-channel time-series data
    output : {'abs', 'complex', 'real', 'imag', 'absreal', 'absimag', 'angle'}
        The transformation after performing the complex Hilbert transform. Choose
        `'angle'` to get the phase.
    timeAxis : int, optional
        Index of running time axis in `dat` (0 or 1)
    noCompute : bool
        If `True`, do not perform actual calculation but
        instead return expected shape and :class:`numpy.dtype` of output
        array.

    Returns
    -------
    rectified : (N, K) :class:`~numpy.ndarray`
        The rectified signals

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

    # operation does not change the shape
    # but may change the number format
    outShape = dat.shape
    fmt = spectralDTypes["fourier"] if output == 'complex' else spectralDTypes["abs"]
    if noCompute:
        return outShape, fmt

    trafo = sci.hilbert(dat, axis=0)

    return spectralConversions[output](trafo)


class Hilbert(ComputationalRoutine):

    """
    Compute class that performs Hilbert transforms
    of :class:`~syncopy.AnalogData` objects

    Sub-class of :class:`~syncopy.shared.computational_routine.ComputationalRoutine`,
    see :doc:`/developer/compute_kernels` for technical details on Syncopy's compute
    classes and metafunctions.

    See also
    --------
    syncopy.preprocessing : parent metafunction
    """

    computeFunction = staticmethod(hilbert_cF)

    # 1st argument,the data, gets omitted
    valid_kws = list(signature(hilbert_cF).parameters.keys())[1:]

    def process_metadata(self, data, out):

        propagate_properties(data, out)


@process_io
def downsample_cF(dat,
                  samplerate=1,
                  new_samplerate=1,
                  timeAxis=0,
                  chunkShape=None,
                  noCompute=False
                  ):
    """
    Provides basic downsampling of signals. The `new_samplerate` should be
    an integer division of the original `samplerate`.

    dat : (N, K) :class:`numpy.ndarray`
        Uniformly sampled multi-channel time-series data
        The 1st dimension is interpreted as the time axis,
        columns represent individual channels.
    samplerate : float
        Sample rate of the input data
    new_samplerate : float
        Sample rate of the output data
    timeAxis : int, optional
        Index of running time axis in `dat` (0 or 1)

    Returns
    -------
    resampled : (X, K) :class:`~numpy.ndarray`
        The downsampled data

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

    if noCompute:
        # we need integers for slicing
        skipped = int(samplerate // new_samplerate)
        outShape = list(dat.shape)
        outShape[0] = int(np.ceil(dat.shape[0] / skipped))
        return tuple(outShape), dat.dtype

    resampled = downsample(dat, samplerate, new_samplerate)

    return resampled


class Downsample(ComputationalRoutine):

    """
    Compute class that performs straightforward (integer division)
    downsampling of :class:`~syncopy.AnalogData` objects

    Sub-class of :class:`~syncopy.shared.computational_routine.ComputationalRoutine`,
    see :doc:`/developer/compute_kernels` for technical details on Syncopy's compute
    classes and metafunctions.

    See also
    --------
    syncopy.preprocessing : parent metafunction
    """

    computeFunction = staticmethod(downsample_cF)

    # 1st argument,the data, gets omitted
    valid_kws = list(signature(downsample_cF).parameters.keys())[1:]

    def process_metadata(self, data, out):

        # we need to re-calculate the downsampling factor
        # that it actually is an 1 / integer gets checked in the frontend
        factor = self.cfg['new_samplerate'] / data.samplerate

        if data.selection is not None:
            chanSec = data.selection.channel
            trl = _resampling_trl_definition(data.selection.trialdefinition, factor)
        else:
            chanSec = slice(None)
            trl = _resampling_trl_definition(data.trialdefinition, factor)

        out.trialdefinition = trl
        # now set new samplerate
        out.samplerate = self.cfg['new_samplerate']
        out.channel = np.array(data.channel[chanSec])


@process_io
def resample_cF(dat,
                samplerate=1,
                new_samplerate=1,
                lpfreq=None,
                order=None,
                timeAxis=0,
                chunkShape=None,
                noCompute=False
                ):
    """
    Provides resampling of signals. The `new_samplerate` can be
    any (rational) factor of the original `samplerate`.

    For the anti-aliasing an explicit low-pass firws filter
    is constructed. Either implicitly with `lpfreq=None`
    which takes the new Nyquist (new_samplerate / 2) as cut-off
    or with an explicit `lpfreq` as the cut-off frequency.

    dat : (N, K) :class:`numpy.ndarray`
        Uniformly sampled multi-channel time-series data
        The 1st dimension is interpreted as the time axis,
        columns represent individual channels.
    samplerate : float
        Sample rate of the input data in Hz
    new_samplerate : float
        Sample rate of the output data in Hz
    lpfreq : None or float, optional
        Leave at `None` for standard anti-alias filtering with
        the new Nyquist or set explicitly in Hz
    order : None or int, optional
        Order (length) of the firws anti-aliasing filter.
        The default `None` will create a filter of
        maximal order which is the number of samples in the trial.
    timeAxis : int, optional
        Index of running time axis in `dat` (0 or 1)

    Returns
    -------
    resampled : (X, K) :class:`~numpy.ndarray`
        The resampled data

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

    nSamples = dat.shape[0]
    fs_ratio = new_samplerate / samplerate

    if noCompute:
        new_nSamples = int(np.ceil(nSamples * fs_ratio))
        return (new_nSamples, dat.shape[1]), dat.dtype

    resampled = resample(dat,
                         samplerate,
                         new_samplerate,
                         lpfreq=lpfreq,
                         order=order)

    return resampled


class Resample(ComputationalRoutine):

    """
    Compute class that performs resampling (up-fir-down)
    of :class:`~syncopy.AnalogData` objects

    Sub-class of :class:`~syncopy.shared.computational_routine.ComputationalRoutine`,
    see :doc:`/developer/compute_kernels` for technical details on Syncopy's compute
    classes and metafunctions.

    See also
    --------
    syncopy.preprocessing : parent metafunction
    """

    computeFunction = staticmethod(resample_cF)

    # 1st argument,the data, gets omitted
    valid_kws = list(signature(downsample_cF).parameters.keys())[1:]

    def process_metadata(self, data, out):

        # we need to re-calculate the resampling factor
        factor = self.cfg['new_samplerate'] / data.samplerate
        trafo_trl = _resampling_trl_definition

        if data.selection is not None:
            chanSec = data.selection.channel
            trl = trafo_trl(data.selection.trialdefinition, factor)
        else:
            chanSec = slice(None)
            trl = trafo_trl(data.trialdefinition, factor)

        out.trialdefinition = trl

        # now set new samplerate
        out.samplerate = self.cfg['new_samplerate']
        out.channel = np.array(data.channel[chanSec])


@process_io
def detrending_cF(dat, polyremoval=None, timeAxis=0, noCompute=False, chunkShape=None):

    """
    Simple cF to wire SciPy's `detrend` to our CRs,
    supported are constant and linear detrending

    Parameters
    ----------
    dat : (N, K) :class:`numpy.ndarray`
        Uniformly sampled multi-channel time-series data
        The 1st dimension is interpreted as the time axis,
        columns represent individual channels.
        Dimensions can be transposed to `(K, N)` with the `timeAxis` parameter
    polyremoval : {0, 1} or None
        Order of polynomial used for de-trending data in the time domain.
        A value of 0 corresponds to subtracting the mean
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
    detrended : (N, K) :class:`~numpy.ndarray`
        The detrended signals

    Notes
    -----
    This method is intended to be used as
    :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
    inside a :class:`~syncopy.shared.computational_routine.ComputationalRoutine`.
    Thus, input parameters are presumed to be forwarded from a parent metafunction.
    Consequently, this function does **not** perform any error checking and operates
    under the assumption that all inputs have been externally validated and cross-checked.
    """

    # should be captured in frontend
    if polyremoval is None:
        return dat

    # Re-arrange array if necessary and get dimensional information
    if timeAxis != 0:
        dat = dat.T       # does not copy but creates view of `dat`
    else:
        dat = dat

    # detrending does not change the shape
    outShape = dat.shape
    if noCompute:
        return outShape, np.float32

    # detrend
    if polyremoval == 0:
        dat = sci.detrend(dat, type='constant', axis=0, overwrite_data=True)
    elif polyremoval == 1:
        dat = sci.detrend(dat, type='linear', axis=0, overwrite_data=True)

    # renaming
    detrended = dat
    return detrended


class Detrending(ComputationalRoutine):

    """
    Compute class that performs constant or linear detrending
    of :class:`~syncopy.AnalogData` objects

    Sub-class of :class:`~syncopy.shared.computational_routine.ComputationalRoutine`,
    see :doc:`/developer/compute_kernels` for technical details on Syncopy's compute
    classes and metafunctions.

    See also
    --------
    syncopy.preprocessing : parent metafunction
    """

    computeFunction = staticmethod(detrending_cF)

    # 1st argument,the data, gets omitted
    valid_kws = list(signature(detrending_cF).parameters.keys())[1:]

    def process_metadata(self, data, out):

        propagate_properties(data, out)


@process_io
def standardize_cF(dat, polyremoval=None, timeAxis=0, noCompute=False, chunkShape=None):

    """
    Yet another simple cF to z-score ('standardize') signals:
    subtracting the mean and normalize by standard deviation.

    Parameters
    ----------
    dat : (N, K) :class:`numpy.ndarray`
        Uniformly sampled multi-channel time-series data
        The 1st dimension is interpreted as the time axis,
        columns represent individual channels.
        Dimensions can be transposed to `(K, N)` with the `timeAxis` parameter
    polyremoval : {0, 1} or None
        Order of polynomial used for de-trending data in the time domain prior
        to z-scoring. A value of 0 corresponds to subtracting the mean
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
    standardized : (N, K) :class:`~numpy.ndarray`
        The standardized signals

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

    # cF does not change the shape
    outShape = dat.shape
    if noCompute:
        return outShape, np.float32

    # detrend
    if polyremoval == 0:
        dat = sci.detrend(dat, type='constant', axis=0, overwrite_data=True)
    elif polyremoval == 1:
        dat = sci.detrend(dat, type='linear', axis=0, overwrite_data=True)

    # standardize
    dat = (dat - np.mean(dat, axis=0)) / np.std(dat, axis=0)

    # renaming
    standardized = dat
    return standardized


class Standardize(ComputationalRoutine):

    """
    Compute class that performs standardizing
    of :class:`~syncopy.AnalogData` objects

    Sub-class of :class:`~syncopy.shared.computational_routine.ComputationalRoutine`,
    see :doc:`/developer/compute_kernels` for technical details on Syncopy's compute
    classes and metafunctions.

    See also
    --------
    syncopy.preprocessing : parent metafunction
    """

    computeFunction = staticmethod(standardize_cF)

    # 1st argument,the data, gets omitted
    valid_kws = list(signature(standardize_cF).parameters.keys())[1:]

    def process_metadata(self, data, out):

        propagate_properties(data, out)


def _resampling_trl_definition(orig_trl, factor):

    """
    Construct new trialdefinition from original
    trialdefinition and the resampling factor
    """

    # start from input trial lengths and scale
    # and ceil them to arrive at new trial lengths
    # important is 1st diff then ceil..
    sinfo = orig_trl[:, :2]
    trl_len = np.ceil(np.diff(sinfo * factor, axis=1)).squeeze()

    # use ceil again to define new trial start
    # and offset samples
    trl_scaled = np.ceil(orig_trl * factor)
    trl_starts = trl_scaled[:, 0]
    offsets = trl_scaled[:, 2]

    # now add new trl_len to get new trial ends
    trl_ends = trl_starts + trl_len

    # finally stack everything back together
    trldef = np.column_stack([trl_starts, trl_ends, offsets])

    return trldef
