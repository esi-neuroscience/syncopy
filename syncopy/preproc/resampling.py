# -*- coding: utf-8 -*-
#
# Backend methods/functions for
# trivial down- and rational (p/q) resampling
#

# Builtin/3rd party package imports
import fractions
import scipy.signal as sci_sig

# Syncopy imports
from syncopy.preproc import firws


def resample(data, orig_fs, new_fs, lpfreq=None, order=None):

    """
    Uses SciPy's polyphase method for the implementation
    of the standard resampling procedure:
        upsampling : FIR filtering : downsampling

    SciPy's default FIR filter has a slow roll-off,
    so the default is to design and use a homegrown firws.

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
    lpfreq : None or float, optional
        Leave at `None` for standard anti-alias filtering with
        the new Nyquist or set explicitly in Hz
        If set to `-1` use SciPy's default kaiser windowed FIR
    order : None or int, optional
        Order (length) of the firws anti-aliasing filter.
        The default `None` will create a filter of
        maximal order which is the number of samples times the upsampling
        factor of the trial, or 10 000 if that is smaller

    Returns
    -------
    resampled : (N, K) :class:`~numpy.ndarray`
        The resampled signals

    See also
    --------
    `SciPy's resample implementation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html>`_
    syncopy.preproc.compRoutines.downsample_cF : Straightforward and cheap downsampling
    """

    nSamples = data.shape[0]

    # get up/down sampling factors
    up, down = _get_updn(orig_fs, new_fs)
    fs_ratio = new_fs / orig_fs

    # -- design firws low-pass filter --

    # default cuts at new Nyquist
    if lpfreq is None:
        f_c = 0.5 * fs_ratio
    # for backend tests only,
    # negative values don't pass the frontend
    elif lpfreq == -1:
        f_c = None
    # explicit cut-off
    else:
        f_c = lpfreq / orig_fs
    if order is None:
        order = nSamples * up
        # limit maximal order
        order = 10000 if order > 10000 else order

    if f_c:
        # filter has to be applied to the upsampled data
        window = firws.design_wsinc("hamming",
                                    order=order,
                                    f_c=f_c / up)
    else:
        window = ('kaiser', 5.0)  # triggers SciPy default filter design

    resampled = sci_sig.resample_poly(data, up, down, window=window, axis=0)

    return resampled


def downsample(
    dat,
    samplerate=1,
    new_samplerate=1,
    ):
    """
    Provides basic downsampling of signals. The `new_samplerate` should be
    an integer division of the original `samplerate`.

    Parameters
    ----------
    dat : (N, K) :class:`numpy.ndarray`
        Uniformly sampled multi-channel time-series data
        The 1st dimension is interpreted as the time axis,
        columns represent individual channels.
    samplerate : float
        Sample rate of the input data
    new_samplerate : float
        Sample rate of the output data

    Returns
    -------
    resampled : (X, K) :class:`~numpy.ndarray`
        The downsampled data

    """

    # we need integers for slicing
    skipped = int(samplerate // new_samplerate)

    return dat[::skipped]


def _get_updn(orig_fs, new_fs):

    """
    Get the up/down sampling
    factors from the original and target
    sampling rate.

    NOTE: Can return very large factors for
          "almost irrational" sampling rate ratios!
    """

    frac = fractions.Fraction.from_float(new_fs / orig_fs)
    # trim down
    frac = frac.limit_denominator()

    return frac.numerator, frac.denominator
