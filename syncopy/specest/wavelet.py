# -*- coding: utf-8 -*-
#
# Time-frequency analysis with wavelets
#

# Builtin/3rd party package imports
import numpy as np

# Local imports
from syncopy.specest.wavelets import cwt


def wavelet(data_arr, samplerate, scales, wavelet):

    """
    Perform time-frequency analysis on multi-channel time series data
    using a wavelet transform

    Parameters
    ----------

    data_arr : 2D :class:`numpy.ndarray`
        Uniformly sampled multi-channel time-series
        The 1st dimension is interpreted as the time axis
    samplerate : float
        Samplerate of `data_arr` in Hz
    scales : 1D :class:`numpy.ndarray`
        Set of scales to use in wavelet transform.
    wavelet : callable
        Wavelet function to use, one of
        :data:`~syncopy.specest.const_def.availableWavelets`

    Returns
    -------
    spec : :class:`numpy.ndarray`
        Complex time-frequency representation of the input data.
        Shape is (len(scales),) + data_arr.shape
    """

    spec = cwt(data_arr, wavelet=wavelet, widths=scales, dt=1 / samplerate, axis=0)

    return spec


def get_optimal_wavelet_scales(scale_from_period, nSamples, dt, dj=0.25, s0=None):
    """
    Local helper to compute an "optimally spaced" set of scales for wavelet analysis

    Parameters
    ----------
    scale_from_period : func
        Function to convert periods to Wavelet specific scales.
    nSamples : int
        Sample-count (i.e., length) of time-series that is analyzed
    dt : float
        Time-series step-size; temporal spacing between consecutive samples
        (1 / sampling rate)
    dj : float
        Spectral resolution of scales. The choice of `dj` depends on the spectral
        width of the employed wavelet function. For instance, ``dj = 0.5`` is the
        largest value that still yields adequate sampling in scale for the Morlet
        wavelet. Other wavelets allow larger values of `dj` while still providing
        sufficient spectral resolution. Small values of `dj` yield finer scale
        resolution.
    s0 : float or None
        Smallest resolvable scale; should be chosen such that the equivalent
        Fourier period is approximately ``2 * dt``. If `None`, `s0` is computed
        to satisfy this criterion.

    Returns
    -------
    scales : 1D :class:`numpy.ndarray`
        Set of scales to use in the wavelet transform, ordered
        from high(low) scale(frequency) to low(high) scale(frequency)

    Notes
    -----
    The calculation of an "optimal" set of scales follows [ToCo98]_.
    This routine is a local auxiliary method that is purely intended for internal
    use. Thus, no error checking is performed.

    .. [ToCo98] C. Torrence and G. P. Compo. A Practical Guide to Wavelet Analysis.
       Bulletin of the American Meteorological Society. Vol. 79, No. 1, January 1998.

    See also
    --------
    syncopy.specest.wavelet.wavelet : :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
                                      performing time-frequency analysis using non-orthogonal continuous wavelet transform
    """

    # Compute `s0` so that the equivalent Fourier period is approximately ``2 * dt```
    if s0 is None:
        s0 = scale_from_period(2 * dt)

    # Largest scale
    J = int((1 / dj) * np.log2(nSamples * dt / s0))
    scales = s0 * 2 ** (dj * np.arange(0, J + 1))
    # we want the low frequencies first
    return scales[::-1]
