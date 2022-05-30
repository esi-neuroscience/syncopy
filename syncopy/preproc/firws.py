# -*- coding: utf-8 -*-
#
# Routines for designing and applying
# FIR windowed sinc filters
#

# Builtin/3rd party package imports
import numpy as np
import scipy.signal.windows as sci_win
from scipy.signal import fftconvolve


def apply_fir(data, fkernel):

    """
    Convolution of the input `data` with a FIR filter.
    The filter's impulse response is given by `fkernel`.

    Parameters
    ----------
    data  : (N, K) :class:`numpy.ndarray`
        Uniformly sampled multi-channel time-series data
        The 1st dimension is interpreted as the time axis,
        columns represent individual channels.
    fkernel : (N,) :class:`numpy.ndarray`
        The time domain representation of the FIR filter

    Returns
    -------
    filtered : (N, K) :class:`~numpy.ndarray`
        The filtered signals

    """

    slices = [None for _ in data.shape]
    slices[0] = slice(None)
    slices = tuple(slices)

    filtered = fftconvolve(data, fkernel[slices], mode="same")
    return filtered


def design_wsinc(window, order, f_c, filter_type="lp"):

    """
    Construct the windowed sinc filter kernel in the time domain

    Parameters
    ----------
    window : str
        One of `scipy.signal.windows`, good choices are
        "blackman", "hamming" and "hann"
    order : int
       The order, or simply length, of the filter
       If not even gets incremented by one
    f_c : float or array_like
       Cut-off frequenc(ies) in sampling units,
       maximum is Nyquist `f_c=0.5`. For band-pass
       and band-stop filters they have to be ordered low to high.
    filter_type : {'lp', 'hp', 'bp, 'bs'}, optional
        Select type of filter, either low-pass `'lp'`,
        high-pass `'hp'`, band-pass `'bp'` or band-stop (Notch) `'bs'`.

    Returns
    ------
    kernel : (order,) :class:`numpy.ndarray`
        The windowed sinc as 1d array
    """

    # order has to be even
    if order % 2 != 0:
        order += 1

    if filter_type == "lp":
        kernel = windowed_sinc(window, order, f_c)
        return kernel

    elif filter_type == "hp":
        lp_kernel = windowed_sinc(window, order, f_c)
        kernel = invert_sinc(lp_kernel)
        return kernel

    if filter_type == "bp":
        # high-pass freq is lower than low-pass freq
        # for band-pass filters
        f_hp, f_lp = f_c
    elif filter_type == "bs":
        # high-pass freq is higher than low-pass freq
        # for band-stop filters
        f_lp, f_hp = f_c

    # construct band filters
    lp_kernel = windowed_sinc(window, order, f_lp)
    kernel = windowed_sinc(window, order, f_hp)
    hp_kernel = invert_sinc(kernel)
    kernel = lp_kernel + hp_kernel

    # subtract dc component from filter addition
    # for band-pass filters in the time-domain
    if filter_type == "bp":
        kernel[len(kernel) // 2] -= 1

    return kernel


def windowed_sinc(window, order, f_c):

    """
    Construct the symmetric windowed sinc filter
    with a cut-off frequency `f_c`

    Parameters
    ----------
    window : str
        One of `scipy.signal.windows`
    order : int
       The order of the filter, has to be strictly even
    f_c : float
       Cut-off frequency in sampling units,
       maximum is Nyquist `f_c=0.5`

    Returns
    -------
    kernel : :class:`numpy.ndarray`
        The windowed filter with length `order + 1`

    """

    # angular cut-off frequency
    omega_c = 2 * np.pi * f_c

    win_func = getattr(sci_win, window)
    win = win_func(order + 1)

    # one-sided support
    m_half = np.arange(1, order / 2 + 1)
    kernel = np.sin(omega_c * m_half) / m_half
    kernel = np.hstack([kernel[::-1], omega_c, kernel]) * win
    # normalize to unity gain
    kernel = kernel / kernel.sum()

    return kernel


def invert_sinc(kernel):

    """
    In frequency space the high-pass filter
    is just 1 - low-pass. Hence, formally we can
    calculate the high-pass version
    in the time domain via the correspinding inverse Fourier:

    Fourier^-1 {1 - Fourier(kernel)} = Fourier^-1(1) - kernel

    This gives a delta-peak at the 0-lag midpoint
    minus the original low-pass impulse response.
    In the DSP world the delta distribution corresponds to a mere `1`.
    """
    kernel = -kernel
    # kernel size is always odd
    kernel[len(kernel) // 2] += 1
    return kernel


def minphaserceps(fkernel):

    """
    Tranform FIR filter to minmum phase (causal) filter

    The original Matlab function was written for FieldTrip in 2013 by
    Andreas Widmann, University of Leipzig, widmann@uni-leipzig.de

    Notes
    -----
    .. [1] Smith III, O. J. (2007). Introduction to Digital Filters with Audio
       Applications. W3K Publishing. Retrieved Nov 11 2013, from
       https://ccrma.stanford.edu/~jos/fp/Matlab_listing_mps_m.html
    .. [2] Vetter, K. (2013, Nov 11). Long FIR filters with low latency.
       Retrieved Nov 11 2013, from
       http://www.katjaas.nl/minimumphase/minimumphase.html
    """

    nSamples = len(fkernel)
    upsamplingFactor = (
        1e3  # Impulse response upsampling/zero padding to reduce time-aliasing
    )
    nFFT = int(2 ** np.ceil(np.log2(nSamples * upsamplingFactor)))  # Power of 2
    clipThresh = 1e-8  # -160 dB

    # Spectrum
    specC = np.abs(np.fft.fft(fkernel, nFFT))
    specC[specC < clipThresh] = clipThresh  # Clip spectrum to reduce time-aliasing

    # Real cepstrum
    specR = np.real(np.fft.ifft(np.log(specC)))

    # Convolve
    ires = np.hstack([specR[1 : nFFT // 2], 0]) + np.conj(
        specR[nFFT // 2 : nFFT + 1][::-1]
    )
    specR = np.hstack([specR[0], ires, np.zeros(nFFT // 2 - 2)])

    # Minimum phase
    MinPhase = np.real(np.fft.ifft(np.exp(np.fft.fft(specR))))

    # Remove zero-padding
    return MinPhase[:nSamples]


def _fir_df(cutoffArray, fs):

    """
    Computes default and maximum possible transition band width from
    FIR filter cutoff frequency(ies) according to the following heuristic:

    Transition band width is 25% of the lower cutoff
    frequency, but not lower than 2 Hz, where possible (for bandpass,
    highpass, and bandstop) and distance from passband edge to critical
    frequency (DC, Nyquist) otherwise.

    This is almost a 1:1 copy of the FT routine `fir_df`, only difference
    is the bandwidths are already normalized with the sampling rate `fs`.

    See also
    --------
    `FieldTrip implementation <https://github.com/fieldtrip/fieldtrip/blob/master/preproc/private/fir_df.m>`_
    """

    # still WIP
    raise NotImplementedError

    TRANSWIDTHRATIO = 0.25
    Fn = fs / 2
    cutoffArray = np.array(cutoffArray) / fs

    # Max possible transition band width
    cutoffArray = np.sort(cutoffArray)
    maxTBWArray = cutoffArray * 2 * (Fn - cutoffArray) * 2 * np.diff(cutoffArray)
    maxDf = np.min(maxTBWArray)

    # Default filter order heuristic
    df = np.min(np.max(cutoffArray[0] * TRANSWIDTHRATIO * 2) * maxDf)

    return df, maxDf
