# -*- coding: utf-8 -*-
#
# Time-frequency analysis based on a short-time Fourier transform
#

# Builtin/3rd party package imports
import numpy as np
import logging
import platform
from scipy import signal

# local imports
from .stft import stft
from ._norm_spec import _norm_taper


def mtmconvol(data_arr, samplerate, nperseg, noverlap=None, taper="hann",
              taper_opt=None, boundary='zeros', padded=True, detrend=False):

    """
    (Multi-)tapered short time fast Fourier transform. Returns
    full complex Fourier transform for each taper.
    Multi-tapering only supported with Slepian windwows (`taper="dpss"`).

    Parameters
    ----------
    data_arr : (N,) :class:`numpy.ndarray`
        Uniformly sampled multi-channel time-series data
        The 1st dimension is interpreted as the time axis
    samplerate : float
        Samplerate in Hz
    nperseg : int
        Sliding window size in sample units
    noverlap : int
        Overlap between consecutive windows, set to ``nperseg - 1``
        to cover the whole signal
    taper : str or None
        Taper function to use, one of `scipy.signal.windows`
        Set to `None` for no tapering.
    taper_opt : dict or None
        Additional keyword arguments passed to the `taper` function.
        For multi-tapering with ``taper='dpss'`` set the keys
        `'Kmax'` and `'NW'`.
        For further details, please refer to the
        `SciPy docs <https://docs.scipy.org/doc/scipy/reference/signal.windows.html>`_
    boundary : str or None
        Wether or not to auto-pad the signal such that a window is centered on each
        sample. If set to `None` half the window size (`nperseg`) will be lost
        on each side of the signal. Defaults `'zeros'`, for zero padding extension.
    padded : bool
        Additional padding in case ``noverlap != nperseg - 1`` to fit an integer number
        of windows.

    Returns
    -------
    ftr : 4D :class:`numpy.ndarray`
         The Fourier transforms, complex output has shape:
         ``(nTime, nTapers x nFreq x nChannels)``
    freqs : 1D :class:`numpy.ndarray`
         Array of Fourier frequencies

    Notes
    -----
    For a (MTM) power spectral estimate average the absolute squared
    transforms across tapers:

    ``Sxx = np.real(ftr * ftr.conj()).mean(axis=0)``

    The STFT result is normalized such that this yields the power
    spectral density. For a clean harmonic and a frequency bin
    width of `dF` this will give a peak power of `A**2 / 2 * dF`,
    with `A` as harmonic ampltiude.
    """

    # attach dummy channel axis in case only a
    # single signal/channel is the input
    if data_arr.ndim < 2:
        data_arr = data_arr[:, np.newaxis]

    nSamples = data_arr.shape[0]
    nChannels = data_arr.shape[1]

    # FFT frequencies from the window size
    freqs = np.fft.rfftfreq(nperseg, 1 / samplerate)
    nFreq = freqs.size
    # frequency bins
    dFreq = freqs[1] - freqs[0]

    if taper is None:
        taper = 'boxcar'

    taper_func = getattr(signal.windows, taper)

    if taper_opt is None:
        taper_opt = {}

    # this parameter mitigates the sum-to-zero problem for the odd slepians
    # as signal.stft has hardcoded scaling='spectrum'
    # -> normalizes with win.sum() :/
    # see also https://github.com/scipy/scipy/issues/14740
    if taper == 'dpss':
        taper_opt['sym'] = False

    # only truly 2d for multi-taper "dpss"
    windows = np.atleast_2d(taper_func(nperseg, **taper_opt))

    # normalize window(s)
    windows = _norm_taper(taper, windows, nperseg)

    # number of time points in the output
    if boundary is None:
        # no padding: we loose half the window on each side
        nTime = int(np.ceil(nSamples / (nperseg - noverlap))) - nperseg
    else:
        # the signal is padded on each side as to cover
        # the whole signal
        nTime = int(np.ceil(nSamples / (nperseg - noverlap)))

    # Short time Fourier transforms (nTime x nTapers x nFreq x nChannels)
    ftr = np.zeros((nTime, windows.shape[0], nFreq, nChannels), dtype='complex64')

    logger = logging.getLogger("syncopy_" + platform.node())
    logger.debug(f"Running mtmconvol on {len(windows)} windows, data chunk has {nSamples} samples and {nChannels} channels.")

    for taperIdx, win in enumerate(windows):
        # ftr has shape (nFreq, nChannels, nTime)
        pxx, _, _ = stft(data_arr, samplerate, window=win,
                         nperseg=nperseg, noverlap=noverlap,
                         boundary=boundary, padded=padded,
                         axis=0, detrend=detrend)

        ftr[:, taperIdx, ...] = pxx.transpose(2, 0, 1)[:nTime, ...]

    return ftr, freqs
