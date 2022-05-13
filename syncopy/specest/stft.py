# -*- coding: utf-8 -*-
#
# Short-time Fourier transform, uses np.fft as backend
#

# Builtin/3rd party package imports
import numpy as np
import scipy.signal as sci_sig

# local imports
from ._norm_spec import _norm_spec


def stft(dat,
         fs=1.,
         window=None,
         nperseg=256,
         noverlap=None,
         boundary='zeros',
         detrend=False,
         padded=True,
         axis=0):

    """
    Implements the short-time (or windowed) Fourier transform

    The interface is designed to be close to SciPy's implementation: :func: `~scipy.signal.stft`

    Parameters
    ----------
    dat : (N, K) :class:`numpy.ndarray`
        Uniformly sampled multi-channel time-series data
        The 1st dimension is interpreted as the time axis
        per default
    fs : float
        Samplerate in Hz
    window : (M,) :class:`numpy.ndarray` or None, optional
        Taper to be multiplied with the
        signal segments, has to be of length `nperseg`
    nperseg : int, optional
        Length of each segment. Defaults to 256.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        `noverlap = nperseg // 2`. Set to `nperseg - 1` to have an output
        with `N` time points. Defaults to `None`.
    boundary : 'zeros' or None
        Specifies whether the input signal is extended at both ends with
        `nperseg // 2` zeros in order to center the first windowed segment on
        the first input point.  If set to `None` half the segment size is
        lost on each side of the input signal. Defaults to `'zeros'`
    detrend : str or `False`, optional
        Optional detrending of the individual segments.
        Sets `type` argument of :func: `~scipy.signal.detrend`,
        acceptable are either `'constant'` or `'linear'`.
        Defaults to  `False` such that no detrending is done.
    padded : bool, optional
        Specifies whether the input signal is zero-padded at the end to
        make the signal fit exactly into an integer number of window
        segments, so that all of the signal is included in the output.
        Defaults to `True`. Padding occurs after boundary extension, if
        `boundary` is not `None`, and `padded` is `True`, as is the
        default.
    axis : int, optional
        Axis along which the STFT is computed; the default is over the
        first axis (i.e. `axis=0`)

    Returns
    -------
    ftr : :class:`numpy.ndarray`
        Short-time fourier transform of the input `dat`
        Per default the first axis corresponds to the segment times
    freqs : :class:`numpy.ndarray`
        Array of sampling frequencies
    times : :class:`numpy.ndarray`
        Array of segment times

    Notes
    -----
    For a power spectral estimate compute:

    ``Sxx = np.real(ftr * ftr.conj())``

    The STFT result is normalized such that this yields the power
    spectral density. For a clean harmonic and a frequency bin
    width of `dF` this will give a peak power of `A**2 / 2 * dF`,
    with `A` as harmonic ampltiude.

    """
    # needed for stride tricks
    # from here on axis=-1 is the data axis!
    if dat.ndim > 1:
        if axis != -1:
            dat = np.moveaxis(dat, axis, -1)

    # extend along time axis to fit in
    # sliding windows at the edges
    if boundary is not None:
        zeros_shape = list(dat.shape)
        zeros_shape[-1] = nperseg // 2
        zeros = np.zeros(zeros_shape, dtype=dat.dtype)
        dat = np.concatenate((zeros, dat, zeros), axis=-1)

    # defaults to half window overlap
    if noverlap is None:
        noverlap = nperseg // 2
    nstep = nperseg - noverlap

    if padded:
        # Pad to integer number of windowed segments
        # I.e make x.shape[-1] = nperseg + (nseg-1)*nstep, with integer nseg
        nadd = (-(dat.shape[-1]-nperseg) % nstep) % nperseg
        zeros_shape = list(dat.shape[:-1]) + [nadd]
        dat = np.concatenate((dat, np.zeros(zeros_shape)), axis=-1)

    # Create strided array of data segments
    if nperseg == 1 and noverlap == 0:
        dat = dat[..., np.newaxis]
    else:
        # https://stackoverflow.com/a/5568169
        step = nperseg - noverlap
        shape = dat.shape[:-1] + ((dat.shape[-1] - noverlap) // step, nperseg)
        strides = dat.strides[:-1] + (step * dat.strides[-1], dat.strides[-1])
        dat = np.lib.stride_tricks.as_strided(dat, shape=shape,
                                              strides=strides)
    # dat now has shape (nChannels, nSamples, nperseg)

    # detrend each segment separately
    if detrend:
        dat = sci_sig.detrend(dat, type=detrend, overwrite_data=True)

    if window is not None:
        # Apply window by multiplication
        dat = dat * window

    times = np.arange(nperseg / 2, dat.shape[-1] - nperseg / 2 + 1,
                      nperseg - noverlap) / fs
    if boundary is not None:
        times -= (nperseg / 2) / fs

    freqs = np.fft.rfftfreq(nperseg, 1 / fs)

    # the complex transforms
    ftr = np.fft.rfft(dat, axis=-1)

    # normalization to power -> squared amplitude / 2
    ftr = _norm_spec(ftr, nperseg, fs)

    # Roll frequency axis back to axis where the data came from
    ftr = np.moveaxis(ftr, -1, 0)

    return ftr, freqs, times
