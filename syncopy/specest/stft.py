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
         nperseg=200,
         noverlap=None,
         boundary='zeros',
         detrend=False,
         padded=True,
         axis=0):

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

    # detrend each window separately
    if detrend:
        dat = sci_sig.detrend(dat, type=detrend, overwrite_data=True)

    if window is not None:
        # Apply window by multiplication
        dat = window * dat

    freqs = np.fft.rfftfreq(nperseg, 1 / fs)

    # the complex transforms
    ftr = np.fft.rfft(dat, axis=-1)

    # normalization to squared amplitude density
    ftr = _norm_spec(ftr, nperseg, freqs)

    # Roll frequency axis back to axis where the data came from
    ftr = np.moveaxis(ftr, -1, 0)

    return ftr, freqs
