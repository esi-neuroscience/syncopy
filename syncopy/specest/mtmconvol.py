# -*- coding: utf-8 -*-
# 
# Time-frequency analysis based on a short-time Fourier transform
# 

# Builtin/3rd party package imports
import numpy as np
from scipy import signal


def mtmconvol(data_arr, samplerate, nperseg, noverlap=None, taper="hann",
              taperopt={}, boundary='zeros', padded=False):

    # attach dummy channel axis in case only a
    # single signal/channel is the input
    if data_arr.ndim < 2:
        data_arr = data_arr[:, np.newaxis]

    nSamples = data_arr.shape[0]
    nChannels = data_arr.shape[1]
    
    # FFT frequencies from the window size
    freqs = np.fft.rfftfreq(nperseg, 1 / samplerate)    
    nFreq = freqs.size

    taper_func = getattr(signal.windows,  taper)
    # only truly 2d for multi-taper "dpss"
    windows = np.atleast_2d(taper_func(nperseg, **taperopt))

    # number of time points in the output    
    if boundary is None:
        # no padding: we loose half the window on each side
        nTime = int(np.ceil(nSamples / (nperseg - noverlap))) - nperseg
    else:
        # the signal is padded on each side as to cover
        # the whole signal
        nTime = int(np.ceil(nSamples / (nperseg - noverlap)))

    # Short time Fourier transforms (nTime x nTapers x nFreq x nChannels)
    ftr = np.zeros((nTime, windows.shape[0], nFreq, nChannels), dtype='complex128')

    for taperIdx, win in enumerate(windows):
        # pxx has shape (nFreq, nChannels, nTime)        
        freq, _, pxx = signal.stft(data_arr, samplerate, win,
                                   nperseg, noverlap, boundary=boundary, padded=padded, axis=0)
        ftr[:, taperIdx, ...] = pxx.transpose(2, 0, 1)[:nTime, ...]

    return ftr, freqs
    



    
