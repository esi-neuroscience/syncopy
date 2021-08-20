# -*- coding: utf-8 -*-
# 
# Spectral estimation with (multi-)tapered FFT
# 

# Builtin/3rd party package imports
import numpy as np
from scipy import signal


def mtmfft(data_arr, taper="hann", taperopt={}):

    '''
    Multi-taper fast Fourier transform. Returns
    full complex Fourier transform in samplerate units.
    Multi-taper only supported for Slepian windwows (`taper="dpss"`).

    Parameters
    ----------
    data_arr : (N,) :class:`numpy.ndarray`
        Uniformly sampled time-series data
        The 1st dimension is interpreted as the time axis
    taper : str
        Taper function to use, one of scipy.signal.windows
    taperopt : dict
        Additional keyword arguments passed to the `taper` function. For further 
        details, please refer to the 
        `SciPy docs <https://docs.scipy.org/doc/scipy/reference/signal.windows.html>`_
    '''

    data_arr = np.atleast_2d(data_arr)
    nSamples = data_arr.shape[0]
    nChannels = data_arr.shape[1]
    taper_func = getattr(signal.windows,  taper)

    nFreq = int(np.floor(nSamples / 2) + 1)
    # only really 2d if taper='dpss' with Kmax > 1
    windows = np.atleast_2d(taper_func(nSamples, **taperopt))
    # (nTapers x nFreq x nChannels)
    spec = np.zeros((windows.shape[0], nFreq, nChannels), dtype='complex128')

    for taperIdx, win in enumerate(windows):
        win = np.tile(win, (nChannels, 1)).T
        spec[taperIdx] = np.fft.rfft(data_arr * win, axis=0)

    return spec

    
        
        

