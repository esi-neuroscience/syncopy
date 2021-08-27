# -*- coding: utf-8 -*-
# 
# Spectral estimation with (multi-)tapered FFT
# 

# Builtin/3rd party package imports
import numpy as np
from scipy import signal


def mtmfft(data_arr, samplerate, taper="dpss", nTaper=None, tapsmofrq=None, taperopt={}):

    '''
    (Multi-)tapered fast Fourier transform. Returns
    full complex Fourier transform for each taper.
    Multi-tapering only supported with Slepian windwows (`taper="dpss"`).
    
    Parameters
    ----------
    data_arr : (N,) :class:`numpy.ndarray`
        Uniformly sampled time-series data
        The 1st dimension is interpreted as the time axis
    samplerate : float
        Samplerate in Hz
    taper : str
        Taper function to use, one of scipy.signal.windows
    nTaper : int
        Only effective if `taper='dpss'`. Number of orthogonal tapers to use.    
    tapsmofrq : float
        Only effective if `taper='dpss'`. The amount of spectral smoothing through  
        multi-tapering (Hz).  Note that smoothing frequency specifications are one-sided, 
        i.e., 4 Hz smoothing means plus-minus 4 Hz, i.e., a 8 Hz smoothing box.
    taperopt : dict
        Additional keyword arguments passed to the `taper` function. 
        For further details, please refer to the 
        `SciPy docs <https://docs.scipy.org/doc/scipy/reference/signal.windows.html>`_

    Returns
    -------

    spec : 3D :class:`numpy.ndarray`
         Complex output has shape (nTapers x nFreq x nChannels).

    freqs : 1D :class:`numpy.ndarray`
         Array of Fourier frequencies
    
    Notes
    -----

    For a (MTM) power spectral estimate average the absolute squared
    transforms across tapers:

    Sxx = np.real(spec * spec.conj()).mean(axis=0)
          
    '''

    # Slepian window parameters
    if taper == "dpss":
        taperopt['Kmax'] = nTaper
        taperopt['NW'] = tapsmofrq

    data_arr = np.atleast_2d(data_arr)
    nSamples = data_arr.shape[0]
    nChannels = data_arr.shape[1]
    taper_func = getattr(signal.windows,  taper)

    freqs = np.fft.rfftfreq(nSamples, 1 / samplerate)
    nFreq = freqs.size
    # only really 2d if taper='dpss' with Kmax > 1
    windows = np.atleast_2d(taper_func(nSamples, **taperopt))
    # (nTapers x nFreq x nChannels)
    spec = np.zeros((windows.shape[0], nFreq, nChannels), dtype='complex128')

    for taperIdx, win in enumerate(windows):
        win = np.tile(win, (nChannels, 1)).T
        spec[taperIdx] = np.fft.rfft(data_arr * win, axis=0)

    return spec, freqs

    
        
        

