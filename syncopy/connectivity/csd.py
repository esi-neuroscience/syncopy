# -*- coding: utf-8 -*-
#
# Cross Spectral Density and Coherency
#

# Builtin/3rd party package imports
import numpy as np

# syncopy imports
from syncopy.specest.mtmfft import mtmfft


def csd(data_arr, samplerate, taper="hann", taperopt={}, norm=False):

    """
    Cross spectral density (CSD) estimate between all channels
    of the input data. First all the individual Fourier transforms
    are calculated via a (multi-)tapered FFT, then the pairwise
    coherence is calculated. Averaging over tapers is done implicitly.
    Output consists of all (nChannels x nChannels+1)/2 different CSD estimates
    aranged in a symmetric fashion (CSD_ij == CSD_ji). The elements on the
    main diagonal (CSD_ii) are the auto-spectra.

    If normalization is required (`norm=True`) the respective coherencies
    are returned.

    Parameters
    ----------
    data_arr : (K,N) :class:`numpy.ndarray`
        Uniformly sampled multi-channel time-series data
        The 1st dimension is interpreted as the time axis,
        columns represent individual channels.
    norm : bool
        Set to `True` to normalize the cross spectra 

    Returns
    -------    
    CSD_ij : (N, N, M) :class:`numpy.ndarray`
        Cross spectral densities or coherencies if `norm=True`. 
        `M = K // 2 + 1` is the number of Fourier frequency bins, 
        `N` corresponds to number of input channels.

    freqs : (M,) :class:`numpy.ndarray`
        The Fourier frequencies

    See also
    --------
    mtmfft : :func:`~syncopy.specest.mtmfft.mtmfft`
             (Multi-)tapered Fourier analysis

    """
    nSamples, nChannels = data_arr.shape

    # has shape (nTapers x nFreq x nChannels)
    specs, freqs = mtmfft(data_arr, samplerate, taper, taperopt)

    # outer product along channel axes
    # has shape (nTapers x nFreq x nChannels x nChannels)        
    CSD_ij = specs[:, :, np.newaxis, :] * specs[:, :, :, np.newaxis].conj()
    
    # average tapers and transpose:
    # now has shape (nChannels x nChannels x nFreq)        
    CSD_ij = CSD_ij.mean(axis=0).T
    
    if norm:
        # main diagonal: the auto spectra
        # has shape (nChannels x nFreq)        
        diag = CSD_ij.diagonal()
        # get the needed product pairs of the autospectra
        Ciijj = np.sqrt(diag[:, :, None] * diag[:, None, :]).T
        CSD_ij = CSD_ij / Ciijj
        
    return CSD_ij, specs
