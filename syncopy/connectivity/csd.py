# -*- coding: utf-8 -*-
#
# Cross Spectral Densities and Coherency
#

# Builtin/3rd party package imports
import numpy as np
from scipy import signal
import itertools

# syncopy imports
from syncopy.specest.mtmfft import mtmfft


def csd(data_arr, samplerate, taper="hann", taperopt={}, norm=False, naive=True):

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

    See also
    --------
    mtmfft : :func:`~syncopy.specest.mtmfft.mtmfft`
             (Multi-)tapered Fourier analysis

    """

    nSamples, nChannels = data_arr.shape

    # has shape (nTapers x nFreq x nChannels)
    specs, freqs = mtmfft(data_arr, samplerate, taper, taperopt)

    # has shape (nChannels x nChannels x nFreq)
    output = np.zeros((nChannels, nChannels, freqs.size))
    
    # somewhat vectorized - not really fast :/                
    if norm:
        # main diagonal: auto spectrum for each taper and averaging
        diag = np.multiply(specs, specs.conj()).mean(axis=0)
        # output[range(nChannels), range(nChannels), :] = np.real(diag.T)
        diag = np.real(diag).T

        for i in range(nChannels):
            idx = slice(i, nChannels)
            row = np.multiply(specs[..., np.tile(i, nChannels - i)],
                              specs.conj()[..., idx])

            # normalization
            denom = np.multiply(np.tile(diag[i], ((nChannels - i), 1)), diag[i:])
            row = row.mean(axis=0).T / np.sqrt(denom)
            output[i, i:, ...] = np.real(row)
        
    elif naive:
        for i in range(nChannels):
            for j in range(i, nChannels):
                output[i, j] = np.real(specs[:, :, i] * specs[:, :, j].conj()).mean(axis=0)
                output[j, i] = output[i, j]
                
    else:
        # build single taper array
        chan_idx = np.arange(nChannels)
        idx1, idx2 = np.meshgrid(chan_idx, chan_idx)        
        output = np.real(specs[:, :, idx1] * specs[:, :, idx2].conj()).mean(axis=0).transpose(2,1,0)
    return output


# dummy input
a = np.ones((10, 3)) * np.arange(1,4)
abig = np.ones((100, 50)) * np.arange(1,51)
# dummy mtmfft result
# b = np.arange(1, 4) * np.ones((2,10,3)).astype('complex')
# dummt csd matrix
c = np.ones((5, 5, 10))

def vectorized(arr):
    nSamples, nChannels = arr.shape
    r = np.multiply.outer(arr,arr.T).reshape(nSamples, nChannels**2 ,nSamples).diagonal(axis1=0, axis2=2)

    return r.reshape((nChannels, nChannels, nSamples))

def arr_loop(arr):
    nChannels = arr.shape[1]
    output = np.zeros((nChannels, nChannels, arr.shape[0]))
    
    for i in range(nChannels):
        for j in range(i, nChannels):
            output[i, j] = np.real(arr[:, i] * arr[:, j])
            output[j, i] = output[i, j]

    return output
    
