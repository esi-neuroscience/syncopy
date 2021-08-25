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


def csd(data_arr, samplerate, taper="hann", taperopt={}, norm=False):

    """
    Cross spectral density (CSD) estimate between all channels
    of the input data. First all the individual Fourier transforms
    are calculated via a (multi-)tapered FFT, then the pairwise
    coherence is calculated. Averaging over tapers is done implicitly.
    Output consists of all (nChannels x nChannels-1)/2 different CSD estimates
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

    for i in range(nChannels):
        for j in range(i, nChannels):
            output[i, j, :] = np.real(specs[0, :, i] * specs[0, :, j].conj())
            output[j, i, :] = output[i, j, :]

    # there is probably a more efficient way
    if norm:
        for i in range(nChannels):
            for j in range(i, nChannels):
                output[i, j, :] = output[i, j, :] / np.sqrt(
                    output[i, i, :] * output[j, j, :]
                )

    return output, freqs
