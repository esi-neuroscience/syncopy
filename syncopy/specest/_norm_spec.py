# -*- coding: utf-8 -*-
#
# Helper routines to normalize Fourier spectra
#

import numpy as np


def _norm_spec(ftr, nSamples, fs, mode='bins'):

    """
    Normalizes the complex Fourier transform to
    power spectral density or 1Hz-bin units.
    """

    # frequency bins
    if mode == 'density':
        delta_f = fs / nSamples
    elif mode == 'bins':
        delta_f = 1

    ftr *= np.sqrt(2) / (nSamples * np.sqrt(delta_f))

    return ftr


def _norm_taper(taper, windows, nSamples):

    """
    Helper function to normalize tapers such
    that the resulting spectra are normalized
    with respect to the sum of the window. Meaning
    that the total original (untapered) power gets
    distributed over the spectral window response.
    """

    if taper == 'dpss':
        windows *= np.sqrt(nSamples)
    elif taper == 'boxcar':
        windows *= np.sqrt(nSamples / windows.sum())
    # weird 3 point normalization,
    # checks out exactly for 'hann' though
    else:
        windows *= np.sqrt(4 / 3) * np.sqrt(nSamples / windows.sum())

    return windows
