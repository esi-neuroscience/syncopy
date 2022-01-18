# -*- coding: utf-8 -*-
#
# Implementation of Granger-Geweke causality
#
#
# Builtin/3rd party package imports
import numpy as np


def granger(CSD, Hfunc, Sigma):
    """
    Computes the pairwise Granger-Geweke causalities
    for all (non-symmetric!) channel combinations
    according to Equation 8 in [1]_.

    The transfer functions `Hfunc` and noise covariance
    `Sigma` are expected to have been already computed.

    Parameters
    ----------
    CSD : (nFreq, N, N) :class:`numpy.ndarray`
        Complex cross spectra for all channel combinations ``i,j``
        `N` corresponds to number of input channels.
    Hfunc : (nFreq, N, N) :class:`numpy.ndarray`
        Spectral transfer functions for all channel combinations ``i,j``
    Sigma :  (N, N) :class:`numpy.ndarray`
        The noise covariances

    Returns
    -------
    Granger : (nFreq, N, N) :class:`numpy.ndarray`
        Spectral Granger-Geweke causality between all channel
        combinations. Directionality follows array
        notation: causality from ``i -> j`` is ``Granger[:,i,j]``,
        causality from ``j -> i`` is ``Granger[:,j,i]``

    See also
    --------
    wilson_sf : :func:`~syncopy.connectivity.wilson_sf.wilson_sf
             Spectral matrix factorization that yields the
             transfer functions and noise covariances
             from a cross spectral density.

    Notes
    -----
    .. [1] Dhamala, Mukeshwar, Govindan Rangarajan, and Mingzhou Ding.
       "Estimating Granger causality from Fourier and wavelet transforms
        of time series data." Physical review letters 100.1 (2008): 018701.

    """

    nChannels = CSD.shape[1]
    auto_spectra = CSD.transpose(1, 2, 0).diagonal()
    auto_spectra = np.abs(auto_spectra) # auto-spectra are real

    # we need the stacked auto-spectra of the form (nChannel=3):
    #           S_11 S_22 S_33
    # Smat(f) = S_11 S_22 S_33
    #           S_11 S_22 S_33
    Smat = auto_spectra[:, None, :] * np.ones(nChannels)[:, None]

    # Granger i->j needs H_ji entry
    Hmat = np.abs(Hfunc.transpose(0, 2, 1))**2
    # Granger i->j needs Sigma_ji entry
    SigmaJI = np.abs(Sigma.T)

    # imag part should be 0
    auto_cov = np.abs(Sigma.diagonal())
    # same stacking as for the auto spectra (without freq axis)
    SigmaII = auto_cov[None, :] * np.ones(nChannels)[:, None]

    # the denominator
    denom = SigmaII.T - SigmaJI**2 / SigmaII
    denom = Smat - denom * Hmat

    # linear causality i -> j
    Granger = np.log(Smat / denom)

    return Granger
