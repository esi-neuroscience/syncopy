# -*- coding: utf-8 -*-
#
# Synthetic data generators for testing and tutorials
#

import numpy as np


# noisy phase evolution <-> phase diffusion
def phase_evo(freq, eps=10, fs=1000, nSamples=1000):
    
    """
    Linear (harmonic) phase evolution + a brownian noise term
    inducing phase diffusion around the deterministic phase drift with
    slope 2pi * freq (angular frequency). 
    
    The linear phase increments are given by dPhase = 2pi * freq/fs,
    the brownian increments are scaled with `eps` relative to these
    phase increments.
 
    Parameters
    ----------
    freq : float
        Harmonic frequency in Hz
    eps : float
        Scaled brownian increments
    fs : float
        Sampling rate in Hz
    nSamples : int
    """

    # white noise
    wn = np.random.randn(nSamples)
    delta_ts = np.ones(nSamples) * 1 / fs
    omega0 = 2 * np.pi * freq
    rel_eps = omega0 / fs * eps
    phase = np.cumsum(omega0 * delta_ts + rel_eps * wn)
    return phase


def AR2_process(AdjMat=None, nSamples=2500, coupling=0.25, alphas=[0.55, -0.8]):
    
    """
    Coupled AR(2) processes, with the default parameters 
    (as in Dhamala 2008) yield a spectral peak at 40Hz 
    when a sampling frequency of 200Hz is assumed (set).

    Parameters
    ----------
    AdjMat : np.ndarray or None
        nChannel x nChannel adjacency matrix
        determining the coupling topology. If left
        at `None`, the default 2 Channel system
        with unidirectional 2->1 coupling is generated.
    nSamples : int, optional
        Number of samples in time
    coupling : float, optional
        Uniform coupling strengths
    alphas : 2-element sequence, optional
       The AR(2) parameters for lag1 and lag2
    """

    # default systen layout as in Dhamala 2008:
    # unidirectional (2->1) coupling
    if AdjMat is None:
        AdjMat = np.identity(2)
        AdjMat[0, 1] = 1 #

    nChannels = AdjMat.shape[0]
    alpha1, alpha2 = alphas
    # off diagonal coupling,
    # diagonal 'self-interaction' with lag 1
    AdjMat = coupling * AdjMat
    np.fill_diagonal(AdjMat, alpha1)
    
    sol = np.zeros((nSamples, nChannels))
    # pick the 1st values at random
    xs_ini = np.random.randn(2, nChannels)
    sol[:2, :] = xs_ini
    for i in range(2, nSamples):
        
        sol[i, :] = AdjMat @ sol[i - 1, :] + alpha2 * sol[i - 2, :]
        sol[i, :] += np.random.randn(nChannels)
        # X2 drives X1

    return sol
