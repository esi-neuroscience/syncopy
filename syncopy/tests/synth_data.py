# -*- coding: utf-8 -*-
#
# Synthetic data generators for testing and tutorials
#

import numpy as np


# noisy phase evolution <-> phase diffusion
def phase_evo(freq, eps=1, fs=1000, nSamples=1000):
    
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


def AR2_network(AdjMat=None, nSamples=2500, alphas=[0.55, -0.8]):
    
    """
    Simulation of a network of coupled AR(2) processes 

    With the default parameters the individual processes 
    (as in Dhamala 2008) have a spectral peak at 40Hz 
    with a sampling frequency of 200Hz.

    Parameters
    ----------
    AdjMat : np.ndarray or None
        nChannel x nChannel adjacency matrix where
        entry (i,j) is the coupling strength 
        from channel i -> j.
        If left at `None`, the default 2 Channel system
        with unidirectional 2->1 coupling is generated.
    nSamples : int, optional
        Number of samples in time
    alphas : 2-element sequence, optional
       The AR(2) parameters for lag1 and lag2

    Returns
    -------
    sol : np.ndarray
        The nSamples x nChannel
        solution of the network dynamics
    """

    # default systen layout as in Dhamala 2008:
    # unidirectional (2->1) coupling
    if AdjMat is None:
        AdjMat = np.zeros((2, 2))
        AdjMat[1, 0] = .25 

    nChannels = AdjMat.shape[0]
    alpha1, alpha2 = alphas
    # diagonal 'self-interaction' with lag 1
    DiagMat = np.diag(nChannels * [alpha1])
    
    sol = np.zeros((nSamples, nChannels))
    # pick the 1st values at random
    sol[:2, :] = np.random.randn(2, nChannels)

    for i in range(2, nSamples):
        
        sol[i, :] = (DiagMat + AdjMat.T) @ sol[i - 1, :] + alpha2 * sol[i - 2, :]
        sol[i, :] += np.random.randn(nChannels)
        # X2 drives X1

    return sol


def mk_RandomAdjMat(nChannels=3, conn_thresh=0.25, max_coupling=0.25):
    """
    Create a random adjacency matrix
    for the network of AR(2) processes
    where entry (i,j) is the coupling
    strength from channel i -> j

    Parameters
    ---------
    conn_thresh : float
        Connectivity threshold for the bernoulli
        sampling of the network connections. Setting
        `conn_thresh=1` yields a fully connected network
        (not recommended).
    max_coupling : float < 0.5, optional
        Total input into single channel 
        normalized by number of couplings
        (for stability).    

    """
        
    # random numbers in [0,1)
    AdjMat = np.random.random_sample((nChannels, nChannels))
    # all smaller than threshold elements get set to 1 (coupled)
    AdjMat = (AdjMat < conn_thresh).astype(int)
    # set diagonal to 0 to easier identify coupling
    np.fill_diagonal(AdjMat, 0)
    
    # normalize such that total input
    # does not exceed max. coupling
    norm = AdjMat.sum(axis=0)
    norm[norm == 0] = 1
    AdjMat = AdjMat / norm[None, :] * max_coupling
    
    return AdjMat
