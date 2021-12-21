# -*- coding: utf-8 -*-
#
# Synthetic data generators
#

import numpy as np

# local imports
from syncopy.datatype import AnalogData


# noisy phase evolution <-> phase diffusion
def phase_evo(freq, eps=10, fs=1000, nSamples=1000):
    
    """
    Linear (harmonic) phase evolution + a brownian noise term
    inducing phase diffusion around the deterministic phase drift with
    slope 2pi * freq (angular frequency). 
    
    The linear phase increments are given by dPhase = 2pi * freq/fs,
    the brownian increments are scaled with `eps` according to these
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


def AR2_process(nSamples=2500, coupling=0.25, fs = 200):
    
    """
    Two unidirectionally (2->1) coupled AR(2) processes 
    With the parameters below yield a spectral peak at 40Hz with fs=200Hz
    """

    # Values from Dhamala et al. 2008 
    alpha1, alpha2 = 0.55, -0.8

    sol = np.zeros((nSamples, 2))
    # pick the 1st values at random
    xs_ini = np.random.randn(2, 2)
    sol[:2, :] = xs_ini
    for i in range(1, nSamples):
        sol[i, 1] = alpha1 * sol[i - 1, 1] + alpha2 * sol[i - 2, 1]
        sol[i, 1] += np.random.randn()
        # X2 drives X1
        sol[i, 0] = alpha1 * sol[i - 1, 0] + alpha2 * sol[i - 2, 0]
        sol[i, 0] += sol[i - 1, 1] * coupling
        sol[i, 0] += np.random.randn()

    return sol
