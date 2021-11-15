# -*- coding: utf-8 -*-
# 
# Performs the numerical inner-outer factorization of a spectral matrix, using
# Wilsons method. This implementation here is a Python version of the original
# Matlab implementation by M. Dhamala (mdhamala@bme.ufl.edu) & G. Rangarajan
# (rangaraj@math.iisc.ernet.in), UF, Aug 3-4, 2006.
#
# The algorithm itself was first presented in:
# The Factorization of Matricial Spectral Densities, SIAM J. Appl. Math,
# Vol. 23, No. 4, pgs 420-426 December 1972 by G T Wilson).

# Builtin/3rd party package imports
import numpy as np


def wilson_sf(CSD, samplerate, nIter=500, tol=1e-9):

    '''
    Wilsons spectral matrix factorization ("analytic method")

    This is a pure backend function and hence no input argument
    checking is performed.

    Parameters
    ----------
    CSD : (nFreq, N, N) :class:`numpy.ndarray`
        Complex cross spectra for all channel combinations i,j.
        `N` corresponds to number of input channels.

    Returns
    -------

    '''

    psi0 = _psi0_initial(CSD)

    g = np.zeros(CSD.shape)

    g = 0 # :D
    
def _psi0_initial(CSD):

    '''
    Initialize Wilson's algorithm with the Cholesky
    decomposition of the 1st Fourier series component
    of the cross spectral density matrix (CSD). This is
    explicitly proposed in section 4. of the original paper.
    '''

    nSamples = CSD.shape[1]
    
    # perform ifft to obtain gammas.
    gamma = np.fft.ifft(CSD, axis=0)
    gamma0 = gamma[0, ...]
	
    # Remove any assymetry due to rounding error.
    # This also will zero out any imaginary values
    # on the diagonal - real diagonals are required for cholesky.
    gamma0 = np.real((gamma0 + gamma0.conj()) / 2)

    # check for positive definiteness
    eivals = np.linalg.eigvals(gamma0)
    if np.all(np.imag(eivals) == 0):    
        psi0 = np.linalg.cholesky(gamma0)
    # otherwise initialize with 1's as a fallback
    else:
        psi0 = np.ones((nSamples, nSamples))
        
    return psi0
    

def _plusOperator(g):

    '''
    The []+ operator from definition 1.2,
    given by explicit Fourier transformations

    The time x nChannel x nChannel matrix `g` is given 
    in the frequency domain.
    '''

    # 'negative lags' from the ifft
    nLag = g.shape[0] // 2
    # the series expansion in beta_k
    beta = np.fft.ifft(g, axis=0)

    # take half of the zero lag
    beta[0, ...] = 0.5 * beta[0, ...]
    g0 = beta[0, ...]

    # Zero out negative powers.
    beta[:nLag + 1, ..., ...] = 0

    gp = np.fft.fft(beta, axis=0)
    return gp, g0
