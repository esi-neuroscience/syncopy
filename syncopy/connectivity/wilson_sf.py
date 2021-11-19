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


def regularize_csd(CSD, cond_max=1e6, reg_max=-3):

    '''
    Brute force regularize CSD matrix
    by inspecting the maximal condition number 
    along the frequency axis.
    Multiply with different epsilon * I, 
    starting with eps = 1e-12 until the
    condition number is smaller than `cond_max`
    or the maximal regularization factor was reached.

    Inspection/Check of the used regularization constant 
    epsilon is highly recommended!
    '''

    reg_factors = np.logspace(-12, reg_max, 100)
    I = np.eye(CSD.shape[1])

    CondNum = np.linalg.cond(CSD).max()

    # nothing to be done
    if CondNum < cond_max:
        return CSD
    
    for factor in reg_factors:        
        CSDreg = CSD + factor * I
        CondNum = np.linalg.cond(CSDreg).max()
        print(f'Factor: {factor}, CN: {CondNum}')
        
        if CondNum < cond_max:
            return CSDreg, factor
        
    # raise sth..

    
def wilson_sf(CSD, samplerate, nIter=100, rtol=1e-9):

    '''
    Wilsons spectral matrix factorization ("analytic method")

    Converges extremely fast, so the default number of
    iterations should be enough in practical situations.

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

    nFreq, nChannels = CSD.shape[:2]

    Ident = np.eye(*CSD.shape[1:])
    
    # nChannel x nChannel
    psi0 = _psi0_initial(CSD)
    
    # initial choice of psi, constant for all z(~f)
    psi = np.tile(psi0, (nFreq, 1, 1))    
    assert psi.shape == CSD.shape

    errs = []
    for _ in range(nIter):
        
        psi_inv = np.linalg.inv(psi)        
        # the bracket of equation 3.1
        g = psi_inv @ CSD @ psi_inv.conj().transpose(0, 2, 1)
        gplus, gplus_0 = _plusOperator(g + Ident)
        
        # the 'any' matrix
        S = np.triu(gplus_0)
        S = S - S.conj().T # S + S* = 0

        psi_old = psi
        # the next step psi_{tau+1}
        psi = psi @ (gplus + S)

        rel_err = np.abs((psi - psi_old) / np.abs(psi))
        # print(rel_err.max())
        # mean relative error
        CSDfac = psi @ psi.conj().transpose(0, 2, 1)
        err = np.abs(CSD - CSDfac)
        err = err / np.abs(CSD) # relative error
        
        print('Cond', np.linalg.cond(psi[0]))        
        print('Error:', err.max(),'\n')
        
        errs.append(err.max())
        
    Aks = np.fft.ifft(psi, axis=0)
    A0 = Aks[0, ...]
    
    # Noise Covariance
    Sigma = A0 * A0.T
    # strip off remaining imaginary parts
    Sigma = np.real(Sigma)

    # Transfer function
    A0inv = np.linalg.inv(A0)
    Hfunc = psi @ A0inv.conj().T

    # print(err.mean())
    
    return Hfunc, Sigma, CSDfac, errs


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
        
    return psi0.T
    
# from scipy.signal import windows
def _plusOperator(g):

    '''
    The []+ operator from definition 1.2,
    given by explicit Fourier transformations

    The nFreq x nChannel x nChannel matrix `g` is given 
    in the frequency domain.
    '''

    # 'negative lags' from the ifft
    nLag = g.shape[0] // 2
    # the series expansion in beta_k 
    beta = np.fft.ifft(g, axis=0)

    # take half of the zero lag
    beta[0, ...] = 0.5 * beta[0, ...]
    g0 = beta[0, ...].copy()

    # Zero out negative lags
    beta[nLag + 1:, ...] = 0

    # beta = beta * windows.tukey(len(beta), alpha=0.2)[:, None, None]

    gp = np.fft.fft(beta, axis=0)

    return gp, g0


def _mem_size(arr):
    '''
    Gives array size in MB
    '''
    return f'{arr.size * arr.itemsize / 1e6:.2f} MB'
