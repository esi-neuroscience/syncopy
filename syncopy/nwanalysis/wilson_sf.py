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


def wilson_sf(CSD, nIter=100, rtol=1e-6, direct_inversion=True):
    """
    Wilsons spectral matrix factorization ("analytic method")

    Converges extremely fast, so the default number of
    iterations should be more than enough in practical situations.

    This is a pure backend function and hence no input argument
    checking is performed.

    Parameters
    ----------
    CSD : (nFreq, N, N) :class:`numpy.ndarray`
        Complex cross spectra for all channel combinations ``i,j``.
        `N` corresponds to number of input channels. Has to be
        positive definite and well conditioned.
    nIter : int
        Maximum number of iterations, factorization result
        is returned also if error tolerance wasn't met.
    rtol : float
        Tolerance of the relative maximal
        error of the factorization.
    direct_inversion : bool
        With `True` a direct matrix inversion is
        performed, `False` solves the associated
        least-square problems.

    Returns
    -------
    Hfunc : (nFreq, N, N) :class:`numpy.ndarray`
        The transfer function
    Sigma : (N, N) :class:`numpy.ndarray`
        Noise covariance
    converged : bool
        Indicates wether the algorithm converged.
        If `False` result was returned after `nIter`
        iterations.
    err : float
        Maximal final relative error
        between input CSD and factorized CSD
    """

    nFreq = CSD.shape[0]

    Ident = np.eye(*CSD.shape[1:])

    # attach negative frequencies
    CSD = np.r_[CSD, CSD[nFreq - 2:0:-1].conj()]

    # nChannel x nChannel
    psi0 = _psi0_initial(CSD)

    # initial choice of psi, constant for all z(~f)
    psi = np.tile(psi0, (nFreq, 1, 1))
    # attach negative frequencies
    psi = np.r_[psi, psi[nFreq - 2:0:-1].conj()]

    g = np.zeros(CSD.shape, dtype=np.complex64)
    converged = False
    # use cholesky for performance
    U = np.linalg.cholesky(CSD)
    for _ in range(nIter):

        if direct_inversion:
            psi_inv = np.linalg.inv(psi)

            # the bracket of equation 3.1
            # g = psi_inv @ CSD @ psi_inv.conj().transpose(0, 2, 1)

            # equivalent using cholesky decomposition
            g = psi_inv @ U
            g = (g @ g.conj().transpose(0, 2, 1))

        else:
            for i in range(g.shape[0]):
                C = np.linalg.lstsq(psi[i], CSD[i], rcond=None)[0]
                g[i] = np.linalg.lstsq(
                    psi[i], C.conj().T, rcond=None)[0].conj().T

        gplus, gplus_0 = _plusOperator(g + Ident)

        # the 'any' matrix
        S = np.triu(gplus_0)
        S = S - S.conj().T   # S + S* = 0

        # the next step psi_{tau+1}
        psi = psi @ (gplus + S)
        psi0 = psi0 @ (gplus_0 + S)

        # max relative error
        CSDfac = psi @ psi.conj().transpose(0, 2, 1)
        err = max_rel_err(CSD, CSDfac)

        # converged
        if err < rtol:
            converged = True
            break

    # Noise Covariance
    Sigma = psi0 @ psi0.T

    # Transfer function
    psi0_inv = np.linalg.inv(psi0)
    Hfunc = psi @ psi0_inv

    return Hfunc[:nFreq], Sigma, converged, err


def _psi0_initial(CSD):

    """
    Initialize Wilson's algorithm with the Cholesky
    decomposition of the 1st Fourier series component
    of the cross spectral density matrix (CSD). This is
    explicitly proposed in section 4. of the original paper.
    """

    nSamples = CSD.shape[1]

    # perform (i)fft to obtain gammas.
    gamma = np.fft.fft(CSD, axis=0)
    gamma0 = gamma[0, ...]

    # Remove any asymmetry due to rounding error.
    # This also will zero out any imaginary values
    # on the diagonal - real diagonals are required for cholesky.
    gamma0 = np.real((gamma0 + gamma0.T.conj()) / 2)

    # check for positive definiteness
    eivals = np.linalg.eigvals(gamma0)
    if np.all(np.imag(eivals) == 0):
        psi0 = np.linalg.cholesky(gamma0)
    # otherwise initialize with 1's as a fallback
    else:
        psi0 = np.ones((nSamples, nSamples))

    return psi0.T


def _plusOperator(g):

    """
    The []+ operator from definition 1.2,
    given by explicit Fourier transformations

    The nFreq x nChannel x nChannel matrix `g` is given
    in the frequency domain.
    """

    # 'negative lags' from the ifft
    nLag = g.shape[0] // 2

    # the series expansion in beta_k
    # is covariance like
    beta = np.real(np.fft.ifft(g, axis=0))

    # take half of the zero lag
    beta[0, ...] = 0.5 * beta[0, ...]
    g0 = beta[0, ...].copy()

    # take half of Nyquist bin
    # Dhamala "NewEdits" 28.01.22
    beta[nLag, ...] = 0.5 * beta[nLag, ...]

    # Zero out negative lags
    beta[nLag + 1:, ...] = 0

    gp = np.fft.fft(beta, axis=0)

    return gp, g0


# --- End of Wilson's Algorithm ---


def max_rel_err(A, B):

    err = np.abs(A - B)
    err = (err / np.abs(A)).max()
    return err


def regularize_csd(CSD, cond_max=1e3, eps_max=1e-3, nSteps=15):

    """
    Brute force regularization of CSD matrix
    by inspecting the maximal condition number
    along the frequency axis.
    Multiply with ``epsilon * I``, starting with ``epsilon = 1e-10``
    up to ``epsilon = eps_max`` on a log-scale of size ``nSteps``
    until the condition number is smaller than `cond_max`.
    If that can not be achieved, return the last regularization
    result and `-1` as factor for downstream (error/warning) handling.


    Parameters
    ----------
    CSD : 3D :class:`numpy.ndarray`
        The cross spectral density matrix
        with shape ``(nFreq, nChannel, nChannel)``
    cond_max : float
        The maximal condition number after regularization
    eps_max : float
        The largest regularization factor to be used. If
        also this value does not regularize the CSD up
        to `cond_max` a `ValueError` is raised.
    nSteps : int
        Number of steps between 1e-10 and `eps_max`.

    Returns
    -------
    CSDreg : 3D :class:`numpy.ndarray`
        The regularized CSD matrix with a maximal
        condition number of `cond_max`
    eps : float
        The regularization factor used
    iniCondNum : float
        The initial condition number of the CSD

    """

    epsilons = np.logspace(-10, np.log10(eps_max), nSteps)
    I = np.eye(CSD.shape[1])

    CondNum = np.linalg.cond(CSD).max()
    iniCondNum = CondNum
    
    # nothing to be done
    if CondNum < cond_max:
        return CSD, 0, iniCondNum

    for eps in epsilons:
        CSDreg = CSD + eps * I
        CondNum = np.linalg.cond(CSDreg).max()

        if CondNum < cond_max:
            return CSDreg, eps, iniCondNum

    # regularization goal not achieved
    return CSDreg, -1, iniCondNum
