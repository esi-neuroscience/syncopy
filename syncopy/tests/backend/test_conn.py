# -*- coding: utf-8 -*-
#
# syncopy.nwanalysis backend method tests
#
import numpy as np
import matplotlib.pyplot as ppl

from syncopy.tests import synth_data
from syncopy.nwanalysis import csd
from syncopy.nwanalysis import ST_compRoutines as stCR
from syncopy.nwanalysis.wilson_sf import (
    wilson_sf,
    regularize_csd,
    max_rel_err
)
from syncopy.nwanalysis.granger import granger


def test_coherence():

    """
    Tests the csd normalization to
    arrive at the coherence given
    a trial averaged csd
    """

    nSamples = 1001
    fs = 1000
    tvec = np.arange(nSamples) / fs
    harm_freq = 40
    phase_shifts = np.array([0, np.pi / 2, np.pi])

    nTrials = 100

    # shape is (1, nFreq, nChannel, nChannel)
    nFreq = nSamples // 2 + 1
    nChannel = len(phase_shifts)
    avCSD = np.zeros((nFreq, nChannel, nChannel), dtype=np.complex64)

    for i in range(nTrials):

        # 1 phase phase shifted harmonics + white noise + constant, SNR = 1
        trl_dat = [np.cos(harm_freq * 2 * np. pi * tvec + ps)
                   for ps in phase_shifts]
        trl_dat = np.array(trl_dat).T
        trl_dat = np.array(trl_dat) + np.random.randn(nSamples, len(phase_shifts))

        # process every trial individually
        CSD, freqs = csd.csd(trl_dat, fs,
                             taper='hann',
                             norm=False)  # this is important!

        assert avCSD.shape == CSD.shape
        avCSD += CSD

    # this is the trial average
    avCSD /= nTrials

    # perform the normalisation on the trial averaged csd's
    Cij = csd.normalize_csd(avCSD)

    # output has shape (1, nFreq, nChannels, nChannels)
    assert Cij.shape == avCSD.shape

    # coherence between channel 0 and 1
    coh = Cij[:, 0, 1]

    fig, ax = ppl.subplots(figsize=(6, 4), num=None)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('coherence')
    ax.set_ylim((-.02, 1.05))
    ax.set_title(f'{nTrials} trials averaged coherence,  SNR=1')

    ax.plot(freqs, coh, lw=1.5, alpha=0.8, c='cornflowerblue')

    # we test for the highest peak sitting at
    # the vicinity (± 5Hz) of the harmonic
    peak_val = np.max(coh)
    peak_idx = np.argmax(coh)
    peak_freq = freqs[peak_idx]

    assert harm_freq - 5 < peak_freq < harm_freq + 5

    # we test that the peak value
    # is at least 0.9 and max 1
    assert 0.9 < peak_val < 1

    # trial averaging should suppress the noise
    # we test that away from the harmonic the coherence is low
    level = 0.4
    assert np.all(coh[:peak_idx - 2] < level)
    assert np.all(coh[peak_idx + 2:] < level)


def test_csd():

    """
    Tests multi-tapered single trial cross spectral
    densities
    """

    nSamples = 1001
    fs = 1000
    tvec = np.arange(nSamples) / fs
    harm_freq = 40
    phase_shifts = np.array([0, np.pi / 2, np.pi])

    # 1 phase phase shifted harmonics + white noise, SNR = 1
    data = [np.cos(harm_freq * 2 * np. pi * tvec + ps)
            for ps in phase_shifts]
    data = np.array(data).T
    data = np.array(data) + np.random.randn(nSamples, len(phase_shifts))

    bw = 8  # Hz
    NW = nSamples * bw / (2 * fs)
    Kmax = int(2 * NW - 1)   # multiple tapers for single trial coherence
    CSD, freqs = csd.csd(data, fs,
                         taper='dpss',
                         taper_opt={'Kmax': Kmax, 'NW': NW},
                         norm=True)

    # output has shape (nFreq, nChannels, nChannels)
    assert CSD.shape == (len(freqs), data.shape[1], data.shape[1])

    # single trial coherence between channel 0 and 1
    coh = np.abs(CSD[:, 0, 1])

    fig, ax = ppl.subplots(figsize=(6, 4), num=None)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('coherence')
    ax.set_ylim((-.02, 1.05))
    ax.set_title(f'MTM coherence, {Kmax} tapers, SNR=1')

    ax.plot(freqs, coh, lw=1.5, alpha=0.8, c='cornflowerblue')

    # we test for the highest peak sitting at
    # the vicinity (± 5Hz) of one the harmonic
    peak_val = np.max(coh)
    peak_idx = np.argmax(coh)
    peak_freq = freqs[peak_idx]
    assert harm_freq - 5 < peak_freq < harm_freq + 5

    # we test that the peak value
    # is at least 0.9 and max 1
    assert 0.9 < peak_val < 1


def test_cross_cov():

    nSamples = 1001
    fs = 1000
    tvec = np.arange(nSamples) / fs

    cosine = np.cos(2 * np.pi * 30 * tvec)
    sine = np.sin(2 * np.pi * 30 * tvec)
    data = np.c_[cosine, sine]

    # output shape is (nLags x 1 x nChannels x nChannels)
    CC, lags = stCR.cross_covariance_cF(data, samplerate=fs, norm=True, fullOutput=True)

    # test for result is returned in the [0, np.ceil(nSamples / 2)] lag interval
    nLags = int(np.ceil(nSamples / 2))

    # output has shape (nLags, 1, nChannels, nChannels)
    assert CC.shape == (nLags, 1, data.shape[1], data.shape[1])

    # cross-correlation (normalized cross-covariance) between
    # cosine and sine analytically equals minus sine
    assert np.all(CC[:, 0, 0, 1] + sine[:nLags] < 1e-5)


def test_wilson():
    """
    Test Wilson's spectral matrix factorization.

    As the routine has relative error-checking
    inbuild, we just need to check for convergence.
    """

    # -- test error testing routine

    A = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)

    assert max_rel_err(A, A + A * 1e-16) < 1e-15

    # --- create test data ---
    fs = 200
    nChannels = 2
    nSamples = 1000
    nTrials = 150
    CSDav = np.zeros((nSamples // 2 + 1, nChannels, nChannels), dtype=np.complex64)
    for _ in range(nTrials):

        sol = synth_data.AR2_network(nSamples=nSamples, seed=None)
        # --- get the (single trial) CSD ---

        CSD, freqs = csd.csd(sol, fs,
                             norm=False)

        CSDav += CSD

    CSDav /= nTrials

    # --- factorize CSD with Wilson's algorithm ---

    H, Sigma, conv, err = wilson_sf(CSDav, rtol=1e-6)
    # converged - \Psi \Psi^* \approx CSD,
    # with relative error <= rtol?
    assert conv

    # reconstitute
    CSDfac = H @ Sigma @ H.conj().transpose(0, 2, 1)
    err = max_rel_err(CSDav, CSDfac)
    assert err < 1e-6

    fig, ax = ppl.subplots(figsize=(6, 4))
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel(r'$|CSD_{ij}(f)|$')
    chan = nChannels // 2
    # show (real) auto-spectra
    ax.plot(freqs, np.abs(CSDav[:, chan, chan]),
            '-o', label='original CSD', ms=3)
    ax.plot(freqs, np.abs(CSDfac[:, chan, chan]),
            '-o', label='factorized CSD', ms=3)
    # ax.set_xlim((350, 450))
    ax.legend()


def test_regularization():

    """
    The dyadic product of single random matrices has rank 1
    and condition number --> np.inf.
    By averaging "many trials" we interestingly "fill the space" and
    one can achieve full rank for white noise.
    However here purposefully we ill-condition to test the regularization,
    """
    nChannels = 20
    nTrials = 10
    CSD = np.zeros((nChannels, nChannels))
    for _ in range(nTrials):
        A = np.random.randn(nChannels)
        CSD += np.outer(A, A)

    # --- regularize CSD ---

    cmax = 1e4
    eps_max = 1e-1
    CSDreg, fac, iniCN = regularize_csd(CSD, cond_max=cmax, eps_max=eps_max)
    # check initial CSD condition number, which is way too large!
    assert iniCN > cmax, f"intial condition number is {iniCN}"

    CNreg = np.linalg.cond(CSDreg).max()
    assert CNreg < cmax, f"regularized condition number is {CNreg}"
    # check that 'small' regularization factor is enough
    assert fac < eps_max


def test_granger():

    """
    Test the granger causality measure
    with uni-directionally coupled AR(2)
    processes akin to the source publication:

    Dhamala, Mukeshwar, Govindan Rangarajan, and Mingzhou Ding.
       "Estimating Granger causality from Fourier and wavelet transforms
        of time series data." Physical review letters 100.1 (2008): 018701.
    """

    fs = 200  # Hz
    nSamples = 1500
    nTrials = 100

    CSDav = np.zeros((nSamples // 2 + 1, 2, 2), dtype=np.complex64)
    for _ in range(nTrials):

        # -- simulate 2 AR(2) processes with 2->1 coupling --
        sol = synth_data.AR2_network(nSamples=nSamples)

        # --- get CSD ---
        bw = 2
        NW = bw * nSamples / (2 * fs)
        Kmax = int(2 * NW - 1)  # optimal number of tapers
        CSD, freqs = csd.csd(sol, fs,
                             taper='dpss',
                             taper_opt={'Kmax': Kmax, 'NW': NW},
                             demean_taper=True)

        CSDav += CSD

    CSDav /= nTrials
    # with only 2 channels this CSD is well conditioned
    assert np.linalg.cond(CSDav).max() < 1e2
    H, Sigma, conv, err = wilson_sf(CSDav, direct_inversion=True)

    G = granger(CSDav, H, Sigma)
    assert G.shape == CSDav.shape

    fig, ax = ppl.subplots(figsize=(6, 4))
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel(r'Granger causality(f)')
    ax.plot(freqs, G[:, 0, 1], label=r'Granger $1\rightarrow2$')
    ax.plot(freqs, G[:, 1, 0], label=r'Granger $2\rightarrow1$')
    ax.legend()

    # check for directional causality at 40Hz
    freq_idx = np.argmin(freqs < 40)
    assert 39 < freqs[freq_idx] < 41

    # check low to no causality for 1->2
    assert G[freq_idx, 0, 1] < 0.1
    # check high causality for 2->1
    assert G[freq_idx, 1, 0] > 0.7

    # repeat test with least-square solution
    H, Sigma, conv, err = wilson_sf(CSDav, direct_inversion=False)
    G2 = granger(CSDav, H, Sigma)

    # check low to no causality for 1->2
    assert G2[freq_idx, 0, 1] < 0.1
    # check high causality for 2->1
    assert G2[freq_idx, 1, 0] > 0.7

    ax.plot(freqs, G2[:, 0, 1], label=r'Granger (LS) $1\rightarrow2$')
    ax.plot(freqs, G2[:, 1, 0], label=r'Granger (LS) $2\rightarrow1$')
    ax.legend()
