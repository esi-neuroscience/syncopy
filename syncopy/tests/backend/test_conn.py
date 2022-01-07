# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as ppl

from syncopy.tests import synth_data
from syncopy.connectivity import csd
from syncopy.connectivity import ST_compRoutines as stCR
from syncopy.connectivity.wilson_sf import wilson_sf, regularize_csd
from syncopy.connectivity.granger import granger


def test_coherence():

    """
    Tests the normalization cF to
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
                             norm=False, # this is important!
                             fullOutput=True)

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
    ax.set_ylim((-.02,1.05))
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

    bw = 8 #Hz
    NW = nSamples * bw / (2 * fs)
    Kmax = int(2 * NW - 1) # multiple tapers for single trial coherence
    CSD, freqs = csd.csd(data, fs,
                         taper='dpss',
                         taper_opt={'Kmax' : Kmax, 'NW' : NW},
                         norm=True,
                         fullOutput=True)

    # output has shape (1, nFreq, nChannels, nChannels)
    assert CSD.shape == (len(freqs), data.shape[1], data.shape[1])

    # single trial coherence between channel 0 and 1
    coh = np.abs(CSD[:, 0, 1])

    fig, ax = ppl.subplots(figsize=(6,4), num=None)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('coherence')
    ax.set_ylim((-.02,1.05))
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

    # --- create test data ---
    fs = 1000
    nChannels = 10
    nSamples = 1000
    f1, f2 = [30 , 40] # 30Hz and 60Hz
    data = np.zeros((nSamples, nChannels))

    # more phase diffusion in the 60Hz band
    p1 = synth_data.phase_evo(f1, eps=.3, fs=fs,
                              nSamples=nSamples, nChannels=nChannels)
    p2 = synth_data.phase_evo(f2, eps=1, fs=fs,
                              nSamples=nSamples, nChannels=nChannels)

    data = np.cos(p1) + 2 * np.sin(p2) + .5 * np.random.randn(nSamples, nChannels)

    # --- get the (single trial) CSD ---

    bw = 5 # 5Hz smoothing
    NW = bw * nSamples / (2 * fs)
    Kmax = int(2 * NW - 1) # optimal number of tapers

    CSD, freqs = csd.csd(data, fs,
                         taper='dpss',
                         taper_opt={'Kmax' : Kmax, 'NW' : NW},
                         norm=False,
                         fullOutput=True)
    
    # get CSD condition number, which is way too large!
    CN = np.linalg.cond(CSD).max()
    assert CN > 1e6

    # --- regularize CSD ---

    CSDreg, fac = regularize_csd(CSD, cond_max=1e6, nSteps=25)
    CNreg = np.linalg.cond(CSDreg).max()
    assert CNreg < 1e6
    # check that 'small' regularization factor is enough
    assert fac < 1e-5

    # --- factorize CSD with Wilson's algorithm ---

    H, Sigma, conv = wilson_sf(CSDreg, rtol=1e-9)

    # converged - \Psi \Psi^* \approx CSD,
    # with relative error <= rtol?
    assert conv

    # reconstitute
    CSDfac = H @ Sigma @ H.conj().transpose(0, 2, 1)

    fig, ax = ppl.subplots(figsize=(6, 4))
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel(r'$|CSD_{ij}(f)|$')
    chan = nChannels // 2
    # show (real) auto-spectra
    ax.plot(freqs, np.abs(CSD[:, chan, chan]),
            '-o', label='original CSD', ms=3)
    ax.plot(freqs, np.abs(CSDreg[:, chan, chan]),
            '--', label='regularized CSD', ms=3)
    ax.plot(freqs, np.abs(CSDfac[:, chan, chan]),
            '-o', label='factorized CSD', ms=3)
    ax.set_xlim((f1 - 5, f2 + 5))
    ax.legend()

    
def test_granger():

    """
    Test the granger causality measure
    with uni-directionally coupled AR(2)
    processes akin to the source publication:

    Dhamala, Mukeshwar, Govindan Rangarajan, and Mingzhou Ding.
       "Estimating Granger causality from Fourier and wavelet transforms
        of time series data." Physical review letters 100.1 (2008): 018701.
    """

    fs = 200 # Hz
    nSamples = 2500
    nTrials = 25

    CSDav = np.zeros((nSamples // 2 + 1, 2, 2), dtype=np.complex64)
    for _ in range(nTrials):

        # -- simulate 2 AR(2) processes with 2->1 coupling --
        sol = synth_data.AR2_network(nSamples=nSamples)

        # --- get CSD ---
        bw = 2
        NW = bw * nSamples / (2 * fs)
        Kmax = int(2 * NW - 1) # optimal number of tapers
        CSD, freqs = csd.csd(sol, fs,
                             taper='dpss',
                             taper_opt={'Kmax' : Kmax, 'NW' : NW},
                             fullOutput=True)

        CSDav += CSD

    CSDav /= nTrials
    # with only 2 channels this CSD is well conditioned
    assert np.linalg.cond(CSDav).max() < 1e2
    H, Sigma, conv = wilson_sf(CSDav)

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
    assert G[freq_idx, 1, 0] > 0.8


# --- Helper routines ---


