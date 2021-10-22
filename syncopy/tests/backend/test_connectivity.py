# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as ppl
from syncopy.connectivity import single_trial_compRoutines as stCR


def test_csd():

    nSamples = 1001
    fs = 1000
    tvec = np.arange(nSamples) / fs    
    harm_freq = 40
    phase_shifts = np.array([0, np.pi / 2, np.pi])

    # 1 phase phase shifted harmonics + white noise + constant, SNR = 1
    data = [10 + np.cos(harm_freq * 2 * np. pi * tvec + ps)
            for ps in phase_shifts] 
    data = np.array(data).T
    data = np.array(data) + np.random.randn(nSamples, len(phase_shifts))
    
    Kmax = 8 # multiple tapers for single trial coherence
    CSD, freqs = stCR.cross_spectra_cF(data, fs,
                                       polyremoval=1,
                                       taper='dpss',
                                       taperopt={'Kmax' : Kmax, 'NW' : 6},
                                       norm=True)

    # output has shape (1, nFreq, nChannels, nChannels)
    assert CSD.shape == (1, len(freqs), data.shape[1], data.shape[1])

    # single trial coherence between channel 0 and 1
    coh = np.abs(CSD[0, :, 0, 1])

    fig, ax = ppl.subplots(figsize=(6,4), num=None)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('coherence')
    ax.set_ylim((-.02,1.05))
    ax.set_title(f'MTM coherence, {Kmax} tapers, SNR=1')

    assert ax.plot(freqs, coh, lw=2, alpha=0.8, c='cornflowerblue')

    # we test for the highest peak sitting at
    # the vicinity (± 5Hz) of one the harmonic
    peak_val = np.max(coh)
    peak_idx = np.argmax(coh)
    peak_freq = freqs[peak_idx]
    print(peak_freq, peak_val)
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
    CC, lags = stCR.cross_covariance_cF(data, samplerate=fs, norm=True)
    # test for result is returned in the [0, np.ceil(nSamples / 2)] lag interval
    nLags = int(np.ceil(nSamples / 2))
    
    # output has shape (nLags, 1, nChannels, nChannels)
    assert CC.shape == (nLags, 1, data.shape[1], data.shape[1])
    
    # cross-correlation (normalized cross-covariance) between
    # cosine and sine analytically equals minus sine    
    assert np.all(CC[:, 0, 0, 1] + sine[:nLags] < 1e-5)
