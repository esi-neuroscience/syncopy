''' This is a temporary development file '''

import numpy as np
import matplotlib.pyplot as ppl

from syncopy.specest.wavelet import _get_optimal_wavelet_scales, wavelet
from syncopy.specest.superlet import SuperletTransform, MorletSL, cwt
from syncopy.specest.wavelets import Morlet


def gen_superlet_testdata(freqs=[20, 40, 60], cycles=11, fs=1000):

    '''
    Harmonic superposition of multiple
    few-cycle oscillations akin to the
    example of Figure 3 in Moca et al. 2021 NatComm
    '''

    signal = []
    for freq in freqs:
        
        # 10 cycles of f1
        tvec = np.arange(cycles / freq, step=1 / fs)

        harmonic = np.cos(2 * np.pi * freq * tvec)
        f_neighbor = np.cos(2 * np.pi * (freq + 10) * tvec) 
        packet = harmonic +  f_neighbor

        # 2 cycles time neighbor
        delta_t = np.zeros(int(2 / freq * fs))
        
        # 5 cycles break
        pad = np.zeros(int(5 / freq * fs))

        signal.extend([pad, packet, delta_t, harmonic])

    # stack the packets together with some padding        
    signal.append(pad)
    signal = np.concatenate(signal)
    
    return signal
                       

# test the Wavelet transform
fs = 1000
s1 = gen_superlet_testdata(fs=fs) # 20Hz, 40Hz and 60Hz
preselect = np.ones(len(s1), dtype=bool)
pads = 0

ts = np.arange(-50,50)
morletTC = Morlet(w0=5)
morletSL = MorletSL(c_i=30)

# frequencies to look at, 10th freq is around 20Hz
freqs = np.linspace(1 / len(s1), 100, 50) # up to 100Hz
scalesTC = morletTC.scale_from_period(1 / freqs)
scalesSL = morletSL.scale_from_period(1 / freqs)


def get_superlet_support(scale, dt, cycles, k_sd=5):

    '''
    this mimics the `cwt` routine support
    '''

    # number of points needed to capture wavelet
    M = 10 * scale * cycles  / dt
    print(M)
    # times to use, centred at zero
    t = np.arange((-M + 1) / 2., (M + 1) / 2.) * dt

    return t


def do_superlet_cwt(wav, scales=None):

    if scales is None: 
        # scales = _get_optimal_wavelet_scales(wav, len(s1), 1/fs)
        scales = wav.scale_from_period(1 / freqs)
    
    res = cwt(s1,
              wav,
              scales=scales,
              dt=1/fs)              

    ppl.figure()
    extent = [0, len(s1) / fs, freqs[-1], freqs[0]]    
    ppl.imshow(np.abs(res), cmap='plasma', aspect='auto', extent=extent)
    ppl.plot([0, len(s1) / fs], [20, 20], 'k--')
    ppl.plot([0, len(s1) / fs], [40, 40], 'k--')
    ppl.plot([0, len(s1) / fs], [60, 60], 'k--')

    return res.T


def do_normal_cwt(wav, scales=None):

    if scales is None: 
        # scales = _get_optimal_wavelet_scales(wav, len(s1), 1/fs)
        scales = wav.scale_from_period(1 / freqs)
    
    res = wavelet(s1[:, np.newaxis],
                  preselect,
                  preselect,
                  pads,
                  pads,
                  samplerate=fs,
                  # toi='some',
                  scales=scales,
                  wav=wav)

    ppl.figure()
    extent = [0, len(s1) / fs, freqs[-1], freqs[0]]
    ppl.imshow(res[:, 0, :, 0].T, cmap='plasma', aspect='auto', extent=extent)

    return res[:, 0, :, 0].T

# do_cwt(morletSL)


def screen_CWT(w0s= [5, 8, 12]):
    for w0 in w0s:
        morletTC = Morlet(w0)
        scales = _get_optimal_wavelet_scales(morlet, len(s1), 1/fs)
        res = wavelet(s1[:, np.newaxis],
                      preselect,
                      preselect,
                      pads,
                      pads,
                      samplerate=fs,
                      # toi='some',
                      scales=scales,
                      wav=morlet)

        ppl.figure()
        ppl.imshow(res[:, 0, :, 0].T, cmap='plasma', aspect='auto')
