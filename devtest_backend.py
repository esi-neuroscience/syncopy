''' This is a temporary development file '''

import numpy as np
import matplotlib.pyplot as ppl

from syncopy.shared.parsers import data_parser, scalar_parser, array_parser 
from syncopy.specest.wavelet import get_optimal_wavelet_scales, wavelet
from syncopy.specest.superlet import SuperletTransform, MorletSL, cwtSL, _get_superlet_support, superlet, compute_adaptive_order, scale_from_period
from syncopy.specest.wavelets import Morlet
from scipy.signal import fftconvolve


def gen_superlet_testdata(freqs=[20, 40, 60],
                          cycles=11, fs=1000,
                          eps = 0):

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

    # additive white noise
    if eps > 0:
        signal = np.random.randn(len(signal)) * eps + signal
    
    return signal
                       

# test the Wavelet transform
fs = 1000
s1 = 1 * gen_superlet_testdata(fs=fs, eps=0) # 20Hz, 40Hz and 60Hz
data = np.c_[3*s1, 50*s1]
preselect = np.ones(len(s1), dtype=bool)
preselect2 = np.ones((len(s1), 2), dtype=bool)
pads = 0

ts = np.arange(-50,50)
morletTC = Morlet()
morletSL = MorletSL(c_i=30)

# frequencies to look at, 10th freq is around 20Hz
freqs = np.linspace(1, 100, 50) # up to 100Hz
scalesTC = morletTC.scale_from_period(1 / freqs)
# scales are cycle independent!
scalesSL = scale_from_period(1 / freqs)

# automatic diadic scales
ssTC = get_optimal_wavelet_scales(Morlet().scale_from_period, len(s1), 1/fs)
ssSL = get_optimal_wavelet_scales(scale_from_period, len(s1), 1/fs)
# a multiplicative Superlet - a set of Morlets, order 1 - 30
c_1 = 1
cycles = c_1 * np.arange(1, 31)
sl = [MorletSL(c) for c in cycles]

res = wavelet(data,
              preselect,
              preselect,
              pads,
              pads,
              samplerate=fs,
              # toi='some',
              output_fmt="pow",
              scales=scalesTC,
              wav=Morlet(),
              noCompute=False)


# unit impulse
# data = np.zeros(500)
# data[248:252] = 1
spec = superlet(s1, samplerate=fs, scales=scalesSL,
                order_max=10,
                order_min=5,
                adaptive=False)
spec2 = superlet(data, samplerate=fs, scales=scalesSL, order_max=20, adaptive=False)

# nc = superlet(data, samplerate=fs, scales=scalesSL, order_max=30)


def do_slt(data, scales=scalesSL, **slkwargs):

    if scales is None: 
        scales = get_optimal_wavelet_scales(scale_from_period,
                                            len(data[:, 0]),
                                            1 / fs)    

    spec = superlet(data, samplerate=fs,
                    scales=scales,
                    **slkwargs)

    print(spec.max(),spec.shape)
    ppl.figure()
    extent = [0, len(s1) / fs, freqs[-1], freqs[0]]    
    ppl.imshow(np.abs(spec[...,0]), cmap='plasma', aspect='auto', extent=extent)
    ppl.plot([0, len(s1) / fs], [20, 20], 'k--')
    ppl.plot([0, len(s1) / fs], [40, 40], 'k--')
    ppl.plot([0, len(s1) / fs], [60, 60], 'k--')

    return spec


def show_MorletSL(morletSL, scale):

    cycle = morletSL.c_i
    ts = _get_superlet_support(scale, 1/fs, cycle)
    ppl.plot(ts, MorletSL(cycle)(ts, scale))

    
def show_MorletTC(morletTC, scale):

    M = 10 * scale * fs
    # times to use, centred at zero
    ts = np.arange((-M + 1) / 2.0, (M + 1) / 2.0) / fs
    ppl.plot(ts, morletTC(ts, scale))
    

def do_superlet_cwt(data, wav, scales=None):

    if scales is None: 
        scales = get_optimal_wavelet_scales(scale_from_period, len(data[:,0]), 1/fs)

    res = cwtSL(data,
                wav,
                scales=scales,
                dt=1 / fs)              

    ppl.figure()
    extent = [0, len(s1) / fs, freqs[-1], freqs[0]]
    channel=0
    ppl.imshow(np.abs(res[:,:, channel]), cmap='plasma', aspect='auto', extent=extent)
    ppl.plot([0, len(s1) / fs], [20, 20], 'k--')
    ppl.plot([0, len(s1) / fs], [40, 40], 'k--')
    ppl.plot([0, len(s1) / fs], [60, 60], 'k--')

    return res.T


def do_normal_cwt(data, wav, scales=None):

    if scales is None: 
        scales = get_optimal_wavelet_scales(wav.scale_from_period,
                                             len(data[:,0]),
                                             1/fs)    
    res = wavelet(data,
                  preselect,
                  preselect,
                  pads,
                  pads,
                  samplerate=fs,
                  # toi='some',
                  output_fmt="pow",
                  scales=scales,
                  wav=wav)

    ppl.figure()
    extent = [0, len(s1) / fs, freqs[-1], freqs[0]]
    ppl.imshow(res[:, 0, :, 0].T, cmap='plasma', aspect='auto', extent=extent)
    ppl.plot([0, len(s1) / fs], [20, 20], 'k--')
    ppl.plot([0, len(s1) / fs], [40, 40], 'k--')
    ppl.plot([0, len(s1) / fs], [60, 60], 'k--')

    return res[:, 0, :, :].T

# do_cwt(morletSL)


def screen_CWT(w0s= [5, 8, 12]):
    for w0 in w0s:
        morletTC = Morlet(w0)
        scales = _get_optimal_wavelet_scales(morletTC, len(s1), 1/fs)
        res = wavelet(s1[:, np.newaxis],
                      preselect,
                      preselect,
                      pads,
                      pads,
                      samplerate=fs,
                      toi=np.array([1,2]),
                      scales=scales,
                      wav=morletTC)

        ppl.figure()
        ppl.imshow(res[:, 0, :, 0].T, cmap='plasma', aspect='auto')
