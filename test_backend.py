import numpy as np
import matplotlib.pyplot as ppl

from syncopy.specest.wavelet import _get_optimal_wavelet_scales, wavelet
from syncopy.specest.superlet import SuperletTransform
from syncopy.specest.wavelets import Morlet


def gen_superlet_testdata(freqs=[20, 40, 60], cycles=11, fs=1000):

    '''
    Harmonic superposition of multiple
    few-cycle oscillations akin to the
    example of Figure 4 in Vale et al. 2021 NatComm
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
s1 = gen_superlet_testdata()
fs = 1000 # sampling frequency
    

preselect = np.ones(len(s1), dtype=bool)

for w0 in [5, 8, 12]:
    morlet = Morlet(w0)
    scales = _get_optimal_wavelet_scales(morlet, len(s1), 1/fs)
    res = wavelet(s1[:, np.newaxis],
                  preselect,
                  preselect,
                  0,
                  0,
                  samplerate=fs,
                  toi='some',
                  scales=scales,
                  wav=morlet)

    ppl.figure()
    ppl.imshow(res[:, 0, :, 0].T, cmap='plasma', aspect='auto')
