import pytest
import numpy as np
import matplotlib.pyplot as ppl

from syncopy.specest import superlet


def gen_testdata(freqs=[20, 40, 60],
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


fs = 1000 # sampling frequency

# generate 3 packets at 20, 40 and 60Hz with 10 cycles each
# Noise variance is given by eps
signal_freqs = np.array([20, 40, 60])
cycles = 10
signal = gen_testdata(freqs=signal_freqs, cycles=cycles, fs=fs, eps=0.) 

# define frequencies of interest for time-frequency methods
foi = np.arange(1, 101, step=1)

# closest spectral indices to validate time-freq results
freq_idx = []
for frequency in signal_freqs:
    freq_idx.append(np.argmax(foi > frequency))
    

def test_superlet():
    
    scalesSL = superlet.scale_from_period(1 / foi)
    
    spec = superlet.superlet(signal,
                             samplerate=fs,
                             scales=scalesSL,
                             order_max=40,
                             order_min=5,
                             c_1=1,
                             adaptive=True)
    # power spectrum
    ampls = np.abs(spec**2)

    ppl.figure()
    extent = [0, len(signal) / fs, foi[0], foi[-1]]

    # test also the plotting
    assert ppl.imshow(ampls,
                      cmap='plasma',
                      aspect='auto',
                      extent=extent,
                      origin='lower')

    for idx, frequency in zip(freq_idx, signal_freqs):

        ppl.plot([0, len(signal) / fs],
                 [frequency, frequency],
                 '--',
                 c='0.5')

        # number of cycles with relevant
        # amplitude at the respective frequency
        cycle_num = (ampls[idx, :] > 1 / np.e).sum() / fs * frequency
        # we have 2 times the cycles for each frequency (temporal neighbor)
        assert cycle_num > 2 * cycles
