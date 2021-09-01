import numpy as np
import matplotlib.pyplot as ppl

from syncopy.connectivity import csd


def gen_testdata():

    '''
    Superposition of harmonics with 
    distinct phase relationships between channel.

    Every channel has a 30Hz and a pi/2 shifted 80Hz band,
    plus an additional channel specific shift:
    Channel1 : 0
    Channel2 : pi/2
    Channel3 : pi

    So the coherencies should be:
    C_12 = 0, C_23 = 0, C_13 = -1
    '''

    fs = 1000 # sampling frequency
    nSamples = fs # for integer Fourier freq bins
    nChannels = 3
    tvec = np.arange(nSamples) / fs
    omegas = np.array([30, 80]) * 2 * np.pi
    phase_shifts = np.array([0, np.pi / 2, np.pi])
    
    data = np.zeros((nSamples, nChannels))

    for i, pshift in enumerate(phase_shifts):
        sig = 2 * np.cos(omegas[0] * tvec + pshift)
        sig += np.cos(omegas[1] * tvec + pshift + np.pi / 2)

        data[:, i] = sig

    return data


data = gen_testdata()
        
                                   
