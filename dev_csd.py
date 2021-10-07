import numpy as np
from scipy.signal import csd as sci_csd
from syncopy.specest import csd
import matplotlib.pyplot as ppl


# white noise ensemble
nSamples = 10000
fs = 1000
nChannels = 5
data1 = np.random.randn(nSamples, nChannels)

x1 = data1[:, 0]
y1 = data1[:, 1]


def sci_est(x, y, nper, norm=False):
    freqs1, csd1 = sci_csd(x, y, fs, window='bartlett', nperseg=nper)
    freqs2, csd2 = sci_csd(x, y, fs, window='bartlett', nperseg=nSamples)

    if norm:
        #  WIP..
        auto1 = sci_csd(x, x, fs, window='bartlett', nperseg=nper)
        auto1 *= sci_csd(y, y, fs, window='bartlett', nperseg=nper)

        auto2 = sci_csd(x, y, fs, window='bartlett', nperseg=nSamples)
        auto2 *= sci_csd(y, y, fs, window='bartlett', nperseg=nSamples)
        
        csd1 = csd1 / np.sqrt(auto1 * auto2)
        
    return (freqs1, np.abs(csd1)), (freqs2, np.abs(csd2))
    

freqs, CSD, specs = csd.csd(data1, fs, 'bartlett', norm=False)

# harmonics
tvec = np.arange(nSamples) / fs
omegas = np.array([30, 80]) * 2 * np.pi
phase_shifts = np.array([0, np.pi / 2, np.pi])

data2 = [np.sum([np.cos(om * tvec + ps) for om in omegas], axis=0) for ps in phase_shifts]
data2 = np.array(data2).T
data2 = data2 + np.random.randn(nSamples, 3) * .3

# x1 = data2[:, 0]
# y1 = data2[:, 1]

# freqs2, csd1 = sci_csd(x1, y1, fs, window='bartlett', nperseg=10000)
# freqs2, csd2 = sci_csd(y1, y1, fs, window='bartlett', nperseg=10000)

# freqs, CSD, specs = csd.csd(data2, fs, 'bartlett', norm=False)
