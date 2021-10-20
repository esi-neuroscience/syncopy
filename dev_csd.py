import numpy as np
from scipy.signal import csd as sci_csd
import scipy.signal as sci
from syncopy.connectivity.single_trial_compRoutines import cross_spectra_cF
from syncopy.connectivity.single_trial_compRoutines import cross_covariance_cF
import matplotlib.pyplot as ppl


# white noise ensemble
nSamples = 1001
fs = 1000
tvec = np.arange(nSamples) / fs
omegas = np.array([21, 42, 59, 78]) * 2 * np.pi
omegas = np.arange(10, 40) * 2 * np.pi
data = np.c_[[1*np.cos(om * tvec) for om in omegas]].T

nChannels = 5
data1 = np.random.randn(nSamples, nChannels)

x1 = data1[:, 0]
y1 = data1[:, 1]


def dev_cc(nSamples=1001):

    nSamples = 1001
    fs = 1000    
    tvec = np.arange(nSamples) / fs
    
    sig1 = np.cos(2 * np.pi * 30 * tvec)
    sig2 = np.sin(2 * np.pi * 30 * tvec)

    mode = 'same'
    t_half = nSamples // 2

    r = sci.correlate(sig2, sig2, mode=mode, method='fft')
    r2 = sci.fftconvolve(sig2, sig2[::-1], mode=mode)
    assert np.all(r == r2)


    lags = np.arange(-nSamples // 2, nSamples // 2)
    if nSamples % 2 != 0:
        lags = lags + 1
    lags = lags * 1 / fs

    if nSamples % 2 == 0:
        half_lags = np.arange(0, nSamples // 2)
    else:
        half_lags = np.arange(0, nSamples // 2 + 1)
    half_lags = half_lags * 1 / fs

    ppl.figure(1)
    ppl.xlabel('lag (s)')
    ppl.ylabel('convolution result')

    ppl.plot(lags, r2, lw = 1.5)
    #  ppl.xlim((490,600))


    norm = np.arange(nSamples, t_half, step = -1) / 2
    # norm = np.r_[norm, norm[::-1]]

    ppl.figure(2)
    ppl.xlabel('lag (s)')
    ppl.ylabel('correlation')
    ppl.plot(half_lags, r2[nSamples // 2:] / norm, lw = 1.5)


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
    

def mtm_csd_harmonics():

    omegas = np.array([30, 80]) * 2 * np.pi
    phase_shifts = np.array([0, np.pi / 2, np.pi])

    NN = 50
    res = np.zeros((nSamples // 2 + 1, NN))
    eps = 1
    Kmax = 15
    data = [np.sum([np.cos(om * tvec + ps) for om in omegas], axis=0) for ps in phase_shifts]
    data = np.array(data).T
    
    for i in range(NN):
        dataR = 5 * (data + np.random.randn(nSamples, 3) * eps)

        CS, freqs = cross_spectra_cF(data, fs, taper='bartlett')
        CS2, freqs = cross_spectra_cF(dataR, fs, taper='dpss', taperopt={'Kmax' : Kmax, 'NW' : 6}, norm=True)

        res[:, i] = np.abs(CS2[:, 0, 1])

    q1 = np.percentile(res, 25, axis=1)
    q3 = np.percentile(res, 75, axis=1)
    med = np.percentile(res, 50, axis=1)

    fig, ax = ppl.subplots(figsize=(6,4), num=None)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('coherence')
    ax.set_ylim((-.02,1.05))
    ax.set_title(f'MTM coherence, {Kmax} tapers, SNR={1/eps**2}')

    c = 'cornflowerblue'
    ax.plot(freqs, med, lw=2, alpha=0.8, c=c)
    ax.fill_between(freqs, q1, q3, color=c, alpha=0.3)


# omegas = np.arange(30, 50, step=1) * 2 * np.pi
# data = np.array([np.cos(om * tvec) for om in omegas]).T
# dataR = 1 * (data + np.random.randn(*data.shape) * 1)
# CS, freqs = cross_spectra_cF(dataR, fs, taper='dpss', taperopt={'Kmax' : 15, 'NW' : 6}, norm=True)

# CS2, freqs = cross_spectra_cF(dataR, fs, taper='dpss', taperopt={'Kmax' : 15, 'NW' : 6}, norm=False)

# noisy phase evolution
def phase_evo(omega0, eps, fs=1000, N=1000):
    wn = np.random.randn(N) * 1 / fs
    delta_ts = np.ones(N) * 1 / fs
    phase = np.cumsum(omega0 * delta_ts + eps * wn)
    return phase


eps = 0.3 * fs
omega = 50 * 2 * np.pi
p1 = phase_evo(omega, eps, N=nSamples)
p2 = phase_evo(omega, eps, N=nSamples)
s1 = np.cos(p1) + np.cos(2 * omega * tvec) + .5 * np.random.randn(nSamples)
s2 = np.cos(p2) + np.cos(2 * omega * tvec) + .5 * np.random.randn(nSamples)
data = np.c_[s1, s2]

CS, freqs = cross_spectra_cF(data, fs, taper='dpss', taperopt={'Kmax' : 15, 'NW' : 6}, norm=True)
CS2, freqs = cross_spectra_cF(data, fs, taper='dpss', taperopt={'Kmax' : 15, 'NW' : 6}, norm=False)

fig, ax = ppl.subplots(figsize=(6,4), num=None)
ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('$|CSD(f)|$')
ax.set_ylim((-.02,1.25))

ax.plot(freqs, np.abs(CS2[:, 0, 0]), label = '$CSD_{00}$', lw = 2, alpha = 0.7)
ax.plot(freqs, np.abs(CS2[:, 1, 1]), label = '$CSD_{11}$', lw = 2, alpha = 0.7)
ax.plot(freqs, np.abs(CS2[:, 0, 1]), label = '$CSD_{01}$', lw = 2, alpha = 0.7)

ax.legend()
