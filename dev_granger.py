import numpy as np
from scipy.signal import csd as sci_csd
import scipy.signal as sci
from syncopy.connectivity.wilson_sf import wilson_sf
from syncopy.connectivity.ST_compRoutines import cross_spectra_cF
#from syncopy.connectivity import wilson_sf
import matplotlib.pyplot as ppl


# noisy phase evolution
def phase_evo(omega0, eps, fs=1000, N=1000):
    wn = np.random.randn(N) 
    delta_ts = np.ones(N) * 1 / fs
    phase = np.cumsum(omega0 * delta_ts + eps * wn)
    return phase


def plot_wilson_errs(errs, label='', c='k'):

    fig, ax = ppl.subplots(figsize=(6,4), num=2)
    ax.set_xlabel('Iteration Step')
    ax.set_ylabel(r'rel. Error $\frac{|CSD - \Psi\Psi^*|}{|CSD|}$')
    ax.semilogy()
    # ax.plot(errs, '-o', label=label, c=c)
    
    fig.subplots_adjust(left=0.15, bottom=0.2)
    return ax


def make_test_data(nChannels=10, nSamples=1000, bw=5):
    
    # white noise ensemble
    fs = 1000
    tvec = np.arange(nSamples) / fs

    data = np.zeros((nSamples, nChannels))
    for i in range(nChannels):
        p1 = phase_evo(30 * 2 * np.pi, 0.1, N=nSamples)
        p2 = phase_evo(60 * 2 * np.pi, 0.25, N=nSamples)

        data[:, i] = np.cos(p1) + np.sin(p2) + .5 * np.random.randn(nSamples)
        #data[:, i] = 2 * np.random.randn(nSamples)

    bw = 5
    NW = bw * nSamples / (2 * fs)
    Kmax = int(2 * NW - 1) # optimal number of tapers

    CS2, freqs = cross_spectra_cF(data, fs, taper='dpss', taper_opt={'Kmax' : Kmax, 'NW' : NW}, norm=False, fullOutput=True)

    CSD = CS2[0, ...]

    return CSD
