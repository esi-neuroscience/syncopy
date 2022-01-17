import numpy as np
import scipy.signal as sci

from syncopy.datatype import CrossSpectralData, padding, SpectralData, AnalogData
from syncopy.connectivity.ST_compRoutines import cross_spectra_cF, ST_CrossSpectra
from syncopy.connectivity.ST_compRoutines import cross_covariance_cF
from syncopy.connectivity import connectivity
from syncopy.specest import freqanalysis
import matplotlib.pyplot as ppl

from syncopy.shared.parsers import data_parser, scalar_parser, array_parser
from syncopy.shared.tools import get_defaults
from syncopy.datatype import SpectralData, padding

from syncopy.tests.misc import generate_artificial_data
from syncopy.tests import synth_data

foilim = [5, 80]
foi = np.arange(5, 80, 1)
foi2 = np.arange(5, 180, .2)
# this still gives type(tsel) = slice :)
sdict1 = {'channels' : ['channel01', 'channel03'], 'toilim' : [-.221, 1.12]}

nSamples = 2500
nChannels = 4
nTrials = 10
fs = 200

f1, f2 = 10, 40
trls = []
for _ in range(nTrials):

    # little phase diffusion
    p1 = synth_data.phase_evo(f1, eps=.01, nChannels=nChannels, nSamples=nSamples)
    # same frequency but more diffusion
    p2 = synth_data.phase_evo(f2, eps=0.001, nChannels=nChannels, nSamples=nSamples)
    # set 2nd channel to higher phase diffusion
    #p1[:, 1] = p2[:, 1]
    # add a pi/2 phase shift for the even channels
    #p1[:, 2::2] += np.pi / 2
    # add a pi phase shift for the odd channels
    #p1[:, 3::2] += np.pi

    trls.append(1 * np.cos(p1) + 1 * np.cos(p2) + 0.6 * np.random.randn(nSamples, nChannels))
    
tdat2 = AnalogData(trls, samplerate=1000)

AdjMat = synth_data.mk_RandomAdjMat(nChannels)
trls = [100 * synth_data.AR2_network(AdjMat) for _ in range(nTrials)]
tdat1 = AnalogData(trls, samplerate=fs)


def call_con(data, method, **kwargs):

    res = connectivity(data=data,
                       method=method,
                       **kwargs)
    return res


def call_freq(data, method, **kwargs):
    res = freqanalysis(data=data, method=method, **kwargs)

    return res

# ampl = np.abs(res2.show())


def plot_coh(res, i, j, label=''):

    dim = res.dimord.index('freq')
    
    ax = ppl.gca()
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('coherence $|CSD|^2$')
    ax.plot(res.freq, res.data[0, :, i, j], label=label)


def plot_corr(res, i, j, label=''):

    ax = ppl.gca()
    ax.set_xlabel('lag (s)')
    ax.set_ylabel('Correlation')
    ax.plot(res.time[0], res.data[:, 0, i, j], label=label)
    
# ppl.xlabel('frequency (Hz)')
