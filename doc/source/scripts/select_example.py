import numpy as np
import syncopy as spy
from syncopy.tests import synth_data

# 100 trials of two phase diffusing signals with 40Hz
adata = synth_data.phase_diffusion(nTrials=100,
                                   freq=40,
                                   samplerate=200,
                                   nSamples=500,
                                   nChannels=2,
                                   eps=0.01)

# coherence for full dataset
coh1 = spy.connectivityanalysis(adata, method='coh')

# plot coherence of channel1 vs channel2
coh1.singlepanelplot(channel_i='channel1', channel_j='channel2')
