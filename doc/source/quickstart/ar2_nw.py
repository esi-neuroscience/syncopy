import numpy as np
import syncopy as spy
from syncopy.tests import synth_data

nTrials = 50
nSamples = 1500

# 2x2 Adjacency matrix to define coupling
AdjMat = np.zeros((2, 2))
# coupling 0 -> 1
AdjMat[0, 1] = 0.2


data = synth_data.ar2_network(nTrials, samplerate=500, AdjMat=AdjMat, nSamples=nSamples)
spec = spy.freqanalysis(data, tapsmofrq=3, keeptrials=False)
