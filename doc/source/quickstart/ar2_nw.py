import numpy as np
import syncopy as spy
from syncopy.tests import synth_data

nTrials = 50
nSamples = 1500
trls = []
# 2x2 Adjacency matrix to define coupling
AdjMat = np.zeros((2, 2))
# coupling 0 -> 1
AdjMat[0, 1] = 0.2

for _ in range(nTrials):

    trl = synth_data.AR2_network(AdjMat, nSamples=nSamples)
    trls.append(trl)

data = spy.AnalogData(trls, samplerate=500)
spec = spy.freqanalysis(data, tapsmofrq=3, keeptrials=False)
