import numpy as np
import syncopy as spy
from syncopy.tests import synth_data

nTrials = 50
nSamples = 1500
trls = []
# empty adjacency matrix - no coupling
AdjMat = np.zeros((2, 2))

for _ in range(nTrials):

    trl = synth_data.AR2_network(AdjMat, nSamples=nSamples)
    trls.append(trl)

data_uc = spy.AnalogData(trls, samplerate=500)
spec_uc = spy.freqanalysis(data_uc, tapsmofrq=3, keeptrials=False)
