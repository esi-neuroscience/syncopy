import numpy as np
import syncopy as spy
from syncopy import synthdata

cfg = spy.StructDict()
cfg.nTrials = 50
cfg.nSamples = 2000
cfg.samplerate = 250

# 3x3 Adjacency matrix to define coupling
AdjMat = np.zeros((3, 3))
# only coupling 0 -> 1
AdjMat[0, 1] = 0.2

data = synthdata.ar2_network(AdjMat, cfg=cfg, seed=42)

# add some red noise as 1/f surrogate
data = data + 2 * synthdata.red_noise(cfg, alpha=0.95, nChannels=3, seed=42)

spec = spy.freqanalysis(data, tapsmofrq=3, keeptrials=False)
