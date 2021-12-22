# -*- coding: utf-8 -*-
#
# Test connectivity measures
#

# 3rd party imports
import numpy as np
import matplotlib.pyplot as ppl

# Local imports
from syncopy import __acme__
if __acme__:
    import dask.distributed as dd
from syncopy.datatype import AnalogData    
from syncopy.connectivity import connectivity
import syncopy.tests.synth_data as synth_data


class TestGranger:

    nTrials = 50
    nChannels = 5
    nSamples = 2500
    fs = 200

    # -- Create a somewhat intricated
    # -- network of AR(2) processes   
        
    # random numbers in [0,1)
    AdjMat = np.random.random_sample((nChannels, nChannels))
    conn_thresh = 0.75
    # all larger elements get set to 1 (coupled)
    AdjMat = (AdjMat > conn_thresh).astype(int)
    # set diagonal to 0 to easier identify coupling
    np.fill_diagonal(AdjMat, 0)
    # channel indices of coupling
    cpl_idx = np.where(AdjMat)
    nocpl_idx = np.where(AdjMat == 0)
    
    trls = []
    for _ in range(nTrials):
        # defaults AR(2) parameters yield 40Hz peak
        trls.append(synth_data.AR2_process(AdjMat, nSamples=nSamples))    
    data = AnalogData(trls, samplerate=fs)
    foi = np.arange(5,75) # in Hz

    def test_solution(self):

        Gcaus = connectivity(self.data, method='granger')
        # check all channel combinations with
        # and w/o coupling
        for i,j in zip(*self.cpl_idx):
            peak = Gcaus.data[0, :, i, j].max()
            peak_frq = Gcaus.freq[Gcaus.data[0, :, i, j].argmax()]
            print(peak, peak_frq, i, j)
        return Gcaus, self.cpl_idx

T = TestGranger()
G, idx = T.test_solution()
