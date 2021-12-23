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
    nChannels = 2
    nSamples = 5000
    fs = 200

    # -- Create a somewhat intricated
    # -- network of AR(2) processes   

    AdjMat = synth_data.mk_AdjMat(nChannels, conn_thresh=0.45)
    print(AdjMat)
    # channel indices of coupling
    # a 1 at AdjMat(i,j) means coupling from j->i
    cpl_idx = np.where(AdjMat.T)
    nocpl_idx = np.where(AdjMat.T == 0)
    
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
            cval = self.AdjMat[j, i]
            print(peak, peak_frq, i, j, self.AdjMat[j, i])            
            # test for directional coupling
            # assert peak > 3 * cval
            # assert 35 < peak_frq < 45

            
        return Gcaus, self.cpl_idx

T = TestGranger()
G, idx = T.test_solution()
