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

    nTrials = 100
    nChannels = 10
    nSamples = 1000
    fs = 200

    # -- Create a random
    # -- network of AR(2) processes   

    # thresh of 1 is fully connected network
    conn_thresh = 0.05
    max_cpl = 0.2
    AdjMat = np.zeros((nChannels, nChannels))    
    AdjMat = synth_data.mk_RandomAdjMat(nChannels, conn_thresh,
                                        max_coupling=max_cpl)

    print(AdjMat)
    # channel indices of coupling
    # a 1 at AdjMat(i,j) means coupling from i->j
    cpl_idx = np.where(AdjMat)
    nocpl_idx = np.where(AdjMat == 0)
    
    trls = []
    for _ in range(nTrials):
        # defaults AR(2) parameters yield 40Hz peak
        trls.append(synth_data.AR2_network(AdjMat, nSamples=nSamples))
        # trls.append(synth_data.AR2_network(None, nSamples=nSamples))
        
    data = AnalogData(trls, samplerate=fs)
    foi = np.arange(5, 75) # in Hz

    def test_solution(self):

        Gcaus = connectivity(self.data, method='granger',
                             taper='dpss', tapsmofrq=3)
        
        # print("peak \t Aij \t peak-frq \t ij")                
        # check all channel combinations with
        # and w/o coupling
        for i, j in zip(*self.cpl_idx):
            peak = Gcaus.data[0, :, i, j].max()
            peak_frq = Gcaus.freq[Gcaus.data[0, :, i, j].argmax()]
            cval = self.AdjMat[i, j]

            dbg_str = f"{peak:.2f}\t{self.AdjMat[i,j]:.2f}\t {peak_frq:.2f}\t"
            print(dbg_str,f'\t {i}', f' {j}')
            
            # test for directional coupling
            # at the right frequency range
            assert peak >= cval
            assert 35 < peak_frq < 45
                        
        return Gcaus, self.cpl_idx

T = TestGranger()
G, idx = T.test_solution()


# -- helper functions --

def plot_G(G, i, j, nTr, color='cornflowerblue'):

    ax = ppl.gca()
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel(r'Granger causality(f)')
    ax.plot(G.freq, G.data[0, :, i, j], label=f'Granger {i}-{j}',
            alpha=0.7, lw=1.2, c=color)
    ax.set_ylim((-.1, 1.3))
