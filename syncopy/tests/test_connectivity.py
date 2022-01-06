# -*- coding: utf-8 -*-
#
# Test connectivity measures
#

# 3rd party imports
import itertools
import numpy as np
import matplotlib.pyplot as ppl

# Local imports
from syncopy import __acme__
if __acme__:
    import dask.distributed as dd
from syncopy.datatype import AnalogData    
from syncopy.connectivity import connectivity
import syncopy.tests.synth_data as synth_data
from syncopy.shared.errors import SPYValueError


class TestGranger:

    nTrials = 100
    nChannels = 5
    nSamples = 1000
    fs = 200

    # -- Create a somewhat intricate
    # -- network of AR(2) processes   

    # the adjacency matrix
    # encodes coupling strength directly
    AdjMat = np.zeros((nChannels, nChannels))
    AdjMat[0, 4] = 0.15
    AdjMat[3, 4] = 0.15
    AdjMat[3, 2] = 0.25    
    AdjMat[1, 0] = 0.25
    
    print(AdjMat)
    # channel indices of coupling
    # a 1 at AdjMat(i,j) means coupling from i->j
    cpl_idx = np.where(AdjMat)
    nocpl_idx = np.where(AdjMat == 0)
    
    trls = []
    for _ in range(nTrials):
        # defaults AR(2) parameters yield 40Hz peak
        trls.append(synth_data.AR2_network(AdjMat, nSamples=nSamples))

    # create syncopy data instance 
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
            peak_frq = Gcaus.freq[Gcaus.data[0, 5:-5, i, j].argmax()]
            cval = self.AdjMat[i, j]

            dbg_str = f"{peak:.2f}\t{self.AdjMat[i,j]:.2f}\t {peak_frq:.2f}\t"
            print(dbg_str,f'\t {i}', f' {j}')
            
            # test for directional coupling
            # at the right frequency range
            assert peak >= cval
            assert 35 < peak_frq < 45
            
    def test_selections(self):

        # create 3 random trial and channel selections
        trials, channels = [], []
        for _ in range(3):

            sizeTr =  np.random.randint(1, self.nTrials + 1) 
            trials.append(list(np.random.choice(self.nTrials, size=sizeTr)))
            
            sizeCh = np.random.randint(1, self.nChannels + 1)        
            channels.append(['channel' + str(i + 1)
                             for i in np.random.choice(self.nChannels, size=sizeCh, replace=False)])
            
        # create toi selections, signal length is 5s at 200Hz
        # with -1s as offset (from synthetic data instantiation)
        # subsampling does NOT WORK due to precision issues :/
        # toi1 = np.linspace(-.4, 2, 100)
        # toi2 = 'all'
        tois = [None, 'all']

        # 2 random toilims
        toilims = [np.sort(np.random.rand(2) * 5 - 1) for _ in range(2)]
        
        
        # combinatorics of all selection options
        # order matters to assign the selection dict keys!
        toilim_combinations = itertools.product(trials,
                                                channels,
                                                toilims)

        # create selections and run frontend
        for comb in toilim_combinations:

            sel_dct = {}
            sel_dct['trials'] = comb[0]
            sel_dct['channels'] = comb[1]
            sel_dct['toilim'] = comb[2]
            
            Gcaus = connectivity(self.data, method='granger', select=sel_dct)

            # check here just for finiteness and positivity
            assert np.all(np.isfinite(Gcaus.data))
            assert np.all(Gcaus.data[0, ...] >= -1e-10)
            
    def test_foi(self):
        # fois
        foi1 = np.arange(10, 60) # 1Hz steps
        foi2 = np.arange(20, 50, 0.5) # 0.5Hz steps
        foi3 = 'all'
        fois = [foi1, foi2, foi3]

        for foi in fois:
            Gcaus = connectivity(self.data, method='granger', foi=foi)
            # check here just for finiteness and positivity
            assert np.all(np.isfinite(Gcaus.data))
            assert np.all(Gcaus.data[0, ...] >= -1e-10)

        # 3 random foilims
        foilims = [np.sort(np.random.rand(2) * 60) for _ in range(2)]
        for foil in foilims:
            Gcaus = connectivity(self.data, method='granger', foilim=foil)
            # check here just for finiteness and positivity
            assert np.all(np.isfinite(Gcaus.data))
            assert np.all(Gcaus.data[0, ...] >= -1e-10)

        # make sure specification of both foi and foilim triggers a
        # Syncopy ValueError
        try:
            Gcaus = connectivity(self.data, method='granger', foi=foi, foilim=foil)
        except SPYValueError as err:
            assert 'foi/foilim' in str(err)
            

T = TestGranger()
T.test_solution()
# T.test_selections()
T.test_foi()

l1 = [1,2,3]
l2 = ['a', 'b']
l3 = ['ccc', 'xyx']

# -- helper functions --

def plot_G(G, i, j, nTr, color='cornflowerblue'):

    ax = ppl.gca()
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel(r'Granger causality(f)')
    ax.plot(G.freq, G.data[0, :, i, j], label=f'Granger {i}-{j}',
            alpha=0.7, lw=1.2, c=color)
    ax.set_ylim((-.1, 1.3))
