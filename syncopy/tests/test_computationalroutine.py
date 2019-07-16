# -*- coding: utf-8 -*-
#
# Test basic functionality of ComputationalRoutine class
#
# Created: 2019-07-03 11:31:33
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-07-10 14:28:53>

import pytest
import numpy as np
import dask.distributed as dd
from scipy import signal
from syncopy.datatype import AnalogData
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.tests.misc import generate_artifical_data, is_slurm_node


# Decorator to run SLURM tests only on cluster nodes
skip_without_slurm = pytest.mark.skipif(not is_slurm_node(),
                                        reason="not running on cluster node")


def lowpass(arr, b, a, noCompute=None, chunkShape=None):
    if noCompute:
        return arr.shape, arr.dtype
    res = signal.filtfilt(b, a, arr.T, padlen=200).T
    return res


class LowPassFilter(ComputationalRoutine):
    computeFunction = staticmethod(lowpass)

    def process_metadata(self, data, out):
        if not self.keeptrials:
            trl = np.array([[0, out.data.shape[0], 0]], dtype=int)
        else:
            trl = np.zeros((len(data.trials), 3), dtype=int)
            trial_lengths = np.diff(data.sampleinfo)
            cnt = 0
            for row, tlen in enumerate(trial_lengths):
                trl[row, 0] = cnt
                trl[row, 1] = cnt + tlen
                cnt += tlen
        out.sampleinfo = trl[:, :2]
        out._t0 = trl[:, 2]
        out.trialinfo = trl[:, 3:]
        out.samplerate = data.samplerate
        out.channel = np.array(data.channel)


class TestComputationalRoutine():

    # Construct linear combination of low- and high-frequency sine waves
    # and use an IIR filter to reconstruct the low-frequency component
    nChannels = 32
    nTrials = 8
    fData = 2
    fNoise = 64
    fs = 1000
    t = np.linspace(-1, 1, fs)
    orig = np.sin(2 * np.pi * fData * t)
    sig = orig + np.sin(2 * np.pi * fNoise * t)
    cutoff = 50
    b, a = signal.butter(8, 2 * cutoff / fs)
    tol = 1e-6

    # Blow up the signal to have "channels" and "trials" and inflate the low-
    # frequency component accordingly for ad-hoc comparisons later
    sig = np.repeat(sig.reshape(-1, 1), axis=1, repeats=nChannels)
    sig = np.tile(sig, (nTrials, 1))
    orig = np.repeat(orig.reshape(-1, 1), axis=1, repeats=nChannels)
    orig = np.tile(orig, (nTrials, 1))

    # Construct artificial equidistant trial-definition array
    trl = np.zeros((nTrials, 3), dtype="int")
    for ntrial in range(nTrials):
        trl[ntrial, :] = np.array([ntrial * fs, (ntrial + 1) * fs, 0])

    # Create reference AnalogData object with equidistant trial spacing
    equidata = AnalogData(data=sig, samplerate=fs, trialdefinition=trl,
                          dimord=["time", "channel"])


    def test_sequential_equidistant(self):
        myfilter = LowPassFilter(self.b, self.a)
        myfilter.initialize(self.equidata)
        out = AnalogData()
        myfilter.compute(self.equidata, out)
        assert np.abs(out.data - self.orig).max() < self.tol

        myfilter = LowPassFilter(self.b, self.a, keeptrials=False)
        myfilter.initialize(self.equidata)
        out = AnalogData()
        myfilter.compute(self.equidata, out)
        assert np.abs(out.data - self.orig[:self.t.size, :]).max() < self.tol

    def test_sequential_nonequidistant(self):
        myfilter = LowPassFilter(self.b, self.a)
        for overlapping in [False, True]:
            nonequidata = generate_artifical_data(nTrials=self.nTrials,
                                                  nChannels=self.nChannels,
                                                  equidistant=False,
                                                  overlapping=overlapping,
                                                  inmemory=False)
            myfilter.initialize(nonequidata)
            out = AnalogData()
            myfilter.compute(nonequidata, out)
            assert out.data.shape[0] == np.diff(nonequidata.sampleinfo).sum()

    @skip_without_slurm
    def test_parallel_equidistant(self, esicluster):
        client = dd.Client(esicluster)
        for parallel_store in [True, False]:
            myfilter = LowPassFilter(self.b, self.a)
            myfilter.initialize(self.equidata)
            out = AnalogData()
            myfilter.compute(self.equidata, out, parallel=True, parallel_store=parallel_store)
            assert np.abs(out.data - self.orig).max() < self.tol
            assert out.data.is_virtual == parallel_store
    
            myfilter = LowPassFilter(self.b, self.a, keeptrials=False)
            myfilter.initialize(self.equidata)
            out = AnalogData()
            myfilter.compute(self.equidata, out, parallel=True, parallel_store=parallel_store)
            assert np.abs(out.data - self.orig[:self.t.size, :]).max() < self.tol
            assert out.data.is_virtual == parallel_store
        client.close()
    
    @skip_without_slurm
    def test_parallel_nonequidistant(self, esicluster):
        client = dd.Client(esicluster)
        for overlapping in [False, True]:
            for parallel_store in [True, False]:
                nonequidata = generate_artifical_data(nTrials=self.nTrials,
                                                      nChannels=self.nChannels,
                                                      equidistant=False,
                                                      overlapping=overlapping,
                                                      inmemory=False)
    
                out = AnalogData()
                myfilter = LowPassFilter(self.b, self.a)
                myfilter.initialize(nonequidata)
                myfilter.compute(nonequidata, out, parallel=True, parallel_store=parallel_store)
                assert out.data.shape[0] == np.diff(nonequidata.sampleinfo).sum()
                for tk, trl in enumerate(out.trials):
                    assert trl.shape[0] == np.diff(nonequidata.sampleinfo[tk, :])
                assert out.data.is_virtual == parallel_store
        client.close()
