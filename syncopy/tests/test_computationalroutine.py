# -*- coding: utf-8 -*-
# 
# Test basic functionality of ComputationalRoutine class
# 
# Created: 2019-07-03 11:31:33
# Last modified by: Joscha Schmiedt [joscha.schmiedt@esi-frankfurt.de]
# Last modification time: <2019-09-18 13:56:13>

import os
import tempfile
import pytest
import numpy as np
from glob import glob
from scipy import signal
from syncopy import __dask__
if __dask__:
    import dask.distributed as dd
from syncopy.datatype import AnalogData
from syncopy.io import load
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.shared.parsers import unwrap_io
from syncopy.tests.misc import generate_artifical_data

# Decorator to decide whether or not to run dask-related tests
skip_without_dask = pytest.mark.skipif(not __dask__, reason="dask not available")


@unwrap_io
def lowpass(arr, b, a, noCompute=None, chunkShape=None):
    if noCompute:
        return arr.shape, arr.dtype
    res = signal.filtfilt(b, a, arr.T, padlen=200).T
    return res


class LowPassFilter(ComputationalRoutine):
    computeFunction = staticmethod(lowpass)

    def process_metadata(self, data, out):
        if not self.keeptrials:
            out.trialdefinition = np.array([[0, out.data.shape[0], 0]], dtype=int)             
        else:
            trl = np.zeros((len(data.trials), 3), dtype=int)
            trial_lengths = np.diff(data.sampleinfo)
            cnt = 0
            for row, tlen in enumerate(trial_lengths):
                trl[row, 0] = cnt
                trl[row, 1] = cnt + tlen
                cnt += tlen
            out.trialdefinition = np.hstack((trl, data.trialinfo))
        
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
    
    # For parallel computation w/concurrent writing: predict no. of generated 
    # HDF5 files that will make up virtual data-set in case of channel-chunking
    chanPerWrkr = 7
    nFiles = nTrials * (int(nChannels/chanPerWrkr) + int(nChannels % chanPerWrkr > 0))


    def test_sequential_equidistant(self):
        myfilter = LowPassFilter(self.b, self.a)
        myfilter.initialize(self.equidata)
        out = AnalogData(dimord=AnalogData._defaultDimord)
        myfilter.compute(self.equidata, out)
        assert np.abs(out.data - self.orig).max() < self.tol
        
        myfilter = LowPassFilter(self.b, self.a)
        myfilter.initialize(self.equidata, keeptrials=False)
        out = AnalogData(dimord=AnalogData._defaultDimord)
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
            out = AnalogData(dimord=AnalogData._defaultDimord)
            myfilter.compute(nonequidata, out)
            assert out.data.shape[0] == np.diff(nonequidata.sampleinfo).sum()
            
    def test_sequential_saveload(self):
        myfilter = LowPassFilter(self.b, self.a)
        myfilter.initialize(self.equidata)
        out = AnalogData(dimord=AnalogData._defaultDimord)
        myfilter.compute(self.equidata, out, log_dict={"a": self.a, "b": self.b})
        assert set(["a", "b"]) == set(out.cfg.keys())
        assert np.array_equal(out.cfg["a"], self.a)
        assert np.array_equal(out.cfg["b"], self.b)
        assert "lowpass" in out._log
        
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "dummy")
            out.save(fname)
            dummy = load(fname)
            assert "a" in dummy.cfg.keys()
            assert "b" in dummy.cfg.keys()
            assert np.array_equal(dummy.cfg["a"], self.a)
            assert np.array_equal(dummy.cfg["b"], self.b)
            assert np.abs(dummy.data - self.orig).max() < self.tol
            del dummy, out

    @skip_without_dask
    def test_parallel_equidistant(self, testcluster):
        client = dd.Client(testcluster)
        for parallel_store in [True, False]:
            for chan_per_worker in [None, self.chanPerWrkr]:
                myfilter = LowPassFilter(self.b, self.a)
                myfilter.initialize(self.equidata, chan_per_worker=chan_per_worker)
                out = AnalogData(dimord=AnalogData._defaultDimord)
                myfilter.compute(self.equidata, out, parallel=True, parallel_store=parallel_store)
                assert np.abs(out.data - self.orig).max() < self.tol
                assert out.data.is_virtual == parallel_store
                if parallel_store:
                    nfiles = len(glob(os.path.join(myfilter.virtualDatasetDir, "*.h5")))
                    if chan_per_worker is None:
                        assert nfiles == self.nTrials
                    else:
                        assert nfiles == self.nFiles
        
                myfilter = LowPassFilter(self.b, self.a)
                myfilter.initialize(self.equidata, 
                                    chan_per_worker=chan_per_worker,
                                    keeptrials=False)
                out = AnalogData(dimord=AnalogData._defaultDimord)
                myfilter.compute(self.equidata, out, parallel=True, parallel_store=parallel_store)
                assert np.abs(out.data - self.orig[:self.t.size, :]).max() < self.tol
                assert out.data.is_virtual == False
        client.close()
    
    @skip_without_dask
    def test_parallel_nonequidistant(self, testcluster):
        client = dd.Client(testcluster)
        for overlapping in [False, True]:
            nonequidata = generate_artifical_data(nTrials=self.nTrials,
                                                    nChannels=self.nChannels,
                                                    equidistant=False,
                                                    overlapping=overlapping,
                                                    inmemory=False)
            for parallel_store in [True, False]:
                for chan_per_worker in [None, self.chanPerWrkr]:
                    out = AnalogData(dimord=AnalogData._defaultDimord)
                    myfilter = LowPassFilter(self.b, self.a)
                    myfilter.initialize(nonequidata, chan_per_worker=chan_per_worker)
                    myfilter.compute(nonequidata, out, parallel=True, parallel_store=parallel_store)
                    assert out.data.shape[0] == np.diff(nonequidata.sampleinfo).sum()
                    for tk, trl in enumerate(out.trials):
                        assert trl.shape[0] == np.diff(nonequidata.sampleinfo[tk, :])
                    assert out.data.is_virtual == parallel_store
                    if parallel_store:
                        nfiles = len(glob(os.path.join(myfilter.virtualDatasetDir, "*.h5")))
                        if chan_per_worker is None:
                            assert nfiles == self.nTrials
                        else:
                            assert nfiles == self.nFiles
        client.close()

    @skip_without_dask
    def test_parallel_saveload(self, testcluster):
        client = dd.Client(testcluster)
        for parallel_store in [True, False]:
            myfilter = LowPassFilter(self.b, self.a)
            myfilter.initialize(self.equidata)
            out = AnalogData(dimord=AnalogData._defaultDimord)
            myfilter.compute(self.equidata, out, parallel=True, parallel_store=parallel_store, 
                             log_dict={"a": self.a, "b": self.b})
            
            assert set(["a", "b"]) == set(out.cfg.keys())
            assert np.array_equal(out.cfg["a"], self.a)
            assert np.array_equal(out.cfg["b"], self.b)
            assert "lowpass" in out._log
            
            with tempfile.TemporaryDirectory() as tdir:
                fname = os.path.join(tdir, "dummy")
                out.save(fname)
                dummy = load(fname)
                assert "a" in dummy.cfg.keys()
                assert "b" in dummy.cfg.keys()
                assert np.array_equal(dummy.cfg["a"], self.a)
                assert np.array_equal(dummy.cfg["b"], self.b)
                assert np.abs(dummy.data - self.orig).max() < self.tol
                assert not out.data.is_virtual
                assert out.filename == dummy.filename
                del out, dummy
        client.close()
