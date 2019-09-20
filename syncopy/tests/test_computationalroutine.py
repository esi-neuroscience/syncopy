# -*- coding: utf-8 -*-
# 
# Test basic functionality of ComputationalRoutine class
# 
# Created: 2019-07-03 11:31:33
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-09-20 13:38:57>

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
from syncopy.datatype.base_data import Selector
from syncopy.io import load
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.shared.parsers import unwrap_io, unwrap_cfg
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
            trl = np.array([[0, out.data.shape[0], 0]], dtype=int)
        else:
            trl = np.zeros((len(self.trialList), 3), dtype=int)
            tidx = data.dimord.index("time")
            trial_lengths = [shp[tidx] for shp in self.targetShapes]
            cnt = 0
            for row, tlen in enumerate(trial_lengths):
                trl[row, 0] = cnt
                trl[row, 1] = cnt + tlen
                cnt += tlen
        out.sampleinfo = trl[:, :2]
        out._t0 = trl[:, 2]
        out.trialinfo = trl[:, 3:]
        out.samplerate = data.samplerate
        if data._selection is not None:
            chanSec = data._selection.channel
        else:
            chanSec = slice(None)
        out.channel = np.array(data.channel[chanSec])


@unwrap_cfg        
def filter_manager(data, b=None, a=None, 
                   out=None, select=None, chan_per_worker=None, keeptrials=True):
    myfilter = LowPassFilter(b, a)
    myfilter.initialize(data, chan_per_worker=chan_per_worker, keeptrials=keeptrials)
    newOut = False
    if out is None:
        newOut = True
        out = AnalogData()
    myfilter.compute(data, out)
    return out if newOut else None


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

    # Blow up the signal to have "channels" and "trials" and inflate the low-
    # frequency component accordingly for ad-hoc comparisons later
    sig = np.repeat(sig.reshape(-1, 1), axis=1, repeats=nChannels)
    sig = np.tile(sig, (nTrials, 1))
    orig = np.repeat(orig.reshape(-1, 1), axis=1, repeats=nChannels)
    orig = np.tile(orig, (nTrials, 1))

    # Construct artificial equidistant trial-definition array
    trl = np.zeros((nTrials, 3), dtype="int")
    for ntrial in range(nTrials):
        trl[ntrial, :] = np.array([ntrial * fs, (ntrial + 1) * fs, -500])
        # trl[ntrial, :] = np.array([ntrial * fs, (ntrial + 1) * fs, 0])

    # Create reference AnalogData objects with equidistant trial spacing
    sigdata = AnalogData(data=sig, samplerate=fs, trialdefinition=trl,
                         dimord=["time", "channel"])
    origdata = AnalogData(data=orig, samplerate=fs, trialdefinition=trl,
                          dimord=["time", "channel"])
    
    # For parallel computation w/concurrent writing: predict no. of generated 
    # HDF5 files that will make up virtual data-set in case of channel-chunking
    chanPerWrkr = 7
    nFiles = nTrials * (int(nChannels/chanPerWrkr) + int(nChannels % chanPerWrkr > 0))
    
    # Data selections to be tested (w/`sigdata` and artificial data generated below)
    sigdataSelections = [None, 
                         {"trials": [3, 1, 0],
                          "channels": ["channel" + str(i) for i in range(12, 28)][::-1]},
                         {"trials": [0, 1, 2],
                          "channels": range(0, int(nChannels / 2)),
                          "toilim": [-0.25, 0.25]}]
    
    seed = np.random.RandomState(13)
    artdataSelections = [None, 
                         {"trials": [3, 1, 0],
                          "channels": ["channel" + str(i) for i in range(12, 28)][::-1],
                          "toi": None},
                         {"trials": [0, 1, 2],
                          "channels": range(0, int(nChannels / 2)),
                          "toilim": [1.0, 1.25]}]
    
    # Error tolerances and respective quality metrics (depend on data selection!)
    tols = [1e-6, 1e-6, 1e-2]
    metrix = [np.max, np.max, np.mean]

    def test_sequential_equidistant(self):
        for sk, select in enumerate(self.sigdataSelections):
            sel = Selector(self.sigdata, select)
            
            out = filter_manager(self.sigdata, self.b, self.a, select=select)
            if select is None:
                reference = self.orig
            else:
                ref = []
                for tk, trlno in enumerate(sel.trials):
                    ref.append(self.origdata.trials[trlno][sel.time[tk], sel.channel])
                reference = np.vstack(ref)
            assert self.metrix[sk](np.abs(out.data - reference)) < self.tols[sk]
            
            # # FIXME: ensure pre-selection is equivalent to in-place selection
            # out_sel = filter_manager(self.equidata.selectdata(select), self.b, self.a)
            # assert np.array_equal(out.data, out_sel.data)
            
            out = filter_manager(self.sigdata, self.b, self.a, select=select, keeptrials=False)
            if select is None:
                reference = self.orig[:self.t.size, :]
            else:
                ref = np.zeros(out.trials[0].shape)
                for tk, trl in enumerate(sel.trials):
                    ref += self.origdata.trials[trl][sel.time[tk], sel.channel]
                reference = ref / len(sel.trials)
            assert self.metrix[sk](np.abs(out.data - reference)) < self.tols[sk]

            # # FIXME: ensure pre-selection is equivalent to in-place selection
            # out_sel = filter_manager(self.equidata.selectdata(select), self.b, self.a, keeptrials=False)
            # assert np.array_equal(out.data, out_sel.data)

    def test_sequential_nonequidistant(self):
        
        
        
        for overlapping in [False, True]:
            nonequidata = generate_artifical_data(nTrials=self.nTrials,
                                                  nChannels=self.nChannels,
                                                  equidistant=False,
                                                  overlapping=overlapping,
                                                  inmemory=False)
            
            # unsorted, w/repetitions
            toi = self.seed.choice(nonequidata.time[0], int(nonequidata.time[0].size))
            self.artdataSelections[1]["toi"] = toi
            
            for select in self.artdataSelections:
                sel = Selector(nonequidata, select)
                out = filter_manager(nonequidata, self.b, self.a, select=select)
                
                reference = 0
                for tk, trlno in enumerate(sel.trials):
                    reference += nonequidata.trials[trlno][sel.time[tk]].shape[0]
                # import pdb; pdb.set_trace()
                assert out.data.shape[0] == reference
                
                # # FIXME: ensure pre-selection is equivalent to in-place selection
                # out_sel = filter_manager(nonequidata.selectdata(select), self.b, self.a)
                # assert np.array_equal(out.data, out_sel.data)
            
    def test_sequential_saveload(self):
        myfilter = LowPassFilter(self.b, self.a)
        myfilter.initialize(self.sigdata)
        out = AnalogData()
        myfilter.compute(self.sigdata, out, log_dict={"a": self.a, "b": self.b})
        assert set(["a", "b"]) == set(out.cfg.keys())
        assert np.array_equal(out.cfg["a"], self.a)
        assert np.array_equal(out.cfg["b"], self.b)
        # FIXME: check out.channel and out.time!
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
                myfilter.initialize(self.sigdata, chan_per_worker=chan_per_worker)
                out = AnalogData()
                myfilter.compute(self.sigdata, out, parallel=True, parallel_store=parallel_store)
                assert np.abs(out.data - self.orig).max() < self.tol
                assert out.data.is_virtual == parallel_store
                if parallel_store:
                    nfiles = len(glob(os.path.join(myfilter.virtualDatasetDir, "*.h5")))
                    if chan_per_worker is None:
                        assert nfiles == self.nTrials
                    else:
                        assert nfiles == self.nFiles
        
                myfilter = LowPassFilter(self.b, self.a)
                myfilter.initialize(self.sigdata, 
                                    chan_per_worker=chan_per_worker,
                                    keeptrials=False)
                out = AnalogData()
                myfilter.compute(self.sigdata, out, parallel=True, parallel_store=parallel_store)
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
                    out = AnalogData()
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
            myfilter.initialize(self.sigdata)
            out = AnalogData()
            myfilter.compute(self.sigdata, out, parallel=True, parallel_store=parallel_store, 
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
