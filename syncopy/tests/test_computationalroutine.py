# -*- coding: utf-8 -*-
# 
# Test basic functionality of ComputationalRoutine class
# 
# Created: 2019-07-03 11:31:33
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-09-25 17:07:11>

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
        if data._selection is not None:
            chanSec = data._selection.channel
            trl = data._selection.trialdefinition
        else:
            chanSec = slice(None)
            trl = np.zeros((len(self.trialList), 3), dtype=int)
            trial_lengths = np.diff(data.sampleinfo)
            cnt = 0
            for row, tlen in enumerate(trial_lengths):
                trl[row, 0] = cnt
                trl[row, 1] = cnt + tlen
                trl[row, 2] = data._t0[row]
                cnt += tlen
        if not self.keeptrials:
            trl = np.array([[0, out.data.shape[0], trl[0, 2]]], dtype=int)
        out.trialdefinition = trl
        out.samplerate = data.samplerate
        out.channel = np.array(data.channel[chanSec])


@unwrap_cfg        
def filter_manager(data, b=None, a=None, 
                   out=None, select=None, chan_per_worker=None, keeptrials=True,
                   parallel=False, parallel_store=None, log_dict=None):
    myfilter = LowPassFilter(b, a)
    myfilter.initialize(data, chan_per_worker=chan_per_worker, keeptrials=keeptrials)
    newOut = False
    if out is None:
        newOut = True
        out = AnalogData(dimord=AnalogData._defaultDimord)
    myfilter.compute(data, out, 
                     parallel=parallel, 
                     parallel_store=parallel_store, 
                     log_dict=log_dict)
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

    # Create reference AnalogData objects with equidistant trial spacing
    sigdata = AnalogData(data=sig, samplerate=fs, trialdefinition=trl,
                         dimord=["time", "channel"])
    origdata = AnalogData(data=orig, samplerate=fs, trialdefinition=trl,
                          dimord=["time", "channel"])
    
    # Set by-worker channel-count for channel-parallelization 
    chanPerWrkr = 7
    
    # Data selections to be tested w/`sigdata` 
    sigdataSelections = [None, 
                         {"trials": [3, 1, 0],
                          "channels": ["channel" + str(i) for i in range(12, 28)][::-1]},
                         {"trials": [0, 1, 2],
                          "channels": range(0, int(nChannels / 2)),
                          "toilim": [-0.25, 0.25]}]
    
    # Data selections to be tested w/`artdata` generated below (use fixed but arbitrary
    # random number seed to randomly select time-points for `toi` (with repetitions)
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
            
            # check correct signal filtering (especially wrt data-selection)
            if select is None:
                reference = self.orig
            else:
                ref = []
                for tk, trlno in enumerate(sel.trials):
                    ref.append(self.origdata.trials[trlno][sel.time[tk], sel.channel])
                    # check for correct time selection
                    assert np.array_equal(out.time[tk], self.sigdata.time[trlno][sel.time[tk]])
                reference = np.vstack(ref)
            assert self.metrix[sk](np.abs(out.data - reference)) < self.tols[sk]
            assert np.array_equal(out.channel, self.sigdata.channel[sel.channel])
            
            # # FIXME: ensure pre-selection is equivalent to in-place selection
            # out_sel = filter_manager(self.equidata.selectdata(select), self.b, self.a)
            # assert np.array_equal(out.data, out_sel.data)
            # assert np.array_equal(out.channel, out_sel.channel)
            # assert np.array_equal(out.time, out_sel.time)
            
            out = filter_manager(self.sigdata, self.b, self.a, select=select, keeptrials=False)

            # check correct signal filtering (especially wrt data-selection)
            if select is None:
                reference = self.orig[:self.t.size, :]
            else:
                ref = np.zeros(out.trials[0].shape)
                for tk, trlno in enumerate(sel.trials):
                    ref += self.origdata.trials[trlno][sel.time[tk], sel.channel]
                    # check for correct time selection (accounting for trial-averaging)
                    assert np.array_equal(out.time[0], self.sigdata.time[0][sel.time[0]])
                reference = ref / len(sel.trials)
            assert self.metrix[sk](np.abs(out.data - reference)) < self.tols[sk]
            assert np.array_equal(out.channel, self.sigdata.channel[sel.channel])
            
            # # FIXME: ensure pre-selection is equivalent to in-place selection
            # out_sel = filter_manager(self.equidata.selectdata(select), self.b, self.a, keeptrials=False)
            # assert np.array_equal(out.data, out_sel.data)
            # assert np.array_equal(out.channel, out_sel.channel)
            # assert np.array_equal(out.time, out_sel.time)

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

                # compare expected w/actual shape of computed data                
                reference = 0
                for tk, trlno in enumerate(sel.trials):
                    reference += nonequidata.trials[trlno][sel.time[tk]].shape[0]
                    # check for correct time selection
                    # FIXME: remove `if` below as soon as `time` prop for lists is fixed
                    if not isinstance(sel.time[0], list):
                        assert np.array_equal(out.time[tk], nonequidata.time[trlno][sel.time[tk]])
                        
                assert out.data.shape[0] == reference
                assert np.array_equal(out.channel, nonequidata.channel[sel.channel])
                
                # # FIXME: ensure pre-selection is equivalent to in-place selection
                # out_sel = filter_manager(nonequidata.selectdata(select), self.b, self.a)
                # assert np.array_equal(out.data, out_sel.data)
                # assert np.array_equal(out.channel, out_sel.channel)
                # assert np.array_equal(out.time, out_sel.time)
            
    def test_sequential_saveload(self):
        for sk, select in enumerate(self.sigdataSelections):
            sel = Selector(self.sigdata, select)
            out = filter_manager(self.sigdata, self.b, self.a, select=select,
                                 log_dict={"a": self.a, "b": self.b})
            
            assert set(["a", "b"]) == set(out.cfg.keys())
            assert np.array_equal(out.cfg["a"], self.a)
            assert np.array_equal(out.cfg["b"], self.b)
            assert len(out.trials) == len(sel.trials)
            assert "lowpass" in out._log

            # # FIXME: ensure pre-selection is equivalent to in-place selection
            # out_sel = filter_manager(self.sigdata.selectdata(select), self.b, self.a, 
            #                          log_dict={"a": self.a, "b": self.b})
            # assert set(["a", "b"]) == set(out_sel.cfg.keys())
            # assert np.array_equal(out.cfg["a"], out_sel.cfg["a"])
            # assert np.array_equal(out.cfg["b"], out.cfg["b"])
            # assert len(out.trials) == len(out_sel.trials)
            # assert "lowpass" in out_sel._log
            
            # save and re-load result, ensure nothing funky happens
            with tempfile.TemporaryDirectory() as tdir:
                fname = os.path.join(tdir, "dummy")
                out.save(fname)
                dummy = load(fname)
                assert "a" in dummy.cfg.keys()
                assert "b" in dummy.cfg.keys()
                assert np.array_equal(dummy.cfg["a"], self.a)
                assert np.array_equal(dummy.cfg["b"], self.b)
                assert out.filename == dummy.filename
                if select is None:
                    reference = self.orig
                else:
                    ref = []
                    for tk, trlno in enumerate(sel.trials):
                        ref.append(self.origdata.trials[trlno][sel.time[tk], sel.channel])
                        assert np.array_equal(dummy.time[tk], self.sigdata.time[trlno][sel.time[tk]])
                    reference = np.vstack(ref)
                assert self.metrix[sk](np.abs(dummy.data - reference)) < self.tols[sk]
                assert np.array_equal(dummy.channel, self.sigdata.channel[sel.channel])
                del dummy, out
                
                # # FIXME: ensure out_sel is written/read correctly
                # fname2 = os.path.join(tdir, "dummy2")
                # out_sel.save(fname2)
                # dummy2 = load(fname2)
                # assert "a" in dummy2.cfg.keys()
                # assert "b" in dummy2.cfg.keys()
                # assert np.array_equal(dummy2.cfg["a"], dummy.cfg["a"])
                # assert np.array_equal(dummy2.cfg["b"], dummy.cfg["b"])
                # assert np.array_equal(dummy.data, dummy2.data)
                # assert np.array_equal(dummy.channel, dummy2.channel)
                # assert np.array_equal(dummy.time, dummy2.time)
                # del dummy, dummy2, out, out_sel

    @skip_without_dask
    def test_parallel_equidistant(self, testcluster):
        client = dd.Client(testcluster)
        for parallel_store in [True, False]:
            for chan_per_worker in [None, self.chanPerWrkr]:
                for sk, select in enumerate(self.sigdataSelections):
                    # FIXME: remove as soon as channel-parallelization works w/channel selectors
                    if chan_per_worker is not None:
                        select = None
                    sel = Selector(self.sigdata, select)
                    out = filter_manager(self.sigdata, self.b, self.a, select=select,
                                         chan_per_worker=chan_per_worker, parallel=True,
                                         parallel_store=parallel_store)
                    
                    assert out.data.is_virtual == parallel_store

                    # check correct signal filtering (especially wrt data-selection)
                    if select is None:
                        reference = self.orig
                    else:
                        ref = []
                        for tk, trlno in enumerate(sel.trials):
                            ref.append(self.origdata.trials[trlno][sel.time[tk], sel.channel])
                            # check for correct time selection
                            assert np.array_equal(out.time[tk], self.sigdata.time[trlno][sel.time[tk]])
                        reference = np.vstack(ref)
                    assert self.metrix[sk](np.abs(out.data - reference)) < self.tols[sk]
                    assert np.array_equal(out.channel, self.sigdata.channel[sel.channel])

                    # ensure correct no. HDF5 files were generated for virtual data-set
                    if parallel_store:
                        nfiles = len(glob(os.path.join(os.path.splitext(out.filename)[0], "*.h5")))
                        if chan_per_worker is None:
                            assert nfiles == len(sel.trials)
                        else:
                            assert nfiles == len(sel.trials) * (int(out.channel.size / 
                                                                    chan_per_worker) + 
                                                                int(out.channel.size % chan_per_worker > 0))

                    # # FIXME: ensure pre-selection is equivalent to in-place selection
                    # out_sel = filter_manager(self.sigdata.selectdata(select), self.b, self.a,
                    #                          chan_per_worker=chan_per_worker, parallel=True,
                    #                          parallel_store=parallel_store)
                    # assert np.array_equal(out.data, out_sel.data)
                    # assert np.array_equal(out.channel, out_sel.channel)
                    # assert np.array_equal(out.time, out_sel.time)

                    out = filter_manager(self.sigdata, self.b, self.a, select=select,
                                         parallel=True, parallel_store=parallel_store,
                                         keeptrials=False)
                    
                    # check correct signal filtering (especially wrt data-selection)
                    if select is None:
                        reference = self.orig[:self.t.size, :]
                    else:
                        ref = np.zeros(out.trials[0].shape)
                        for tk, trlno in enumerate(sel.trials):
                            ref += self.origdata.trials[trlno][sel.time[tk], sel.channel]
                            # check for correct time selection (accounting for trial-averaging)
                            assert np.array_equal(out.time[0], self.sigdata.time[0][sel.time[0]])
                        reference = ref / len(sel.trials)
                    assert self.metrix[sk](np.abs(out.data - reference)) < self.tols[sk]
                    assert np.array_equal(out.channel, self.sigdata.channel[sel.channel])
                    assert out.data.is_virtual == False

                    # # FIXME: ensure pre-selection is equivalent to in-place selection
                    # out_sel = filter_manager(self.sigdata.selectdata(select), self.b, self.a,
                    #                          parallel=True, parallel_store=parallel_store, 
                    #                          keeptrials=False)
                    # assert np.array_equal(out.data, out_sel.data)
                    # assert np.array_equal(out.channel, out_sel.channel)
                    # assert np.array_equal(out.time, out_sel.time)

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
            
            # unsorted, w/repetitions
            toi = self.seed.choice(nonequidata.time[0], int(nonequidata.time[0].size))
            self.artdataSelections[1]["toi"] = toi
            
            for parallel_store in [True, False]:
                for chan_per_worker in [None, self.chanPerWrkr]:
                    for select in self.artdataSelections:
                        # FIXME: remove as soon as channel-parallelization works w/channel selectors
                        if chan_per_worker is not None:
                            select = None
                        sel = Selector(nonequidata, select)
                        out = filter_manager(nonequidata, self.b, self.a, select=select,
                                             chan_per_worker=chan_per_worker, parallel=True,
                                             parallel_store=parallel_store)

                        # compare expected w/actual shape of computed data                
                        reference = 0
                        for tk, trlno in enumerate(sel.trials):
                            reference += nonequidata.trials[trlno][sel.time[tk]].shape[0]
                            # check for correct time selection
                            # FIXME: remove `if` below as soon as `time` prop for lists is fixed
                            if not isinstance(sel.time[0], list):
                                assert np.array_equal(out.time[tk], nonequidata.time[trlno][sel.time[tk]])
                        assert out.data.shape[0] == reference
                        assert np.array_equal(out.channel, nonequidata.channel[sel.channel])
                        assert out.data.is_virtual == parallel_store
                        
                        if parallel_store:
                            nfiles = len(glob(os.path.join(os.path.splitext(out.filename)[0], "*.h5")))
                            if chan_per_worker is None:
                                assert nfiles == len(sel.trials)
                            else:
                                assert nfiles == len(sel.trials) * (int(out.channel.size / 
                                                                        chan_per_worker) + 
                                                                    int(out.channel.size % chan_per_worker > 0))

                        # # FIXME: ensure pre-selection is equivalent to in-place selection
                        # out_sel = filter_manager(nonequidata.selectdata(select), self.b, self.a,
                        #                          chan_per_worker=chan_per_worker, parallel=True,
                        #                          parallel_store=parallel_store)
                        # assert np.array_equal(out.data, out_sel.data)
                        # assert np.array_equal(out.channel, out_sel.channel)
                        # assert np.array_equal(out.time, out_sel.time)

        client.close()

    @skip_without_dask
    def test_parallel_saveload(self, testcluster):
        client = dd.Client(testcluster)
        for parallel_store in [True, False]:
            for sk, select in enumerate(self.sigdataSelections):
                sel = Selector(self.sigdata, select)
                out = filter_manager(self.sigdata, self.b, self.a, select=select,
                                     log_dict={"a": self.a, "b": self.b},
                                     parallel=True, parallel_store=parallel_store)
                
                assert set(["a", "b"]) == set(out.cfg.keys())
                assert np.array_equal(out.cfg["a"], self.a)
                assert np.array_equal(out.cfg["b"], self.b)
                assert len(out.trials) == len(sel.trials)
                assert "lowpass" in out._log

                # # FIXME: ensure pre-selection is equivalent to in-place selection
                # out_sel = filter_manager(self.sigdata.selectdata(select), self.b, self.a,
                #                          log_dict={"a": self.a, "b": self.b},
                #                          parallel=True, parallel_store=parallel_store)
                # assert set(["a", "b"]) == set(out_sel.cfg.keys())
                # assert np.array_equal(out.cfg["a"], out_sel.cfg["a"])
                # assert np.array_equal(out.cfg["b"], out.cfg["b"])
                # assert len(out.trials) == len(out_sel.trials)
                # assert "lowpass" in out_sel._log
                
                # save and re-load result, ensure nothing funky happens
                with tempfile.TemporaryDirectory() as tdir:
                    fname = os.path.join(tdir, "dummy")
                    out.save(fname)
                    dummy = load(fname)
                    assert "a" in dummy.cfg.keys()
                    assert "b" in dummy.cfg.keys()
                    assert np.array_equal(dummy.cfg["a"], self.a)
                    assert np.array_equal(dummy.cfg["b"], self.b)
                    assert out.filename == dummy.filename
                    assert not out.data.is_virtual
                    if select is None:
                        reference = self.orig
                    else:
                        ref = []
                        for tk, trlno in enumerate(sel.trials):
                            ref.append(self.origdata.trials[trlno][sel.time[tk], sel.channel])
                            assert np.array_equal(dummy.time[tk], self.sigdata.time[trlno][sel.time[tk]])
                        reference = np.vstack(ref)
                    assert self.metrix[sk](np.abs(dummy.data - reference)) < self.tols[sk]
                    assert np.array_equal(dummy.channel, self.sigdata.channel[sel.channel])
                    del dummy, out
                    
                    # # FIXME: ensure out_sel is written/read correctly
                    # fname2 = os.path.join(tdir, "dummy2")
                    # out_sel.save(fname2)
                    # dummy2 = load(fname2)
                    # assert "a" in dummy2.cfg.keys()
                    # assert "b" in dummy2.cfg.keys()
                    # assert np.array_equal(dummy2.cfg["a"], dummy.cfg["a"])
                    # assert np.array_equal(dummy2.cfg["b"], dummy.cfg["b"])
                    # assert np.array_equal(dummy.data, dummy2.data)
                    # assert np.array_equal(dummy.channel, dummy2.channel)
                    # assert np.array_equal(dummy.time, dummy2.time)
                    # assert not dummy2.data.is_virtual
                    # del dummy, dummy2, out, out_sel
                    
        client.close()
