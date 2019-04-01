# -*- coding: utf-8 -*-
#
# Test proper functionality of SyNCoPy DiscreteData-type classes
# 
# Created: 2019-03-21 15:44:03
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-04-01 16:44:20>

import os
import tempfile
import pytest
import time
import numpy as np
from syncopy.datatype import AnalogData, SpikeData, EventData
from syncopy.utils import SPYValueError, SPYTypeError

class TestSpikeData(object):

    # Allocate test-dataset
    nc = 10
    ns = 30
    nd = 50
    seed = np.random.RandomState(13)
    data = np.vstack([seed.choice(ns, size=nd),
                      seed.choice(nc, size=nd),
                      seed.choice(int(nc/2), size=nd)]).T
    data2 = data.copy()
    data2[:, -1] = data[:, 0]
    data2[:, 0] = data[:, -1]
    trl = np.vstack([np.arange(0, ns, 5),
                     np.arange(5, ns + 5, 5),
                     np.ones((int(ns/5), )),
                     np.ones((int(ns/5), )) * np.pi]).T
    num_smp = np.unique(data[:, 0]).size
    num_chn = data[:, 1].max() + 1
    num_unt = data[:, 2].max() + 1

    def test_empty(self):
        dummy = SpikeData()
        assert len(dummy.cfg) == 0
        assert dummy.dimord == ["sample", "channel", "unit"]
        for attr in ["channel", "data", "sampleinfo", "samplerate", \
                     "trialid", "trialinfo", "unit"]:
            assert getattr(dummy, attr) is None
        with pytest.raises(SPYTypeError):
            SpikeData({})

    def test_nparray(self):
        dummy = SpikeData(self.data)
        assert dummy.channel.size == self.num_chn
        assert dummy.sample.size == self.num_smp
        assert dummy.unit.size == self.num_unt
        assert (dummy.sampleinfo == [0, self.data[:, 0].max()]).min()
        assert dummy.trialinfo.shape == (1, 0)
        assert dummy._filename is None
        assert np.array_equal(dummy.data, self.data)
        
        # wrong shape for data-type
        with pytest.raises(SPYValueError):
            SpikeData(np.ones((3,)))

    def test_trialretrieval(self):
        # test ``_get_trial`` with NumPy array: regular order
        dummy = SpikeData(self.data, trialdefinition=self.trl)
        smp = self.data[:, 0]
        for trlno, start in enumerate(range(0, self.ns, 5)):
            idx = np.intersect1d(np.where(smp >= start)[0],
                                 np.where(smp < start + 5)[0])
            trl_ref = self.data[idx, ...]
            assert np.array_equal(dummy._get_trial(trlno), trl_ref)

        # test ``_get_trial`` with NumPy array: swapped dimensions
        dummy = SpikeData(self.data2, trialdefinition=self.trl,
                           dimord=["unit", "channel", "sample"])
        smp = self.data2[:, -1]
        for trlno, start in enumerate(range(0, self.ns, 5)):
            idx = np.intersect1d(np.where(smp >= start)[0],
                                 np.where(smp < start + 5)[0])
            trl_ref = self.data2[idx, ...]
            assert np.array_equal(dummy._get_trial(trlno), trl_ref)
            
    def test_saveload(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "dummy")
            
            # basic but most important: ensure object integrity is preserved
            dummy = SpikeData(self.data)
            dummy.save(fname)
            dummy2 = SpikeData(fname)
            for attr in ["channel", "data", "dimord", "sampleinfo", \
                         "samplerate", "trialinfo", "unit"]:
                assert np.array_equal(getattr(dummy, attr), getattr(dummy2, attr))
            
            # newer files must be loaded from existing "dummy.spy" folder
            # (enforce one second pause to prevent race-condition)
            time.sleep(1)
            dummy.samplerate = 20
            dummy.save(fname)
            dummy2 = SpikeData(filename=fname)
            assert dummy2.samplerate == 20

            # ensure trialdefinition is saved and loaded correctly
            dummy = SpikeData(self.data, trialdefinition=self.trl)
            dummy.save(fname)
            dummy2 = SpikeData(fname)
            assert np.array_equal(dummy.sampleinfo, dummy2.sampleinfo)
            assert np.array_equal(dummy.t0, dummy2.t0)
            assert np.array_equal(dummy.trialinfo, dummy2.trialinfo)

            # swap dimensions and ensure `dimord` is preserved
            dummy = SpikeData(self.data, dimord=["unit", "channel", "sample"])
            dummy.save(fname + "_dimswap")
            dummy2 = SpikeData(fname + "_dimswap")
            assert dummy2.dimord == dummy.dimord
            assert dummy2.unit.size == self.num_smp # swapped
            assert dummy2.data.shape == dummy.data.shape
            

class TestEventData(object):

    # Allocate test-datasets
    nc = 10
    ns = 30
    data = np.vstack([np.arange(0, ns, 5),
                      np.zeros((int(ns/5), ))]).T
    data[1::2, 1] = 1 
    data2 = data.copy()
    data2[:, -1] = data[:, 0]
    data2[:, 0] = data[:, -1]
    trl = np.vstack([np.arange(0, ns, 5),
                     np.arange(5, ns + 5, 5),
                     np.ones((int(ns/5), )),
                     np.ones((int(ns/5), )) * np.pi]).T
    num_smp = np.unique(data[:, 0]).size
    num_evt = np.unique(data[:, 1]).size

    adata = np.arange(1, nc*ns + 1).reshape(nc, ns)

    def test_empty(self):
        dummy = EventData()
        assert len(dummy.cfg) == 0
        assert dummy.dimord == ["sample", "eventid"]
        for attr in ["data", "sampleinfo", "samplerate", "trialid", "trialinfo"]:
            assert getattr(dummy, attr) is None
        with pytest.raises(SPYTypeError):
            EventData({})
    
    def test_nparray(self):
        dummy = EventData(self.data)
        assert dummy.eventid.size == self.num_evt
        assert dummy.sample.size == self.num_smp
        assert (dummy.sampleinfo == [0, self.data[:, 0].max()]).min()
        assert dummy.trialinfo.shape == (1, 0)
        assert dummy._filename is None
        assert np.array_equal(dummy.data, self.data)
        
        # wrong shape for data-type
        with pytest.raises(SPYValueError):
            EventData(np.ones((3,)))

    def test_trialretrieval(self):
        # test ``_get_trial`` with NumPy array: regular order
        dummy = EventData(self.data, trialdefinition=self.trl)
        smp = self.data[:, 0]
        for trlno, start in enumerate(range(0, self.ns, 5)):
            idx = np.intersect1d(np.where(smp >= start)[0],
                                 np.where(smp < start + 5)[0])
            trl_ref = self.data[idx, ...]
            assert np.array_equal(dummy._get_trial(trlno), trl_ref)

        # test ``_get_trial`` with NumPy array: swapped dimensions
        dummy = EventData(self.data2, trialdefinition=self.trl,
                           dimord=["eventid", "sample"])
        smp = self.data2[:, -1]
        for trlno, start in enumerate(range(0, self.ns, 5)):
            idx = np.intersect1d(np.where(smp >= start)[0],
                                 np.where(smp < start + 5)[0])
            trl_ref = self.data2[idx, ...]
            assert np.array_equal(dummy._get_trial(trlno), trl_ref)
            
    def test_saveload(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "dummy")
            
            # basic but most important: ensure object integrity is preserved
            dummy = EventData(self.data)
            dummy.save(fname)
            dummy2 = EventData(fname)
            for attr in ["data", "dimord", "sampleinfo", "samplerate", "trialinfo"]:
                assert np.array_equal(getattr(dummy, attr), getattr(dummy2, attr))
            
            # newer files must be loaded from existing "dummy.spy" folder
            # (enforce one second pause to prevent race-condition)
            time.sleep(1)
            dummy.samplerate = 20
            dummy.save(fname)
            dummy2 = EventData(filename=fname)
            assert dummy2.samplerate == 20

            # ensure trialdefinition is saved and loaded correctly
            dummy = EventData(self.data, trialdefinition=self.trl)
            dummy.save(fname)
            dummy2 = EventData(fname)
            assert np.array_equal(dummy.sampleinfo, dummy2.sampleinfo)
            assert np.array_equal(dummy.t0, dummy2.t0)
            assert np.array_equal(dummy.trialinfo, dummy2.trialinfo)

            # swap dimensions and ensure `dimord` is preserved
            dummy = EventData(self.data, dimord=["eventid", "sample"])
            dummy.save(fname + "_dimswap")
            dummy2 = EventData(fname + "_dimswap")
            assert dummy2.dimord == dummy.dimord
            assert dummy2.eventid.size == self.num_smp # swapped
            assert dummy2.data.shape == dummy.data.shape

    def test_trialsetting(self):

        # Create sampleinfo w/ EventData vs. AnalogData samplerate
        sr_e = 2
        sr_a = 1
        pre = 2
        post = 1
        msk = self.data[:,1] == 1
        sinfo = np.vstack([self.data[msk,0]/sr_e - pre,
                           self.data[msk,0]/sr_e + post]).T
        sinfo_e = np.round(sinfo*sr_e).astype(int)
        sinfo_a = np.round(sinfo*sr_a).astype(int)

        # Compute sampleinfo w/pre, post and trigger
        evt_dummy = EventData(self.data, samplerate=sr_e, mode="r")
        evt_dummy.redefinetrial(pre=pre, post=post, trigger=1)
        assert np.array_equal(evt_dummy.sampleinfo, sinfo_e)

        # Compute sampleinfo w/ start/stop combination
        evt_dummy = EventData(self.data, samplerate=sr_e)
        evt.redefinetrial(start=0, stop=1)

        # Compute sampleinfo w/start array
        samples = np.arange(0, int(ns/3), 3)[1:]
        dappend = np.vstack([samples, np.full(samples.shape, 2)]).T
        data3 = np.vstack([self.data, dappend])
        idx = np.argsort(data3[:,0])
        data3 = data3[idx, :]
        evt_dummy = EventData(data3, samplerate=sr_e)
        evt_dummy.redefinetrial(start=[2,2,1], stop=[1,2,0])
        


        # Attach computed sampleinfo to AnalogData
        ang_dummy = AnalogData(self.adata, samplerate=sr_a)
        ang_dummy.redefinetrial(evt_dummy)
        assert np.array_equal(ang_dummy.sampleinfo, sinfo_a)

        # Compute and attach sampleinfo on the fly
        ang_dummy = AnalogData(self.adata, samplerate=sr_a)
        ang_dummy.redefinetrial(EventData(samplerate=sr_e), pre=pre, post=post, trigger=1)
        assert np.array_equal(ang_dummy.sampleinfo, sinfo_a)
        
    # FIXME: check both pre/start and/or post/stop being None
    # FIXME: try to define trials w/o samplerate set
    # FIXME: trigger error w/ only pre+post AND pre=x, post=None
    # FIXME: test clip_edges
