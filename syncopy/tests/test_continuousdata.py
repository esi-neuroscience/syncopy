# -*- coding: utf-8 -*-
#
# Test proper functionality of SyNCoPy ContinousData-type classes
# 
# Created: 2019-03-20 11:46:31
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-03-20 13:34:38>

import os
import tempfile
import pytest
import time
import numpy as np
from numpy.lib.format import open_memmap
from syncopy.datatype import AnalogData
from syncopy.datatype.base_data import VirtualData
from syncopy.utils import SPYValueError, SPYTypeError

class TestAnalogData(object):

    # Allocate test-dataset
    nc = 10
    ns = 30
    data = np.arange(1, nc*ns + 1).reshape(nc, ns)
    trl = np.vstack([np.arange(0, ns, 5),
                     np.arange(5, ns + 5, 5),
                     np.ones((int(ns/5), )),
                     np.ones((int(ns/5), )) * np.pi]).T

    def test_empty(self):
        dummy = AnalogData()
        assert len(dummy.cfg) == 0
        assert dummy.dimord == ["channel", "time"]
        for attr in ["channel", "data", "hdr", "sampleinfo", "samplerate", "trialinfo"]:
            assert getattr(dummy, attr) is None
        with pytest.raises(SPYTypeError):
            AnalogData({})

    def test_nparray(self):
        dummy = AnalogData(self.data)
        assert dummy.channel.size == self.nc
        assert (dummy.sampleinfo == [0, self.ns]).min()
        assert dummy.trialinfo.shape == (1, 0)
        assert dummy._filename is None
        assert np.array_equal(dummy.data, self.data)
        
        # wrong shape for data-type
        with pytest.raises(SPYValueError):
            AnalogData(np.ones((3,)))

    def test_virtualdata(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "dummy.npy")
            np.save(fname, self.data)
            dmap = open_memmap(fname, mode="r")
            vdata = VirtualData([dmap, dmap])
            dummy = AnalogData(vdata)
            assert dummy.channel.size == 2*self.nc
            assert len(dummy._filename) == 2
    
    def test_trialretrieval(self):
        # test ``_get_trial`` with NumPy array
        dummy = AnalogData(self.data, trialdefinition=self.trl)
        for trlno, start in enumerate(range(0, self.ns, 5)):
            trl_ref = self.data[:, start:start + 5]
            assert np.array_equal(dummy._get_trial(trlno), trl_ref)
            
        # test ``_copy_trial`` with memmap'ed data
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "dummy.npy")
            np.save(fname, self.data)
            mm = open_memmap(fname, mode="r")
            dummy = AnalogData(mm, trialdefinition=self.trl)
            for trlno, start in enumerate(range(0, self.ns, 5)):
                trl_ref = self.data[:, start:start + 5]
                trl_tmp = dummy._copy_trial(trlno,
                                            dummy._filename,
                                            dummy.dimord,
                                            dummy.sampleinfo,
                                            dummy.hdr)
                assert np.array_equal(trl_tmp, trl_ref)
            
    def test_saveload(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "dummy")
            
            # basic but most important: ensure object integrity is preserved
            dummy = AnalogData(self.data)
            dummy.save(fname)
            dummy2 = AnalogData(fname)
            for attr in ["channel", "data", "dimord", "sampleinfo", "samplerate", "trialinfo"]:
                assert np.array_equal(getattr(dummy, attr), getattr(dummy2, attr))
            
            # save object hosting VirtualData; preference must be given to
            # spy container over identically named npy file
            np.save(fname + ".npy", self.data)
            dmap = open_memmap(fname + ".npy", mode="r")
            vdata = VirtualData([dmap, dmap])
            dummy = AnalogData(vdata)
            dummy.save(fname)
            dummy2 = AnalogData(fname)
            assert dummy2.mode == "w"
            assert np.array_equal(dummy.data, vdata[:,:])
            
            # newer files must be loaded from existing "dummy.spy" folder
            # (enforce one second pause to prevent race-condition)
            time.sleep(1)
            dummy.samplerate = 20
            dummy.save(fname)
            dummy2 = AnalogData(filename=fname)
            assert dummy2.samplerate == 20

            # ensure trialdefinition is saved and loaded correctly
            dummy = AnalogData(self.data, trialdefinition=self.trl)
            dummy.save(fname + "_trl")
            dummy2 = AnalogData(fname + "_trl")
            assert np.array_equal(dummy.sampleinfo, dummy2.sampleinfo)
            assert np.array_equal(dummy.t0, dummy2.t0)
            assert np.array_equal(dummy.trialinfo, dummy2.trialinfo)

