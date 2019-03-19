# -*- coding: utf-8 -*-
#
# Test proper functionality of SyNCoPy data-types
# 
# Created: 2019-03-19 10:43:22
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-03-19 17:07:00>

import os
import tempfile
import time
import pytest
import numpy as np
from numpy.lib.format import open_memmap
from memory_profiler import memory_usage
from syncopy.datatype import AnalogData, SpectralData, SpikeData
from syncopy.datatype.data_classes import VirtualData
from syncopy.utils import SPYValueError, SPYTypeError

class TestVirtualData(object):

    # Allocate test-dataset
    nc = 5
    ns = 30
    data = np.arange(1, nc*ns + 1).reshape(nc, ns)

    def test_alloc(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "vdat")
            np.save(fname, self.data)
            dmap = open_memmap(fname + ".npy")
        
            # illegal type
            with pytest.raises(SPYTypeError):
                VirtualData({})

            # 2darray expected
            d3 = np.ones((2,3,4))
            np.save(fname + "3", d3)
            d3map = open_memmap(fname + "3.npy")
            with pytest.raises(SPYValueError):
                VirtualData([d3map])

            # rows/cols don't match up
            with pytest.raises(SPYValueError):
                VirtualData([dmap, dmap.T])

            # check consistency of VirtualData object
            vdata = VirtualData([dmap, dmap])
            assert vdata.dtype == dmap.dtype
            assert vdata.M == 2*dmap.shape[0]
            assert vdata.N == dmap.shape[1]

    def test_retrieval(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "vdat.npy")
            np.save(fname, self.data)
            dmap = open_memmap(fname)

            # ensure stacking is performed correctly
            vdata = VirtualData([dmap, dmap])
            assert np.array_equal(vdata[:self.nc, :], self.data)
            assert np.array_equal(vdata[self.nc:, :], self.data)
            assert np.array_equal(vdata[0, :].flatten(), self.data[0, :].flatten())
            assert np.array_equal(vdata[self.nc, :].flatten(), self.data[0, :].flatten())
            assert np.array_equal(vdata[:, 0].flatten(), np.hstack([self.data[:, 0], self.data[:, 0]]))

            # illegal indexing type
            with pytest.raises(SPYTypeError):
                vdata[{}, :]

            # queried indices out of bounds
            with pytest.raises(SPYValueError):
                vdata[self.nc*3, :]
            with pytest.raises(SPYValueError):
                vdata[0, self.ns*2]

    def test_memory(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "vdat.npy")
            data = np.ones((5000, 1000)) # ca. 38.2 MB
            np.save(fname, data)
            del data
            dmap = open_memmap(fname)

            # allocation of VirtualData object must not consume memory
            mem = memory_usage()[0]
            vdata = VirtualData([dmap, dmap])
            assert np.abs(mem - memory_usage()[0]) < 1

            # test consistency and efficacy of clear method
            vd = vdata[:, :]
            vdata.clear()
            assert np.array_equal(vd, vdata[:,:])
            mem = memory_usage()[0]
            vdata.clear()
            assert np.abs(mem - memory_usage()[0]) > 30
            

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

    def test_mmap(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "dummy.npy")
            
            # attempt allocation with random file
            with open(fname, "w") as f:
                f.write("dummy")
            with pytest.raises(SPYValueError):
                AnalogData(fname)
                
            # allocation with memmaped npy file
            np.save(fname, self.data)
            dummy = AnalogData(fname)
            assert np.array_equal(dummy.data, self.data)
            assert dummy._filename == fname
            
            # allocation using memmap directly
            mm = open_memmap(fname, mode="r")
            dummy = AnalogData(mm)
            assert np.array_equal(dummy.data, self.data)
            assert dummy.mode == "r"
            
            # attempt assigning data to read-only object
            with pytest.raises(SPYValueError):
                dummy.data = self.data

            # allocation using array + filename for target memmap
            dummy = AnalogData(self.data, fname)
            assert dummy._filename == fname
            assert np.array_equal(dummy.data, self.data)
            
            # attempt allocation using memmap of wrong shape
            np.save(fname, np.ones((self.nc,)))
            with pytest.raises(SPYValueError):
                AnalogData(open_memmap(fname))

    def test_virtualdata(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "dummy.npy")
            np.save(fname, self.data)
            dmap = open_memmap(fname, mode="r")
            vdata = VirtualData([dmap, dmap])
            dummy = AnalogData(vdata)
            assert dummy.channel.size == 2*self.nc
            assert len(dummy._filename) == 2

    def test_trialdef(self):
        dummy = AnalogData(self.data, trialdefinition=self.trl)
        assert np.array_equal(dummy.sampleinfo, self.trl[:, :2])
        assert np.array_equal(dummy.t0, self.trl[:, 2])
        assert np.array_equal(dummy.trialinfo.flatten(), self.trl[:, 3])

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

    
