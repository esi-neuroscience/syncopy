# -*- coding: utf-8 -*-
#
# Test proper functionality of SyNCoPy BaseData class + helper
# 
# Created: 2019-03-19 10:43:22
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-04-15 14:09:55>

import os
import tempfile
import pytest
import numpy as np
from numpy.lib.format import open_memmap
from memory_profiler import memory_usage
from syncopy.datatype import AnalogData
import syncopy.datatype as swd
from syncopy.datatype.base_data import VirtualData
from syncopy.utils import SPYValueError, SPYTypeError

class TestVirtualData(object):

    # Allocate test-dataset
    nc = 5
    ns = 30
    data = np.arange(1, nc*ns + 1).reshape(ns, nc)

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
            for vk in range(2,6):
                vdata = VirtualData([dmap] * vk)
                assert vdata.dtype == dmap.dtype
                assert vdata.M == dmap.shape[0]
                assert vdata.N == vk*dmap.shape[1]

    def test_retrieval(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "vdat.npy")
            fname2 = os.path.join(tdir, "vdat2.npy")
            np.save(fname, self.data)
            np.save(fname2, self.data*2)
            dmap = open_memmap(fname)
            dmap2 = open_memmap(fname2)

            # ensure stacking is performed correctly
            vdata = VirtualData([dmap, dmap2])
            assert np.array_equal(vdata[:, :self.nc], self.data)
            assert np.array_equal(vdata[:, self.nc:], 2*self.data)
            assert np.array_equal(vdata[:, 0].flatten(), self.data[:, 0].flatten())
            assert np.array_equal(vdata[:, self.nc].flatten(), 2*self.data[:, 0].flatten())
            assert np.array_equal(vdata[0, :].flatten(), np.hstack([self.data[0, :], 2*self.data[0, :]]))
            vdata = VirtualData([dmap, dmap2, dmap])
            assert np.array_equal(vdata[:, :self.nc], self.data)
            assert np.array_equal(vdata[:, self.nc:2*self.nc], 2*self.data)
            assert np.array_equal(vdata[:, 2*self.nc:], self.data)
            assert np.array_equal(vdata[:, 0].flatten(), self.data[:, 0].flatten())
            assert np.array_equal(vdata[:, self.nc].flatten(), 2*self.data[:, 0].flatten())
            assert np.array_equal(vdata[0, :].flatten(), np.hstack([self.data[0, :], 2*self.data[0, :], self.data[0, :]]))

            # illegal indexing type
            with pytest.raises(SPYTypeError):
                vdata[{}, :]

            # queried indices out of bounds
            with pytest.raises(SPYValueError):
                vdata[:, self.nc*3]
            with pytest.raises(SPYValueError):
                vdata[self.ns*2, 0]

    def test_memory(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "vdat.npy")
            data = np.ones((1000, 5000)) # ca. 38.2 MB
            np.save(fname, data)
            del data
            dmap = open_memmap(fname)

            # allocation of VirtualData object must not consume memory
            mem = memory_usage()[0]
            vdata = VirtualData([dmap, dmap, dmap])
            assert np.abs(mem - memory_usage()[0]) < 1

            # test consistency and efficacy of clear method
            vd = vdata[:, :]
            vdata.clear()
            assert np.array_equal(vd, vdata[:,:])
            mem = memory_usage()[0]
            vdata.clear()
            assert (mem - memory_usage()[0]) > 100
            

# Test BaseData methods that work identically for all regular classes            
class TestBaseData(object):

    # Allocate test-datasets for AnalogData, SpectralData, SpikeData and EventData objects
    nc = 10
    ns = 30
    nt = 5
    nf = 15
    nd = 50
    data = {}
    trl = {}

    # Generate 2D array simulating an AnalogData array
    data["AnalogData"] = np.arange(1, nc*ns + 1).reshape(ns, nc)
    trl["AnalogData"] = np.vstack([np.arange(0, ns, 5),
                                   np.arange(5, ns + 5, 5),
                                   np.ones((int(ns/5), )),
                                   np.ones((int(ns/5), )) * np.pi]).T

    # Generate a 4D array simulating a SpectralData array
    data["SpectralData"] = np.arange(1, nc*ns*nt*nf + 1).reshape(ns, nt, nf, nc)
    trl["SpectralData"] = trl["AnalogData"]

    # Use a fixed random number generator seed to simulate a 2D SpikeData array
    seed = np.random.RandomState(13)
    data["SpikeData"] = np.vstack([seed.choice(ns, size=nd),
                                   seed.choice(nc, size=nd),
                                   seed.choice(int(nc/2), size=nd)]).T
    trl["SpikeData"] = trl["AnalogData"]

    # Use a simple binary trigger pattern to simulate EventData
    data["EventData"] = np.vstack([np.arange(0, ns, 5),
                                   np.zeros((int(ns/5), ))]).T
    data["EventData"][1::2, 1] = 1 
    trl["EventData"] = trl["AnalogData"]

    # Define data classes to be used in tests below
    classes = ["AnalogData", "SpectralData", "SpikeData", "EventData"]

    # Allocation to `data` property is tested with all members of `classes`
    def test_mmap(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "dummy.npy")
            
            for dclass in self.classes:
                # attempt allocation with random file
                with open(fname, "w") as f:
                    f.write("dummy")
                with pytest.raises(SPYValueError):
                    getattr(swd, dclass)(fname)
                
                # allocation with memmaped npy file
                np.save(fname, self.data[dclass])
                dummy = getattr(swd, dclass)(fname)
                assert np.array_equal(dummy.data, self.data[dclass])
                assert dummy._filename == fname

                # allocation using memmap directly
                mm = open_memmap(fname, mode="r")
                dummy = getattr(swd, dclass)(mm)
                assert np.array_equal(dummy.data, self.data[dclass])
                assert dummy.mode == "r"

                # attempt assigning data to read-only object
                with pytest.raises(SPYValueError):
                    dummy.data = self.data[dclass]

                # allocation using array + filename for target memmap
                dummy = getattr(swd, dclass)(self.data[dclass], fname)
                assert dummy._filename == fname
                assert np.array_equal(dummy.data, self.data[dclass])

                # attempt allocation using memmap of wrong shape
                np.save(fname, np.ones((self.nc,)))
                with pytest.raises(SPYValueError):
                    getattr(swd, dclass)(open_memmap(fname))

    # Assignment of trialdefinition array is tested with all members of `classes`
    def test_trialdef(self):
        for dclass in self.classes:
            dummy = getattr(swd, dclass)(self.data[dclass],
                                         trialdefinition=self.trl[dclass])
            assert np.array_equal(dummy.sampleinfo, self.trl[dclass][:, :2])
            assert np.array_equal(dummy.t0, self.trl[dclass][:, 2])
            assert np.array_equal(dummy.trialinfo.flatten(), self.trl[dclass][:, 3])

    # Test ``clear`` with `AnalogData` only - method is independent from concrete data object
    def test_clear(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "dummy.npy")
            data = np.ones((5000, 1000)) # ca. 38.2 MB
            np.save(fname, data)
            del data
            dmap = open_memmap(fname)
    
            # test consistency and efficacy of clear method
            dummy = AnalogData(dmap)
            data = np.array(dummy.data)
            dummy.clear()
            assert np.array_equal(data, dummy.data)
            mem = memory_usage()[0]
            dummy.clear()
            assert np.abs(mem - memory_usage()[0]) > 30
            
    # Test ``_gen_filename`` with `AnalogData` only - method is independent from concrete data object
    def test_filename(self):
        # ensure we're salting sufficiently to create at least `numf`
        # distinct pseudo-random filenames in `__storage__`
        numf = 1000
        dummy = AnalogData()
        fnames = []
        for k in range(numf):
            fnames.append(dummy._gen_filename())
        assert np.unique(fnames).size == numf
    
    # Object copying is tested with all members of `classes`
    def test_copy(self):

        # test shallow copy of data arrays (hashes must match up, since
        # shallow copies are views in memory)
        for dclass in self.classes:
            dummy = getattr(swd, dclass)(self.data[dclass],
                                         trialdefinition=self.trl[dclass])
            dummy2 = dummy.copy()
            assert dummy._filename == dummy2._filename
            assert hash(str(dummy.data)) == hash(str(dummy2.data))
            assert hash(str(dummy.sampleinfo)) == hash(str(dummy2.sampleinfo))
            assert hash(str(dummy.t0)) == hash(str(dummy2.t0))
            assert hash(str(dummy.trialinfo)) == hash(str(dummy2.trialinfo))
    
        # test shallow + deep copies of memmaps
        with tempfile.TemporaryDirectory() as tdir:
            for dclass in self.classes:
                fname = os.path.join(tdir, "dummy.npy")
                np.save(fname, self.data[dclass])
                mm = open_memmap(fname, mode="r")
                dummy = getattr(swd, dclass)(mm, trialdefinition=self.trl[dclass])

                # hash-matching of shallow-copied memmap
                dummy2 = dummy.copy()
                assert dummy._filename == dummy2._filename
                assert hash(str(dummy.data)) == hash(str(dummy2.data))
                assert hash(str(dummy.sampleinfo)) == hash(str(dummy2.sampleinfo))
                assert hash(str(dummy.t0)) == hash(str(dummy2.t0))
                assert hash(str(dummy.trialinfo)) == hash(str(dummy2.trialinfo))

                # test integrity of deep-copy
                dummy3 = dummy.copy(deep=True)
                assert dummy3._filename != dummy._filename
                assert np.array_equal(dummy.sampleinfo, dummy3.sampleinfo)
                assert np.array_equal(dummy.t0, dummy3.t0)
                assert np.array_equal(dummy.trialinfo, dummy3.trialinfo)
                assert np.array_equal(dummy.data, dummy3.data)
