# -*- coding: utf-8 -*-
# 
# Test proper functionality of SyNCoPy's `BaseData` class + helpers
# 
# Created: 2019-03-19 10:43:22
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-11-11 15:23:21>

import os
import tempfile
import h5py
import time
import pytest
import numpy as np
from numpy.lib.format import open_memmap
from memory_profiler import memory_usage
from syncopy.datatype import AnalogData
import syncopy.datatype as spd
from syncopy.datatype.base_data import VirtualData
from syncopy.shared.errors import SPYValueError, SPYTypeError
from syncopy.tests.misc import is_win_vm, is_slurm_node

# Construct decorators for skipping certain tests
skip_in_vm = pytest.mark.skipif(is_win_vm(), reason="running in Win VM")
skip_in_slurm = pytest.mark.skipif(is_slurm_node(), reason="running on cluster node")


class TestVirtualData():

    # Allocate test-dataset
    nChannels = 5
    nSamples = 30
    data = np.arange(1, nChannels * nSamples + 1).reshape(nSamples, nChannels)

    def test_alloc(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "vdat")
            np.save(fname, self.data)
            dmap = open_memmap(fname + ".npy")

            # illegal type
            with pytest.raises(SPYTypeError):
                VirtualData({})

            # 2darray expected
            d3 = np.ones((2, 3, 4))
            np.save(fname + "3", d3)
            d3map = open_memmap(fname + "3.npy")
            with pytest.raises(SPYValueError):
                VirtualData([d3map])

            # rows/cols don't match up
            with pytest.raises(SPYValueError):
                VirtualData([dmap, dmap.T])

            # check consistency of VirtualData object
            for vk in range(2, 6):
                vdata = VirtualData([dmap] * vk)
                assert vdata.dtype == dmap.dtype
                assert vdata.M == dmap.shape[0]
                assert vdata.N == vk * dmap.shape[1]

            # Delete all open references to file objects b4 closing tmp dir
            del dmap, vdata, d3map

    def test_retrieval(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "vdat.npy")
            fname2 = os.path.join(tdir, "vdat2.npy")
            np.save(fname, self.data)
            np.save(fname2, self.data * 2)
            dmap = open_memmap(fname)
            dmap2 = open_memmap(fname2)

            # ensure stacking is performed correctly
            vdata = VirtualData([dmap, dmap2])
            assert np.array_equal(vdata[:, :self.nChannels], self.data)
            assert np.array_equal(vdata[:, self.nChannels:], 2 * self.data)
            assert np.array_equal(vdata[:, 0].flatten(), self.data[:, 0].flatten())
            assert np.array_equal(vdata[:, self.nChannels].flatten(), 2 * self.data[:, 0].flatten())
            assert np.array_equal(vdata[0, :].flatten(),
                                  np.hstack([self.data[0, :], 2 * self.data[0, :]]))
            vdata = VirtualData([dmap, dmap2, dmap])
            assert np.array_equal(vdata[:, :self.nChannels], self.data)
            assert np.array_equal(vdata[:, self.nChannels:2 * self.nChannels], 2 * self.data)
            assert np.array_equal(vdata[:, 2 * self.nChannels:], self.data)
            assert np.array_equal(vdata[:, 0].flatten(), self.data[:, 0].flatten())
            assert np.array_equal(vdata[:, self.nChannels].flatten(),
                                  2 * self.data[:, 0].flatten())
            assert np.array_equal(vdata[0, :].flatten(),
                                  np.hstack([self.data[0, :], 2 * self.data[0, :], self.data[0, :]]))

            # illegal indexing type
            with pytest.raises(SPYTypeError):
                vdata[{}, :]

            # queried indices out of bounds
            with pytest.raises(SPYValueError):
                vdata[:, self.nChannels * 3]
            with pytest.raises(SPYValueError):
                vdata[self.nSamples * 2, 0]

            # Delete all open references to file objects b4 closing tmp dir
            del dmap, dmap2, vdata

    @skip_in_vm
    @skip_in_slurm
    def test_memory(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "vdat.npy")
            data = np.ones((1000, 5000))  # ca. 38.2 MB
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
            assert np.array_equal(vd, vdata[:, :])
            mem = memory_usage()[0]
            vdata.clear()
            assert (mem - memory_usage()[0]) > 100

            # Delete all open references to file objects b4 closing tmp dir
            del dmap, vdata


# Test BaseData methods that work identically for all regular classes
class TestBaseData():

    # Allocate test-datasets for AnalogData, SpectralData, SpikeData and EventData objects
    nChannels = 10
    nSamples = 30
    nTrials = 5
    nFreqs = 15
    nSpikes = 50
    data = {}
    trl = {}

    # Generate 2D array simulating an AnalogData array
    data["AnalogData"] = np.arange(1, nChannels * nSamples + 1).reshape(nSamples, nChannels)
    trl["AnalogData"] = np.vstack([np.arange(0, nSamples, 5),
                                   np.arange(5, nSamples + 5, 5),
                                   np.ones((int(nSamples / 5), )),
                                   np.ones((int(nSamples / 5), )) * np.pi]).T

    # Generate a 4D array simulating a SpectralData array
    data["SpectralData"] = np.arange(1, nChannels * nSamples * nTrials * nFreqs + 1).reshape(nSamples, nTrials, nFreqs, nChannels)
    trl["SpectralData"] = trl["AnalogData"]

    # Use a fixed random number generator seed to simulate a 2D SpikeData array
    seed = np.random.RandomState(13)
    data["SpikeData"] = np.vstack([seed.choice(nSamples, size=nSpikes),
                                   seed.choice(nChannels, size=nSpikes),
                                   seed.choice(int(nChannels/2), size=nSpikes)]).T
    trl["SpikeData"] = trl["AnalogData"]

    # Use a simple binary trigger pattern to simulate EventData
    data["EventData"] = np.vstack([np.arange(0, nSamples, 5),
                                   np.zeros((int(nSamples / 5), ))]).T
    data["EventData"][1::2, 1] = 1
    trl["EventData"] = trl["AnalogData"]

    # Define data classes to be used in tests below
    classes = ["AnalogData", "SpectralData", "SpikeData", "EventData"]

    # Allocation to `data` property is tested with all members of `classes`
    def test_data_alloc(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "dummy.npy")
            hname = os.path.join(tdir, "dummy.h5")

            for dclass in self.classes:
                # attempt allocation with random file
                with open(fname, "w") as f:
                    f.write("dummy")
                # with pytest.raises(SPYValueError):
                #     getattr(spd, dclass)(fname)

                # allocation with HDF5 file
                h5f = h5py.File(hname, mode="w")
                h5f.create_dataset("dummy", data=self.data[dclass])
                h5f.close()
                
                # dummy = getattr(spd, dclass)(filename=hname)
                # assert np.array_equal(dummy.data, self.data[dclass])
                # assert dummy.filename == hname
                # del dummy

                # allocation using HDF5 dataset directly
                dset = h5py.File(hname, mode="r+")["dummy"]
                dummy = getattr(spd, dclass)(data=dset)
                assert np.array_equal(dummy.data, self.data[dclass])
                assert dummy.mode == "r+", dummy.data.file.mode
                del dummy               
                
                # # allocation with memmaped npy file
                # np.save(fname, self.data[dclass])
                # dummy = getattr(spd, dclass)(filename=fname)
                # assert np.array_equal(dummy.data, self.data[dclass])
                # assert dummy.filename == fname
                # del dummy

                # allocation using memmap directly
                np.save(fname, self.data[dclass])
                mm = open_memmap(fname, mode="r")
                dummy = getattr(spd, dclass)(data=mm)
                assert np.array_equal(dummy.data, self.data[dclass])
                assert dummy.mode == "r"

                # attempt assigning data to read-only object
                with pytest.raises(SPYValueError):
                    dummy.data = self.data[dclass]

                # allocation using array + filename
                del dummy, mm
                dummy = getattr(spd, dclass)(data=self.data[dclass], filename=fname)
                assert dummy.filename == fname
                assert np.array_equal(dummy.data, self.data[dclass])
                del dummy

                # attempt allocation using HDF5 dataset of wrong shape
                h5f = h5py.File(hname, mode="r+")
                del h5f["dummy"]
                dset = h5f.create_dataset("dummy", data=np.ones((self.nChannels,)))
                with pytest.raises(SPYValueError):
                    getattr(spd, dclass)(data=dset)

                # # attempt allocation using illegal HDF5 container
                del h5f["dummy"]
                h5f.create_dataset("dummy1", data=self.data[dclass])
                # FIXME: unused: h5f.create_dataset("dummy2", data=self.data[dclass])
                h5f.close()
                # with pytest.raises(SPYValueError):
                #     getattr(spd, dclass)(hname)

                # allocate with valid dataset of "illegal" container
                dset = h5py.File(hname, mode="r")["dummy1"]
                dummy = getattr(spd, dclass)(data=dset, filename=fname)

                # attempt data access after backing file of dataset has been closed
                dset.file.close()
                with pytest.raises(SPYValueError):
                    dummy.data[0, ...]

                # attempt allocation with HDF5 dataset of closed container
                with pytest.raises(SPYValueError):
                    getattr(spd, dclass)(data=dset)

                # attempt allocation using memmap of wrong shape
                np.save(fname, np.ones((self.nChannels,)))
                with pytest.raises(SPYValueError):
                    getattr(spd, dclass)(data=open_memmap(fname))
                
            time.sleep(0.01)
            del dummy

    # Assignment of trialdefinition array is tested with all members of `classes`
    def test_trialdef(self):
        for dclass in self.classes:
            dummy = getattr(spd, dclass)(self.data[dclass],
                                         trialdefinition=self.trl[dclass])
            assert np.array_equal(dummy.sampleinfo, self.trl[dclass][:, :2])
            assert np.array_equal(dummy._t0, self.trl[dclass][:, 2])
            assert np.array_equal(dummy.trialinfo.flatten(), self.trl[dclass][:, 3])

    # Test ``clear`` with `AnalogData` only - method is independent from concrete data object
    @skip_in_vm
    def test_clear(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "dummy.npy")
            data = np.ones((5000, 1000))  # ca. 38.2 MB
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
            time.sleep(1)
            assert np.abs(mem - memory_usage()[0]) > 30

            # Delete all open references to file objects b4 closing tmp dir
            del dmap, dummy

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
            dummy = getattr(spd, dclass)(self.data[dclass],
                                         trialdefinition=self.trl[dclass])
            dummy2 = dummy.copy()
            assert dummy.filename == dummy2.filename
            assert hash(str(dummy.data)) == hash(str(dummy2.data))
            assert hash(str(dummy.sampleinfo)) == hash(str(dummy2.sampleinfo))
            assert hash(str(dummy._t0)) == hash(str(dummy2._t0))
            assert hash(str(dummy.trialinfo)) == hash(str(dummy2.trialinfo))

        # test shallow + deep copies of memmaps + HDF5 containers
        with tempfile.TemporaryDirectory() as tdir:
            for dclass in self.classes:
                fname = os.path.join(tdir, "dummy.npy")
                hname = os.path.join(tdir, "dummy.h5")
                np.save(fname, self.data[dclass])
                h5f = h5py.File(hname, mode="w")
                h5f.create_dataset("dummy", data=self.data[dclass])
                h5f.close()
                mm = open_memmap(fname, mode="r")

                # hash-matching of shallow-copied memmap
                dummy = getattr(spd, dclass)(data=mm, trialdefinition=self.trl[dclass])
                dummy2 = dummy.copy()
                assert dummy.filename == dummy2.filename
                assert hash(str(dummy.data)) == hash(str(dummy2.data))
                assert hash(str(dummy.sampleinfo)) == hash(str(dummy2.sampleinfo))
                assert hash(str(dummy._t0)) == hash(str(dummy2._t0))
                assert hash(str(dummy.trialinfo)) == hash(str(dummy2.trialinfo))

                # test integrity of deep-copy
                dummy3 = dummy.copy(deep=True)
                assert dummy3.filename != dummy.filename
                assert np.array_equal(dummy.trialdefinition, dummy3.trialdefinition)
                assert np.array_equal(dummy.data, dummy3.data)
                assert np.array_equal(dummy._t0, dummy3._t0)
                assert np.array_equal(dummy.trialinfo, dummy3.trialinfo)
                assert np.array_equal(dummy.sampleinfo, dummy3.sampleinfo)

                # hash-matching of shallow-copied HDF5 dataset
                dummy = getattr(spd, dclass)(data=h5py.File(hname)["dummy"],
                                             trialdefinition=self.trl[dclass])
                dummy2 = dummy.copy()
                assert dummy.filename == dummy2.filename
                assert hash(str(dummy.data)) == hash(str(dummy2.data))
                assert hash(str(dummy.sampleinfo)) == hash(str(dummy2.sampleinfo))
                assert hash(str(dummy._t0)) == hash(str(dummy2._t0))
                assert hash(str(dummy.trialinfo)) == hash(str(dummy2.trialinfo))

                # test integrity of deep-copy
                dummy3 = dummy.copy(deep=True)
                assert dummy3.filename != dummy.filename
                assert np.array_equal(dummy.sampleinfo, dummy3.sampleinfo)
                assert np.array_equal(dummy._t0, dummy3._t0)
                assert np.array_equal(dummy.trialinfo, dummy3.trialinfo)
                assert np.array_equal(dummy.data, dummy3.data)

                # Delete all open references to file objects b4 closing tmp dir
                del mm, dummy, dummy2, dummy3
                time.sleep(0.01)

                # remove container for next round
                os.unlink(hname)
