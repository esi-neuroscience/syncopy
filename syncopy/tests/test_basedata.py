# -*- coding: utf-8 -*-
# 
# Test proper functionality of SyNCoPy BaseData class + helper
# 
# Created: 2019-03-19 10:43:22
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-09-25 16:53:04>

import os
import tempfile
import h5py
import time
import pytest
import numpy as np
from numpy.lib.format import open_memmap
from memory_profiler import memory_usage
from syncopy.datatype import AnalogData, SpectralData
import syncopy.datatype as spd
from syncopy.datatype.base_data import VirtualData, Selector
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

                # remove container for next round
                os.unlink(hname)


# Test Selector class
class TestSelector():

    # Set up "global" parameters for data objects to be tested (we only test
    # equidistant trials here)
    nChannels = 10
    nSamples = 30
    nTrials = 5
    lenTrial = int(nSamples / nTrials) - 1
    nFreqs = 15
    nSpikes = 50
    samplerate = 2.0
    data = {}
    trl = {}
    
    # Prepare selector results for valid/invalid selections
    selectDict = {}
    selectDict["channel"] = {"valid": (["channel03", "channel01"], 
                                       ["channel03", "channel01", "channel01", "channel02"],  # repetition
                                       ["channel01", "channel01", "channel02", "channel03"],  # preserve repetition
                                       [4, 2, 5], 
                                       [4, 2, 2, 5, 5],   # repetition
                                       [0, 0, 1, 2, 3],  # preserve repetition, don't convert to slice
                                       range(0, 3), 
                                       range(5, 8), 
                                       slice(None), 
                                       slice(0, 5), 
                                       slice(7, None), 
                                       slice(2, 8),
                                       slice(0, 10, 2),
                                       slice(-2, None),
                                       [0, 1, 2, 3],  # contiguous list...
                                       [2, 3, 5]),  # non-contiguous list...
                             "result": ([2, 0],
                                        [2, 0, 0, 1],
                                        [0, 0, 1, 2],
                                        [4, 2, 5],
                                        [4, 2, 2, 5, 5], 
                                        [0, 0, 1, 2, 3],
                                        slice(0, 3, 1),
                                        slice(5, 8, 1), 
                                        slice(None, None, 1), 
                                        slice(0, 5, 1),
                                        slice(7, None, 1), 
                                        slice(2, 8, 1),
                                        slice(0, 10, 2),
                                        slice(-2, None, 1),
                                        slice(0, 4, 1),  # ...gets converted to slice
                                        [2, 3, 5]),  # stays as is
                             "invalid": (["channel200", "channel400"],
                                         ["invalid"],
                                         "wrongtype",
                                         range(0, 100), 
                                         slice(80, None),
                                         slice(-20, None),
                                         slice(-15, -2),
                                         slice(5, 1), 
                                         [40, 60, 80]),
                             "errors": (SPYValueError,
                                        SPYValueError,
                                        SPYTypeError,
                                        SPYValueError,
                                        SPYValueError,
                                        SPYValueError,
                                        SPYValueError,
                                        SPYValueError,
                                        SPYValueError)}
    
    selectDict["taper"] = {"valid": ([4, 2, 3], 
                                     [4, 2, 2, 3],  # repetition
                                     [0, 1, 1, 2, 3],  # preserve repetition, don't convert to slice
                                     range(0, 3), 
                                     range(2, 5), 
                                     slice(None), 
                                     slice(0, 5), 
                                     slice(3, None), 
                                     slice(2, 4),
                                     slice(0, 5, 2),
                                     slice(-2, None),
                                     [0, 1, 2, 3],  # contiguous list...
                                     [1, 3, 4]),  # non-contiguous list...
                           "result": ([4, 2, 3], 
                                      [4, 2, 2, 3],
                                      [0, 1, 1, 2, 3],
                                      slice(0, 3, 1),
                                      slice(2, 5, 1), 
                                      slice(None, None, 1), 
                                      slice(0, 5, 1),
                                      slice(3, None, 1), 
                                      slice(2, 4, 1),
                                      slice(0, 5, 2),
                                      slice(-2, None, 1),
                                      slice(0, 4, 1),  # ...gets converted to slice
                                      [1, 3, 4]),  # stays as is
                           "invalid": (["taper_typo", "channel400"],
                                       "wrongtype",
                                       range(0, 100), 
                                       slice(80, None),
                                       slice(-20, None),
                                       slice(-15, -2),
                                       slice(5, 1), 
                                       [40, 60, 80]),
                           "errors": (SPYValueError,
                                      SPYTypeError,
                                      SPYValueError,
                                      SPYValueError,
                                      SPYValueError,
                                      SPYValueError,
                                      SPYValueError,
                                      SPYValueError)}
    
    # only define valid inputs, the expected (trial-dependent) results are computed below
    selectDict["unit"] = {"valid": (["unit3", "unit1"],
                                    ["unit3", "unit1", "unit1", "unit2"],  # repetition
                                    ["unit1", "unit1", "unit2", "unit3"],  # preserve repetition
                                    [4, 2, 3], 
                                    [4, 2, 2, 3],  # repetition
                                    [0, 0, 2, 3],  # preserve repetition, don't convert to slice
                                    range(0, 3), 
                                    range(2, 5), 
                                    slice(None), 
                                    slice(0, 5), 
                                    slice(3, None), 
                                    slice(2, 4),
                                    slice(0, 5, 2),
                                    slice(-2, None),
                                    [0, 1, 2, 3],  # contiguous list...
                                    [1, 3, 4]),  # non-contiguous list...
                          "invalid": (["unit7", "unit77"],
                                      "wrongtype",
                                      range(0, 100), 
                                      slice(80, None),
                                      slice(-20, None),
                                      slice(-15, -2),
                                      slice(5, 1), 
                                      [40, 60, 80]),
                          "errors": (SPYValueError,
                                     SPYTypeError,
                                     SPYValueError,
                                     SPYValueError,
                                     SPYValueError,
                                     SPYValueError,
                                     SPYValueError,
                                     SPYValueError)}

    # only define valid inputs, the expected (trial-dependent) results are computed below
    selectDict["eventid"] = {"valid": ([1, 0], 
                                       [1, 1, 0],  # repetition
                                       [0, 0, 1, 2],  # preserve repetition, don't convert to slice
                                       range(0, 2),
                                       range(1, 2), 
                                       slice(None), 
                                       slice(0, 2), 
                                       slice(1, None), 
                                       slice(0, 1),
                                       slice(-1, None),
                                       [0, 1]),  # contiguous list...
                             "invalid": (["eventid", "eventid"],
                                         "wrongtype",
                                         range(0, 100), 
                                         slice(80, None),
                                         slice(-20, None),
                                         slice(-15, -2),
                                         slice(5, 1), 
                                         [40, 60, 80]),
                             "errors": (SPYValueError,
                                        SPYTypeError,
                                        SPYValueError,
                                        SPYValueError,
                                        SPYValueError,
                                        SPYValueError,
                                        SPYValueError,
                                        SPYValueError)}

    # in the general test routine, only check correct handling of invalid `toi`/`toilim`
    # and `foi`/`foilim` selections - valid selectors are strongly object-dependent
    # and thus tested in separate methods below
    selectDict["toi"] = {"invalid": (["notnumeric", "stillnotnumeric"],
                                     "wrongtype",
                                     range(0, 10), 
                                     slice(0, 5),
                                     [0, np.inf],
                                     [np.nan, 1]),
                         "errors": (SPYValueError,
                                    SPYTypeError,
                                    SPYTypeError,
                                    SPYTypeError,
                                    SPYValueError,
                                    SPYValueError)}
    selectDict["toilim"] = {"invalid": (["notnumeric", "stillnotnumeric"],
                                        "wrongtype",
                                        range(0, 10), 
                                        slice(0, 5),
                                        [np.nan, 1],
                                        [0.5, 1.5 , 2.0],  # more than 2 components
                                        [2.0, 1.5]),  # lower bound > upper bound
                            "errors": (SPYValueError,
                                       SPYTypeError,
                                       SPYTypeError,
                                       SPYTypeError,
                                       SPYValueError,
                                       SPYValueError,
                                       SPYValueError)}
    selectDict["foi"] = {"invalid": (["notnumeric", "stillnotnumeric"],
                                     "wrongtype",
                                     range(0, 10), 
                                     slice(0, 5),
                                     [0, np.inf],
                                     [np.nan, 1],
                                     [-1, 2],  # out of bounds
                                     [2, 900]),  # out of bounds                                     
                         "errors": (SPYValueError,
                                    SPYTypeError,
                                    SPYTypeError,
                                    SPYTypeError,
                                    SPYValueError,
                                    SPYValueError,
                                    SPYValueError,
                                    SPYValueError)}
    selectDict["foilim"] = {"invalid": (["notnumeric", "stillnotnumeric"],
                                     "wrongtype",
                                     range(0, 10), 
                                     slice(0, 5),
                                     [np.nan, 1],
                                     [-1, 2],  # lower limit out of bounds
                                     [2, 900],  # upper limit out of bounds
                                     [2, 7, 6],  # more than 2 components
                                     [9, 2]),  # lower bound > upper bound
                         "errors": (SPYValueError,
                                    SPYTypeError,
                                    SPYTypeError,
                                    SPYTypeError,
                                    SPYValueError,
                                    SPYValueError,
                                    SPYValueError,
                                    SPYValueError,
                                    SPYValueError)}
    
    # Generate 2D array simulating an AnalogData array
    data["AnalogData"] = np.arange(1, nChannels * nSamples + 1).reshape(nSamples, nChannels)
    trl["AnalogData"] = np.vstack([np.arange(0, nSamples, nTrials),
                                   np.arange(lenTrial, nSamples + nTrials, nTrials),
                                   np.ones((lenTrial + 1, )),
                                   np.ones((lenTrial + 1, )) * np.pi]).T

    # Generate a 4D array simulating a SpectralData array
    data["SpectralData"] = np.arange(1, nChannels * nSamples * nTrials * nFreqs + 1).reshape(nSamples, nTrials, nFreqs, nChannels)
    trl["SpectralData"] = trl["AnalogData"]

    # Use a fixed random number generator seed to simulate a 2D SpikeData array
    seed = np.random.RandomState(13)
    data["SpikeData"] = np.vstack([seed.choice(nSamples, size=nSpikes),
                                   seed.choice(np.arange(1, nChannels + 1), size=nSpikes), 
                                   seed.choice(int(nChannels/2), size=nSpikes)]).T
    trl["SpikeData"] = trl["AnalogData"]

    # Use a triple-trigger pattern to simulate EventData w/non-uniform trials
    data["EventData"] = np.vstack([np.arange(0, nSamples, 2), 
                                   np.zeros((int(nSamples / 2), ))]).T  
    data["EventData"][1::3, 1] = 1
    data["EventData"][2::3, 1] = 2
    trl["EventData"] = trl["AnalogData"]
    
    # Define data classes to be used in tests below
    classes = ["AnalogData", "SpectralData", "SpikeData", "EventData"]
    
    # test `Selector` constructor w/all data classes    
    def test_general(self):
        
        # construct expected results for `DiscreteData` objects constructed above
        mapDict = {"unit": "SpikeData", "eventid": "EventData"}
        for prop, dclass in mapDict.items():
            discrete = getattr(spd, dclass)(data=self.data[dclass],
                                            trialdefinition=self.trl[dclass],
                                            samplerate=self.samplerate)
            propIdx = discrete.dimord.index(prop)

            # convert selection from `selectDict` to a usable integer-list
            allResults = []
            for selection in self.selectDict[prop]["valid"]:
                if isinstance(selection, slice):
                    if selection.start is selection.stop is None:
                        selects = [None]
                    else:
                        selects = list(range(getattr(discrete, prop).size))[selection]
                elif isinstance(selection, range):
                    selects = list(selection)
                else: # selection is list/ndarray
                    if isinstance(selection[0], str):
                        avail = getattr(discrete, prop)
                    else:
                        avail = np.arange(getattr(discrete, prop).size)
                    selects = []
                    for sel in selection:
                        selects += list(np.where(avail == sel)[0])
                
                # alternate (expensive) way to get by-trial selection indices
                result = []    
                for trial in discrete.trials:
                    if selects[0] is None:
                        res = slice(None, None, 1)
                    else:
                        res = []
                        for sel in selects:
                            res += list(np.where(trial[:, propIdx] == sel)[0])
                        if len(res) > 1:
                            steps = np.diff(res)
                            if steps.min() == steps.max() == 1:
                                res = slice(res[0], res[-1] + 1, 1)
                    result.append(res)
                allResults.append(result)
                
            self.selectDict[prop]["result"] = tuple(allResults)
        
        # wrong type of data and/or selector
        with pytest.raises(SPYTypeError):
            Selector(np.empty((3,)), {})
        with pytest.raises(SPYValueError):
            Selector(spd.AnalogData(), {})
        ang = AnalogData(data=self.data["AnalogData"], 
                         trialdefinition=self.trl["AnalogData"], 
                         samplerate=self.samplerate)
        with pytest.raises(SPYTypeError):
            Selector(ang, ())
        with pytest.raises(SPYValueError):
            Selector(ang, {"wrongkey": [1]})

        # go through all data-classes defined above            
        for dclass in self.classes:
            dummy = getattr(spd, dclass)(data=self.data[dclass],
                                         trialdefinition=self.trl[dclass],
                                         samplerate=self.samplerate)
            
            # test trial selection
            selection = Selector(dummy, {"trials": [3, 1]})
            assert selection.trials == [3, 1]
            with pytest.raises(SPYValueError):
                Selector(dummy, {"trials": [-1, 9]})

            # test "simple" property setters handled by `_selection_setter`
            for prop in ["channel", "taper", "unit", "eventid"]:
                if hasattr(dummy, prop):
                    expected = self.selectDict[prop]["result"]
                    for sk, sel in enumerate(self.selectDict[prop]["valid"]):
                        assert getattr(Selector(dummy, {prop + "s": sel}), prop) == expected[sk]
                    for ik, isel in enumerate(self.selectDict[prop]["invalid"]):
                        with pytest.raises(self.selectDict[prop]["errors"][ik]):
                            Selector(dummy, {prop + "s": isel})
                else:
                    with pytest.raises(SPYValueError):
                        Selector(dummy, {prop + "s": [0]})

            # test `toi` + `toilim`
            if hasattr(dummy, "time") or hasattr(dummy, "trialtime"):
                for selection in ["toi", "toilim"]:
                    for ik, isel in enumerate(self.selectDict[selection]["invalid"]):
                        with pytest.raises(self.selectDict[selection]["errors"][ik]):
                            Selector(dummy, {selection: isel})
                # provide both `toi` and `toilim`
                with pytest.raises(SPYValueError):
                    Selector(dummy, {"toi": [0], "toilim": [0, 1]})
            else:
                with pytest.raises(SPYValueError):
                    Selector(dummy, {"toi": [0]})
                with pytest.raises(SPYValueError):
                    Selector(dummy, {"toilim": [0]})
                
            # test `foi` + `foilim`
            if hasattr(dummy, "freq"):
                for selection in ["foi", "foilim"]:
                    for ik, isel in enumerate(self.selectDict[selection]["invalid"]):
                        with pytest.raises(self.selectDict[selection]["errors"][ik]):
                            Selector(dummy, {selection: isel})
                # provide both `foi` and `foilim`
                with pytest.raises(SPYValueError):
                    Selector(dummy, {"foi": [0], "foilim": [0, 1]})
            else:
                with pytest.raises(SPYValueError):
                    Selector(dummy, {"foi": [0]})
                with pytest.raises(SPYValueError):
                    Selector(dummy, {"foilim": [0]})

    def test_continuous_toitoilim(self):
        
        # this only works w/the equidistant trials constructed above!!!
        selDict = {"toi": ([0.5],  # single entry lists
                           [0.6],  # inexact match
                           [1.0, 2.5],  # two disjoint time-points
                           [1.2, 2.7],  # inexact from above
                           [1.9, 2.4],  # inexact from below
                           [0.4, 2.1],  # inexact from below, inexact from above
                           [1.6, 1.9],  # inexact from above, inexact from below
                           [-0.2, 0.6, 0.9, 1.1, 1.3, 1.6, 1.8, 2.2, 2.45, 3.],  # alternating madness
                           [2.0, 0.5, 2.5],  # unsorted list
                           [1.0, 0.5, 0.5, 1.5],  # repetition
                           [0.5, 0.5, 1.0, 1.5], # preserve repetition, don't convert to slice
                           [0.5, 1.0, 1.5]),  # sorted list (should be converted to slice-selection)
                   "toilim": ([0.5, 1.5],  # regular range
                              [1.5, 2.0],  # minimal range (just two-time points)
                              [1.0, np.inf],  # unbounded from above
                              [-np.inf, 1.0])}  # unbounded from below
        
        # all trials have same time-scale: take 1st one as reference
        trlTime = (np.arange(0, self.trl["AnalogData"][0, 1] - self.trl["AnalogData"][0, 0])
                        + self.trl["AnalogData"][0, 2]) / self.samplerate
        
        ang = AnalogData(data=self.data["AnalogData"], 
                         trialdefinition=self.trl["AnalogData"], 
                         samplerate=self.samplerate)
        
        # the below check only works for equidistant trials!
        for tselect in ["toi", "toilim"]:
            for timeSel in selDict[tselect]:
                sel = Selector(ang, {tselect: timeSel}).time
                if tselect == "toi":
                    idx = []
                    for tp in timeSel:
                        idx.append(np.abs(trlTime - tp).argmin())
                else:
                    idx = np.intersect1d(np.where(trlTime >= timeSel[0])[0],
                                         np.where(trlTime <= timeSel[1])[0])
                # check that correct data was selected (all trials identical, just take 1st one)
                assert np.array_equal(ang.trials[0][idx, :],
                                      ang.trials[0][sel[0], :])
                if len(idx) > 1:
                    timeSteps = np.diff(idx)
                    if timeSteps.min() == timeSteps.max() == 1:
                        idx = slice(idx[0], idx[-1] + 1, 1)
                result = [idx] * len(ang.trials)
                # check correct format of selector (list -> slice etc.)
                assert result == sel
                
        # FIXME: test time-frequency data selection as soon as we support this object type                
    
    # test `toi`/`toilim` selection w/`SpikeData` and `EventData`
    def test_discrete_toitoilim(self):
        
        # this only works w/the equidistant trials constructed above!!!
        selDict = {"toi": ([0.5],  # single entry lists
                           [0.6],  # inexact match
                           [1.0, 2.5],  # two disjoint time-points
                           [1.2, 2.7],  # inexact from above
                           [1.9, 2.4],  # inexact from below
                           [0.4, 2.1],  # inexact from below, inexact from above
                           [1.6, 1.9],  # inexact from above, inexact from below
                           [-0.2, 0.6, 0.9, 1.1, 1.3, 1.6, 1.8, 2.2, 2.45, 3.],  # alternating madness
                           [2.0, 0.5, 2.5],  # unsorted list
                           [1.0, 0.5, 0.5, 1.5],  # repetition
                           [0.5, 0.5, 1.0, 1.5], # preserve repetition, don't convert to slice
                           [0.5, 1.0, 1.5]),  # sorted list (should be converted to slice-selection)
                   "toilim": ([0.5, 1.5],  # regular range
                              [1.5, 2.0],  # minimal range (just two-time points)
                              [1.0, np.inf],  # unbounded from above
                              [-np.inf, 1.0])}  # unbounded from below
        
        # all trials have same time-scale for both `EventData` and `SpikeData`: take 1st one as reference
        trlTime = list((np.arange(0, self.trl["SpikeData"][0, 1] - self.trl["SpikeData"][0, 0])
                        + self.trl["SpikeData"][0, 2])/2 )

        # the below method of extracting spikes satisfying `toi`/`toilim` only works w/equidistant trials!
        for dclass in ["SpikeData", "EventData"]:
            discrete = getattr(spd, dclass)(data=self.data[dclass],
                                            trialdefinition=self.trl[dclass],
                                            samplerate=self.samplerate)
            for tselect in ["toi", "toilim"]:
                for timeSel in selDict[tselect]:
                    smpIdx = []
                    for tp in timeSel:
                        if np.isfinite(tp):
                            smpIdx.append(np.abs(np.array(trlTime) - tp).argmin())
                        else:
                            smpIdx.append(tp)
                    result = []
                    sel = Selector(discrete, {tselect: timeSel}).time
                    for trlno in range(len(discrete.trials)):
                        thisTrial = discrete.trials[trlno][:, 0]
                        if tselect == "toi":
                            trlRes = []
                            for idx in smpIdx:
                                trlRes += list(np.where(thisTrial == idx + trlno * self.lenTrial)[0])
                        else:
                            start = smpIdx[0] + trlno * self.lenTrial
                            stop = smpIdx[1] + trlno * self.lenTrial
                            candidates = np.intersect1d(thisTrial[thisTrial >= start], 
                                                        thisTrial[thisTrial <= stop])
                            trlRes = []
                            for cand in candidates:
                                trlRes += list(np.where(thisTrial == cand)[0])
                        # check that actually selected data is correct
                        assert np.array_equal(discrete.trials[trlno][trlRes, :], 
                                            discrete.trials[trlno][sel[trlno], :])
                        if len(trlRes) > 1:
                            sampSteps = np.diff(trlRes)
                            if sampSteps.min() == sampSteps.max() == 1:
                                trlRes = slice(trlRes[0], trlRes[-1] + 1, 1)
                        result.append(trlRes)
                    # check correct format of selector (list -> slice etc.)
                    assert result == sel
    
    def test_spectral_foifoilim(self):
        
        # this selection only works w/the dummy frequency data constructed above!!!
        selDict = {"foi": ([1],  # single entry lists
                           [2.6],  # inexact match
                           [2, 9],  # two disjoint frequencies
                           [7.2, 8.3],  # inexact from above
                           [6.8, 11.9],  # inexact from below
                           [0.4, 13.1],  # inexact from below, inexact from above
                           [1.2, 2.9],  # inexact from above, inexact from below
                           [1.1, 1.9, 2.1, 3.9, 9.2, 11.8, 12.9, 5.1, 13.8],  # alternating madness
                           [2, 1, 11],  # unsorted list
                           [5, 2, 2, 3],  # repetition
                           [1, 1, 2, 3], # preserve repetition, don't convert to slice
                           [2, 3, 4]),  # sorted list (should be converted to slice-selection)
                   "foilim": ([2, 11],  # regular range
                              [1, 2],  # minimal range (just two-time points)
                              [1.0, np.inf],  # unbounded from above
                              [-np.inf, 12])}  # unbounded from below
        
        spc = SpectralData(data=self.data['SpectralData'], 
                           trialdefinition=self.trl['SpectralData'], 
                           samplerate=self.samplerate)
        allFreqs = spc.freq
        
        for fselect in ["foi", "foilim"]:
            for freqSel in selDict[fselect]:
                sel = Selector(spc, {fselect: freqSel}).freq
                if fselect == "foi":
                    idx = []
                    for fq in freqSel:
                        idx.append(np.abs(allFreqs - fq).argmin())
                else:
                    idx = np.intersect1d(np.where(allFreqs >= freqSel[0])[0],
                                         np.where(allFreqs <= freqSel[1])[0])
                # check that correct data was selected (all trials identical, just take 1st one)
                assert np.array_equal(spc.freq[idx], spc.freq[sel])
                if len(idx) > 1:
                    freqSteps = np.diff(idx)
                    if freqSteps.min() == freqSteps.max() == 1:
                        idx = slice(idx[0], idx[-1] + 1, 1)
                # check correct format of selector (list -> slice etc.)
                assert idx == sel
