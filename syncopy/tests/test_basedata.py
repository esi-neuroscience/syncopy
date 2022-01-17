# -*- coding: utf-8 -*-
#
# Test proper functionality of Syncopy's `BaseData` class + helpers
#

# Builtin/3rd party package imports
import os
import tempfile
from attr import has
import h5py
import time
import pytest
import numpy as np
from numpy.lib.format import open_memmap
from memory_profiler import memory_usage

# Local imports
from syncopy.datatype import AnalogData
import syncopy.datatype as spd
from syncopy.datatype.base_data import VirtualData
from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYError
from syncopy.tests.misc import is_win_vm, is_slurm_node

# Construct decorators for skipping certain tests
skip_in_vm = pytest.mark.skipif(is_win_vm(), reason="running in Win VM")
skip_in_slurm = pytest.mark.skipif(is_slurm_node(), reason="running on cluster node")

# Collect all supported binary arithmetic operators
arithmetics = [lambda x, y : x + y,
               lambda x, y : x - y,
               lambda x, y : x * y,
               lambda x, y : x / y,
               lambda x, y : x ** y]


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
    samplerate = 1.0

    # Generate 2D array simulating an AnalogData array
    data["AnalogData"] = np.arange(1, nChannels * nSamples + 1).reshape(nSamples, nChannels)
    trl["AnalogData"] = np.vstack([np.arange(0, nSamples, 5),
                                   np.arange(5, nSamples + 5, 5),
                                   np.ones((int(nSamples / 5), )),
                                   np.ones((int(nSamples / 5), )) * np.pi]).T

    # Generate a 4D array simulating a SpectralData array (`nTrials` stands in for tapers)
    data["SpectralData"] = np.arange(1, nChannels * nSamples * nTrials * nFreqs + 1).reshape(nSamples, nTrials, nFreqs, nChannels)
    trl["SpectralData"] = trl["AnalogData"]

    # Generate a 4D array simulating a CorssSpectralData array
    data["CrossSpectralData"] = np.arange(1, nChannels * nChannels * nSamples * nFreqs + 1).reshape(nSamples, nFreqs, nChannels, nChannels)
    trl["CrossSpectralData"] = trl["AnalogData"]

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
    classes = ["AnalogData", "SpectralData", "CrossSpectralData", "SpikeData", "EventData"]

    # Allocation to `data` property is tested with all members of `classes`
    def test_data_alloc(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "dummy.npy")
            hname = os.path.join(tdir, "dummy.h5")

            for dclass in self.classes:

                # allocation with HDF5 file
                h5f = h5py.File(hname, mode="w")
                h5f.create_dataset("dummy", data=self.data[dclass])
                h5f.close()

                # allocation using HDF5 dataset directly
                dset = h5py.File(hname, mode="r+")["dummy"]
                dummy = getattr(spd, dclass)(data=dset)
                assert np.array_equal(dummy.data, self.data[dclass])
                assert dummy.mode == "r+", dummy.data.file.mode
                del dummy

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

                # allocate with valid dataset of "illegal" file
                del h5f["dummy"]
                h5f.create_dataset("dummy1", data=self.data[dclass])
                h5f.close()
                dset = h5py.File(hname, mode="r")["dummy1"]
                dummy = getattr(spd, dclass)(data=dset, filename=fname)

                # attempt data access after backing file of dataset has been closed
                dset.file.close()
                with pytest.raises(SPYValueError):
                    dummy.data[0, ...]

                # attempt allocation with HDF5 dataset of closed file
                with pytest.raises(SPYValueError):
                    getattr(spd, dclass)(data=dset)

                # attempt allocation using memmap of wrong shape
                np.save(fname, np.ones((self.nChannels,)))
                with pytest.raises(SPYValueError):
                    getattr(spd, dclass)(data=open_memmap(fname))

                # ensure synthetic data allocation via list of arrays works
                dummy = getattr(spd, dclass)(data=[self.data[dclass], self.data[dclass]])
                assert len(dummy.trials) == 2

                dummy = getattr(spd, dclass)(data=[self.data[dclass], self.data[dclass]],
                                samplerate=10.0)
                assert len(dummy.trials) == 2
                assert dummy.samplerate == 10

                if any(["ContinuousData" in str(base) for base in self.__class__.__mro__]):
                    nChan = self.data[dclass].shape[dummy.dimord.index("channel")]
                    dummy = getattr(spd, dclass)(data=[self.data[dclass], self.data[dclass]],
                                    channel=['label']*nChan)
                    assert len(dummy.trials) == 2
                    assert np.array_equal(dummy.channel, np.array(['label']*nChan))

                # the most egregious input errors are caught by `array_parser`; only
                # test list-routine-specific stuff: complex/real mismatch
                with pytest.raises(SPYValueError) as spyval:
                    getattr(spd, dclass)(data=[self.data[dclass], np.complex64(self.data[dclass])])
                    assert "same numeric type (real/complex)" in str(spyval.value)

                # shape mismatch
                with pytest.raises(SPYValueError):
                    getattr(spd, dclass)(data=[self.data[dclass], self.data[dclass].T])


            time.sleep(0.01)
            del dummy

    # Assignment of trialdefinition array is tested with all members of `classes`
    def test_trialdef(self):
        for dclass in self.classes:
            dummy = getattr(spd, dclass)(self.data[dclass],
                                         samplerate=self.samplerate)
            dummy.trialdefinition = self.trl[dclass]
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
        numf = 10000
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
                                         samplerate=self.samplerate)
            dummy.trialdefinition = self.trl[dclass]
            dummy2 = dummy.copy()
            assert dummy.filename == dummy2.filename
            assert hash(str(dummy.data)) == hash(str(dummy2.data))
            assert hash(str(dummy.sampleinfo)) == hash(str(dummy2.sampleinfo))
            assert hash(str(dummy._t0)) == hash(str(dummy2._t0))
            assert hash(str(dummy.trialinfo)) == hash(str(dummy2.trialinfo))
            assert hash(str(dummy.samplerate)) == hash(str(dummy2.samplerate))

        # test shallow + deep copies of memmaps + HDF5 files
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
                dummy = getattr(spd, dclass)(data=mm,
                                             samplerate=self.samplerate)
                dummy.trialdefinition = self.trl[dclass]
                dummy2 = dummy.copy()
                assert dummy.filename == dummy2.filename
                assert hash(str(dummy.data)) == hash(str(dummy2.data))
                assert hash(str(dummy.sampleinfo)) == hash(str(dummy2.sampleinfo))
                assert hash(str(dummy._t0)) == hash(str(dummy2._t0))
                assert hash(str(dummy.trialinfo)) == hash(str(dummy2.trialinfo))
                assert hash(str(dummy.samplerate)) == hash(str(dummy2.samplerate))

                # test integrity of deep-copy
                dummy3 = dummy.copy(deep=True)
                assert dummy3.filename != dummy.filename
                assert np.array_equal(dummy.trialdefinition, dummy3.trialdefinition)
                assert np.array_equal(dummy.data, dummy3.data)
                assert np.array_equal(dummy._t0, dummy3._t0)
                assert np.array_equal(dummy.trialinfo, dummy3.trialinfo)
                assert np.array_equal(dummy.sampleinfo, dummy3.sampleinfo)
                assert dummy.samplerate == dummy3.samplerate

                # hash-matching of shallow-copied HDF5 dataset
                dummy = getattr(spd, dclass)(data=h5py.File(hname)["dummy"],
                                             samplerate=self.samplerate)
                dummy.trialdefinition = self.trl[dclass]
                dummy2 = dummy.copy()
                assert dummy.filename == dummy2.filename
                assert hash(str(dummy.data)) == hash(str(dummy2.data))
                assert hash(str(dummy.sampleinfo)) == hash(str(dummy2.sampleinfo))
                assert hash(str(dummy._t0)) == hash(str(dummy2._t0))
                assert hash(str(dummy.trialinfo)) == hash(str(dummy2.trialinfo))
                assert hash(str(dummy.samplerate)) == hash(str(dummy2.samplerate))

                # test integrity of deep-copy
                dummy3 = dummy.copy(deep=True)
                assert dummy3.filename != dummy.filename
                assert np.array_equal(dummy.sampleinfo, dummy3.sampleinfo)
                assert np.array_equal(dummy._t0, dummy3._t0)
                assert np.array_equal(dummy.trialinfo, dummy3.trialinfo)
                assert np.array_equal(dummy.data, dummy3.data)
                assert dummy.samplerate == dummy3.samplerate

                # Delete all open references to file objects b4 closing tmp dir
                del mm, dummy, dummy2, dummy3
                time.sleep(0.01)

                # remove file for next round
                os.unlink(hname)

    # Test basic error handling of arithmetic ops
    def test_arithmetic(self):

        # Define list of classes arithmetic ops should and should not work with
        continuousClasses = ["AnalogData", "SpectralData", "CrossSpectralData"]
        discreteClasses = ["SpikeData", "EventData"]

        # Illegal classes for arithmetics
        for dclass in discreteClasses:
            dummy = getattr(spd, dclass)(self.data[dclass],
                                         trialdefinition=self.trl[dclass],
                                         samplerate=self.samplerate)
            for operation in arithmetics:
                with pytest.raises(SPYTypeError) as spytyp:
                    operation(dummy, 2)
                    assert "Wrong type of base: expected `AnalogData`, `SpectralData` or `CrossSpectralData`" in str(spytyp.value)

        # Now, test basic error handling for allowed classes
        for dclass in continuousClasses:
            dummy = getattr(spd, dclass)(self.data[dclass],
                                         samplerate=self.samplerate)
            dummy.trialdefinition = self.trl[dclass]
            otherClass = list(set(self.classes).difference([dclass]))[0]
            other = getattr(spd, otherClass)(self.data[otherClass],
                                             samplerate=self.samplerate)
            other.trialdefinition = self.trl[dclass]
            complexArr = np.complex64(dummy.trials[0])
            complexNum = 3+4j

            # Start w/the one operator that does not handle zeros well...
            with pytest.raises(SPYValueError) as spyval:
                dummy / 0
                assert "expected non-zero scalar for division" in str(spyval.value)

            # Go through all supported operators and try to sabotage them
            for operation in arithmetics:

                # Completely wrong operand
                with pytest.raises(SPYTypeError) as spytyp:
                    operation(dummy, np.sin)
                    assert "expected Syncopy object, scalar or array-like found ufunc" in str(spytyp.value)

                # Empty object
                with pytest.raises(SPYValueError) as spyval:
                    operation(getattr(spd, dclass)(), np.sin)
                    assert "expected non-empty Syncopy data object" in str(spyval.value)

                # Unbounded scalar
                with pytest.raises(SPYValueError) as spyval:
                    operation(dummy, np.inf)
                    assert "'inf'; expected finite scalar" in str(spyval.value)

                # Complex scalar (all test data are real)
                with pytest.raises(SPYTypeError) as spytyp:
                    operation(dummy, complexNum)
                    assert "expected scalar of same mathematical type (real/complex)" in str(spytyp.value)

                # Array w/wrong numeric type
                with pytest.raises(SPYTypeError) as spytyp:
                    operation(dummy, complexArr)
                    assert "array of same numerical type (real/complex) found ndarray" in str(spytyp.value)

                # Syncopy object of different type
                with pytest.raises(SPYTypeError) as spytyp:
                    operation(dummy, other)
                    err = "expected Syncopy {} object found {}"
                    assert err.format(dclass, otherClass)  in str(spytyp.value)

        # Next, validate proper functionality of `==` operator for Syncopy objects
        for dclass in self.classes:

            # Start simple compare obj to itself, to empty object and compare two empties
            dummy = getattr(spd, dclass)(self.data[dclass],
                                         samplerate=self.samplerate)
            dummy.trialdefinition = self.trl[dclass]
            assert dummy == dummy
            assert dummy != getattr(spd, dclass)()
            assert getattr(spd, dclass)() == getattr(spd, dclass)()

            # Basic type mismatch
            assert dummy != complexArr
            assert dummy != complexNum

            # Two differing Syncopy object classes
            otherClass = list(set(self.classes).difference([dclass]))[0]
            other = getattr(spd, otherClass)(self.data[otherClass],
                                             samplerate=self.samplerate)
            other.trialdefinition = self.trl[otherClass]
            assert dummy != other

            # Ensure shallow and deep copies are "==" to their origin
            dummy2 = dummy.copy()
            assert dummy2 == dummy
            dummy3 = dummy.copy(deep=True)
            assert dummy3 == dummy

            # Ensure differing samplerate evaluates to `False`
            dummy3.samplerate = 2*dummy.samplerate
            assert dummy3 != dummy
            dummy3.samplerate = dummy.samplerate

            # In-place selections are invalid for `==` comparisons
            dummy3.selectdata(inplace=True)
            with pytest.raises(SPYError) as spe:
                dummy3 == dummy
                assert "Cannot perform object comparison" in str(spe.value)

            # Abuse existing in-place selection to alter dimensional props of dummy3
            # and ensure inequality
            dimProps = dummy3._selector._dimProps
            dummy3.selectdata(clear=True)
            for prop in dimProps:
                if hasattr(dummy3, prop):
                    setattr(dummy3, prop, getattr(dummy, prop)[::-1])
                    assert dummy3 != dummy
                    setattr(dummy3, prop, getattr(dummy, prop))

            # Different trials
            dummy3 = dummy.selectdata(trials=list(range(len(dummy.trials) - 1)))
            assert dummy3 != dummy

            # Different trial offsets
            trl = self.trl[dclass]
            trl[:, 1] -= 1
            dummy3 = getattr(spd, dclass)(self.data[dclass],
                                          samplerate=self.samplerate)
            dummy3.trialdefinition = trl
            assert dummy3 != dummy

            # Different trial annotations
            trl = self.trl[dclass]
            trl[:, -1] = np.sqrt(2)
            dummy3 = getattr(spd, dclass)(self.data[dclass],
                                          samplerate=self.samplerate)
            dummy3.trialdefinition = trl
            assert dummy3 != dummy

            # Difference in actual numerical data
            dummy3 = dummy.copy(deep=True)
            for dsetName in dummy3._hdfFileDatasetProperties:
                getattr(dummy3, dsetName)[0] = 2 * np.pi
            assert dummy3 != dummy

            del dummy, dummy2, dummy3, other

        # Same objects but different dimords: `ContinuousData`` children
        for dclass in continuousClasses:
            dummy = getattr(spd, dclass)(self.data[dclass],
                                         samplerate=self.samplerate)
            dummy.trialdefinition = self.trl[dclass]
            ymmud = getattr(spd, dclass)(self.data[dclass].T,
                                         dimord=dummy.dimord[::-1],
                                         samplerate=self.samplerate)
            ymmud.trialdefinition = self.trl[dclass]
            assert dummy != ymmud

        # Same objects but different dimords: `DiscreteData` children
        for dclass in discreteClasses:
            dummy = getattr(spd, dclass)(self.data[dclass],
                                         trialdefinition=self.trl[dclass],
                                         samplerate=self.samplerate)
            ymmud = getattr(spd, dclass)(self.data[dclass],
                                         dimord=dummy.dimord[::-1],
                                         trialdefinition=self.trl[dclass],
                                         samplerate=self.samplerate)
            assert dummy != ymmud



