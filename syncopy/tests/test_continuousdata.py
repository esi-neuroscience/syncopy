# -*- coding: utf-8 -*-
#
# Test proper functionality of Syncopy's `ContinuousData` class + subclasses
#

# Builtin/3rd party package imports
import os
import tempfile
import time
import pytest
import random
import numpy as np
import h5py
import dask.distributed as dd

# Local imports
import syncopy as spy
from syncopy.datatype import AnalogData, SpectralData, CrossSpectralData, TimeLockData
from syncopy.io import save, load
from syncopy.datatype.methods.selectdata import selectdata
from syncopy.shared.errors import SPYValueError, SPYTypeError
from syncopy.shared.tools import StructDict
from syncopy.tests.misc import flush_local_cluster, generate_artificial_data, construct_spy_filename
from syncopy.tests import helpers


# Construct decorators for skipping certain tests
skip_legacy = pytest.mark.skipif(True, reason="code not used atm")

# Collect all supported binary arithmetic operators
arithmetics = [lambda x, y: x + y,
               lambda x, y: x - y,
               lambda x, y: x * y,
               lambda x, y: x / y,
               lambda x, y: x ** y]

# Module-wide set of testing selections
trialSelections = [
    "all",  # enforce below selections in all trials of `dummy`
    [3, 1, 2]  # minimally unordered
]
chanSelections = [
    ["channel03", "channel01", "channel01", "channel02"],  # string selection w/repetition + unordered
    [4, 2, 2, 5, 5],   # repetition + unordered
    range(5, 8),  # narrow range
    "channel02",  # str selection
    1  # scalar selection
]
latencySelections = [
    'all',
    'minperiod',
    [0.5, 1.5],  # regular range - 'maxperiod'
    [1., 1.5],
]
frequencySelections = [
    [2, 11],  # regular range
    [1, 2.0],  # minimal range (just two-time points)
    # [1.0, np.inf]  # unbounded from above, dropped support
]
taperSelections = [
    ["TestTaper_03", "TestTaper_01", "TestTaper_01", "TestTaper_02"],  # string selection w/repetition + unordered
    "TestTaper_03",  # singe str
    0,  # scalar selection
    [0, 1, 1, 2, 3],  # preserve repetition, don't convert to slice
    range(2, 5),  # narrow range
]
timeSelections = list(zip(["latency"] * len(latencySelections), latencySelections))
freqSelections = list(zip(["frequency"] * len(frequencySelections), frequencySelections))


# Local helper function for performing basic arithmetic tests
def _base_op_tests(dummy, ymmud, dummy2, ymmud2, dummyC, operation):

    dummyArr = 2 * np.ones((dummy.trials[0].shape))
    ymmudArr = 2 * np.ones((ymmud.trials[0].shape))
    scalarOperands = [2, np.pi]
    dummyOperands = [dummyArr, dummyArr.tolist()]
    ymmudOperands = [ymmudArr, ymmudArr.tolist()]

    # Ensure trial counts are properly vetted
    dummy2.selectdata(trials=0, inplace=True)
    with pytest.raises(SPYValueError) as spyval:
        operation(dummy, dummy2)
        assert "Syncopy object with same number of trials (selected)" in str(spyval.value)
    dummy2.selection = None

    # Scalar algebra must be commutative (except for pow)
    for operand in scalarOperands:
        result = operation(dummy, operand)  # perform operation from right
        for tk, trl in enumerate(result.trials):
            assert np.array_equal(trl, operation(dummy.trials[tk], operand))
        # Don't try to compute `2 ** data``
        if operation(2, 3) != 8:
            result2 = operation(operand, dummy)  # perform operation from left
            assert np.array_equal(result2.data, result.data)

        # Same as above, but swapped `dimord`
        result = operation(ymmud, operand)
        for tk, trl in enumerate(result.trials):
            assert np.array_equal(trl, operation(ymmud.trials[tk], operand))
        if operation(2, 3) != 8:
            result2 = operation(operand, ymmud)
            assert np.array_equal(result2.data, result.data)

    # Careful: NumPy tries to avoid failure by broadcasting; instead of relying
    # on an existing `__radd__` method, it performs arithmetic component-wise, i.e.,
    # ``np.ones((3,3)) + data`` performs ``1 + data`` nine times, so don't
    # test for left/right arithmetics...
    for operand in dummyOperands:
        result = operation(dummy, operand)
        for tk, trl in enumerate(result.trials):
            assert np.array_equal(trl, operation(dummy.trials[tk], operand))
    for operand in ymmudOperands:
        result = operation(ymmud, operand)
        for tk, trl in enumerate(result.trials):
            assert np.array_equal(trl, operation(ymmud.trials[tk], operand))

    # Ensure erroneous object type-casting is prevented
    if dummyC is not None:
        with pytest.raises(SPYTypeError) as spytyp:
            operation(dummy, dummyC)
            assert "Syncopy data object of same numerical type (real/complex)" in str(spytyp.value)

    # Most severe safety hazard: throw two objects at each other (with regular and
    # swapped dimord)
    result = operation(dummy, dummy2)
    for tk, trl in enumerate(result.trials):
        assert np.array_equal(trl, operation(dummy.trials[tk], dummy2.trials[tk]))
    result = operation(ymmud, ymmud2)
    for tk, trl in enumerate(result.trials):
        assert np.array_equal(trl, operation(ymmud.trials[tk], ymmud2.trials[tk]))


def _selection_op_tests(dummy, ymmud, dummy2, ymmud2, kwdict, operation):

    # Perform in-place selection and construct array based on new subset
    selected = dummy.selectdata(**kwdict)
    dummy.selectdata(inplace=True, **kwdict)
    arr = 2 * np.ones((selected.trials[0].shape), dtype=np.intp)
    for operand in [np.pi, arr]:
        result = operation(dummy, operand)
        for tk, trl in enumerate(result.trials):
            assert np.array_equal(trl, operation(selected.trials[tk], operand))

    # Most most complicated: subset selection present in base object
    # and operand thrown at it: only attempt to do this if the selection
    # is "well-behaved", i.e., is ordered and does not contain repetitions
    # The operator code checks for this, so catch the corresponding
    # `SpyValueError` and only attempt to test if coast is clear
    dummy2.selectdata(inplace=True, **kwdict)
    try:
        result = operation(dummy, dummy2)
        cleanSelection = True
    except SPYValueError:
        cleanSelection = False
    if cleanSelection:
        for tk, trl in enumerate(result.trials):
            assert np.array_equal(trl, operation(selected.trials[tk],
                                                 selected.trials[tk]))
        selected = ymmud.selectdata(**kwdict)
        ymmud.selectdata(inplace=True, **kwdict)
        ymmud2.selectdata(inplace=True, **kwdict)
        result = operation(ymmud, ymmud2)
        for tk, trl in enumerate(result.trials):
            assert np.array_equal(trl, operation(selected.trials[tk],
                                                 selected.trials[tk]))

    # Very important: clear manually set selections for next iteration
    dummy.selection = None
    dummy2.selection = None
    ymmud.selection = None
    ymmud2.selection = None


class TestAnalogData():

    # Allocate test-dataset
    nc = 10
    ns = 30
    data = np.arange(1, nc * ns + 1, dtype="float").reshape(ns, nc)
    trl = np.vstack([np.arange(0, ns, 5),
                     np.arange(5, ns + 5, 5),
                     np.ones((int(ns / 5), )),
                     np.ones((int(ns / 5), )) * np.pi]).T
    samplerate = 2.0

    def test_constructor(self):

        # -- test empty --
        dummy = AnalogData()
        assert len(dummy.cfg) == 0
        for attr in ["channel", "data", "sampleinfo", "trialinfo"]:
            assert getattr(dummy, attr) is None
        with pytest.raises(SPYTypeError):
            AnalogData({})

        # -- test with single array--

        dummy = AnalogData(data=self.data)
        assert dummy.dimord == AnalogData._defaultDimord
        assert dummy.channel.size == self.nc
        assert (dummy.sampleinfo == [0, self.ns]).min()
        assert dummy.trialinfo.shape == (1, 0)
        assert np.array_equal(dummy.data, self.data)

        # wrong shape for data-type
        with pytest.raises(SPYValueError):
            AnalogData(np.ones((3,)))

        # -- test with list of arrays

        # 5 trials of shape (10, 2)
        nTrials = 3
        nSamples = 10
        data_list = [i * np.ones((nSamples, 2)) for i in range(nTrials)]

        dummy = AnalogData(data_list, samplerate=1)
        assert len(dummy.trials) == nTrials
        assert np.all([dummy.trials[i][0, 0] == i for i in range(nTrials)])
        # test for correct trial size
        assert np.all([len(dummy.trials[i]) == nSamples for i in range(nTrials)])

        with pytest.raises(SPYValueError, match="mismatching shapes"):
            _ = AnalogData(data=[np.ones((2, 2)), np.ones((3, 2))])

        # -- test with generator --

        # can accomodate variable trial sizes
        gen = (i * np.ones((i + 1, 2)) for i in range(nTrials))

        dummy2 = AnalogData(gen, samplerate=1)
        assert len(dummy2.trials) == nTrials
        assert np.all([dummy2.trials[i][0, 0] == i for i in range(nTrials)])
        # test for correct trial size
        assert np.all([len(dummy2.trials[i]) == i + 1 for i in range(nTrials)])

        # however dims which are not the stacking dim still have to match
        with pytest.raises(SPYValueError, match="mismatching shapes"):
            gen1 = (np.ones((2, i + 1)) for i in range(nTrials))
            _ = AnalogData(data=gen1)

        # if we change the dimord/stacking dim, this is fine
        gen1 = (np.ones((2, i + 1)) for i in range(nTrials))
        dummy3 = AnalogData(data=gen1, dimord=['channel', 'time'])
        assert len(dummy3.trials) == nTrials

        # -- test with list of syncopy objects --

        concat = AnalogData([dummy2, dummy])
        assert len(concat.trials) == len(dummy.trials) + len(dummy2.trials)
        # check trial sizes kept consistent
        assert np.all([len(concat.trials[i]) == i + 1 for i in range(nTrials)])
        assert np.all([len(concat.trials[i]) == nSamples for i in range(nTrials, 2 * nTrials)])
        # check values are there
        assert np.all([concat.trials[i][0, 0] == i for i in range(nTrials)])
        assert np.all([concat.trials[i][0, 0] == i - nTrials for i in range(nTrials, 2 * nTrials)])

        # mismatching attributes are not allowed
        dummy4 = AnalogData(data_list)

        # samplerate is missing
        with pytest.raises(SPYValueError, match="missing attribute"):
            _ = AnalogData([dummy, dummy4])

        dummy4.samplerate = 1
        # channel labels are not the same
        with pytest.raises(SPYValueError, match="different attribute"):
            dummy4.channel = ['c1', 'c2']
            _ = AnalogData([dummy, dummy4])

        # mismatching shape
        dummy5 = AnalogData([np.ones((2, 3))], samplerate=1)
        with pytest.raises(SPYValueError, match="mismatching shapes"):
            _ = AnalogData([dummy, dummy5])

        # wrong stacking dim
        with pytest.raises(SPYValueError, match="different stacking"):
            _ = AnalogData([dummy, dummy3])

        # test channel property propagation
        dummy.channel = ['c1', 'c2']
        dummy2.channel = ['c1', 'c2']

        concat2 = AnalogData([dummy, dummy2])
        assert np.all(concat2.channel == dummy.channel)

    def test_trialretrieval(self):
        # test ``_get_trial`` with NumPy array: regular order
        dummy = AnalogData(data=self.data, trialdefinition=self.trl)
        for trlno, start in enumerate(range(0, self.ns, 5)):
            trl_ref = self.data[start:start + 5, :]
            assert np.array_equal(dummy._get_trial(trlno), trl_ref)

        # test ``_get_trial`` with NumPy array: swapped dimensions
        dummy = AnalogData(self.data.T, trialdefinition=self.trl,
                           dimord=["channel", "time"])
        for trlno, start in enumerate(range(0, self.ns, 5)):
            trl_ref = self.data.T[:, start:start + 5]
            assert np.array_equal(dummy._get_trial(trlno), trl_ref)

        del dummy

    def test_saveload(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "dummy")

            # basic but most important: ensure object integrity is preserved
            checkAttr = ["channel", "data", "dimord", "sampleinfo", "samplerate", "trialinfo"]
            dummy = AnalogData(data=self.data, samplerate=1000)
            dummy.save(fname)
            filename = construct_spy_filename(fname, dummy)
            # NOTE: We removed support for loading data via the constructor
            # dummy2 = AnalogData(filename)
            # for attr in checkAttr:
            #     assert np.array_equal(getattr(dummy, attr), getattr(dummy2, attr))
            dummy3 = load(fname)
            for attr in checkAttr:
                assert np.array_equal(getattr(dummy3, attr), getattr(dummy, attr))
            save(dummy3, container=os.path.join(tdir, "ymmud"))
            dummy4 = load(os.path.join(tdir, "ymmud"))
            for attr in checkAttr:
                assert np.array_equal(getattr(dummy4, attr), getattr(dummy, attr))
            del dummy, dummy3, dummy4  # avoid PermissionError in Windows

            # ensure trialdefinition is saved and loaded correctly
            dummy = AnalogData(data=self.data, trialdefinition=self.trl, samplerate=1000)
            dummy.save(fname + "_trl")
            filename = construct_spy_filename(fname + "_trl", dummy)
            dummy2 = load(filename)
            assert np.array_equal(dummy.trialdefinition, dummy2.trialdefinition)

            # test getters
            assert np.array_equal(dummy.sampleinfo, dummy2.sampleinfo)
            assert np.array_equal(dummy._t0, dummy2._t0)
            assert np.array_equal(dummy.trialinfo, dummy2.trialinfo)

            del dummy, dummy2  # avoid PermissionError in Windows

            # swap dimensions and ensure `dimord` is preserved
            dummy = AnalogData(data=self.data,
                               dimord=["channel", "time"], samplerate=1000)
            dummy.save(fname + "_dimswap")
            filename = construct_spy_filename(fname + "_dimswap", dummy)
            dummy2 = load(filename)
            assert dummy2.dimord == dummy.dimord
            assert dummy2.channel.size == self.ns  # swapped
            assert dummy2.data.shape == dummy.data.shape

            # Delete all open references to file objects and wait 0.1s for changes
            # to take effect (thanks, Windows!)
            del dummy, dummy2
            time.sleep(0.1)

    # test arithmetic operations
    def test_ang_arithmetic(self):

        # Create testing objects and corresponding arrays to perform arithmetics with
        dummy = AnalogData(data=self.data,
                           trialdefinition=self.trl,
                           samplerate=self.samplerate)
        ymmud = AnalogData(data=self.data.T,
                           trialdefinition=self.trl,
                           samplerate=self.samplerate,
                           dimord=AnalogData._defaultDimord[::-1])
        dummy2 = AnalogData(data=self.data,
                            trialdefinition=self.trl,
                            samplerate=self.samplerate)
        ymmud2 = AnalogData(data=self.data.T,
                            trialdefinition=self.trl,
                            samplerate=self.samplerate,
                            dimord=AnalogData._defaultDimord[::-1])

        # Perform basic arithmetic with +, -, *, / and ** (pow)
        for operation in arithmetics:

            # First, ensure `dimord` is respected
            with pytest.raises(SPYValueError) as spyval:
                operation(dummy, ymmud)
                assert "expected Syncopy 'time' x 'channel' data object" in str(spyval.value)

            _base_op_tests(dummy, ymmud, dummy2, ymmud2, None, operation)

            kwdict = {}
            kwdict["trials"] = trialSelections[1]
            kwdict["channel"] = chanSelections[3]
            kwdict[timeSelections[2][0]] = timeSelections[2][1]
            _selection_op_tests(dummy, ymmud, dummy2, ymmud2, kwdict, operation)

        # Finally, perform a representative chained operation to ensure chaining works
        result = (dummy + dummy2) / dummy ** 3
        for tk, trl in enumerate(result.trials):
            assert np.array_equal(trl,
                                  (dummy.trials[tk] + dummy2.trials[tk]) / dummy.trials[tk] ** 3)

    def test_parallel(self, testcluster):
        # repeat selected test w/parallel processing engine
        client = dd.Client(testcluster)
        slow_tests = ["test_dataselection",
                      "test_ang_arithmetic"]
        for test in slow_tests:
            getattr(self, test)()
            flush_local_cluster(testcluster)
        client.close()


class TestSpectralData():

    # Allocate test-dataset
    nc = 10
    ns = 30
    nt = 5
    nf = 15
    data = np.arange(1, nc * ns * nt * nf + 1, dtype="float").reshape(ns, nt, nf, nc)
    trl = np.vstack([np.arange(0, ns, 5),
                     np.arange(5, ns + 5, 5),
                     np.ones((int(ns / 5), )),
                     np.ones((int(ns / 5), )) * np.pi]).T
    data2 = np.moveaxis(data, 0, -1)
    samplerate = 2.0

    def test_sd_empty(self):
        dummy = SpectralData()
        assert len(dummy.cfg) == 0
        for attr in ["channel", "data", "freq", "sampleinfo", "taper", "trialinfo"]:
            assert getattr(dummy, attr) is None
        with pytest.raises(SPYTypeError):
            SpectralData({})

    def test_sd_nparray(self):
        dummy = SpectralData(self.data)
        assert dummy.dimord == SpectralData._defaultDimord
        assert dummy.channel.size == self.nc
        assert dummy.taper.size == self.nt
        assert dummy.freq.size == self.nf
        assert (dummy.sampleinfo == [0, self.ns]).min()
        assert dummy.trialinfo.shape == (1, 0)
        assert np.array_equal(dummy.data, self.data)

        # wrong shape for data-type
        with pytest.raises(SPYValueError):
            SpectralData(data=np.ones((3,)))

    def test_sd_concat(self):

        # time x taper x freq x channel
        nTrials = 3
        gen = (i * np.ones((10, 2, 10, 3), dtype=np.complex64) for i in range(nTrials))
        # use generator
        dummy = SpectralData(data=gen, samplerate=11)
        dummy.freq = dummy.freq * 1.239

        # raw copy the array
        dummy2 = SpectralData(dummy.data[()], samplerate=11)
        dummy2.trialdefinition = dummy.trialdefinition
        dummy2.freq = dummy.freq

        concat = SpectralData([dummy, dummy2])
        # check attributes
        assert concat.samplerate == 11
        assert np.all(concat.freq == dummy.freq)
        # check trial sizes
        assert len(concat.trials) == 2 * nTrials

    def test_sd_trialretrieval(self):
        # test ``_get_trial`` with NumPy array: regular order
        dummy = SpectralData(self.data, trialdefinition=self.trl)
        for trlno, start in enumerate(range(0, self.ns, 5)):
            trl_ref = self.data[start:start + 5, ...]
            assert np.array_equal(dummy._get_trial(trlno), trl_ref)

        # test ``_get_trial`` with NumPy array: swapped dimensions
        dummy = SpectralData(self.data2, trialdefinition=self.trl,
                             dimord=["taper", "channel", "freq", "time"])
        for trlno, start in enumerate(range(0, self.ns, 5)):
            trl_ref = self.data2[..., start:start + 5]
            assert np.array_equal(dummy._get_trial(trlno), trl_ref)

        del dummy

    def test_sd_saveload(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "dummy")

            # basic but most important: ensure object integrity is preserved
            checkAttr = ["channel", "data", "dimord", "freq", "sampleinfo",
                         "samplerate", "taper", "trialinfo"]
            dummy = SpectralData(self.data, samplerate=1000)
            dummy.save(fname)
            filename = construct_spy_filename(fname, dummy)
            # dummy2 = SpectralData(filename)
            # for attr in checkAttr:
            #     assert np.array_equal(getattr(dummy, attr), getattr(dummy2, attr))
            dummy3 = load(fname)
            for attr in checkAttr:
                assert np.array_equal(getattr(dummy3, attr), getattr(dummy, attr))
            save(dummy3, container=os.path.join(tdir, "ymmud"))
            dummy4 = load(os.path.join(tdir, "ymmud"))
            for attr in checkAttr:
                assert np.array_equal(getattr(dummy4, attr), getattr(dummy, attr))
            del dummy, dummy3, dummy4  # avoid PermissionError in Windows

            # ensure trialdefinition is saved and loaded correctly
            dummy = SpectralData(self.data, trialdefinition=self.trl, samplerate=1000)
            dummy.save(fname, overwrite=True)
            dummy2 = load(filename)
            assert np.array_equal(dummy.trialdefinition, dummy2.trialdefinition)

            # test getters
            assert np.array_equal(dummy.sampleinfo, dummy2.sampleinfo)
            assert np.array_equal(dummy._t0, dummy2._t0)
            assert np.array_equal(dummy.trialinfo, dummy2.trialinfo)

            # swap dimensions and ensure `dimord` is preserved
            dummy = SpectralData(self.data, dimord=["time", "channel", "taper", "freq"],
                                 samplerate=1000)
            dummy.save(fname + "_dimswap")
            filename = construct_spy_filename(fname + "_dimswap", dummy)
            dummy2 = load(filename)
            assert dummy2.dimord == dummy.dimord
            assert dummy2.channel.size == self.nt  # swapped
            assert dummy2.taper.size == self.nf  # swapped
            assert dummy2.data.shape == dummy.data.shape

            # Delete all open references to file objects b4 closing tmp dir
            del dummy, dummy2

    # test arithmetic operations
    def test_sd_arithmetic(self):

        # Create testing objects and corresponding arrays to perform arithmetics with
        dummy = SpectralData(data=self.data,
                             trialdefinition=self.trl,
                             samplerate=self.samplerate,
                             taper=["TestTaper_0{}".format(k) for k in range(1, self.nt + 1)])
        dummyC = SpectralData(data=np.complex64(self.data),
                              trialdefinition=self.trl,
                              samplerate=self.samplerate,
                              taper=["TestTaper_0{}".format(k) for k in range(1, self.nt + 1)])
        ymmud = SpectralData(data=np.transpose(self.data, [3, 2, 1, 0]),
                             trialdefinition=self.trl,
                             samplerate=self.samplerate,
                             taper=["TestTaper_0{}".format(k) for k in range(1, self.nt + 1)],
                             dimord=SpectralData._defaultDimord[::-1])
        dummy2 = SpectralData(data=self.data,
                              trialdefinition=self.trl,
                              samplerate=self.samplerate,
                              taper=["TestTaper_0{}".format(k) for k in range(1, self.nt + 1)])
        ymmud2 = SpectralData(data=np.transpose(self.data, [3, 2, 1, 0]),
                              trialdefinition=self.trl,
                              samplerate=self.samplerate,
                              taper=["TestTaper_0{}".format(k) for k in range(1, self.nt + 1)],
                              dimord=SpectralData._defaultDimord[::-1])

        # Perform basic arithmetic with +, -, *, / and ** (pow)
        for operation in arithmetics:

            # First, ensure `dimord` is respected
            with pytest.raises(SPYValueError) as spyval:
                operation(dummy, ymmud)
                assert "expected Syncopy 'time' x 'channel' data object" in str(spyval.value)

            _base_op_tests(dummy, ymmud, dummy2, ymmud2, dummyC, operation)


            kwdict = {}
            kwdict["trials"] = trialSelections[1]
            kwdict["channel"] = chanSelections[3]
            kwdict[timeSelections[2][0]] = timeSelections[2][1]
            kwdict[freqSelections[1][0]] = freqSelections[1][1]
            kwdict["taper"] = taperSelections[2]
            _selection_op_tests(dummy, ymmud, dummy2, ymmud2, kwdict, operation)

        # Finally, perform a representative chained operation to ensure chaining works
        result = (dummy + dummy2) / dummy ** 3
        for tk, trl in enumerate(result.trials):
            assert np.array_equal(trl,
                                  (dummy.trials[tk] + dummy2.trials[tk]) / dummy.trials[tk] ** 3)

    def test_sd_parallel(self, testcluster):
        # repeat selected test w/parallel processing engine
        client = dd.Client(testcluster)
        par_tests = ["test_sd_arithmetic", "test_sd_concat"]
        for test in par_tests:
            getattr(self, test)()
            flush_local_cluster(testcluster)
        client.close()


class TestCrossSpectralData():

    # Allocate test-dataset
    nci = 10
    ncj = 12
    nl = 3
    nt = 6
    ns = nt * nl
    nf = 15
    data = np.arange(1, nci * ncj * ns * nf + 1, dtype="float").reshape(ns, nf, nci, ncj)
    trl = np.vstack([np.arange(0, ns, nl),
                     np.arange(nl, ns + nl, nl),
                     np.ones((int(ns / nl), )),
                     np.ones((int(ns / nl), )) * np.pi]).T
    data2 = np.moveaxis(data, 0, -1)
    samplerate = 2.0

    def test_csd_empty(self):
        dummy = CrossSpectralData()
        assert len(dummy.cfg) == 0
        assert dummy.dimord is None
        for attr in ["channel_i", "channel_j", "data", "freq", "sampleinfo", "trialinfo"]:
            assert getattr(dummy, attr) is None
        with pytest.raises(SPYTypeError):
            CrossSpectralData({})

    def test_csd_nparray(self):
        dummy = CrossSpectralData(self.data)
        assert dummy.dimord == CrossSpectralData._defaultDimord
        assert dummy.channel_i.size == self.nci
        assert dummy.channel_j.size == self.ncj
        assert dummy.freq.size == self.nf
        assert (dummy.sampleinfo == [0, self.ns]).min()
        assert dummy.trialinfo.shape == (1, 0)
        assert np.array_equal(dummy.data, self.data)

        # wrong shape for data-type
        with pytest.raises(SPYValueError):
            CrossSpectralData(data=np.ones((3,)))

    def test_csd_concat(self):

        # time x freq x channel x channel
        nTrials = 3
        gen = (i * np.ones((2, 2, 3, 3), dtype=np.complex64) for i in range(nTrials))
        # use generator
        dummy = CrossSpectralData(data=gen, samplerate=11)
        dummy.freq = dummy.freq * 1.239

        # raw copy the array
        dummy2 = CrossSpectralData(dummy.data[()], samplerate=11)
        dummy2.trialdefinition = dummy.trialdefinition
        dummy2.freq = dummy.freq

        concat = CrossSpectralData([dummy, dummy2])
        # check attributes
        assert concat.samplerate == 11
        assert np.all(concat.freq == dummy.freq)
        # check trial sizes
        assert len(concat.trials) == 2 * nTrials

        # try to concat SpectralData and CrossSpectralData
        dummy3 = SpectralData(dummy.data[()], samplerate=11)
        with pytest.raises(SPYValueError, match="different attribute values"):
            concat2 = CrossSpectralData([dummy, dummy3])

    def test_csd_trialretrieval(self):
        # test ``_get_trial`` with NumPy array: regular order
        dummy = CrossSpectralData(self.data)
        dummy.trialdefinition = self.trl
        for trlno, start in enumerate(range(0, self.ns, self.nl)):
            trl_ref = self.data[start:start + self.nl, ...]
            assert np.array_equal(dummy.trials[trlno], trl_ref)

        # test ``_get_trial`` with NumPy array: swapped dimensions
        dummy = CrossSpectralData(self.data2, dimord=["freq", "channel_i", "channel_j", "time"])
        dummy.trialdefinition = self.trl
        for trlno, start in enumerate(range(0, self.ns, self.nl)):
            trl_ref = self.data2[..., start:start + self.nl]
            assert np.array_equal(dummy.trials[trlno], trl_ref)

        del dummy

    def test_csd_saveload(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "dummy")

            # basic but most important: ensure object integrity is preserved
            checkAttr = ["channel_i", "channel_i", "data", "dimord", "freq", "sampleinfo",
                         "samplerate", "trialinfo"]
            dummy = CrossSpectralData(self.data, samplerate=1000)
            dummy.save(fname)
            filename = construct_spy_filename(fname, dummy)
            dummy3 = load(fname)
            for attr in checkAttr:
                assert np.array_equal(getattr(dummy3, attr), getattr(dummy, attr))
            save(dummy3, container=os.path.join(tdir, "ymmud"))
            dummy4 = load(os.path.join(tdir, "ymmud"))
            for attr in checkAttr:
                assert np.array_equal(getattr(dummy4, attr), getattr(dummy, attr))
            del dummy, dummy3, dummy4  # avoid PermissionError in Windows

            # ensure trialdefinition is saved and loaded correctly
            dummy = CrossSpectralData(self.data, samplerate=1000)
            dummy.trialdefinition = self.trl
            dummy.save(fname, overwrite=True)
            dummy2 = load(filename)
            assert np.array_equal(dummy.trialdefinition, dummy2.trialdefinition)

            # test getters
            assert np.array_equal(dummy.sampleinfo, dummy2.sampleinfo)
            assert np.array_equal(dummy._t0, dummy2._t0)
            assert np.array_equal(dummy.trialinfo, dummy2.trialinfo)

            # swap dimensions and ensure `dimord` is preserved
            dummy = CrossSpectralData(self.data, dimord=["freq", "channel_j", "channel_i", "time"],
                                      samplerate=1000)
            dummy.save(fname + "_dimswap")
            filename = construct_spy_filename(fname + "_dimswap", dummy)
            dummy2 = load(filename)
            assert dummy2.dimord == dummy.dimord
            assert dummy2.channel_i.size == self.nci  # swapped
            assert dummy2.channel_j.size == self.nf  # swapped
            assert dummy2.freq.size == self.ns  # swapped
            assert dummy2.data.shape == dummy.data.shape

            # Delete all open references to file objects b4 closing tmp dir
            del dummy, dummy2

    # test arithmetic operations
    def test_csd_arithmetic(self):

        # Create testing objects and corresponding arrays to perform arithmetics with
        dummy = CrossSpectralData(data=self.data,
                                  samplerate=self.samplerate)
        dummy.trialdefinition = self.trl
        dummyC = CrossSpectralData(data=np.complex64(self.data),
                                   samplerate=self.samplerate)
        dummyC.trialdefinition = self.trl
        ymmud = CrossSpectralData(data=np.transpose(self.data, [3, 2, 1, 0]),
                                  samplerate=self.samplerate,
                                  dimord=CrossSpectralData._defaultDimord[::-1])
        ymmud.trialdefinition = self.trl
        dummy2 = CrossSpectralData(data=self.data,
                                   samplerate=self.samplerate)
        dummy2.trialdefinition = self.trl
        ymmud2 = CrossSpectralData(data=np.transpose(self.data, [3, 2, 1, 0]),
                                   samplerate=self.samplerate,
                                   dimord=CrossSpectralData._defaultDimord[::-1])
        ymmud2.trialdefinition = self.trl

        # Perform basic arithmetic with +, -, *, / and ** (pow)
        for operation in arithmetics:

            # First, ensure `dimord` is respected
            with pytest.raises(SPYValueError) as spyval:
                operation(dummy, ymmud)
                assert "expected Syncopy 'time' x 'freq' x 'channel_i' x 'channel_j' data object" in str(spyval.value)

            _base_op_tests(dummy, ymmud, dummy2, ymmud2, dummyC, operation)

            # Go through full selection stack - WARNING: this takes > 1 hour
            kwdict = {}
            kwdict["trials"] = trialSelections[1]
            kwdict["channel_i"] = chanSelections[3]
            kwdict["channel_j"] = chanSelections[4]
            kwdict[timeSelections[2][0]] = timeSelections[2][1]
            kwdict[freqSelections[1][0]] = freqSelections[1][1]
            _selection_op_tests(dummy, ymmud, dummy2, ymmud2, kwdict, operation)

        # Finally, perform a representative chained operation to ensure chaining works
        result = (dummy + dummy2) / dummy ** 3
        for tk, trl in enumerate(result.trials):
            assert np.array_equal(trl,
                                  (dummy.trials[tk] + dummy2.trials[tk]) / dummy.trials[tk] ** 3)

    def test_csd_parallel(self, testcluster):
        # repeat selected test w/parallel processing engine
        client = dd.Client(testcluster)
        par_tests = ["test_csd_arithmetic", "test_csd_concat"]
        for test in par_tests:
            getattr(self, test)
            flush_local_cluster(testcluster)
        client.close()


class TestTimeLockData:
    """Tests for the `TimeLockData` data type, which is derived from `ContinuousData`."""

    def test_create(self):
        """Test instantiation, and that expected properties/datasets specific to this data type exist."""
        tld = TimeLockData()

        assert hasattr(tld, '_avg')
        assert hasattr(tld, '_var')
        assert hasattr(tld, '_cov')
        assert tld.avg is None
        assert tld.var is None
        assert tld.cov is None

    def test_modify_properties(self):
        """Test modification of the extra datasets avg, var, cov."""
        tld = TimeLockData()

        avg_data = np.zeros((3, 3), dtype=np.float64)
        tld._update_dataset("avg", avg_data)
        assert isinstance(tld.avg, h5py.Dataset)
        assert np.array_equal(avg_data, tld.avg)

        # Try to overwrite data via setter, which should not work.
        avg_data2 = np.zeros((4, 4, 4), dtype=np.float32)
        with pytest.raises(AttributeError, match="can't set attribute"):
            tld.avg = avg_data2

        # But we can do it with _update_dataset:
        tld._update_dataset("avg", avg_data2)
        assert np.array_equal(avg_data2, tld.avg)

        # ... or of course, directly using '_avg':
        tld2 = TimeLockData()
        avg_data3 = np.zeros((2, 2), dtype=np.float32)
        tld2._avg = avg_data3
        assert np.array_equal(avg_data3, tld2.avg)


if __name__ == '__main__':

    T1 = TestAnalogData()
    T2 = TestSpectralData()
    T3 = TestTimeLockData()
    T4 = TestCrossSpectralData()
