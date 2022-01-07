# -*- coding: utf-8 -*-
#
# Test proper functionality of Syncopy's `ContinuousData` class + subclasses
#

# Builtin/3rd party package imports
import os
import tempfile
import time
import pytest
import numpy as np
from numpy.lib.format import open_memmap

# Local imports
from syncopy.datatype import AnalogData, SpectralData, CrossSpectralData, padding
from syncopy.io import save, load
from syncopy.datatype.base_data import VirtualData, Selector
from syncopy.datatype.methods.selectdata import selectdata
from syncopy.shared.errors import SPYValueError, SPYTypeError
from syncopy.shared.tools import StructDict
from syncopy.tests.misc import flush_local_cluster, generate_artificial_data, construct_spy_filename
from syncopy import __acme__
if __acme__:
    import dask.distributed as dd

# Decorator to decide whether or not to run dask-related tests
skip_without_acme = pytest.mark.skipif(
    not __acme__, reason="acme not available")

# Collect all supported binary arithmetic operators
arithmetics = [lambda x, y : x + y,
               lambda x, y : x - y,
               lambda x, y : x * y,
               lambda x, y : x / y,
               lambda x, y : x ** y]

# Module-wide set of testing selections
trialSelections = [
    "all",  # enforce below selections in all trials of `dummy`
    [3, 1, 2]  # minimally unordered
]
chanSelections = [
    ["channel03", "channel01", "channel01", "channel02"],  # string selection w/repetition + unordered
    [4, 2, 2, 5, 5],   # repetition + unorderd
    range(5, 8),  # narrow range
    slice(-2, None)  # negative-start slice
    ]
toiSelections = [
    "all",  # non-type-conform string
    [0.6],  # single inexact match
    [-0.2, 0.6, 0.9, 1.1, 1.3, 1.6, 1.8, 2.2, 2.45, 3.]  # unordered, inexact, repetions
    ]
toilimSelections = [
    [0.5, 1.5],  # regular range
    [1.5, 2.0],  # minimal range (just two-time points)
    [1.0, np.inf]  # unbounded from above
    ]
foiSelections = [
    "all",  # non-type-conform string
    [2.6],  # single inexact match
    [1.1, 1.9, 2.1, 3.9, 9.2, 11.8, 12.9, 5.1, 13.8]  # unordered, inexact, repetions
    ]
foilimSelections = [
    [2, 11],  # regular range
    [1, 2.0],  # minimal range (just two-time points)
    [1.0, np.inf]  # unbounded from above
    ]
taperSelections = [
    ["TestTaper_03", "TestTaper_01", "TestTaper_01", "TestTaper_02"],  # string selection w/repetition + unordered
    [0, 1, 1, 2, 3],  # preserve repetition, don't convert to slice
    range(2, 5),  # narrow range
    slice(0, 5, 2),  # slice w/non-unitary step-size
    ]
timeSelections = list(zip(["toi"] * len(toiSelections), toiSelections)) \
    + list(zip(["toilim"] * len(toilimSelections), toilimSelections))
freqSelections = list(zip(["foi"] * len(foiSelections), foiSelections)) \
    + list(zip(["foilim"] * len(foilimSelections), foilimSelections))


# Local helper function for performing basic arithmetic tests
def _base_op_tests(dummy, ymmud, dummy2, ymmud2, dummyC, operation):

    dummyArr = 2 * np.ones((dummy.trials[0].shape))
    ymmudArr = 2 * np.ones((ymmud.trials[0].shape))
    scalarOperands = [2, np.pi]
    dummyOperands = [dummyArr, dummyArr.tolist()]
    ymmudOperands = [ymmudArr, ymmudArr.tolist()]

    # Ensure trial counts are properly vetted
    dummy2.selectdata(trials=[0], inplace=True)
    with pytest.raises(SPYValueError) as spyval:
        operation(dummy, dummy2)
        assert "Syncopy object with same number of trials (selected)" in str (spyval.value)
    dummy2._selection = None

    # Scalar algebra must be commutative (except for pow)
    for operand in scalarOperands:
        result = operation(dummy, operand) # perform operation from right
        for tk, trl in enumerate(result.trials):
            assert np.array_equal(trl, operation(dummy.trials[tk], operand))
        # Don't try to compute `2 ** data``
        if operation(2,3) != 8:
            result2 = operation(operand, dummy) # perform operation from left
            assert np.array_equal(result2.data, result.data)

        # Same as above, but swapped `dimord`
        result = operation(ymmud, operand)
        for tk, trl in enumerate(result.trials):
            assert np.array_equal(trl, operation(ymmud.trials[tk], operand))
        if operation(2,3) != 8:
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
    dummy._selection = None
    dummy2._selection = None
    ymmud._selection = None
    ymmud2._selection = None


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

    def test_empty(self):
        dummy = AnalogData()
        assert len(dummy.cfg) == 0
        assert dummy.dimord == None
        for attr in ["channel", "data", "hdr", "sampleinfo", "trialinfo"]:
            assert getattr(dummy, attr) is None
        with pytest.raises(SPYTypeError):
            AnalogData({})

    def test_nparray(self):
        dummy = AnalogData(data=self.data)
        assert dummy.dimord == AnalogData._defaultDimord
        assert dummy.channel.size == self.nc
        assert (dummy.sampleinfo == [0, self.ns]).min()
        assert dummy.trialinfo.shape == (1, 0)
        assert np.array_equal(dummy.data, self.data)

        # wrong shape for data-type
        with pytest.raises(SPYValueError):
            AnalogData(np.ones((3,)))

    @pytest.mark.skip(reason="VirtualData is currently not supported")
    def test_virtualdata(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "dummy.npy")
            np.save(fname, self.data)
            dmap = open_memmap(fname, mode="r")
            vdata = VirtualData([dmap, dmap])
            dummy = AnalogData(vdata)
            assert dummy.channel.size == 2 * self.nc
            assert len(dummy._filename) == 2
            assert isinstance(dummy.filename, str)
            del dmap, dummy, vdata

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

        # # test ``_copy_trial`` with memmap'ed data
        # with tempfile.TemporaryDirectory() as tdir:
        #     fname = os.path.join(tdir, "dummy.npy")
        #     np.save(fname, self.data)
        #     mm = open_memmap(fname, mode="r")
        #     dummy = AnalogData(mm, trialdefinition=self.trl)
        #     for trlno, start in enumerate(range(0, self.ns, 5)):
        #         trl_ref = self.data[start:start + 5, :]
        #         trl_tmp = dummy._copy_trial(trlno,
        #                                     dummy.filename,
        #                                     dummy.dimord,
        #                                     dummy.sampleinfo,
        #                                     dummy.hdr)
        #         assert np.array_equal(trl_tmp, trl_ref)
        #
        #    # Delete all open references to file objects b4 closing tmp dir
        #    del mm, dummy
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

    def test_relative_array_padding(self):

        # no. of samples to pad
        n_center = 5
        n_pre = 2
        n_post = 3
        n_half = int(n_center / 2)

        # dict for for calling `padding`
        lockws = {"center": {"padlength": n_center},
                  "pre": {"prepadlength": n_pre},
                  "post": {"postpadlength": n_post},
                  "prepost": {"prepadlength": n_pre, "postpadlength": 3}
                  }

        # expected results for padding technique (pre/post/center/prepost) and
        # all available `padtype`'s
        expected_vals = {
            "center": {"zero": [0, 0],
                       "nan": [np.nan, np.nan],
                       "mean": [np.tile(self.data.mean(axis=0), (n_half, 1)),
                                np.tile(self.data.mean(axis=0), (n_half, 1))],
                       "localmean": [np.tile(self.data[:n_half, :].mean(axis=0), (n_half, 1)),
                                     np.tile(self.data[-n_half:, :].mean(axis=0), (n_half, 1))],
                       "edge": [np.tile(self.data[0, :], (n_half, 1)),
                                np.tile(self.data[-1, :], (n_half, 1))],
                       "mirror": [self.data[1:1 + n_half, :][::-1],
                                  self.data[-1 - n_half:-1, :][::-1]]
                       },
            "pre": {"zero": [0],
                    "nan": [np.nan],
                    "mean": [np.tile(self.data.mean(axis=0), (n_pre, 1))],
                    "localmean": [np.tile(self.data[:n_pre, :].mean(axis=0), (n_pre, 1))],
                    "edge": [np.tile(self.data[0, :], (n_pre, 1))],
                    "mirror": [self.data[1:1 + n_pre, :][::-1]]
                    },
            "post": {"zero": [0],
                     "nan": [np.nan],
                     "mean": [np.tile(self.data.mean(axis=0), (n_post, 1))],
                     "localmean": [np.tile(self.data[-n_post:, :].mean(axis=0), (n_post, 1))],
                     "edge": [np.tile(self.data[-1, :], (n_post, 1))],
                     "mirror": [self.data[-1 - n_post:-1, :][::-1]]
                     },
            "prepost": {"zero": [0, 0],
                        "nan": [np.nan, np.nan],
                        "mean": [np.tile(self.data.mean(axis=0), (n_pre, 1)),
                                 np.tile(self.data.mean(axis=0), (n_post, 1))],
                        "localmean": [np.tile(self.data[:n_pre, :].mean(axis=0), (n_pre, 1)),
                                      np.tile(self.data[-n_post:, :].mean(axis=0), (n_post, 1))],
                        "edge": [np.tile(self.data[0, :], (n_pre, 1)),
                                 np.tile(self.data[-1, :], (n_post, 1))],
                        "mirror": [self.data[1:1 + n_pre, :][::-1],
                                   self.data[-1 - n_post:-1, :][::-1]]
                        }
        }

        # indices for slicing resulting array to extract padded values for validation
        expected_idx = {"center": [slice(None, n_half), slice(-n_half, None)],
                        "pre": [slice(None, n_pre)],
                        "post": [slice(-n_post, None)],
                        "prepost": [slice(None, n_pre), slice(-n_post, None)]}

        # expected shape of resulting array
        expected_shape = {"center": self.data.shape[0] + 2 * n_half,
                          "pre": self.data.shape[0] + n_pre,
                          "post": self.data.shape[0] + n_post,
                          "prepost": self.data.shape[0] + n_pre + n_post}

        # happy padding
        for loc, kws in lockws.items():
            for ptype in ["zero", "mean", "localmean", "edge", "mirror"]:
                arr = padding(self.data, ptype, pad="relative", **kws)
                for k, idx in enumerate(expected_idx[loc]):
                    assert np.all(arr[idx, :] == expected_vals[loc][ptype][k])
                assert arr.shape[0] == expected_shape[loc]
            arr = padding(self.data, "nan", pad="relative", **kws)
            for idx in expected_idx[loc]:
                assert np.all(np.isnan(arr[idx, :]))
            assert arr.shape[0] == expected_shape[loc]

        # overdetermined padding
        with pytest.raises(SPYTypeError):
            padding(self.data, "zero", pad="relative", padlength=5,
                    prepadlength=2)
        with pytest.raises(SPYTypeError):
            padding(self.data, "zero", pad="relative", padlength=5,
                    postpadlength=2)
        with pytest.raises(SPYTypeError):
            padding(self.data, "zero", pad="relative", padlength=5,
                    prepadlength=2, postpadlength=2)

        # float input for sample counts
        with pytest.raises(SPYValueError):
            padding(self.data, "zero", pad="relative", padlength=2.5)
        with pytest.raises(SPYValueError):
            padding(self.data, "zero", pad="relative", prepadlength=2.5)
        with pytest.raises(SPYValueError):
            padding(self.data, "zero", pad="relative", postpadlength=2.5)
        with pytest.raises(SPYValueError):
            padding(self.data, "zero", pad="relative", prepadlength=2.5,
                    postpadlength=2.5)

        # time-based padding w/array input
        with pytest.raises(SPYValueError):
            padding(self.data, "zero", pad="relative", padlength=2, unit="time")

    def test_absolute_nextpow2_array_padding(self):

        pad_count = {"absolute": self.ns + 20,
                     "nextpow2": int(2**np.ceil(np.log2(self.ns)))}
        kws = {"absolute": pad_count["absolute"],
               "nextpow2": None}

        for pad, n_total in pad_count.items():

            n_fillin = n_total - self.ns
            n_half = int(n_fillin / 2)

            arr = padding(self.data, "zero", pad=pad, padlength=kws[pad])
            assert np.all(arr[:n_half, :] == 0)
            assert np.all(arr[-n_half:, :] == 0)
            assert arr.shape[0] == n_total

            arr = padding(self.data, "zero", pad=pad, padlength=kws[pad],
                          prepadlength=True)
            assert np.all(arr[:n_fillin, :] == 0)
            assert arr.shape[0] == n_total

            arr = padding(self.data, "zero", pad=pad, padlength=kws[pad],
                          postpadlength=True)
            assert np.all(arr[-n_fillin:, :] == 0)
            assert arr.shape[0] == n_total

            arr = padding(self.data, "zero", pad=pad, padlength=kws[pad],
                          prepadlength=True, postpadlength=True)
            assert np.all(arr[:n_half, :] == 0)
            assert np.all(arr[-n_half:, :] == 0)
            assert arr.shape[0] == n_total

        # 'absolute'-specific errors: `padlength` too short, wrong type, wrong combo with `prepadlength`
        with pytest.raises(SPYValueError):
            padding(self.data, "zero", pad="absolute", padlength=self.ns - 1)
        with pytest.raises(SPYTypeError):
            padding(self.data, "zero", pad="absolute", prepadlength=self.ns)
        with pytest.raises(SPYTypeError):
            padding(self.data, "zero", pad="absolute", padlength=n_total, prepadlength=n_total)

        # 'nextpow2'-specific errors: `padlength` wrong type, wrong combo with `prepadlength`
        with pytest.raises(SPYTypeError):
            padding(self.data, "zero", pad="nextpow2", padlength=self.ns)
        with pytest.raises(SPYTypeError):
            padding(self.data, "zero", pad="nextpow2", prepadlength=self.ns)
        with pytest.raises(SPYTypeError):
            padding(self.data, "zero", pad="nextpow2", padlength=n_total, prepadlength=True)

    def test_object_padding(self):

        # construct AnalogData object w/trials of unequal lengths
        adata = generate_artificial_data(nTrials=7, nChannels=16,
                                        equidistant=False, inmemory=False)
        timeAxis = adata.dimord.index("time")
        chanAxis = adata.dimord.index("channel")

        # Define trial/channel selections for tests
        trialSel = [0, 2, 1]
        chanSel = range(4)

        # test dictionary generation for `create_new = False`: ensure all trials
        # have padded length of `total_time` seconds (1 sample tolerance)
        total_time = 30
        pad_list = padding(adata, "zero", pad="absolute", padlength=total_time,
                           unit="time", create_new=False)
        for tk, trl in enumerate(adata.trials):
            assert "pad_width" in pad_list[tk].keys()
            assert "constant_values" in pad_list[tk].keys()
            trl_time = (pad_list[tk]["pad_width"][timeAxis, :].sum() + trl.shape[timeAxis]) / adata.samplerate
            assert trl_time - total_time < 1 / adata.samplerate

        # real thing: pad object with standing channel selection
        res = padding(adata, "zero", pad="absolute", padlength=total_time,unit="time",
                      create_new=True, select={"trials": trialSel, "channels": chanSel})
        for tk, trl in enumerate(res.trials):
            adataTrl = adata.trials[trialSel[tk]]
            nSamples = pad_list[trialSel[tk]]["pad_width"][timeAxis, :].sum() + adataTrl.shape[timeAxis]
            assert trl.shape[timeAxis] == nSamples
            assert trl.shape[chanAxis] == len(list(chanSel))

        # test correct update of trigger onset w/pre-padding
        adataTimes = adata.time
        prepadTime = 5
        res = padding(adata, "zero", pad="relative", prepadlength=prepadTime,
                      unit="time", create_new=True)
        resTimes = res.time
        adataTimes = adata.time
        for tk, timeArr in enumerate(resTimes):
            assert timeArr[0] == adataTimes[tk][0] - prepadTime
            assert np.array_equal(timeArr[timeArr >= 0], adataTimes[tk][adataTimes[tk] >= 0])

        # postpadding must not change trigger onset timing
        postpadTime = 5
        res = padding(adata, "zero", pad="relative", postpadlength=postpadTime,
                      unit="time", create_new=True)
        resTimes = res.time
        for tk, timeArr in enumerate(resTimes):
            assert timeArr[0] == adataTimes[tk][0]
            assert np.array_equal(timeArr[timeArr <= 0], adataTimes[tk][adataTimes[tk] <= 0])

        # jumble axes of `AnalogData` object and compute max. trial length
        adata2 = generate_artificial_data(nTrials=7, nChannels=16,
                                          equidistant=False, inmemory=False,
                                          dimord=adata.dimord[::-1])
        timeAxis2 = adata2.dimord.index("time")
        chanAxis2 = adata2.dimord.index("channel")
        maxtrllen = 0
        for trl in adata2.trials:
            maxtrllen = max(maxtrllen, trl.shape[timeAxis2])

        # same as above, but this time w/swapped dimensions
        res2 = padding(adata2, "zero", pad="absolute", padlength=total_time, unit="time",
                       create_new=True, select={"trials": trialSel, "channels": chanSel})
        pad_list2 = padding(adata2, "zero", pad="absolute", padlength=total_time,
                            unit="time", create_new=False)
        for tk, trl in enumerate(res2.trials):
            adataTrl = adata2.trials[trialSel[tk]]
            nSamples = pad_list2[trialSel[tk]]["pad_width"][timeAxis2, :].sum() + adataTrl.shape[timeAxis2]
            assert trl.shape[timeAxis2] == nSamples
            assert trl.shape[chanAxis2] == len(list(chanSel))

        # symmetric `maxlen` padding: 1 sample tolerance
        pad_list2 = padding(adata2, "zero", pad="maxlen", create_new=False)
        for tk, trl in enumerate(adata2.trials):
            trl_len = pad_list2[tk]["pad_width"][timeAxis2, :].sum() + trl.shape[timeAxis2]
            assert (trl_len - maxtrllen) <= 1
        pad_list2 = padding(adata2, "zero", pad="maxlen", prepadlength=True,
                            postpadlength=True, create_new=False)
        for tk, trl in enumerate(adata2.trials):
            trl_len = pad_list2[tk]["pad_width"][timeAxis2, :].sum() + trl.shape[timeAxis2]
            assert (trl_len - maxtrllen) <= 1

        # pre- and post- `maxlen` padding: no tolerance
        pad_list2 = padding(adata2, "zero", pad="maxlen", prepadlength=True,
                            create_new=False)
        for tk, trl in enumerate(adata2.trials):
            trl_len = pad_list2[tk]["pad_width"][timeAxis2, :].sum() + trl.shape[timeAxis2]
            assert trl_len == maxtrllen
        pad_list2 = padding(adata2, "zero", pad="maxlen", postpadlength=True,
                            create_new=False)
        for tk, trl in enumerate(adata2.trials):
            trl_len = pad_list2[tk]["pad_width"][timeAxis2, :].sum() + trl.shape[timeAxis2]
            assert trl_len == maxtrllen

        # make things maximally intersting: relative + time + non-equidistant +
        # overlapping + selection + nonstandard dimord
        adata3 = generate_artificial_data(nTrials=7, nChannels=16,
                                          equidistant=False, overlapping=True,
                                          inmemory=False, dimord=adata2.dimord)
        res3 = padding(adata3, "zero", pad="absolute", padlength=total_time, unit="time",
                       create_new=True, select={"trials": trialSel, "channels": chanSel})
        pad_list3 = padding(adata3, "zero", pad="absolute", padlength=total_time,
                            unit="time", create_new=False)
        for tk, trl in enumerate(res3.trials):
            adataTrl = adata3.trials[trialSel[tk]]
            nSamples = pad_list3[trialSel[tk]]["pad_width"][timeAxis2, :].sum() + adataTrl.shape[timeAxis2]
            assert trl.shape[timeAxis2] == nSamples
            assert trl.shape[chanAxis2] == len(list(chanSel))

        # `maxlen'-specific errors: `padlength` wrong type, wrong combo with `prepadlength`
        with pytest.raises(SPYTypeError):
            padding(adata, "zero", pad="maxlen", padlength=self.ns, create_new=False)
        with pytest.raises(SPYTypeError):
            padding(adata, "zero", pad="maxlen", prepadlength=self.ns, create_new=False)
        with pytest.raises(SPYTypeError):
            padding(adata, "zero", pad="maxlen", padlength=self.ns, prepadlength=True,
                    create_new=False)

    # test data-selection via class method
    def test_dataselection(self):

        # Create testing objects (regular and swapped dimords)
        dummy = AnalogData(data=self.data,
                           trialdefinition=self.trl,
                           samplerate=self.samplerate)
        ymmud = AnalogData(data=self.data.T,
                           trialdefinition=self.trl,
                           samplerate=self.samplerate,
                           dimord=AnalogData._defaultDimord[::-1])

        for obj in [dummy, ymmud]:
            idx = [slice(None)] * len(obj.dimord)
            timeIdx = obj.dimord.index("time")
            chanIdx = obj.dimord.index("channel")
            for trialSel in trialSelections:
                for chanSel in chanSelections:
                    for timeSel in timeSelections:
                        kwdict = {}
                        kwdict["trials"] = trialSel
                        kwdict["channels"] = chanSel
                        kwdict[timeSel[0]] = timeSel[1]
                        cfg = StructDict(kwdict)
                        # data selection via class-method + `Selector` instance for indexing
                        selected = obj.selectdata(**kwdict)
                        time.sleep(0.05)
                        selector = Selector(obj, kwdict)
                        idx[chanIdx] = selector.channel
                        for tk, trialno in enumerate(selector.trials):
                            idx[timeIdx] = selector.time[tk]
                            assert np.array_equal(selected.trials[tk].squeeze(),
                                                  obj.trials[trialno][idx[0], :][:, idx[1]].squeeze())
                        cfg.data = obj
                        cfg.out = AnalogData(dimord=obj.dimord)
                        # data selection via package function and `cfg`: ensure equality
                        selectdata(cfg)
                        assert np.array_equal(cfg.out.channel, selected.channel)
                        assert np.array_equal(cfg.out.data, selected.data)
                        time.sleep(0.05)

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
                assert "expected Syncopy 'time' x 'channel' data object" in str (spyval.value)

            _base_op_tests(dummy, ymmud, dummy2, ymmud2, None, operation)

            # Now the most complicated case: user-defined subset selections are present
            kwdict = {}
            kwdict["trials"] = trialSelections[1]
            kwdict["channels"] = chanSelections[3]
            kwdict[timeSelections[4][0]] = timeSelections[4][1]
            _selection_op_tests(dummy, ymmud, dummy2, ymmud2, kwdict, operation)

            # # Go through full selection stack - WARNING: this takes > 15 minutes
            # for trialSel in trialSelections:
            #     for chanSel in chanSelections:
            #         for timeSel in timeSelections:
            #             kwdict = {}
            #             kwdict["trials"] = trialSel
            #             kwdict["channels"] = chanSel
            #             kwdict[timeSel[0]] = timeSel[1]
            #             _selection_op_tests(dummy, ymmud, dummy2, ymmud2, kwdict, operation)

        # Finally, perform a representative chained operation to ensure chaining works
        result = (dummy + dummy2) / dummy ** 3
        for tk, trl in enumerate(result.trials):
            assert np.array_equal(trl,
                                  (dummy.trials[tk] + dummy2.trials[tk]) / dummy.trials[tk] ** 3)

    @skip_without_acme
    def test_parallel(self, testcluster):
        # repeat selected test w/parallel processing engine
        client = dd.Client(testcluster)
        par_tests = ["test_relative_array_padding",
                     "test_absolute_nextpow2_array_padding",
                     "test_object_padding",
                     "test_dataselection",
                     "test_ang_arithmetic"]
        for test in par_tests:
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
        assert dummy.dimord == None
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

    # test data-selection via class method
    def test_sd_dataselection(self):

        # Create testing objects (regular and swapped dimords)
        dummy = SpectralData(data=self.data,
                             trialdefinition=self.trl,
                             samplerate=self.samplerate,
                             taper=["TestTaper_0{}".format(k) for k in range(1, self.nt + 1)])
        ymmud = SpectralData(data=np.transpose(self.data, [3, 2, 1, 0]),
                             trialdefinition=self.trl,
                             samplerate=self.samplerate,
                             taper=["TestTaper_0{}".format(k) for k in range(1, self.nt + 1)],
                             dimord=SpectralData._defaultDimord[::-1])

        for obj in [dummy, ymmud]:
            idx = [slice(None)] * len(obj.dimord)
            timeIdx = obj.dimord.index("time")
            chanIdx = obj.dimord.index("channel")
            freqIdx = obj.dimord.index("freq")
            taperIdx = obj.dimord.index("taper")
            for trialSel in trialSelections:
                for chanSel in chanSelections:
                    for timeSel in timeSelections:
                        for freqSel in freqSelections:
                            for taperSel in taperSelections:
                                kwdict = {}
                                kwdict["trials"] = trialSel
                                kwdict["channels"] = chanSel
                                kwdict[timeSel[0]] = timeSel[1]
                                kwdict[freqSel[0]] = freqSel[1]
                                kwdict["tapers"] = taperSel
                                cfg = StructDict(kwdict)
                                # data selection via class-method + `Selector` instance for indexing
                                selected = obj.selectdata(**kwdict)
                                time.sleep(0.05)
                                selector = Selector(obj, kwdict)
                                idx[chanIdx] = selector.channel
                                idx[freqIdx] = selector.freq
                                idx[taperIdx] = selector.taper
                                for tk, trialno in enumerate(selector.trials):
                                    idx[timeIdx] = selector.time[tk]
                                    indexed = obj.trials[trialno][idx[0], ...][:, idx[1], ...][:, :, idx[2], :][..., idx[3]]
                                    assert np.array_equal(selected.trials[tk].squeeze(),
                                                        indexed.squeeze())
                                cfg.data = obj
                                cfg.out = SpectralData(dimord=obj.dimord)
                                # data selection via package function and `cfg`: ensure equality
                                selectdata(cfg)
                                assert np.array_equal(cfg.out.channel, selected.channel)
                                assert np.array_equal(cfg.out.freq, selected.freq)
                                assert np.array_equal(cfg.out.taper, selected.taper)
                                assert np.array_equal(cfg.out.data, selected.data)
                                time.sleep(0.05)

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

            # Now the most complicated case: user-defined subset selections are present
            kwdict = {}
            kwdict["trials"] = trialSelections[1]
            kwdict["channels"] = chanSelections[3]
            kwdict[timeSelections[4][0]] = timeSelections[4][1]
            kwdict[freqSelections[4][0]] = freqSelections[4][1]
            kwdict["tapers"] = taperSelections[2]
            _selection_op_tests(dummy, ymmud, dummy2, ymmud2, kwdict, operation)

            # # Go through full selection stack - WARNING: this takes > 1 hour
            # for trialSel in trialSelections:
            #     for chanSel in chanSelections:
            #         for timeSel in timeSelections:
            #             for freqSel in freqSelections:
            #                 for taperSel in taperSelections:
            #                     kwdict = {}
            #                     kwdict["trials"] = trialSel
            #                     kwdict["channels"] = chanSel
            #                     kwdict[timeSel[0]] = timeSel[1]
            #                     kwdict[freqSel[0]] = freqSel[1]
            #                     kwdict["tapers"] = taperSel
            #                     _selection_op_tests(dummy, ymmud, dummy2, ymmud2, kwdict, operation)

        # Finally, perform a representative chained operation to ensure chaining works
        result = (dummy + dummy2) / dummy ** 3
        for tk, trl in enumerate(result.trials):
            assert np.array_equal(trl,
                                  (dummy.trials[tk] + dummy2.trials[tk]) / dummy.trials[tk] ** 3)

    @skip_without_acme
    def test_sd_parallel(self, testcluster):
        # repeat selected test w/parallel processing engine
        client = dd.Client(testcluster)
        par_tests = ["test_sd_dataselection", "test_sd_arithmetic"]
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
        assert dummy.dimord == None
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
            assert dummy2.channel_i.size == self.nci # swapped
            assert dummy2.channel_j.size == self.nf # swapped
            assert dummy2.freq.size == self.ns  # swapped
            assert dummy2.data.shape == dummy.data.shape

            # Delete all open references to file objects b4 closing tmp dir
            del dummy, dummy2

    # test data-selection via class method
    def test_csd_dataselection(self):

        # Create testing objects (regular and swapped dimords)
        dummy = CrossSpectralData(data=self.data,
                                  samplerate=self.samplerate)
        dummy.trialdefinition = self.trl
        ymmud = CrossSpectralData(data=np.transpose(self.data, [3, 2, 1, 0]),
                                  samplerate=self.samplerate,
                                  dimord=CrossSpectralData._defaultDimord[::-1])
        ymmud.trialdefinition = self.trl

        for obj in [dummy, ymmud]:
            idx = [slice(None)] * len(obj.dimord)
            timeIdx = obj.dimord.index("time")
            chanIdx = obj.dimord.index("channel_i")
            chanJdx = obj.dimord.index("channel_j")
            freqIdx = obj.dimord.index("freq")
            for trialSel in trialSelections:
                for chaniSel in chanSelections:
                    for chanjSel in chanSelections:
                        for timeSel in timeSelections:
                            for freqSel in freqSelections:
                                kwdict = {}
                                kwdict["trials"] = trialSel
                                kwdict["channels_i"] = chaniSel
                                kwdict["channels_j"] = chanjSel
                                kwdict[timeSel[0]] = timeSel[1]
                                kwdict[freqSel[0]] = freqSel[1]
                                cfg = StructDict(kwdict)
                                # data selection via class-method + `Selector` instance for indexing
                                selected = obj.selectdata(**kwdict)
                                time.sleep(0.05)
                                selector = Selector(obj, kwdict)
                                idx[chanIdx] = selector.channel_i
                                idx[chanJdx] = selector.channel_j
                                idx[freqIdx] = selector.freq
                                for tk, trialno in enumerate(selector.trials):
                                    idx[timeIdx] = selector.time[tk]
                                    indexed = obj.trials[trialno][idx[0], ...][:, idx[1], ...][:, :, idx[2], :][..., idx[3]]
                                    assert np.array_equal(selected.trials[tk].squeeze(),
                                                        indexed.squeeze())
                                cfg.data = obj
                                cfg.out = CrossSpectralData(dimord=obj.dimord)
                                # data selection via package function and `cfg`: ensure equality
                                selectdata(cfg)
                                assert np.array_equal(cfg.out.channel_i, selected.channel_i)
                                assert np.array_equal(cfg.out.channel_j, selected.channel_j)
                                assert np.array_equal(cfg.out.freq, selected.freq)
                                assert np.array_equal(cfg.out.data, selected.data)
                                time.sleep(0.05)

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

            # Now the most complicated case: user-defined subset selections are present
            kwdict = {}
            kwdict["trials"] = trialSelections[1]
            kwdict["channels_i"] = chanSelections[3]
            kwdict["channels_j"] = chanSelections[2]
            kwdict[timeSelections[4][0]] = timeSelections[4][1]
            kwdict[freqSelections[4][0]] = freqSelections[4][1]
            _selection_op_tests(dummy, ymmud, dummy2, ymmud2, kwdict, operation)

        # Finally, perform a representative chained operation to ensure chaining works
        result = (dummy + dummy2) / dummy ** 3
        for tk, trl in enumerate(result.trials):
            assert np.array_equal(trl,
                                  (dummy.trials[tk] + dummy2.trials[tk]) / dummy.trials[tk] ** 3)

    @skip_without_acme
    def test_csd_parallel(self, testcluster):
        # repeat selected test w/parallel processing engine
        client = dd.Client(testcluster)
        par_tests = ["test_csd_dataselection", "test_csd_arithmetic"]
        for test in par_tests:
            getattr(self, test)()
            flush_local_cluster(testcluster)
        client.close()
