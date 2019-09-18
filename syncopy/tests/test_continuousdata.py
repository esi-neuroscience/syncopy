# -*- coding: utf-8 -*-
# 
# 
# 
# Created: 2019-03-20 11:46:31
# Last modified by: Joscha Schmiedt [joscha.schmiedt@esi-frankfurt.de]
# Last modification time: <2019-09-18 14:53:54>

import os
import tempfile
import time
import pytest
import numpy as np
from numpy.lib.format import open_memmap
from syncopy.datatype import AnalogData, SpectralData, padding
from syncopy.io import save, load
from syncopy.datatype.base_data import VirtualData
from syncopy.shared.errors import SPYValueError, SPYTypeError
from syncopy.tests.misc import generate_artifical_data, construct_spy_filename


class TestAnalogData():

    # Allocate test-dataset
    nc = 10
    ns = 30
    data = np.arange(1, nc * ns + 1, dtype="float").reshape(ns, nc)
    trl = np.vstack([np.arange(0, ns, 5),
                     np.arange(5, ns + 5, 5),
                     np.ones((int(ns / 5), )),
                     np.ones((int(ns / 5), )) * np.pi]).T

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

        # test ``_copy_trial`` with memmap'ed data
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "dummy.npy")
            np.save(fname, self.data)
            mm = open_memmap(fname, mode="r")
            dummy = AnalogData(mm, trialdefinition=self.trl)
            for trlno, start in enumerate(range(0, self.ns, 5)):
                trl_ref = self.data[start:start + 5, :]
                trl_tmp = dummy._copy_trial(trlno,
                                            dummy.filename,
                                            dummy.dimord,
                                            dummy.sampleinfo,
                                            dummy.hdr)
                assert np.array_equal(trl_tmp, trl_ref)

            # Delete all open references to file objects b4 closing tmp dir
            del mm, dummy

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
            
            # # FIXME: either remove or repair this
            # # save object hosting VirtualData
            # np.save(fname + ".npy", self.data)
            # dmap = open_memmap(fname + ".npy", mode="r")
            # vdata = VirtualData([dmap, dmap])
            # dummy = AnalogData(vdata, samplerate=1000)
            # dummy.save(fname, overwrite=True)
            # dummy2 = AnalogData(filename)
            # assert dummy2.mode == "r+"
            # assert np.array_equal(dummy2.data, vdata[:, :])
            # del dummy, dummy2  # avoid PermissionError in Windows
            
            # ensure trialdefinition is saved and loaded correctly
            dummy = AnalogData(data=self.data, trialdefinition=self.trl, samplerate=1000)
            dummy.save(fname + "_trl")
            filename = construct_spy_filename(fname + "_trl", dummy)
            dummy2 = load(filename)
            assert np.array_equal(dummy.trialdefinition, dummy2.trialdefinition)
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
        adata = generate_artifical_data(nTrials=7, nChannels=16,
                                        equidistant=False, inmemory=False)
        timeAxis = adata.dimord.index("time")

        # test dictionary generation for `create_new = False`: ensure all trials
        # have padded length of `total_time` seconds (1 sample tolerance)
        total_time = 30
        pad_list = padding(adata, "zero", pad="absolute", padlength=total_time,
                           unit="time", create_new=False)
        for tk, trl in enumerate(adata.trials):
            assert "pad_width" in pad_list[tk].keys()
            assert "constant_values" in pad_list[tk].keys()
            trl_time = (pad_list[tk]["pad_width"][timeAxis, :].sum() + trl.shape[timeAxis]) / adata.samplerate
            assert trl_time - total_time < 1/adata.samplerate

        # jumble axes of `AnalogData` object and compute max. trial length
        adata2 = generate_artifical_data(nTrials=7, nChannels=16,
                                         equidistant=False, inmemory=False,
                                         dimord=adata.dimord[::-1])
        timeAxis2 = adata2.dimord.index("time")
        maxtrllen = 0
        for trl in adata2.trials:
            maxtrllen = max(maxtrllen, trl.shape[timeAxis2])

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

        # `maxlen'-specific errors: `padlength` wrong type, wrong combo with `prepadlength`
        with pytest.raises(SPYTypeError):
            padding(adata, "zero", pad="maxlen", padlength=self.ns, create_new=False)
        with pytest.raises(SPYTypeError):
            padding(adata, "zero", pad="maxlen", prepadlength=self.ns, create_new=False)
        with pytest.raises(SPYTypeError):
            padding(adata, "zero", pad="maxlen", padlength=self.ns, prepadlength=True,
                    create_new=False)

        # FIXME: implement as soon as object padding is supported:
        # test absolute + time + non-equidistant!
        # test relative + time + non-equidistant + overlapping!


class TestSpectralData():

    # Allocate test-dataset
    nc = 10
    ns = 30
    nt = 5
    nf = 15
    data = np.arange(1, nc * ns * nt * nf + 1).reshape(ns, nt, nf, nc)
    trl = np.vstack([np.arange(0, ns, 5),
                     np.arange(5, ns + 5, 5),
                     np.ones((int(ns / 5), )),
                     np.ones((int(ns / 5), )) * np.pi]).T
    data2 = np.moveaxis(data, 0, -1)

    def test_empty(self):
        dummy = SpectralData()
        assert len(dummy.cfg) == 0
        assert dummy.dimord == None
        for attr in ["channel", "data", "freq", "sampleinfo", "taper", "trialinfo"]:
            assert getattr(dummy, attr) is None
        with pytest.raises(SPYTypeError):
            SpectralData({})

    def test_nparray(self):
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

    def test_trialretrieval(self):
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

        # test ``_copy_trial`` with memmap'ed data
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "dummy.npy")
            np.save(fname, self.data)
            mm = open_memmap(fname, mode="r")
            dummy = SpectralData(mm, trialdefinition=self.trl)
            for trlno, start in enumerate(range(0, self.ns, 5)):
                trl_ref = self.data[start:start + 5, ...]
                trl_tmp = dummy._copy_trial(trlno,
                                            dummy.filename,
                                            dummy.dimord,
                                            dummy.sampleinfo,
                                            None)
                assert np.array_equal(trl_tmp, trl_ref)
            del mm, dummy

    def test_saveload(self):
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
