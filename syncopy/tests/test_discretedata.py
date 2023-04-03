# -*- coding: utf-8 -*-
#
# Test proper functionality of Syncopy DiscreteData-type classes
#

# Builtin/3rd party package imports
import os
import tempfile
import time
import h5py
import pytest
import numpy as np


# Local imports
import syncopy as spy
from syncopy.datatype import AnalogData, SpikeData, EventData
from syncopy.io import save, load
from syncopy.shared.errors import SPYValueError, SPYTypeError
from syncopy.tests.misc import construct_spy_filename
from syncopy.tests.test_selectdata import getSpikeData


class TestSpikeData():

    # Allocate test-dataset
    nc = 10
    ns = 30
    nd = 50
    seed = np.random.RandomState(13)
    data = np.vstack([seed.choice(ns, size=nd),
                      seed.choice(nc, size=nd),
                      seed.choice(int(nc / 2), size=nd)]).T
    data = data[data[:,0].argsort()]
    data2 = data.copy()
    data2[:, -1] = data[:, 0]
    data2[:, 0] = data[:, -1]
    trl = np.vstack([np.arange(0, ns, 5),
                     np.arange(5, ns + 5, 5),
                     np.ones((int(ns / 5), )),
                     np.ones((int(ns / 5), )) * np.pi]).T
    num_smp = np.unique(data[:, 0]).size
    num_chn = data[:, 1].max() + 1
    num_unt = data[:, 2].max() + 1

    def test_init(self):

        # data and no labels triggers default labels
        dummy = SpikeData(data=4  * np.ones((2, 3), dtype=int))
        # labels are 0-based
        assert dummy.channel == 'channel05'
        assert dummy.unit == 'unit05'

        # data and fitting labels is fine
        assert isinstance(SpikeData(data=np.ones((2, 3), dtype=int), channel=['only_channel']),
                          SpikeData)

        # --- invalid inits ---

        # non-integer types
        with pytest.raises(SPYTypeError, match='expected integer like'):
            _ = SpikeData(data=np.ones((2, 3)), unit=['unit1', 'unit2'])

        with pytest.raises(SPYTypeError, match='expected integer like'):
            data = np.array([np.nan, 2, np.nan])[:, np.newaxis]
            _ = SpikeData(data=data, unit=['unit1', 'unit2'])

        # data and too many labels
        with pytest.raises(SPYValueError, match='expected exactly 1 unit'):
            _ = SpikeData(data=np.ones((2, 3), dtype=int), unit=['unit1', 'unit2'])

        # no data but labels
        with pytest.raises(SPYValueError, match='cannot assign `channel` without data'):
            _ = SpikeData(channel=['a', 'b', 'c'])

    def test_register_dset(self):
        sdata = SpikeData(self.data, samplerate=10)
        assert not sdata._is_empty()
        sdata._register_dataset("blah", np.zeros((3,3), dtype=float))


    def test_empty(self):
        dummy = SpikeData()
        assert len(dummy.cfg) == 0
        assert dummy.dimord is None
        for attr in ["channel", "data", "sampleinfo", "samplerate",
                     "trialid", "trialinfo", "unit"]:
            assert getattr(dummy, attr) is None
        with pytest.raises(SPYTypeError):
            SpikeData({})

    def test_issue_257_fixed_no_error_for_empty_data(self):
        """This tests that empty datasets are not allowed"""
        with pytest.raises(SPYValueError, match='non empty'):
            data = SpikeData(np.column_stack(([],[],[])).astype(int),
                             dimord=['sample', 'channel', 'unit'],
                             samplerate=30000)

    def test_nparray(self):
        dummy = SpikeData(self.data)
        assert dummy.dimord == ["sample", "channel", "unit"]
        assert dummy.channel.size == self.num_chn
        # NOTE: SpikeData.sample is currently empty
        # assert dummy.sample.size == self.num_smp
        assert dummy.unit.size == self.num_unt
        assert (dummy.sampleinfo == [0, self.data[:, 0].max()]).min()
        assert dummy.trialinfo.shape == (1, 0)
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
            checkAttr = ["channel", "data", "dimord", "sampleinfo",
                         "samplerate", "trialinfo", "unit"]
            dummy = SpikeData(self.data, samplerate=10)
            dummy.save(fname)
            filename = construct_spy_filename(fname, dummy)
            # dummy2 = SpikeData(filename)
            # for attr in checkAttr:
            #     assert np.array_equal(getattr(dummy, attr), getattr(dummy2, attr))
            dummy3 = load(fname)
            for attr in checkAttr:
                assert np.array_equal(getattr(dummy3, attr), getattr(dummy, attr))
            save(dummy3, container=os.path.join(tdir, "ymmud"))
            dummy4 = load(os.path.join(tdir, "ymmud"))
            for attr in checkAttr:
                assert np.array_equal(getattr(dummy4, attr), getattr(dummy, attr))
            del dummy3, dummy4  # avoid PermissionError in Windows
            time.sleep(0.1)  # wait to kick-off garbage collection

            # overwrite existing container w/new data
            dummy.samplerate = 20
            dummy.save()
            dummy2 = load(filename=filename)
            assert dummy2.samplerate == 20
            del dummy, dummy2
            time.sleep(0.1)  # wait to kick-off garbage collection

            # ensure trialdefinition is saved and loaded correctly
            dummy = SpikeData(self.data, trialdefinition=self.trl, samplerate=10)
            dummy.save(fname, overwrite=True)
            dummy2 = load(filename)
            assert np.array_equal(dummy.sampleinfo, dummy2.sampleinfo)
            assert np.array_equal(dummy._t0, dummy2._t0)
            assert np.array_equal(dummy.trialinfo, dummy2.trialinfo)
            del dummy, dummy2
            time.sleep(0.1)  # wait to kick-off garbage collection

            # swap dimensions and ensure `dimord` is preserved
            dummy = SpikeData(self.data, dimord=["unit", "channel", "sample"], samplerate=10)
            dummy.save(fname + "_dimswap")
            filename = construct_spy_filename(fname + "_dimswap", dummy)
            dummy2 = load(filename)
            assert dummy2.dimord == dummy.dimord
            assert dummy2.unit.size == self.num_smp  # swapped
            assert dummy2.data.shape == dummy.data.shape

            # Delete all open references to file objects b4 closing tmp dir
            del dummy, dummy2
            time.sleep(0.1)


class TestEventData():

    # Allocate test-datasets
    nc = 10
    ns = 30
    data = np.vstack([np.arange(0, ns, 5),
                      np.zeros((int(ns / 5), ))]).T.astype(int)
    data[1::2, 1] = 1
    data2 = data.copy()
    data2[:, -1] = data[:, 0]
    data2[:, 0] = data[:, -1]
    data3 = np.hstack([data2, data2])
    trl = np.vstack([np.arange(0, ns, 5),
                     np.arange(5, ns + 5, 5),
                     np.ones((int(ns / 5), )),
                     np.ones((int(ns / 5), )) * np.pi]).T
    num_smp = np.unique(data[:, 0]).size
    num_evt = np.unique(data[:, 1]).size
    customDimord = ["sample", "eventid", "custom1", "custom2"]

    adata = np.arange(1, nc * ns + 1).reshape(ns, nc)

    def test_ed_empty(self):
        dummy = EventData()
        assert len(dummy.cfg) == 0
        assert dummy.dimord is None
        for attr in ["data", "sampleinfo", "samplerate", "trialid", "trialinfo"]:
            assert getattr(dummy, attr) is None
        with pytest.raises(SPYTypeError):
            EventData({})

    def test_ed_nparray(self):
        dummy = EventData(self.data)
        assert dummy.dimord == ["sample", "eventid"]
        assert dummy.eventid.size == self.num_evt
        # NOTE: EventData.sample is currently empty
        # assert dummy.sample.size == self.num_smp
        assert (dummy.sampleinfo == [0, self.data[:, 0].max()]).min()
        assert dummy.trialinfo.shape == (1, 0)
        assert np.array_equal(dummy.data, self.data)

        # wrong shape for data-type
        with pytest.raises(SPYValueError):
            EventData(np.ones((3,)))

    def test_register_dset(self):
        edata = EventData(self.data, samplerate=10)
        assert not edata._is_empty()
        edata._register_dataset("blah", np.zeros((3,3), dtype=float))

    def test_ed_trialretrieval(self):
        # test ``_get_trial`` with NumPy array: regular order
        dummy = EventData(self.data, trialdefinition=self.trl)
        smp = self.data[:, 0]
        for trlno, start in enumerate(range(0, self.ns, 5)):
            idx = np.intersect1d(np.where(smp >= start)[0],
                                 np.where(smp < start + 5)[0])
            trl_ref = self.data[idx, ...]
            assert np.array_equal(dummy._get_trial(trlno), trl_ref)

        # test `_get_trial` with NumPy array: swapped dimensions
        dummy = EventData(self.data2, trialdefinition=self.trl,
                          dimord=["eventid", "sample"])
        smp = self.data2[:, -1]
        for trlno, start in enumerate(range(0, self.ns, 5)):
            idx = np.intersect1d(np.where(smp >= start)[0],
                                 np.where(smp < start + 5)[0])
            trl_ref = self.data2[idx, ...]
            assert np.array_equal(dummy._get_trial(trlno), trl_ref)

        # test `_get_trial` with NumPy array: customized columns names
        nuDimord = ["eventid", "sample", "custom1", "custom2"]
        dummy = EventData(self.data3, trialdefinition=self.trl,
                          dimord=nuDimord)
        assert dummy.dimord == nuDimord
        smp = self.data3[:, -1]
        for trlno, start in enumerate(range(0, self.ns, 5)):
            idx = np.intersect1d(np.where(smp >= start)[0],
                                 np.where(smp < start + 5)[0])
            trl_ref = self.data3[idx, ...]
            assert np.array_equal(dummy._get_trial(trlno), trl_ref)

    def test_ed_saveload(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "dummy")

            # basic but most important: ensure object integrity is preserved
            checkAttr = ["data", "dimord", "sampleinfo", "samplerate", "trialinfo"]
            dummy = EventData(self.data, samplerate=10)
            dummy.save(fname)
            filename = construct_spy_filename(fname, dummy)
            dummy2 = load(filename)
            for attr in checkAttr:
                assert np.array_equal(getattr(dummy, attr), getattr(dummy2, attr))
            dummy3 = load(fname)
            for attr in checkAttr:
                assert np.array_equal(getattr(dummy3, attr), getattr(dummy, attr))
            save(dummy3, container=os.path.join(tdir, "ymmud"))
            dummy4 = load(os.path.join(tdir, "ymmud"))
            for attr in checkAttr:
                assert np.array_equal(getattr(dummy4, attr), getattr(dummy, attr))
            del dummy2, dummy3, dummy4  # avoid PermissionError in Windows

            # overwrite existing file w/new data
            dummy.samplerate = 20
            dummy.save()
            dummy2 = load(filename=filename)
            assert dummy2.samplerate == 20
            del dummy, dummy2
            time.sleep(0.1)  # wait to kick-off garbage collection

            # ensure trialdefinition is saved and loaded correctly
            dummy = EventData(self.data, trialdefinition=self.trl, samplerate=10)
            dummy.save(fname, overwrite=True)
            dummy2 = load(filename)
            assert np.array_equal(dummy.sampleinfo, dummy2.sampleinfo)
            assert np.array_equal(dummy._t0, dummy2._t0)
            assert np.array_equal(dummy.trialinfo, dummy2.trialinfo)
            del dummy, dummy2

            # swap dimensions and ensure `dimord` is preserved
            dummy = EventData(self.data, dimord=["eventid", "sample"], samplerate=10)
            dummy.save(fname + "_dimswap")
            filename = construct_spy_filename(fname + "_dimswap", dummy)
            dummy2 = load(filename)
            assert dummy2.dimord == dummy.dimord
            assert dummy2.eventid.size == self.num_smp  # swapped
            assert dummy2.data.shape == dummy.data.shape
            del dummy, dummy2

            # save dataset w/custom column names and ensure `dimord` is preserved
            dummy = EventData(np.hstack([self.data, self.data]), dimord=self.customDimord, samplerate=10)
            dummy.save(fname + "_customDimord")
            filename = construct_spy_filename(fname + "_customDimord", dummy)
            dummy2 = load(filename)
            assert dummy2.dimord == dummy.dimord
            assert dummy2.eventid.size == self.num_evt
            assert dummy2.data.shape == dummy.data.shape

            # Delete all open references to file objects b4 closing tmp dir
            del dummy, dummy2
            time.sleep(0.1)

    def test_ed_trialsetting(self):

        # Create sampleinfo w/ EventData vs. AnalogData samplerate
        sr_e = 2
        sr_a = 1
        pre = 2
        post = 1
        msk = self.data[:, 1] == 1
        sinfo = np.vstack([self.data[msk, 0] / sr_e - pre,
                           self.data[msk, 0] / sr_e + post]).T
        sinfo_e = np.round(sinfo * sr_e).astype(int)
        sinfo_a = np.round(sinfo * sr_a).astype(int)

        # Compute sampleinfo w/pre, post and trigger
        evt_dummy = EventData(self.data, samplerate=sr_e)
        evt_dummy.definetrial(pre=pre, post=post, trigger=1)
        assert np.array_equal(evt_dummy.sampleinfo, sinfo_e)

        # Compute sampleinfo w/ start/stop combination
        evt_dummy = EventData(self.data, samplerate=sr_e)
        evt_dummy.definetrial(start=0, stop=1)
        sinfo2 = np.vstack([self.data[np.where(self.data[:, 1] == 0)[0], 0],
                            self.data[np.where(self.data[:, 1] == 1)[0], 0]]).T
        assert np.array_equal(sinfo2, evt_dummy.sampleinfo)

        # Same w/ more complicated data array
        samples = np.arange(0, int(self.ns / 3), 3)[1:]
        dappend = np.vstack([samples, np.full(samples.shape, 2)]).T
        data3 = np.vstack([self.data, dappend])
        data3 = np.hstack([data3, data3])
        idx = np.argsort(data3[:, 0])
        data3 = data3[idx, :]
        evt_dummy = EventData(data3, dimord=self.customDimord, samplerate=sr_e)
        evt_dummy.definetrial(start=0, stop=1)
        assert np.array_equal(sinfo2, evt_dummy.sampleinfo)

        # Compute sampleinfo w/start/stop arrays instead of scalars
        starts = [2, 2, 1]
        stops = [1, 2, 0]
        sinfo3 = np.empty((3, 2))
        dsamps = list(data3[:, 0])
        dcodes = list(data3[:, 1])
        for sk, (start, stop) in enumerate(zip(starts, stops)):
            idx = dcodes.index(start)
            start = dsamps[idx]
            dcodes = dcodes[idx + 1:]
            dsamps = dsamps[idx + 1:]
            idx = dcodes.index(stop)
            stop = dsamps[idx]
            dcodes = dcodes[idx + 1:]
            dsamps = dsamps[idx + 1:]
            sinfo3[sk, :] = [start, stop]
        evt_dummy = EventData(data3, dimord=self.customDimord, samplerate=sr_e)
        evt_dummy.definetrial(start=[2, 2, 1], stop=[1, 2, 0])
        assert np.array_equal(evt_dummy.sampleinfo, sinfo3)

        # Attach computed sampleinfo to AnalogData (data and data3 must yield identical resutls)
        evt_dummy = EventData(data=self.data, samplerate=sr_e)
        evt_dummy.definetrial(pre=pre, post=post, trigger=1)
        ang_dummy = AnalogData(self.adata, samplerate=sr_a)
        ang_dummy.definetrial(evt_dummy)
        assert np.array_equal(ang_dummy.sampleinfo, sinfo_a)
        evt_dummy = EventData(data=data3, dimord=self.customDimord, samplerate=sr_e)
        evt_dummy.definetrial(pre=pre, post=post, trigger=1)
        ang_dummy.definetrial(evt_dummy)
        assert np.array_equal(ang_dummy.sampleinfo, sinfo_a)

        # Compute and attach sampleinfo on the fly
        evt_dummy = EventData(data=self.data, samplerate=sr_e)
        ang_dummy = AnalogData(self.adata, samplerate=sr_a)
        ang_dummy.definetrial(evt_dummy, pre=pre, post=post, trigger=1)
        assert np.array_equal(ang_dummy.sampleinfo, sinfo_a)
        evt_dummy = EventData(data=data3, dimord=self.customDimord, samplerate=sr_e)
        ang_dummy = AnalogData(self.adata, samplerate=sr_a)
        ang_dummy.definetrial(evt_dummy, pre=pre, post=post, trigger=1)
        assert np.array_equal(ang_dummy.sampleinfo, sinfo_a)

        # Extend data and provoke an exception due to out of bounds error
        smp = np.vstack([np.arange(self.ns, int(2.5 * self.ns), 5),
                         np.zeros((int((1.5 * self.ns) / 5),))]).T.astype(int)
        smp[1::2, 1] = 1
        smp = np.hstack([smp, smp])
        data4 = np.vstack([data3, smp])
        evt_dummy = EventData(data=data4, dimord=self.customDimord, samplerate=sr_e)
        evt_dummy.definetrial(pre=pre, post=post, trigger=1)
        # with pytest.raises(SPYValueError):
        # ang_dummy.definetrial(evt_dummy)

        # Trimming edges produces zero-length trial
        with pytest.raises(SPYValueError):
            ang_dummy.definetrial(evt_dummy, clip_edges=True)

        # We need `clip_edges` to make trial-definition work
        data4 = data4[:-2, :]
        data4[-2, 0] = data4[-1, 0]
        evt_dummy = EventData(data=data4, dimord=self.customDimord, samplerate=sr_e)
        evt_dummy.definetrial(pre=pre, post=post, trigger=1)
        # with pytest.raises(SPYValueError):
        # ang_dummy.definetrial(evt_dummy)
        ang_dummy.definetrial(evt_dummy, clip_edges=True)
        assert ang_dummy.sampleinfo[-1, 1] == self.ns

        # Check both pre/start and/or post/stop being None
        evt_dummy = EventData(data=self.data, samplerate=sr_e)
        with pytest.raises(SPYValueError):
            evt_dummy.definetrial(trigger=1, post=post)
        with pytest.raises(SPYValueError):
            evt_dummy.definetrial(pre=pre, trigger=1)
        with pytest.raises(SPYValueError):
            evt_dummy.definetrial(start=0)
        with pytest.raises(SPYValueError):
            evt_dummy.definetrial(stop=1)
        with pytest.raises(SPYValueError):
            evt_dummy.definetrial(trigger=1)
        with pytest.raises(SPYValueError):
            evt_dummy.definetrial(pre=pre, post=post)

        # Try to define trials w/o samplerate set
        evt_dummy = EventData(data=self.data)
        with pytest.raises(SPYValueError):
            evt_dummy.definetrial(pre=pre, post=post, trigger=1)
        evt_dummy = EventData(data=self.data, samplerate=sr_e)
        ang_dummy = AnalogData(self.adata)
        with pytest.raises(SPYValueError):
            ang_dummy.definetrial(evt_dummy, pre=pre, post=post, trigger=1)

        # Try to define trials w/o data
        evt_dummy = EventData(samplerate=sr_e)
        with pytest.raises(SPYValueError):
            evt_dummy.definetrial(pre=pre, post=post, trigger=1)
        ang_dummy = AnalogData(self.adata, samplerate=sr_a)
        with pytest.raises(SPYValueError):
            ang_dummy.definetrial(evt_dummy, pre=pre, post=post, trigger=1)

class TestWaveform():

    def test_waveform_invalid_set(self):
        """Sets invalid waveform for data: dimension mismatch"""
        spiked = SpikeData(data=np.ones((2, 3), dtype=int), samplerate=10)
        assert spiked.data.shape == (2, 3,)
        with pytest.raises(SPYValueError, match="wrong size waveform"):
            spiked.waveform = np.ones((3, 3), dtype=int)

    def test_waveform_invalid_set_emptydata(self):
        """Tries to set waveform without any data."""
        spiked = SpikeData()
        with pytest.raises(SPYValueError, match="Please assign data first"):
            spiked.waveform = np.ones((3, 3), dtype=int)

    def test_waveform_invalid_set_1dim(self):
        """Tries to set waveform with data that has ndim=1."""
        spiked = SpikeData(data=np.ones((2, 3), dtype=int), samplerate=10)
        with pytest.raises(SPYValueError, match="waveform data with at least 2 dimensions"):
            spiked.waveform = np.ones((3), dtype=int)

    def test_waveform_valid_set(self):
        """Sets waveform in a correct way."""
        spiked = SpikeData(data=np.ones((2, 3), dtype=int), samplerate=10)

        assert not spiked._is_empty()
        assert spiked.data.shape == (2, 3,)
        assert type(spiked.data) == h5py.Dataset
        assert spiked._get_backing_hdf5_file_handle() is not None
        spiked.waveform = np.ones((2, 3), dtype=int)
        assert "waveform" in spiked._hdfFileDatasetProperties
        assert spiked.waveform.shape == (2, 3,)

    def test_waveform_valid_set_with_None(self):
        """Sets waveform to None, which is valid."""
        spiked = SpikeData(data=np.ones((2, 3), dtype=int), samplerate=10)
        assert not spiked._is_empty()
        assert spiked.data.shape == (2, 3,)
        spiked.waveform = np.ones((2, 3), dtype=int)
        assert spiked.waveform.shape == (2, 3,)
        spiked.waveform = None
        assert spiked.waveform is None
        # try to set again
        spiked.waveform = np.ones((2, 3), dtype=int)
        assert spiked.waveform.shape == (2, 3,)


    def test_waveform_selection_trial(self):
        numSpikes, waveform_dimsize = 20, 50
        spiked = getSpikeData(nSpikes = numSpikes)
        assert sum([s.shape[0] for s in spiked.trials]) == numSpikes
        assert spiked.waveform is None
        spiked.waveform = np.ones((numSpikes, 3, waveform_dimsize), dtype=int)
        for spikeidx in range(numSpikes):
            spiked.waveform[spikeidx, :, :] = np.ones((3, waveform_dimsize), dtype=int) * spikeidx

        trial0_nspikes = spiked.trials[0].shape[0]
        trial2_nspikes = spiked.trials[2].shape[0]

        # Select 2 trials and verify that the number of spikes is correct.
        selection = { 'trials': [0, 2] }
        res = spiked.selectdata(selection)
        assert len(res.trials) == 2
        assert res.trials[0].shape[0] == trial0_nspikes
        assert res.trials[1].shape[0] == trial2_nspikes

        # Verify that the waveform selection is also correct.
        assert res.waveform is not None
        assert res.waveform.shape[0] == trial0_nspikes + trial2_nspikes  # Verify selection on waveform

        # Verify on data level.
        expected_data_indices = np.where((spiked.trialid == 0) | (spiked.trialid == 2))[0]
        for spike_idx in range(res.waveform.shape[0]):
            assert np.all(res.waveform[spike_idx, :, :] == spiked.waveform[expected_data_indices][spike_idx, :, :])

    def test_save_load_with_waveform(self):
        """Test saving file with waveform data."""
        numSpikes, waveform_dimsize = 20, 50
        spiked = getSpikeData(nSpikes = numSpikes)
        spiked.waveform = np.ones((numSpikes, 3, waveform_dimsize), dtype=int)

        tfile1 = tempfile.NamedTemporaryFile(suffix=".spike", delete=True)
        tfile1.close()
        tmp_spy_filename = tfile1.name
        save(spiked, filename=tmp_spy_filename)
        assert "waveform" in h5py.File(tmp_spy_filename, mode="r").keys()
        spkd2 = load(filename=tmp_spy_filename)
        assert isinstance(spkd2.waveform, h5py.Dataset), f"Expected h5py.Dataset, got {type(spkd2.waveform)}"
        assert np.array_equal(spiked.waveform[()], spkd2.waveform[()])

        # Test delete/unregister when setting to None.
        spkd2.waveform = None
        assert spkd2.waveform is None
        assert "waveform" not in h5py.File(tmp_spy_filename, mode="r").keys()

        # Test that we can set waveform again after deleting it.
        spkd2.waveform = np.ones((numSpikes, 3, waveform_dimsize), dtype=int)

        tfile1.close()

    def test_psth_with_waveform(self):
        """Test that the waveform does not break frontend functions, like PSTH.
           The waveform should just be ignored, and the resulting TimeLockData
           will of course NOT have a waveform.
        """
        numSpikes, waveform_dimsize = 20, 50
        spiked = getSpikeData(nSpikes = numSpikes)
        spiked.waveform = np.ones((numSpikes, 3, waveform_dimsize), dtype=int)

        cfg = spy.StructDict()
        cfg.binsize = 0.1
        cfg.latency = 'maxperiod'  # frontend default
        res = spy.spike_psth(spiked, cfg)
        assert type(res) == spy.TimeLockData
        assert not hasattr(res, "waveform")


if __name__ == '__main__':

    T1 = TestSpikeData()
    T2 = TestEventData()
    T3 = TestWaveform()