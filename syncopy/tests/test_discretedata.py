# -*- coding: utf-8 -*-
#
# Test proper functionality of Syncopy DiscreteData-type classes
#

# Builtin/3rd party package imports
import os
import tempfile
import time
import pytest
import numpy as np

# Local imports
from syncopy.datatype import AnalogData, SpikeData, EventData
from syncopy.datatype.base_data import Selector
from syncopy.shared.tools import StructDict
from syncopy.datatype.methods.selectdata import selectdata
from syncopy.io import save, load
from syncopy.shared.errors import SPYValueError, SPYTypeError
from syncopy.tests.misc import construct_spy_filename, flush_local_cluster
from syncopy import __acme__
if __acme__:
    import dask.distributed as dd

# Decorator to decide whether or not to run dask-related tests
skip_without_acme = pytest.mark.skipif(
    not __acme__, reason="acme not available")


class TestSpikeData():

    # Allocate test-dataset
    nc = 10
    ns = 30
    nd = 50
    seed = np.random.RandomState(13)
    data = np.vstack([seed.choice(ns, size=nd),
                      seed.choice(nc, size=nd),
                      seed.choice(int(nc / 2), size=nd)]).T
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

    def test_empty(self):
        dummy = SpikeData()
        assert len(dummy.cfg) == 0
        assert dummy.dimord is None
        for attr in ["channel", "data", "sampleinfo", "samplerate",
                     "trialid", "trialinfo", "unit"]:
            assert getattr(dummy, attr) is None
        with pytest.raises(SPYTypeError):
            SpikeData({})

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

    # test data-selection via class method
    def test_dataselection(self):

        # Create testing objects (regular and swapped dimords)
        dummy = SpikeData(data=self.data,
                          trialdefinition=self.trl,
                          samplerate=2.0)
        ymmud = SpikeData(data=self.data[:, ::-1],
                          trialdefinition=self.trl,
                          samplerate=2.0,
                          dimord=dummy.dimord[::-1])

        # selections are chosen so that result is not empty
        trialSelections = [
            "all",  # enforce below selections in all trials of `dummy`
            [3, 1]  # minimally unordered
        ]
        chanSelections = [
            ["channel03", "channel01", "channel01", "channel02"],  # string selection w/repetition + unordered
            [4, 2, 2, 5, 5],   # repetition + unorderd
            range(5, 8),  # narrow range
            slice(-5, None)  # negative-start slice
            ]
        toiSelections = [
            "all",  # non-type-conform string
            [-0.2, 0.6, 0.9, 1.1, 1.3, 1.6, 1.8, 2.2, 2.45, 3.]  # unordered, inexact, repetions
            ]
        toilimSelections = [
            [0.5, 3.5],  # regular range
            [1.0, np.inf]  # unbounded from above
            ]
        unitSelections = [
            ["unit1", "unit1", "unit2", "unit3"],  # preserve repetition
            [0, 0, 2, 3],  # preserve repetition, don't convert to slice
            range(1, 4),  # narrow range
            slice(-2, None)  # negative-start slice
            ]
        timeSelections = list(zip(["toi"] * len(toiSelections), toiSelections)) \
            + list(zip(["toilim"] * len(toilimSelections), toilimSelections))

        for obj in [dummy, ymmud]:
            chanIdx = obj.dimord.index("channel")
            unitIdx = obj.dimord.index("unit")
            chanArr = np.arange(obj.channel.size)
            for trialSel in trialSelections:
                for chanSel in chanSelections:
                    for unitSel in unitSelections:
                        for timeSel in timeSelections:
                            kwdict = {}
                            kwdict["trials"] = trialSel
                            kwdict["channels"] = chanSel
                            kwdict["units"] = unitSel
                            kwdict[timeSel[0]] = timeSel[1]
                            cfg = StructDict(kwdict)
                            # data selection via class-method + `Selector` instance for indexing
                            selected = obj.selectdata(**kwdict)
                            selector = Selector(obj, kwdict)
                            tk = 0
                            for trialno in selector.trials:
                                if selector.time[tk]:
                                    assert np.array_equal(obj.trials[trialno][selector.time[tk], :],
                                                        selected.trials[tk])
                                    tk += 1
                            assert set(selected.data[:, chanIdx]).issubset(chanArr[selector.channel])
                            assert set(selected.channel) == set(obj.channel[selector.channel])
                            assert np.array_equal(selected.unit,
                                                obj.unit[np.unique(selected.data[:, unitIdx])])
                            cfg.data = obj
                            cfg.out = SpikeData(dimord=obj.dimord)
                            # data selection via package function and `cfg`: ensure equality
                            selectdata(cfg)
                            assert np.array_equal(cfg.out.channel, selected.channel)
                            assert np.array_equal(cfg.out.unit, selected.unit)
                            assert np.array_equal(cfg.out.data, selected.data)

    @skip_without_acme
    def test_parallel(self, testcluster):
        # repeat selected test w/parallel processing engine
        client = dd.Client(testcluster)
        par_tests = ["test_dataselection"]
        for test in par_tests:
            getattr(self, test)()
            flush_local_cluster(testcluster)
        client.close()

class TestEventData():

    # Allocate test-datasets
    nc = 10
    ns = 30
    data = np.vstack([np.arange(0, ns, 5),
                      np.zeros((int(ns / 5), ))]).T
    data[1::2, 1] = 1
    data2 = data.copy()
    data2[:, -1] = data[:, 0]
    data2[:, 0] = data[:, -1]
    trl = np.vstack([np.arange(0, ns, 5),
                     np.arange(5, ns + 5, 5),
                     np.ones((int(ns / 5), )),
                     np.ones((int(ns / 5), )) * np.pi]).T
    num_smp = np.unique(data[:, 0]).size
    num_evt = np.unique(data[:, 1]).size

    adata = np.arange(1, nc * ns + 1).reshape(ns, nc)

    def test_ed_empty(self):
        dummy = EventData()
        assert len(dummy.cfg) == 0
        assert dummy.dimord == None
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

    def test_ed_trialretrieval(self):
        # test ``_get_trial`` with NumPy array: regular order
        dummy = EventData(self.data, trialdefinition=self.trl)
        smp = self.data[:, 0]
        for trlno, start in enumerate(range(0, self.ns, 5)):
            idx = np.intersect1d(np.where(smp >= start)[0],
                                 np.where(smp < start + 5)[0])
            trl_ref = self.data[idx, ...]
            assert np.array_equal(dummy._get_trial(trlno), trl_ref)

        # test ``_get_trial`` with NumPy array: swapped dimensions
        dummy = EventData(self.data2, trialdefinition=self.trl,
                          dimord=["eventid", "sample"])
        smp = self.data2[:, -1]
        for trlno, start in enumerate(range(0, self.ns, 5)):
            idx = np.intersect1d(np.where(smp >= start)[0],
                                 np.where(smp < start + 5)[0])
            trl_ref = self.data2[idx, ...]
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
            assert dummy2.eventid.size == self.num_smp # swapped
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
        idx = np.argsort(data3[:, 0])
        data3 = data3[idx, :]
        evt_dummy = EventData(data3, samplerate=sr_e)
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
        evt_dummy = EventData(data3, samplerate=sr_e)
        evt_dummy.definetrial(start=[2, 2, 1], stop=[1, 2, 0])
        assert np.array_equal(evt_dummy.sampleinfo, sinfo3)

        # Attach computed sampleinfo to AnalogData (data and data3 must yield identical resutls)
        evt_dummy = EventData(data=self.data, samplerate=sr_e)
        evt_dummy.definetrial(pre=pre, post=post, trigger=1)
        ang_dummy = AnalogData(self.adata, samplerate=sr_a)
        ang_dummy.definetrial(evt_dummy)
        assert np.array_equal(ang_dummy.sampleinfo, sinfo_a)
        evt_dummy = EventData(data=data3, samplerate=sr_e)
        evt_dummy.definetrial(pre=pre, post=post, trigger=1)
        ang_dummy.definetrial(evt_dummy)
        assert np.array_equal(ang_dummy.sampleinfo, sinfo_a)

        # Compute and attach sampleinfo on the fly
        evt_dummy = EventData(data=self.data, samplerate=sr_e)
        ang_dummy = AnalogData(self.adata, samplerate=sr_a)
        ang_dummy.definetrial(evt_dummy, pre=pre, post=post, trigger=1)
        assert np.array_equal(ang_dummy.sampleinfo, sinfo_a)
        evt_dummy = EventData(data=data3, samplerate=sr_e)
        ang_dummy = AnalogData(self.adata, samplerate=sr_a)
        ang_dummy.definetrial(evt_dummy, pre=pre, post=post, trigger=1)
        assert np.array_equal(ang_dummy.sampleinfo, sinfo_a)

        # Extend data and provoke an exception due to out of bounds erro
        smp = np.vstack([np.arange(self.ns, int(2.5 * self.ns), 5),
                         np.zeros((int((1.5 * self.ns) / 5),))]).T
        smp[1::2, 1] = 1
        data4 = np.vstack([data3, smp])
        evt_dummy = EventData(data=data4, samplerate=sr_e)
        evt_dummy.definetrial(pre=pre, post=post, trigger=1)
        # with pytest.raises(SPYValueError):
            # ang_dummy.definetrial(evt_dummy)

        # Trimming edges produces zero-length trial
        with pytest.raises(SPYValueError):
            ang_dummy.definetrial(evt_dummy, clip_edges=True)

        # We need `clip_edges` to make trial-definition work
        data4 = data4[:-2, :]
        data4[-2, 0] = data4[-1, 0]
        evt_dummy = EventData(data=data4, samplerate=sr_e)
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

    # test data-selection via class method
    def test_ed_dataselection(self):

        # Create testing objects (regular and swapped dimords)
        dummy = EventData(data=self.data,
                          trialdefinition=self.trl,
                          samplerate=2.0)
        ymmud = EventData(data=self.data[:, ::-1],
                          trialdefinition=self.trl,
                          samplerate=2.0,
                          dimord=dummy.dimord[::-1])

        # selections are chosen so that result is not empty
        trialSelections = [
            "all",  # enforce below selections in all trials of `dummy`
            [3, 1]  # minimally unordered
        ]
        eventidSelections = [
            [0, 0, 1],  # preserve repetition, don't convert to slice
            range(0, 2),  # narrow range
            slice(-2, None)  # negative-start slice
            ]
        toiSelections = [
            "all",  # non-type-conform string
            [-0.2, 0.6, 0.9, 1.1, 1.3, 1.6, 1.8, 2.2, 2.45, 3.]  # unordered, inexact, repetions
            ]
        toilimSelections = [
            [0.5, 3.5],  # regular range
            [0.0, np.inf]  # unbounded from above
            ]
        timeSelections = list(zip(["toi"] * len(toiSelections), toiSelections)) \
            + list(zip(["toilim"] * len(toilimSelections), toilimSelections))

        for obj in [dummy, ymmud]:
            eventidIdx = obj.dimord.index("eventid")
            for trialSel in trialSelections:
                for eventidSel in eventidSelections:
                    for timeSel in timeSelections:
                        kwdict = {}
                        kwdict["trials"] = trialSel
                        kwdict["eventids"] = eventidSel
                        kwdict[timeSel[0]] = timeSel[1]
                        cfg = StructDict(kwdict)
                        # data selection via class-method + `Selector` instance for indexing
                        selected = obj.selectdata(**kwdict)
                        selector = Selector(obj, kwdict)
                        tk = 0
                        for trialno in selector.trials:
                            if selector.time[tk]:
                                assert np.array_equal(obj.trials[trialno][selector.time[tk], :],
                                                    selected.trials[tk])
                                tk += 1
                        assert np.array_equal(selected.eventid,
                                            obj.eventid[np.unique(selected.data[:, eventidIdx]).astype(np.intp)])
                        cfg.data = obj
                        cfg.out = EventData(dimord=obj.dimord)
                        # data selection via package function and `cfg`: ensure equality
                        selectdata(cfg)
                        assert np.array_equal(cfg.out.eventid, selected.eventid)
                        assert np.array_equal(cfg.out.data, selected.data)

    @skip_without_acme
    def test_ed_parallel(self, testcluster):
        # repeat selected test w/parallel processing engine
        client = dd.Client(testcluster)
        par_tests = ["test_ed_dataselection"]
        for test in par_tests:
            getattr(self, test)()
            flush_local_cluster(testcluster)
        client.close()
