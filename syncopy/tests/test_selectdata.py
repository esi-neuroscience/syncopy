# -*- coding: utf-8 -*-
#
# Test functionality of data selection features in Syncopy
#

# Builtin/3rd party package imports
import pytest
import numpy as np
import inspect
import dask.distributed as dd


# Local imports
from syncopy.datatype.selector import Selector
from syncopy.shared.errors import SPYValueError, SPYTypeError
from syncopy.tests.misc import flush_local_cluster

import syncopy as spy

# map selection keywords to selector attributes (holding the idx to access selected data)
map_sel_attr = dict(trials='trial_ids',
                    channel='channel',
                    latency='time',
                    taper='taper',
                    frequency='freq',
                    channel_i='channel_i',
                    channel_j='channel_j',
                    unit='unit',
                    eventid='eventid'
                    )


class TestGeneral:

    adata = spy.AnalogData(data=np.ones((2, 2)), samplerate=1)
    csd_data = spy.CrossSpectralData(data=np.ones((2, 2, 2, 2)), samplerate=1)

    def test_Selector_init(self):

        with pytest.raises(SPYTypeError, match="Wrong type of `data`"):
            Selector(np.arange(10), {'latency': [0, 4]})

    def test_invalid_sel_key(self):

        # AnalogData has no `frequency`
        with pytest.raises(SPYValueError, match="no `frequency` selection available"):
            spy.selectdata(self.adata, frequency=[1, 10])
        # CrossSpectralData has no `channel` (but channel_i, channel_j)
        with pytest.raises(SPYValueError, match="no `channel` selection available"):
            spy.selectdata(self.csd_data, channel=0)


class TestAnalogSelections:

    nChannels = 10
    nSamples = 5  # per trial
    nTrials = 3
    samplerate = 2.0

    trldef = np.vstack([np.arange(0, nSamples * nTrials, nSamples),
                        np.arange(0, nSamples * nTrials, nSamples) + nSamples,
                        np.ones(nTrials) * -1]).T

    # this is an array running from 1 - nChannels * nSamples * nTrials
    # with shape: nSamples*nTrials x nChannels
    # and with data[i, j] = i+1 + j * nSamples*nTrials
    data = np.arange(1, nTrials * nChannels * nSamples + 1).reshape(nChannels, nSamples * nTrials).T

    adata = spy.AnalogData(data=data, samplerate=samplerate,
                           trialdefinition=trldef)

    def test_ad_selection(self):

        """
        Create a typical selection and check that the returned data is correct
        """

        selection = {'trials': 1, 'channel': [6, 2], 'latency': [0, 1]}
        res = spy.selectdata(self.adata, selection)

        # pick the data by hand, latency [0, 1] covers 2nd - 4th sample index
        # as time axis is array([-0.5,  0. ,  0.5,  1. ,  1.5])

        # pick trial
        solution = self.adata.data[self.nSamples:self.nSamples * 2]
        # pick channels and latency
        solution = np.column_stack([solution[1:4, 6], solution[1:4, 2]])

        assert np.all(solution == res.data)

    def test_ad_valid(self):

        """
        Instantiate Selector class and check only its attributes (the idx)
        """

        # each selection test is a 2-tuple: (selection kwargs, dict with same kws and the idx "solutions")
        valid_selections = [
            (
                {'channel': ["channel03", "channel01"],
                 'latency': [0, 1],
                 'trials': np.arange(2)},
                # these are the idx used to access the actual data
                {'channel': [2, 0],
                 'latency': 2 * [slice(1, 4, 1)],
                 'trials': [0, 1]}
            ),
            (
                # 2nd selection with some repetitions
                {'channel': [7, 3, 3],
                 'trials': [0, 1, 1]},
                # 'solutions'
                {'channel': [7, 3, 3],
                 'trials': [0, 1, 1]}
            )
        ]

        for selection in valid_selections:
            # instantiate Selector and check attributes
            sel_kwargs, solution = selection
            selector_object = Selector(self.adata, sel_kwargs)
            for sel_kw in sel_kwargs.keys():
                attr_name = map_sel_attr[sel_kw]
                assert getattr(selector_object, attr_name) == solution[sel_kw]

    def test_ad_invalid(self):

        # each selection test is a 3-tuple: (selection kwargs, Error, error message sub-string)
        invalid_selections = [
            ({'channel': ["channel33", "channel01"]},
             SPYValueError, "existing names or indices"),
            ({'channel': "my-non-existing-channel"},
             SPYValueError, "existing names or indices"),
            ({'channel': 99},
             SPYValueError, "existing names or indices"),
            ({'latency': 1}, SPYTypeError, "expected array_like"),
            ({'latency': [0, 10]}, SPYValueError, "at least one trial covering the latency window"),
            ({'latency': 'sth-wrong'}, SPYValueError, "'maxperiod'"),
            ({'trials': [-3]}, SPYValueError, "all array elements to be bound"),
            ({'trials': ['1', '6']}, SPYValueError, "expected dtype = numeric"),
            ({'trials': slice(2)}, SPYTypeError, "expected serializable data type")
        ]

        for selection in invalid_selections:
            sel_kw, error, err_str = selection
            with pytest.raises(error, match=err_str):
                spy.selectdata(self.adata, sel_kw)

    def test_ad_parallel(self, testcluster):

        # collect all tests of current class and repeat them in parallel
        client = dd.Client(testcluster)
        all_tests = [attr for attr in self.__dir__()
                     if (inspect.ismethod(getattr(self, attr)) and "parallel" not in attr)]
        for test in all_tests:
            getattr(self, test)()
            flush_local_cluster(testcluster)
        client.close()


class TestSpectralSelections:

    nChannels = 3
    nSamples = 3  # per trial
    nTrials = 3
    nTaper = 2
    nFreqs = 3
    samplerate = 2.0

    trldef = np.vstack([np.arange(0, nSamples * nTrials, nSamples),
                        np.arange(0, nSamples * nTrials, nSamples) + nSamples,
                        np.ones(nTrials) * 2]).T

    # this is an array running from 1 - nChannels * nSamples * nTrials * nFreq * nTaper
    data = np.arange(1, nChannels * nSamples * nTrials * nFreqs * nTaper + 1).reshape(nSamples * nTrials, nTaper, nFreqs, nChannels)
    sdata = spy.SpectralData(data=data, samplerate=samplerate,
                             trialdefinition=trldef)
    # freq labels
    sdata.freq = [20, 40, 60]

    def test_spectral_selection(self):

        """
        Create a typical selection and check that the returned data is correct
        """

        selection = {'trials': 1,
                     'channel': [1, 0],
                     'latency': [1, 1.5],
                     'frequency': [25, 50]}
        res = spy.selectdata(self.sdata, selection)

        # pick the data by hand, dimord is: ['time', 'taper', 'freq', 'channel']
        # latency [1, 1.5] covers 1st - 2nd sample index
        # as time axis is array([1., 1.5, 2.])
        # frequency covers only 2nd index (40 Hz)

        # pick trial
        solution = self.sdata.data[self.nSamples:self.nSamples * 2]
        # pick channels, frequency and latency and re-stack
        solution = np.stack([solution[:2, :, [1], 1], solution[:2, :, [1], 0]], axis=-1)

        assert np.all(solution == res.data)

    def test_spectral_valid(self):

        """
        Instantiate Selector class and check only its attributes (the idx)
        test mainly additional dimensions (taper and freq) here
        """

        # each selection test is a 2-tuple: (selection kwargs, dict with same kws and the idx "solutions")
        valid_selections = [
            (
                {'frequency': np.array([30, 60]),
                 'taper': [1, 0]},
                # the 'solutions'
                {'frequency': slice(1, 3, 1),
                 'taper': [1, 0]},
            ),
            # 2nd selection
            (
                {'frequency': 'all',
                 'taper': 'taper2',
                 'latency': [1.2, 1.7],
                 'trials': np.arange(1, 3)},
                # the 'solutions'
                {'frequency': slice(None),
                 'taper': [1],
                 'latency': [[1], [1]],
                 'trials': [1, 2]},
            )
        ]

        for selection in valid_selections:
            # instantiate Selector and check attributes
            sel_kwargs, solution = selection
            selector_object = Selector(self.sdata, sel_kwargs)
            for sel_kw in sel_kwargs.keys():
                attr_name = map_sel_attr[sel_kw]
                assert getattr(selector_object, attr_name) == solution[sel_kw]

    def test_spectral_invalid(self):

        # each selection test is a 3-tuple: (selection kwargs, Error, error message sub-string)
        invalid_selections = [
            ({'frequency': '40Hz'}, SPYValueError, "'all' or `None` or float or list/array"),
            ({'frequency': 4}, SPYValueError, "all array elements to be bounded"),
            ({'frequency': slice(None)}, SPYTypeError, "expected serializable data type"),
            ({'frequency': range(20, 60)}, SPYTypeError, "expected array_like"),
            ({'frequency': np.arange(20, 60)}, SPYValueError, "expected array of shape"),
            ({'taper': 'taper13'}, SPYValueError, "existing names or indices"),
            ({'taper': [18, 99]}, SPYValueError, "existing names or indices"),
        ]

        for selection in invalid_selections:
            sel_kw, error, err_str = selection
            with pytest.raises(error, match=err_str):
                spy.selectdata(self.sdata, sel_kw)


class TestCrossSpectralSelections:

    nChannels = 3
    nSamples = 3  # per trial
    nTrials = 3
    nFreqs = 3
    samplerate = 2.0

    trldef = np.vstack([np.arange(0, nSamples * nTrials, nSamples),
                        np.arange(0, nSamples * nTrials, nSamples) + nSamples,
                        np.ones(nTrials) * 2]).T

    # this is an array running from 1 - nChannels * nSamples * nTrials * nFreq * nTaper
    data = np.arange(1, nChannels**2 * nSamples * nTrials * nFreqs + 1).reshape(nSamples * nTrials, nFreqs, nChannels, nChannels)
    csd_data = spy.CrossSpectralData(data=data, samplerate=samplerate)
    csd_data.trialdefinition = trldef

    # freq labels
    csd_data.freq = [20, 40, 60]

    def test_csd_selection(self):

        """
        Create a typical selection and check that the returned data is correct
        """

        selection = {'trials': [1, 0],
                     'channel_i': [0, 1],
                     'latency': [1.5, 2],
                     'frequency': [25, 60]}

        res = spy.selectdata(self.csd_data, selection)

        # pick the data by hand, dimord is: ['time', 'freq', 'channel_i', 'channel_j']
        # latency [1, 1.5] covers 2nd - 3rd sample index
        # as time axis is array([1., 1.5, 2.])
        # frequency covers 2nd and 3rd index (40 and 60Hz)

        # pick trials
        solution = np.concatenate([self.csd_data.data[self.nSamples: self.nSamples * 2],
                                   self.csd_data.data[: self.nSamples]], axis=0)

        # pick channels, frequency and latency
        solution = np.concatenate([solution[1:3, 1:3, :2, :], solution[4:6, 1:3, :2, :]])
        assert np.all(solution == res.data)

    def test_csd_valid(self):

        """
        Instantiate Selector class and check only its attributes (the idx)
        test mainly additional dimensions (channel_i, channel_j) here
        """

        # each selection test is a 2-tuple: (selection kwargs, dict with same kws and the idx "solutions")
        valid_selections = [
            (
                {'channel_i': [0, 1], 'channel_j': [1, 2], 'latency': [1, 2]},
                # the 'solutions'
                {'channel_i': slice(0, 2, 1), 'channel_j': slice(1, 3, 1),
                 'latency': 3 * [slice(0, 3, 1)]},
            ),
            # 2nd selection
            (
                {'channel_i': ['channel2', 'channel3'], 'channel_j': 1},
                # the 'solutions'
                {'channel_i': slice(1, 3, 1), 'channel_j': 1},
            )
        ]

        for selection in valid_selections:
            # instantiate Selector and check attributes
            sel_kwargs, solution = selection
            selector_object = Selector(self.csd_data, sel_kwargs)
            for sel_kw in sel_kwargs.keys():
                attr_name = map_sel_attr[sel_kw]
                assert getattr(selector_object, attr_name) == solution[sel_kw]

    def test_csd_invalid(self):

        # each selection test is a 3-tuple: (selection kwargs, Error, error message sub-string)
        invalid_selections = [
            (
                {'channel_i': [0, 2]}, NotImplementedError,
                r"Unordered \(low to high\) or non-contiguous multi-channel-pair selections not supported"
            ),
            (
                {'channel_i': [1, 0]}, NotImplementedError,
                r"Unordered \(low to high\) or non-contiguous multi-channel-pair selections not supported"
            ),
            (
                {'channel_j': ['channel3', 'channel1']}, NotImplementedError,
                r"Unordered \(low to high\) or non-contiguous multi-channel-pair selections not supported"
            )

        ]

        for selection in invalid_selections:
            sel_kw, error, err_str = selection
            with pytest.raises(error, match=err_str):
                spy.selectdata(self.csd_data, sel_kw)


def getSpikeData(nChannels = 10, nTrials = 5, samplerate = 1.0, nSpikes = 20):
    T_max = 2 * nSpikes   # in samples, not seconds!
    nSamples = T_max / nTrials
    rng = np.random.default_rng(42)

    data = np.vstack([np.sort(rng.choice(range(T_max), size=nSpikes)),
                      rng.choice(np.arange(0, nChannels), size=nSpikes),
                      rng.choice(nChannels // 2, size=nSpikes)]).T

    trldef = np.vstack([np.arange(0, T_max, nSamples),
                        np.arange(0, T_max, nSamples) + nSamples,
                        np.ones(nTrials) * -2]).T

    return(spy.SpikeData(data=data,
                         samplerate=samplerate,
                         trialdefinition=trldef))


class TestSpikeSelections:


    spike_data = getSpikeData()

    def test_spike_selection(self):

        """
        Create a typical selection and check that the returned data is correct
        """

        selection = {'trials': [2, 4],
                     'channel': [6, 2],
                     'unit': [0, 3],
                     'latency': [-1, 4]}
        spkd = getSpikeData()
        res = spkd.selectdata(selection)

        # hand pick selection from the arrays
        dat_arr = spkd.data[()] # convert h5py to np.ndarray, see https://github.com/h5py/h5py/issues/474

        # these are trial intervals in sample indices!
        trial2 = spkd.trialdefinition[2, :2]
        trial4 = spkd.trialdefinition[4, :2]

        # create boolean mask for trials [2, 4]
        bm = (dat_arr[:, 0] >= trial2[0]) & (dat_arr[:, 0] <= trial2[1])
        bm = bm | (dat_arr[:, 0] >= trial4[0]) & (dat_arr[:, 0] <= trial4[1])

        # add channels [6, 2]
        bm = bm & ((dat_arr[:, 1] == 6) | (dat_arr[:, 1] == 2))

        # units [0, 3]
        bm = bm & ((dat_arr[:, 2] == 0) | (dat_arr[:, 2] == 3))

        # latency [-1, 4]
        # to index all trials at once
        time_vec = np.concatenate([t for t in spkd.time])
        bm = bm & ((time_vec >= -1) & (time_vec <= 4))

        # finally compare to selection result
        assert np.all(dat_arr[bm] == res.data[()])

    def test_spike_valid(self):

        """
        Instantiate Selector class and check only its attributes, the idx
        used by `_preview_trial` in the end
        """

        # each selection test is a 2-tuple: (selection kwargs, dict with same kws and the idx "solutions")
        valid_selections = [
            (
                # units get apparently indexed on a per trial basis
                {'trials': np.arange(1, 4), 'channel': ['channel03', 'channel01'], 'unit': [2, 0]},
                {'trials': [1, 2, 3], 'channel': [2, 0], 'unit': [[], [], [1, 5]]},
            ),
            # 2nd selection
            (
                # time/latency idx can be mixed lists and slices O.0
                # and channel 'all' selections can still be effectively subsets..
                {'trials': [0, 4], 'latency': [0, 3], 'channel': 'all'},
                {'trials': [0, 4], 'latency': [slice(0, 4, 1), [1]], 'channel': [1, 2, 3, 5, 9]},
            )
        ]

        for selection in valid_selections:
            # instantiate Selector and check attributes
            sel_kwargs, solution = selection
            selector_object = Selector(self.spike_data, sel_kwargs)
            for sel_kw in sel_kwargs.keys():
                attr_name = map_sel_attr[sel_kw]
                assert getattr(selector_object, attr_name) == solution[sel_kw]

    def test_spike_invalid(self):

        # each selection test is a 3-tuple: (selection kwargs, Error, error message sub-string)
        invalid_selections = [
            ({'channel': ["channel33", "channel01"]}, SPYValueError, "existing names or indices"),
            ({'channel': "my-non-existing-channel"}, SPYValueError, "existing names or indices"),
            ({'channel': slice(None)}, SPYTypeError, "expected serializable data type"),
            ({'unit': 99}, SPYValueError, "existing names or indices"),
            ({'unit': slice(None)}, SPYTypeError, "expected serializable data type"),
            ({'latency': [-1, 10]}, SPYValueError, "at least one trial covering the latency window"),
        ]

        for selection in invalid_selections:
            sel_kw, error, err_str = selection
            with pytest.raises(error, match=err_str):
                spy.selectdata(self.spike_data, sel_kw)


def _getEventData():
    nSamples = 4
    nTrials = 5
    samplerate = 1.0
    eIDs = [0, 111, 31]  # event ids
    rng = np.random.default_rng(42)

    trldef = np.vstack([np.arange(0, nSamples * nTrials, nSamples),
                        np.arange(0, nSamples * nTrials, nSamples) + nSamples,
                        np.ones(nTrials) * -1]).T

    # Use a triple-trigger pattern to simulate EventData w/non-uniform trials
    data = np.vstack([np.arange(0, nSamples * nTrials, 1),
                      rng.choice(eIDs, size=nSamples * nTrials)]).T
    edata = spy.EventData(data=data, samplerate=samplerate, trialdefinition=trldef)
    return edata

class TestEventSelections:

    edata = _getEventData()

    def test_event_selection(self):

        edata = _getEventData()

        # eIDs[1] = 111, a bit funny that here we need an index actually...
        selection = {'eventid': 1, 'latency': [0, 1], 'trials': [0, 3]}
        res = spy.selectdata(edata, selection)

        # hand pick selection from the arrays
        dat_arr = edata.data[()]

        # these are trial intervals in sample indices!
        trial0 = edata.trialdefinition[0, :2]
        trial3 = edata.trialdefinition[3, :2]

        # create boolean mask for trials [0, 3]
        bm = (dat_arr[:, 0] >= trial0[0]) & (dat_arr[:, 0] <= trial0[1])
        bm = bm | (dat_arr[:, 0] >= trial3[0]) & (dat_arr[:, 0] <= trial3[1])

        # add eventid eIDs[1]
        bm = bm & (dat_arr[:, 1] == 111)

        # latency [0, 1]
        # to index all trials at once
        time_vec = np.concatenate([t for t in self.edata.time])
        bm = bm & ((time_vec >= 0) & (time_vec <= 1))

        # finally compare to selection result
        assert np.all(dat_arr[bm] == res.data[()])

    def test_event_valid(self):
        """
        Instantiate Selector class and check only its attributes, the idx
        used by `_preview_trial` in the end
        """

        # each selection test is a 2-tuple: (selection kwargs, dict with same kws and the idx "solutions")
        valid_selections = [
            (
                # eventids get apparently indexed on a per trial basis
                {'trials': np.arange(1, 4), 'eventid': [0, 2]},
                {'trials': [1, 2, 3], 'eventid': [[2], slice(0, 2, 1), []]}
            ),
        ]

        for selection in valid_selections:
            # instantiate Selector and check attributes
            sel_kwargs, solution = selection
            selector_object = Selector(self.edata, sel_kwargs)
            for sel_kw in sel_kwargs.keys():
                attr_name = map_sel_attr[sel_kw]
                assert getattr(selector_object, attr_name) == solution[sel_kw]

    def test_event_invalid(self):

        """
        eventid seems to be only indexable ([0, 1, 2]) instead of using the actual
        numerical values ([0, 111, 31]), this should most likely change in the future..
        """
        # each selection test is a 3-tuple: (selection kwargs, Error, error message sub-string)
        invalid_selections = [
            ({'eventid': [111, 31]}, SPYValueError, "existing names or indices"),
            ({'eventid': '111'}, SPYValueError, "expected dtype = numeric"),
        ]

        for selection in invalid_selections:
            sel_kw, error, err_str = selection
            with pytest.raises(error, match=err_str):
                spy.selectdata(self.edata, sel_kw)


if __name__ == '__main__':
    T1 = TestGeneral()
    T2 = TestAnalogSelections()
    T3 = TestSpectralSelections()
    T4 = TestCrossSpectralSelections()
    T5 = TestSpikeSelections()
    T6 = TestEventSelections()
