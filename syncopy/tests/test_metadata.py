# -*- coding: utf-8 -*-
#
# Test metadata implementation.


import tempfile
import pytest
import numpy as np
import matplotlib.pyplot as plt
import math
import h5py
import tempfile
import os
import dask.distributed as dd

# Local imports
from syncopy import freqanalysis
from syncopy.datatype.methods.copy import copy
from syncopy.shared.tools import get_defaults
from syncopy.tests.synth_data import AR2_network, phase_diffusion
from syncopy.shared.metadata import encode_unique_md_label, decode_unique_md_label, parse_cF_returns, _parse_backend_metadata, _merge_md_list, metadata_from_hdf5_file, metadata_nest, metadata_unnest
from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYWarning
import syncopy as spy


def _get_fooof_signal(nTrials=100, nChannels = 1, nSamples = 1000, seed=None):
    """
    Produce suitable test signal for fooof, with peaks at 30 and 50 Hz.

    Note: One must perform trial averaging during the FFT to get realistic
    data out of it (and reduce noise). Then work with the averaged data.

    Returns AnalogData instance.
    """
    samplerate = 1000
    ar1_part = AR2_network(AdjMat=np.zeros(nChannels), nSamples=nSamples, alphas=[0.9, 0], nTrials=nTrials, seed=seed)
    pd1 = phase_diffusion(freq=30., eps=.1, samplerate=samplerate, nChannels=nChannels, nSamples=nSamples, nTrials=nTrials, seed=seed)
    pd2 = phase_diffusion(freq=50., eps=.1, samplerate=samplerate, nChannels=nChannels, nSamples=nSamples, nTrials=nTrials, seed=seed)
    signal = ar1_part + .8 * pd1 + 0.6 * pd2
    return signal


class TestMetadataHelpers():
    def test_encode_unique_md_label(self):
        assert encode_unique_md_label("label", "1", "2") == "label__1_2"
        assert encode_unique_md_label("label", 1, 2) == "label__1_2"
        assert encode_unique_md_label("label", 1) == "label__1_0"
        assert encode_unique_md_label("label", "1") == "label__1_0"

    def test_decode_unique_md_label(self):
        assert ("label", "1", "2") == decode_unique_md_label("label__1_2")

    def test_merge_md_list(self):
        assert _merge_md_list(None) is None
        md1 = {'1c': np.zeros(3), '1d': np.zeros(3)}
        md2 = {'2c': np.zeros(5), '2d': np.zeros(5)}
        md3 = {}
        mdl = [md1, md2, md3]
        merged = _merge_md_list(mdl)
        assert len(merged) == 4
        for k in ['1c', '1d', '2c', '2d']:
            assert k in merged

    def test_metadata_from_hdf5_file(self):
        # Test for correct error on hdf5 file without 'data' dataset.
        fd, h5py_filename = tempfile.mkstemp()
        with h5py.File(h5py_filename, "w") as f:
            f.create_dataset("mydataset", (100,), dtype='i')

        with pytest.raises(SPYValueError, match="dataset in hd5f file"):
            _ = metadata_from_hdf5_file(h5py_filename)

        os.close(fd)
        os.remove(h5py_filename)

    def test_get_res_details(self):
        # Test error on invalid input: a dict instead of tuple
        with pytest.raises(SPYValueError) as err:
            _, b = parse_cF_returns({"a": 2})
        assert "user-supplied compute function must return a single ndarray or a tuple with length exactly 2" in str(err.value)

        # Test with tuple of incorrect length 3
        with pytest.raises(SPYValueError) as err:
            _, b = parse_cF_returns((1, 2, 3))
        assert "user-supplied compute function must return a single ndarray or a tuple with length exactly 2" in str(err.value)

        # Test with tuple, 2nd arg is None
        _, b = parse_cF_returns((np.zeros(3)))
        assert b is None

        # Test with ndarray only
        _, b = parse_cF_returns(np.zeros(3))
        assert b is None

        # Test with tuple of correct length, but 2nd value is not dict
        with pytest.raises(SPYValueError) as err:
            a, b = parse_cF_returns((np.zeros(3), np.zeros(4)))
        assert "the second return value of user-supplied compute functions must be a dict" in str(err.value)

        # Test with tuple of correct length, 2nd value is dict, but values are not ndarray
        with pytest.raises(SPYValueError) as err:
            a, b = parse_cF_returns((np.zeros(3), {'a': dict()}))
        assert "the second return value of user-supplied compute functions must be a dict containing np.ndarrays" in str(err.value)

        # Test with numpy array with datatype np.object, which is not valid.
        invalid_val = np.array([np.zeros((5,3)), np.zeros((8,3))], dtype = object)  # The dtype is required to silence numpy deprecation warnings, the dtype will be object even without it.
        assert invalid_val.dtype == object
        with pytest.raises(SPYValueError) as err:
            a, b = parse_cF_returns((np.zeros(3), {'a': invalid_val}))
        assert "the second return value of user-supplied compute functions must be a dict containing np.ndarrays with datatype other than 'np.object'" in str(err.value)

        # Test with tuple of correct length, 2nd value is dict, and values in ndarray are string (but not object). This is fine.
        a, b = parse_cF_returns((np.zeros(3), {'a': np.array(['apples', 'foobar', 'cowboy'])}))
        assert 'a' in b

    def test_parse_backend_metadata(self):
        assert _parse_backend_metadata(None) == dict()

        # Test for error if input is not dict.
        with pytest.raises(SPYTypeError) as err:
            attrs = _parse_backend_metadata(np.zeros(3))
        assert "expected dict found ndarray" in str(err.value)

        # Test for error if dict keys are not string
        with pytest.raises(SPYValueError) as err:
            attrs = _parse_backend_metadata({5: np.zeros(3)})
        assert "keys in metadata must be strings" in str(err.value)

        # Test that empty input leads to empty output
        attrs = _parse_backend_metadata(dict())
        assert isinstance(attrs, dict)
        assert not attrs

        # Test that string-only keys are treated as attributes (not dsets)
        attrs = _parse_backend_metadata({'attr1': np.zeros(3)})
        assert 'attr1' in attrs and len(attrs) == 1

        # Test that error is raised if implicit 'attr 'values are not ndarray
        with pytest.raises(SPYTypeError, match="value in metadata"):
            attrs = _parse_backend_metadata({'attr1': dict()})

        # Test that warning is raised for large data
        _parse_backend_metadata({'attr1': np.arange(100000)})

    def test_metadata_nest(self):
        # Test with valid input
        md = { 'ap__0_0': 1, 'ap__0_1': 2, 'pp__0_0': 3, 'pp__0_1': 4}
        md_nested = metadata_nest(md)
        expected = { 'ap' : { 'ap__0_0': 1, 'ap__0_1': 2}, 'pp': {'pp__0_0': 3, 'pp__0_1': 4}}
        assert md_nested == expected

        # Test exc: key with name identical to unique_label_part of another key already in dict.
        # This leads to an error because the key with name identical to unique_label_part cannot conform
        # to the format expected by the function that splits it into label, trial_idx, chunk_idx.
        md_dupl = { 'ap__0_0': 1, 'ap__0_1': 2, 'pp__0_0': 3, 'pp__0_1': 4, 'pp' : 3}
        with pytest.raises(SPYValueError, match="input string in format `<label>__<trial_idx>_<chunk_idx>'"):
            _ = metadata_nest(md_dupl)

    def test_metadata_unnest(self):
        # Test with valid input
        md_nested = { 'ap' : { 'ap__0_0': 1, 'ap__0_1': 2}, 'pp': {'pp__0_0': 3, 'pp__0_1': 4}}
        md_unnested = metadata_unnest(md_nested)
        expected = { 'ap__0_0': 1, 'ap__0_1': 2, 'pp__0_0': 3, 'pp__0_1': 4}
        assert md_unnested == expected

        # Test exc: with duplicate key in several nested dicts.
        md_nested_dupl_key = { 'ap' : { 'ap__0_0': 1, 'ap__0_1': 2}, 'pp': {'ap__0_0': 3, 'pp__0_1': 4}}
        with pytest.raises(SPYValueError, match="Duplicate key"):
            _ = metadata_unnest(md_nested_dupl_key)

        # Test exc: input not a nested dict
        md_not_nested = { 'ap' : { 'ap__0_0': 1, 'ap__0_1': 2}, 'pp': 1 }
        with pytest.raises(SPYValueError, match="is not a dict"):
            _ = metadata_unnest(md_not_nested)


class TestMetadataUsingFooof():
    """
    Test 2nd cF function return value, with FOOOF as example compute method.
    """

    tfData = _get_fooof_signal()
    expected_metadata_keys = spy.specest.compRoutines.FooofSpy.metadata_keys  # ("aperiodic_params", "gaussian_params", "peak_params", "n_peaks", "r_squared", "error",)
    num_expected_metadata_keys = len(expected_metadata_keys)

    @staticmethod
    def get_fooof_cfg():
        cfg = get_defaults(freqanalysis)
        cfg.method = "mtmfft"
        cfg.taper = "hann"
        cfg.select = {"channel": 0}
        cfg.keeptrials = False
        cfg.output = "fooof"
        cfg.foilim = [1., 100.]
        return cfg

    def test_sequential(self):
        """
        Test metadata propagation with fooof in sequential compute mode.
        """
        cfg = TestMetadataUsingFooof.get_fooof_cfg()
        cfg.parallel = False
        cfg.pop('fooof_opt', None)
        fooof_opt = {'peak_width_limits': (1.0, 12.0)}  # Increase lower limit to avoid fooof warning.
        spec_dt = freqanalysis(cfg, self.tfData, fooof_opt=fooof_opt)

        # These are known from the input data and cfg.
        data_size = 100  # Number of samples (per trial) seen by fooof. The full signal returned by _get_fooof_signal() is
                         # larger, but the cfg.frequency setting (in get_fooof_cfg()) limits to 100 samples.
        num_trials_fooof = 1 # Because of keeptrials = False in cfg.

        assert spec_dt.data.shape == (num_trials_fooof, 1, data_size, 1)

        # check metadata from 2nd cF return value, added to the hdf5 dataset as attribute.
        keys_unique = [kv + "__0_0" for kv in self.expected_metadata_keys]

        # Now for the metadata. This got attached to the syncopy data instance as the 'metadata' attribute. It is a hdf5 group.
        assert spec_dt.metadata is not None
        assert isinstance(spec_dt.metadata, dict)  # Make sure it is a standard dict, not a hdf5 group.
        num_metadata_attrs = len(spec_dt.metadata.keys())  # Get keys of dict
        assert num_metadata_attrs == self.num_expected_metadata_keys
        spec_dt_metadata_unnested = metadata_unnest(spec_dt.metadata)
        for kv in keys_unique:
            assert kv in spec_dt_metadata_unnested.keys()
            # Note that the cF-specific unpacking may convert ndarray values into something else. In case of fooof, we convert
            # some ndarrays (all the peak_params__n_m and gaussian_params__n_m) to list, so we accept both types here.
            assert isinstance(spec_dt_metadata_unnested.get(kv), (list, np.ndarray))

        # check that the cfg is correct (required for replay)
        assert spec_dt.cfg['freqanalysis']['output'] == 'fooof'

        # Test the metadata_keys entry of the CR:
        for k in spy.specest.compRoutines.FooofSpy.metadata_keys:
            assert k in spec_dt.metadata
        assert len(spec_dt.metadata) == len(spy.specest.compRoutines.FooofSpy.metadata_keys)


    def test_par_compute_with_sequential_storage(self):
        """
        Test metadata propagation in with parallel compute and sequential storage.
        With trial averaging (`keeptrials=false` in cfg), sequential storage is used.
        """
        cfg = TestMetadataUsingFooof.get_fooof_cfg()
        cfg.pop('fooof_opt', None)
        cfg.parallel = True
        fooof_opt = {'peak_width_limits': (1.0, 12.0)}  # Increase lower limit to avoid fooof warning.
        spec_dt = freqanalysis(cfg, self.tfData, fooof_opt=fooof_opt)

        # These are known from the input data and cfg.
        data_size = 100
        num_trials_fooof = 1 # Because of keeptrials = False in cfg.

        # check frequency axis
        assert spec_dt.freq.size == data_size
        assert spec_dt.freq[0] == 1
        assert spec_dt.freq[99] == 100.

        assert spec_dt.data.shape == (num_trials_fooof, 1, data_size, 1)
        assert not np.isnan(spec_dt.data).any()

        # check metadata from 2nd cF return value, added to the hdf5 dataset as attribute.
        keys_unique = [kv + "__0_0" for kv in self.expected_metadata_keys]

        expected_num_metadata_keys = len(self.expected_metadata_keys)
        spec_dt_metadata_unnested = metadata_unnest(spec_dt.metadata)

        # Now for the metadata. This got attached to the syncopy data instance as the 'metadata' attribute. It is a hdf5 group.
        assert spec_dt.metadata is not None
        assert isinstance(spec_dt.metadata, dict)  # Make sure it is a standard dict, not a hdf5 group.
        num_metadata_attrs = len(spec_dt.metadata.keys())
        assert num_metadata_attrs == expected_num_metadata_keys
        for kv in keys_unique:
            assert kv in spec_dt_metadata_unnested.keys()
            # Note that the cF-specific unpacking may convert ndarray values into something else. In case of fooof, we convert
            # some ndarrays (all the peak_params__n_m and gaussian_params__n_m) to list, so we accept both types here.
            assert isinstance(spec_dt_metadata_unnested.get(kv), list) or isinstance(spec_dt_metadata_unnested.get(kv), np.ndarray)

        # check that the cfg is correct (required for replay)
        assert spec_dt.cfg['freqanalysis']['output'] == 'fooof'

    def test_par_compute_with_par_storage(self):
        """
        Test metadata propagation in with parallel compute and parallel storage.
        Without trial averaging (`keeptrials=True` in cfg), parallel storage is used.
        """
        cfg = TestMetadataUsingFooof.get_fooof_cfg()
        cfg.pop('fooof_opt', None)
        cfg.parallel = True  # Enable parallel computation
        cfg.keeptrials = True  # Enable parallel storage (is turned off when trial averaging is happening)
        fooof_opt = {'peak_width_limits': (1.0, 12.0)}  # Increase lower limit to avoid fooof warning.
        spec_dt = freqanalysis(cfg, self.tfData, fooof_opt=fooof_opt)

        # These are known from the input data and cfg.
        num_trials_fooof = 100
        data_size = 100

        # check frequency axis
        assert spec_dt.freq.size == data_size
        assert spec_dt.freq[0] == 1
        assert spec_dt.freq[99] == 100.

        # check the data
        assert spec_dt.data.shape == (num_trials_fooof, 1, data_size, 1) # Differs from other tests due to `keeptrials=True`.
        assert not np.isnan(spec_dt.data).any()

        # check metadata from 2nd cF return value, added to the hdf5 dataset 'data' as attributes.
        keys_unique = [kv + "__0_0" for kv in self.expected_metadata_keys]

        # Now for the metadata. This got attached to the syncopy data instance as the 'metadata' attribute. It is a hdf5 group.
        assert spec_dt.metadata is not None
        assert isinstance(spec_dt.metadata, dict)  # Make sure it is a standard dict, not a hdf5 group.
        num_metadata_attrs = len(spec_dt.metadata.keys())

        expected_num_metadata_keys = len(self.expected_metadata_keys)
        spec_dt_metadata_unnested = metadata_unnest(spec_dt.metadata)
        assert num_metadata_attrs == expected_num_metadata_keys
        assert len(spec_dt_metadata_unnested.keys()) == expected_num_metadata_keys * num_trials_fooof
        for kv in keys_unique:
            assert (kv) in spec_dt_metadata_unnested.keys()
            # Note that the cF-specific unpacking may convert ndarray values into something else. In case of fooof, we convert
            # some ndarrays (all the peak_params__n_m and gaussian_params__n_m) to list, so we accept both types here.
            assert isinstance(spec_dt_metadata_unnested.get(kv), list) or isinstance(spec_dt_metadata_unnested.get(kv), np.ndarray)

        # check that the cfg is correct (required for replay)
        assert spec_dt.cfg['freqanalysis']['output'] == 'fooof'

    def test_par_with_selections(self):
        """
        Test metadata propagation in with parallel compute and parallel storage,
        and trial selections.

        In the case of trial selections, the computation of absolute trial indices
        from relative ones uses the trivial branch for fooof, bacause the selection
        is handled (and consumed) by the mtmfft running before fooof. See the test
        in the `TestMetadataUsingMtmfft` class below for the other branch.
        """
        cfg = TestMetadataUsingFooof.get_fooof_cfg()
        cfg.pop('fooof_opt', None)
        cfg.parallel = True  # Enable parallel computation
        cfg.keeptrials = True  # Enable parallel storage (is turned off when trial averaging is happening)
        fooof_opt = {'peak_width_limits': (1.0, 12.0)}  # Increase lower limit to avoid fooof warning.
        data = self.tfData.copy()

        selected_trials = [3, 5, 7]
        cfg.select = {'trials': selected_trials }

        spec_dt = freqanalysis(cfg, data, fooof_opt=fooof_opt)

        # These are known from the input data and cfg.
        num_trials_fooof_selected = len(selected_trials)
        data_size = 100

        # check frequency axis
        assert spec_dt.freq.size == data_size
        assert spec_dt.freq[0] == 1
        assert spec_dt.freq[99] == 100.

        assert spec_dt.data.shape == (num_trials_fooof_selected, 1, data_size, 1)

        # check metadata from 2nd cF return value, added to the hdf5 dataset 'data' as attributes.
        keys_unique = [kv + "__0_0" for kv in self.expected_metadata_keys]

        # Now for the metadata. This got attached to the syncopy data instance as the 'metadata' attribute. It is a hdf5 group.
        assert spec_dt.metadata is not None
        assert isinstance(spec_dt.metadata, dict)  # Make sure it is a standard dict, not a hdf5 group.
        num_metadata_attrs = len(spec_dt.metadata.keys())
        assert num_metadata_attrs == self.num_expected_metadata_keys
        spec_dt_metadata_unnested = metadata_unnest(spec_dt.metadata)
        assert len(spec_dt_metadata_unnested.keys()) == self.num_expected_metadata_keys * num_trials_fooof_selected
        for kv in keys_unique:
            assert (kv) in spec_dt_metadata_unnested.keys()
            assert isinstance(spec_dt_metadata_unnested.get(kv), (list, np.ndarray))

    def test_channel_par(self):
        """
        Test metadata propagation in with channel parallelization. This implies
        parallel compute and parallel storage.

        For syncopy to use channel parallelization, we must make sure that:
        - we use parallel mode (`cfg.parallel` = `True`)
        - 'chan_per_worker' is set to a positive integer (it is a kwarg of `freqanalysis`)
        - `keeptrials=True`, otherwise it makes no sense
        - We also do not select specific channels in `cfg.select`, as that is not supported with channel parallelisation.
        """
        cfg = TestMetadataUsingFooof.get_fooof_cfg()
        cfg.pop('fooof_opt', None)
        cfg.parallel = True  # Enable parallel computation
        cfg.keeptrials = True  # Enable parallel storage (is turned off when trial averaging is happening)
        cfg.select = None
        fooof_opt = {'peak_width_limits': (1.0, 12.0)}  # Increase lower limit to avoid fooof warning.
        num_trials_fooof = 100
        num_channels = 5
        chan_per_worker = 2
        multi_chan_data = _get_fooof_signal(nTrials=num_trials_fooof, nChannels = num_channels)
        spec_dt = freqanalysis(cfg, multi_chan_data, fooof_opt=fooof_opt, chan_per_worker=chan_per_worker)

        # How many more calls we expect due to channel parallelization.
        used_parallel = 'used_parallel = True' in spec_dt._log
        if used_parallel:
            calls_per_trial = int(math.ceil(num_channels / chan_per_worker))
        else:
            calls_per_trial = 1
        data_size = 100

        # check frequency axis
        assert spec_dt.freq.size == data_size
        assert spec_dt.freq[0] == 1
        assert spec_dt.freq[99] == 100.

        # check the data
        assert spec_dt.data.shape == (num_trials_fooof, 1, data_size, num_channels) # Differs from other tests due to `keeptrials=True` and channels.
        assert not np.isnan(spec_dt.data).any()

        # check metadata from 2nd cF return value, added to the hdf5 dataset 'data' as attributes.
        keys_unique = list()
        for fde in self.expected_metadata_keys:
            for trial_idx in range(num_trials_fooof):
                for call_idx in range(calls_per_trial):
                    keys_unique.append(encode_unique_md_label(fde, trial_idx, call_idx))

        assert spec_dt.metadata is not None
        assert isinstance(spec_dt.metadata, dict)  # Make sure it is a standard dict, not a hdf5 group.
        num_metadata_attrs = len(spec_dt.metadata.keys())

        spec_dt_metadata_unnested = metadata_unnest(spec_dt.metadata)
        assert num_metadata_attrs == self.num_expected_metadata_keys
        assert len(spec_dt_metadata_unnested.keys()) == self.num_expected_metadata_keys * num_trials_fooof * calls_per_trial
        for kv in keys_unique:
            assert kv in spec_dt_metadata_unnested.keys()
            assert isinstance(spec_dt_metadata_unnested.get(kv), (list, np.ndarray))

        # check that the cfg is correct (required for replay)
        assert spec_dt.cfg['freqanalysis']['output'] == 'fooof'

    def test_metadata_parallel(self, testcluster):

        plt.ioff()
        client = dd.Client(testcluster)
        all_tests = ["test_par_compute_with_sequential_storage",
                     "test_par_compute_with_par_storage",
                     "test_channel_par"]

        for test_name in all_tests:
            test_method = getattr(self, test_name)
            test_method()
        client.close()
        plt.ion()


if __name__ == "__main__":
    T1 = TestMetadataHelpers()
    T2 = TestMetadataUsingFooof()
