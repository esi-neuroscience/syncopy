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

# Local imports
from syncopy import freqanalysis
from syncopy.shared.tools import get_defaults
from syncopy.tests.synth_data import AR2_network, phase_diffusion
from syncopy.shared.metadata import encode_unique_md_label, decode_unique_md_label, get_res_details, _parse_backend_metadata, _merge_md_list, metadata_from_hdf5_file
from syncopy.shared.errors import SPYValueError, SPYTypeError
import syncopy as spy
from syncopy import __acme__
if __acme__:
    import dask.distributed as dd

# Decorator to decide whether or not to run dask-related tests
skip_without_acme = pytest.mark.skipif(not __acme__, reason="acme not available")


def _get_fooof_signal(nTrials=100, nChannels = 1):
    """
    Produce suitable test signal for fooof, with peaks at 30 and 50 Hz.

    Note: One must perform trial averaging during the FFT to get realistic
    data out of it (and reduce noise). Then work with the averaged data.

    Returns AnalogData instance.
    """
    nSamples = 1000
    samplerate = 1000
    ar1_part = AR2_network(AdjMat=np.zeros(nChannels), nSamples=nSamples, alphas=[0.9, 0], nTrials=nTrials)
    pd1 = phase_diffusion(freq=30., eps=.1, fs=samplerate, nChannels=nChannels, nSamples=nSamples, nTrials=nTrials)
    pd2 = phase_diffusion(freq=50., eps=.1, fs=samplerate, nChannels=nChannels, nSamples=nSamples, nTrials=nTrials)
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
        md1 = {'dsets': {'1a': np.zeros(3), '1b': np.zeros(3)}, 'attrs': {'1c': np.zeros(3), '1d': np.zeros(3)}}
        md2 = {'dsets': {'2a': np.zeros(5), '2b': np.zeros(5)}, 'attrs': {'2c': np.zeros(5), '2d': np.zeros(5)}}
        md3 = {'dsets': {'3a': np.zeros(1)}, 'attrs': {}}
        mdl = [md1, md2, md3]
        merged = _merge_md_list(mdl)
        assert len(merged['dsets']) == 5
        for k in ['1a', '1b', '2a', '2b', '3a']:
            assert k in merged['dsets']
        assert len(merged['attrs']) == 4
        for k in ['1c', '1d', '2c', '2d']:
            assert k in merged['attrs']

    def test_metadata_from_hdf5_file(self):
        # Test for correct error on hdf5 file without 'data' dataset.
        _, h5py_filename = tempfile.mkstemp()
        with h5py.File(h5py_filename, "w") as f:
            f.create_dataset("mydataset", (100,), dtype='i')

        with pytest.raises(SPYValueError, match="dataset in hd5f file"):
            _ = metadata_from_hdf5_file(h5py_filename)
        os.remove(h5py_filename)

    def test_get_res_details(self):
        # Test error on invalid input: a dict instead of tuple
        with pytest.raises(SPYValueError) as err:
            _, b = get_res_details({"a": 2})
        assert "user-supplied compute function must return a single ndarray or a tuple with length exactly 2" in str(err.value)

        # Test with tuple of incorrect length 3
        with pytest.raises(SPYValueError) as err:
            _, b = get_res_details((1, 2, 3))
        assert "user-supplied compute function must return a single ndarray or a tuple with length exactly 2" in str(err.value)

        # Test with tuple, 2nd arg is None
        _, b = get_res_details((np.zeros(3)))
        assert b is None

        # Test with ndarray only
        _, b = get_res_details(np.zeros(3))
        assert b is None

        # Test with tuple of correct length, but 2nd value is not dict
        with pytest.raises(SPYValueError) as err:
            a, b = get_res_details((np.zeros(3), np.zeros(4)))
        assert "the second return value of user-supplied compute functions must be a dict" in str(err.value)

        # Test with tuple of correct length, 2nd value is dict, but values are not ndarray
        with pytest.raises(SPYValueError) as err:
            a, b = get_res_details((np.zeros(3), {'a': dict()}))
        assert "the second return value of user-supplied compute functions must be a dict containing np.ndarrays" in str(err.value)

        # Test with tuple of correct length, 2nd value is dict, but values ndarray but not numeric
        with pytest.raises(SPYValueError) as err:
            a, b = get_res_details((np.zeros(3), {'a': np.array(['apples', 'foobar', 'cowboy'])}))
        assert "the second return value of user-supplied compute functions must be a dict containing np.ndarrays containing numbers" in str(err.value)

    def test_parse_details(self):
        # Test for error if input is not dict.
        with pytest.raises(SPYTypeError) as err:
            attrs, dsets = _parse_backend_metadata(np.zeros(3))
        assert "details" in str(err.value)
        assert "dict" in str(err.value)

        # Test that empty input leads to empty output
        attrs, dsets = _parse_backend_metadata(dict())
        assert isinstance(attrs, dict)
        assert isinstance(dsets, dict)
        assert not attrs
        assert not dsets

        # Test that string-only keys are treated as attributes (not dsets)
        attrs, dsets = _parse_backend_metadata({'attr1': np.zeros(3)})
        assert 'attr1' in attrs and len(attrs) == 1
        assert not dsets

        # Test that tuple keys lead to proper sorting into dsets and attrs
        attrs, dsets = _parse_backend_metadata({('attr1', 'attr'): np.zeros(3), ('dset1', 'data'): np.zeros(3), ('attr2', 'attr'): np.zeros(3)})
        assert 'attr1' in attrs and 'attr2' in attrs and len(attrs) == 2
        assert 'dset1' in dsets and len(dsets) == 1

        # Test that error is raised if implicit 'attr 'values are not ndarray
        with pytest.raises(SPYTypeError, match="value in details"):
            attrs, dsets = _parse_backend_metadata({'attr1': dict()})
        # Test that error is raised if explicit 'attr 'values are not ndarray
        with pytest.raises(SPYTypeError, match="value in details"):
            attrs, dsets = _parse_backend_metadata({('attr1', 'attr'): dict()})
        # Test that error is raised if explicit 'dset 'values are not ndarray
        with pytest.raises(SPYTypeError, match="value in details"):
            attrs, dsets = _parse_backend_metadata({('dset1', 'data'): dict()})



class TestMetadataUsingFooof():
    """
    Test passing on 2nd cF function return value in sequential mode.

    Note: This test implicitely also tests the case that no metadata is attached by the cF, because
          before FOOOF, we also call mtmfft, which does not attach metadata.
    """

    tfData = _get_fooof_signal()

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

    def test_metadata_1call_sequential(self):
        """
        Test metadata propagation in with sequential compute.
        """
        cfg = TestMetadataUsingFooof.get_fooof_cfg()
        cfg.parallel = False
        cfg.pop('fooof_opt', None)
        fooof_opt = {'peak_width_limits': (1.0, 12.0)}  # Increase lower limit to avoid fooof warning.
        spec_dt = freqanalysis(cfg, self.tfData, fooof_opt=fooof_opt)

        # These are known from the input data and cfg.
        data_size = 100  # number of samples (per trial) seen by fooof. the full signal returned by _get_fooof_signal() is
                         # larger, but the cfg.foilim setting (in get_fooof_cfg()) limits to 100 samples.
        num_trials_fooof = 1 # Because of keeptrials = False in cfg.

        # check frequency axis
        assert spec_dt.freq.size == data_size
        assert spec_dt.freq[0] == 1
        assert spec_dt.freq[99] == 100.

        # check the log
        assert "fooof_method = fooof" in spec_dt._log
        assert "fooof_aperiodic" not in spec_dt._log
        assert "fooof_peaks" not in spec_dt._log
        assert "fooof_opt" in spec_dt._log

        # check the data
        assert spec_dt.data.ndim == 4
        assert spec_dt.data.shape == (num_trials_fooof, 1, data_size, 1)
        assert not np.isnan(spec_dt.data).any()

        # check metadata from 2nd cF return value, added to the hdf5 dataset as attribute.
        k_unique = "__0_0"  # TODO: this is currently still hardcoded, and the _0 is the one added by the first cF function call.
                         #       depending on data size and RAM, there may or may not be several calls, and "_1" , "_2", ... exist.
        expected_fooof_dict_entries = ["aperiodic_params", "gaussian_params", "peak_params", "n_peaks", "r_squared", "error"]
        keys_unique = [kv + k_unique for kv in expected_fooof_dict_entries]

        test_metadata_on_main_dset = False

        if test_metadata_on_main_dset:
            assert len(spec_dt.data.attrs.keys()) == len(expected_fooof_dict_entries)

            for kv in keys_unique:
                assert (kv) in spec_dt.data.attrs.keys()
                assert isinstance(spec_dt.data.attrs.get(kv), np.ndarray)
            # Expect one entry in detail.
            n_peaks = spec_dt.data.attrs.get("n_peaks" + k_unique)
            assert isinstance(n_peaks, np.ndarray)
            assert n_peaks.size == 1 # cfg.keeptrials is False, so FOOOF operates on a single trial and we expect only one value here.
            assert spec_dt.data.attrs.get("r_squared" + k_unique).size == num_trials_fooof  # Same, see line above.
            assert spec_dt.data.attrs.get("error" + k_unique).size == num_trials_fooof  # Same, see line above.

        # Now for the metadata. This got attached to the syncopy data instance as the 'metadata' attribute. It is a hdf5 group.
        assert spec_dt.metadata is not None
        assert isinstance(spec_dt.metadata, dict)  # Make sure it is a standard dict, not a hdf5 group.
        num_metadata_dsets = len(spec_dt.metadata['dsets'].keys())
        num_metadata_dsets = 0
        num_metadata_attrs = len(spec_dt.metadata['attrs'].keys())  # Get keys of dict
        assert num_metadata_dsets == 0
        assert num_metadata_attrs == 6
        for kv in keys_unique:
            assert kv in spec_dt.metadata['attrs'].keys()
            # Note that the cF-specific unpacking may convert ndarray values into something else. In case of fooof, we convert
            # some ndarrays (all the peak_params__n_m and gaussian_params__n_m) to list, so we accept both types here.
            assert isinstance(spec_dt.metadata['attrs'].get(kv), list) or isinstance(spec_dt.metadata['attrs'].get(kv), np.ndarray)

        # check that the cfg is correct (required for replay)
        assert spec_dt.cfg['freqanalysis']['output'] == 'fooof'
        return spec_dt

    def test_metadata_parallel_with_sequential_storage(self):
        """
        Test metadata propagation in with parallel compute and sequential storage.
        With trial averaging (`keeptrials=false` in cfg), sequential storage is used.

        Note: This function is currently identical to 'test_metadata_1call_sequential()',
              the only difference is the `cfg.parallel = True` before the call to freqanalysis().
              TODO: We should refactor this.

        Note2: This test implicitely also tests the case that no metadata is attached by the cF, because
               before FOOOF, we also call mtmfft, which does not attach metadata.
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

        # check the log
        assert "fooof_method = fooof" in spec_dt._log
        assert "fooof_aperiodic" not in spec_dt._log
        assert "fooof_peaks" not in spec_dt._log
        assert "fooof_opt" in spec_dt._log

        # check the data
        assert spec_dt.data.ndim == 4
        assert spec_dt.data.shape == (num_trials_fooof, 1, data_size, 1)
        assert not np.isnan(spec_dt.data).any()

        # check metadata from 2nd cF return value, added to the hdf5 dataset as attribute.
        k_unique = "__0_0"  # TODO: this is currently still hardcoded, and the _0 is the one added by the first cF function call.
                         #       depending on data size and RAM, there may or may not be several calls, and "_1" , "_2", ... exist.
        expected_fooof_dict_entries = ["aperiodic_params", "gaussian_params", "peak_params", "n_peaks", "r_squared", "error"]
        keys_unique = [kv + k_unique for kv in expected_fooof_dict_entries]

        # Now for the metadata. This got attached to the syncopy data instance as the 'metadata' attribute. It is a hdf5 group.
        assert spec_dt.metadata is not None
        assert isinstance(spec_dt.metadata, dict)  # Make sure it is a standard dict, not a hdf5 group.
        num_metadata_dsets = len(spec_dt.metadata['dsets'].keys())
        num_metadata_attrs = len(spec_dt.metadata['attrs'].keys())
        assert num_metadata_dsets == 0
        assert num_metadata_attrs == 6
        for kv in keys_unique:
            assert kv in spec_dt.metadata['attrs'].keys()
            # Note that the cF-specific unpacking may convert ndarray values into something else. In case of fooof, we convert
            # some ndarrays (all the peak_params__n_m and gaussian_params__n_m) to list, so we accept both types here.
            assert isinstance(spec_dt.metadata['attrs'].get(kv), list) or isinstance(spec_dt.metadata['attrs'].get(kv), np.ndarray)

        # check that the cfg is correct (required for replay)
        assert spec_dt.cfg['freqanalysis']['output'] == 'fooof'
        return spec_dt

    def test_metadata_parallel_with_parallel_storage(self):
        """
        Test metadata propagation in with parallel compute and parallel storage.
        Without trial averaging (`keeptrials=True` in cfg), parallel storage is used.

        Note: This function is currently identical to 'test_metadata_parallel_with_sequential_storage()',
              the only difference is the `cfg.keeptrials = True`.
              TODO: We should refactor this.

        Note2: This test implicitely also tests the case that no metadata is attached by the cF, because
               before FOOOF, we also call mtmfft, which does not attach metadata.
        """
        cfg = TestMetadataUsingFooof.get_fooof_cfg()
        cfg.pop('fooof_opt', None)
        cfg.parallel = True # enable parallel computation
        cfg.keeptrials = True # enable parallel storage (is turned off when trial averaging is happening)
        fooof_opt = {'peak_width_limits': (1.0, 12.0)}  # Increase lower limit to avoid fooof warning.
        spec_dt = freqanalysis(cfg, self.tfData, fooof_opt=fooof_opt)

        # These are known from the input data and cfg.
        num_trials_fooof = 100
        data_size = 100

        # check frequency axis
        assert spec_dt.freq.size == data_size
        assert spec_dt.freq[0] == 1
        assert spec_dt.freq[99] == 100.

        # check the log
        assert "fooof_method = fooof" in spec_dt._log
        assert "fooof_aperiodic" not in spec_dt._log
        assert "fooof_peaks" not in spec_dt._log
        assert "fooof_opt" in spec_dt._log

        # check the data
        assert spec_dt.data.ndim == 4
        #print("spec_dt.data.shape is {}.".format(spec_dt.data.shape))
        assert spec_dt.data.shape == (num_trials_fooof, 1, data_size, 1) # Differs from other tests due to `keeptrials=True`.
        assert not np.isnan(spec_dt.data).any()

        # check metadata from 2nd cF return value, added to the hdf5 dataset 'data' as attributes.
        k_unique = "__0_0"  # TODO: this is currently still hardcoded, and the _0 is the one added by the first cF function call.
                         #       depending on data size and RAM, there may or may not be several calls, and "_1" , "_2", ... exist.
        expected_fooof_dict_entries = ["aperiodic_params", "gaussian_params", "peak_params", "n_peaks", "r_squared", "error"]
        keys_unique = [kv + k_unique for kv in expected_fooof_dict_entries]

        test_metadata_on_metadata_group = True

        if test_metadata_on_metadata_group:
            # Now for the metadata. This got attached to the syncopy data instance as the 'metadata' attribute. It is a hdf5 group.
            assert spec_dt.metadata is not None
            assert isinstance(spec_dt.metadata, dict)  # Make sure it is a standard dict, not a hdf5 group.
            num_metadata_dsets = len(spec_dt.metadata['dsets'].keys())
            num_metadata_attrs = len(spec_dt.metadata['attrs'].keys())  # Get keys of hdf5 attribute manager.
            assert num_metadata_dsets == 0
            assert num_metadata_attrs == 6 * num_trials_fooof
            print("keys={k}".format(k=",".join(spec_dt.metadata['attrs'].keys())))
            for kv in keys_unique:
                assert (kv) in spec_dt.metadata['attrs'].keys()
                # Note that the cF-specific unpacking may convert ndarray values into something else. In case of fooof, we convert
                # some ndarrays (all the peak_params__n_m and gaussian_params__n_m) to list, so we accept both types here.
                assert isinstance(spec_dt.metadata['attrs'].get(kv), list) or isinstance(spec_dt.metadata['attrs'].get(kv), np.ndarray)

        # check that the cfg is correct (required for replay)
        assert spec_dt.cfg['freqanalysis']['output'] == 'fooof'
        return spec_dt

    def test_metadata_parallel_with_parallel_storage_and_channel_parallelisation(self):
        """
        Test metadata propagation in with channel parallelisation. This implies
        parallel compute and parallel storage.

        For syncopy to use channel parallelization, we must make sure that:
        - we use parallel mode (`cfg.parallel` = `True`)
        - 'chan_per_worker' is set to a positive integer (it is a kwarg of `freqanalysis()`)
        - `keeptrials=True`, otherwise it makes no sense
        - We also do not select specific channels in `cfg.select`, as that is not supported with channel parallelisation.

        Note2: This test implicitely also tests the case that no metadata is attached by the cF, because
               before FOOOF, we also call mtmfft, which does not attach metadata.
        """
        cfg = TestMetadataUsingFooof.get_fooof_cfg()
        cfg.pop('fooof_opt', None)
        cfg.parallel = True # enable parallel computation
        cfg.keeptrials = True # enable parallel storage (is turned off when trial averaging is happening)
        cfg.select = None
        fooof_opt = {'peak_width_limits': (1.0, 12.0)}  # Increase lower limit to avoid fooof warning.
        num_trials_fooof = 100
        num_channels = 5
        chan_per_worker = 2
        multi_chan_data = _get_fooof_signal(nTrials=num_trials_fooof, nChannels = num_channels)
        spec_dt = freqanalysis(cfg, multi_chan_data, fooof_opt=fooof_opt, chan_per_worker=chan_per_worker)

        # How many more calls we expect due to channel parallelization.
        calls_per_trial = int(math.ceil(num_channels / chan_per_worker))

        # These are known from the input data and cfg.

        data_size = 100

        # check frequency axis
        assert spec_dt.freq.size == data_size
        assert spec_dt.freq[0] == 1
        assert spec_dt.freq[99] == 100.

        # check the log
        assert "fooof_method = fooof" in spec_dt._log
        assert "fooof_aperiodic" not in spec_dt._log
        assert "fooof_peaks" not in spec_dt._log
        assert "fooof_opt" in spec_dt._log

        # check the data
        assert spec_dt.data.ndim == 4
        assert spec_dt.data.shape == (num_trials_fooof, 1, data_size, num_channels) # Differs from other tests due to `keeptrials=True` and channels.
        assert not np.isnan(spec_dt.data).any()

        # check metadata from 2nd cF return value, added to the hdf5 dataset 'data' as attributes.
        expected_fooof_dict_entries = ["aperiodic_params", "gaussian_params", "peak_params", "n_peaks", "r_squared", "error"]
        keys_unique = list()
        for fde in expected_fooof_dict_entries:
            for trial_idx in range(num_trials_fooof):
                for call_idx in range(calls_per_trial):
                    keys_unique.append(encode_unique_md_label(fde, trial_idx, call_idx))

        test_metadata_on_metadata_group = True

        if test_metadata_on_metadata_group:
            # Now for the metadata. This got attached to the syncopy data instance as the 'metadata' attribute,
            # and was collected from the metadata of each part of the virtual hdf5 dataset. It is a standard dictionary (not a hdf5 group).
            assert spec_dt.metadata is not None
            assert isinstance(spec_dt.metadata, dict)  # Make sure it is a standard dict, not a hdf5 group.
            num_metadata_dsets = len(spec_dt.metadata['dsets'].keys())
            num_metadata_attrs = len(spec_dt.metadata['attrs'].keys())  # Get keys of hdf5 attribute manager.
            assert num_metadata_dsets == 0
            assert num_metadata_attrs == 6 * num_trials_fooof * calls_per_trial
            print("keys={k}".format(k=",".join(spec_dt.metadata['attrs'].keys())))
            for kv in keys_unique:
                assert kv in spec_dt.metadata['attrs'].keys()
                # Note that the cF-specific unpacking may convert ndarray values into something else. In case of fooof, we convert
                # some ndarrays (all the peak_params__n_m and gaussian_params__n_m) to list, so we accept both types here.
                assert isinstance(spec_dt.metadata['attrs'].get(kv), list) or isinstance(spec_dt.metadata['attrs'].get(kv), np.ndarray)

        # check that the cfg is correct (required for replay)
        assert spec_dt.cfg['freqanalysis']['output'] == 'fooof'
        return spec_dt

    @pytest.mark.skip(reason="we only care about sequential for now")
    @skip_without_acme
    def test_fooof_parallel(self, testcluster=None):

        plt.ioff()
        client = dd.Client(testcluster)
        all_tests = [self.test_metadata_1call_sequential]

        for test_name in all_tests:
            test_method = getattr(self, test_name)
            test_method()
        client.close()
        plt.ion()

if __name__ == "__main__":
    print("---------------Testing---------------")
    #TestMetadataUsingFooof().test_metadata_1call_sequential()
    #TestMetadataUsingFooof().test_metadata_parallel_with_sequential_storage()
    #TestMetadataUsingFooof().test_metadata_parallel_with_parallel_storage()
    TestMetadataUsingFooof().test_metadata_parallel_with_parallel_storage_and_channel_parallelisation()
    print("------------Testing done------------")
