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
from syncopy.datatype.methods.copy import copy
from syncopy.shared.tools import get_defaults
from syncopy.tests.synth_data import AR2_network, phase_diffusion
from syncopy.shared.metadata import encode_unique_md_label, decode_unique_md_label, get_res_details, _parse_backend_metadata, _merge_md_list, metadata_from_hdf5_file
from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYWarning
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

        # Test with numpy array with datatype np.object, which is not valid.
        invalid_val = np.array([np.zeros((5,3)), np.zeros((8,3))], dtype = object)  # The dtype is required to silence numpy deprecation warnings, the dtype will be object even without it.
        assert invalid_val.dtype == object
        with pytest.raises(SPYValueError) as err:
            a, b = get_res_details((np.zeros(3), {'a': invalid_val}))
        assert "the second return value of user-supplied compute functions must be a dict containing np.ndarrays with datatype other than 'np.object'" in str(err.value)

        # Test with tuple of correct length, 2nd value is dict, and values in ndarray are string (but not object). This is fine.
        a, b = get_res_details((np.zeros(3), {'a': np.array(['apples', 'foobar', 'cowboy'])}))
        assert 'a' in b

    def test_parse_details(self):
        # Test for error if input is not dict.
        with pytest.raises(SPYTypeError) as err:
            attrs = _parse_backend_metadata(np.zeros(3))
        assert "expected dict found ndarray" in str(err.value)

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

class TestMetadataUsingFooof():
    """
    Test 2nd cF function return value, with FOOOF as example compute method.
    """

    tfData = _get_fooof_signal()
    expected_fooof_dict_entries = ["aperiodic_params", "gaussian_params", "peak_params", "n_peaks", "r_squared", "error"]

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
                         # larger, but the cfg.foilim setting (in get_fooof_cfg()) limits to 100 samples.
        num_trials_fooof = 1 # Because of keeptrials = False in cfg.

        assert spec_dt.data.shape == (num_trials_fooof, 1, data_size, 1)

        # check metadata from 2nd cF return value, added to the hdf5 dataset as attribute.
        keys_unique = [kv + "__0_0" for kv in self.expected_fooof_dict_entries]

        # Now for the metadata. This got attached to the syncopy data instance as the 'metadata' attribute. It is a hdf5 group.
        assert spec_dt.metadata is not None
        assert isinstance(spec_dt.metadata, dict)  # Make sure it is a standard dict, not a hdf5 group.
        num_metadata_attrs = len(spec_dt.metadata.keys())  # Get keys of dict
        assert num_metadata_attrs == 6
        for kv in keys_unique:
            assert kv in spec_dt.metadata.keys()
            # Note that the cF-specific unpacking may convert ndarray values into something else. In case of fooof, we convert
            # some ndarrays (all the peak_params__n_m and gaussian_params__n_m) to list, so we accept both types here.
        assert isinstance(spec_dt.metadata.get(kv), (list, np.ndarray))

        # check that the cfg is correct (required for replay)
        assert spec_dt.cfg['freqanalysis']['output'] == 'fooof'

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
        keys_unique = [kv + "__0_0" for kv in self.expected_fooof_dict_entries]

        # Now for the metadata. This got attached to the syncopy data instance as the 'metadata' attribute. It is a hdf5 group.
        assert spec_dt.metadata is not None
        assert isinstance(spec_dt.metadata, dict)  # Make sure it is a standard dict, not a hdf5 group.
        num_metadata_attrs = len(spec_dt.metadata.keys())
        assert num_metadata_attrs == 6
        for kv in keys_unique:
            assert kv in spec_dt.metadata.keys()
            # Note that the cF-specific unpacking may convert ndarray values into something else. In case of fooof, we convert
            # some ndarrays (all the peak_params__n_m and gaussian_params__n_m) to list, so we accept both types here.
            assert isinstance(spec_dt.metadata.get(kv), list) or isinstance(spec_dt.metadata.get(kv), np.ndarray)

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
        keys_unique = [kv + "__0_0" for kv in self.expected_fooof_dict_entries]

        # Now for the metadata. This got attached to the syncopy data instance as the 'metadata' attribute. It is a hdf5 group.
        assert spec_dt.metadata is not None
        assert isinstance(spec_dt.metadata, dict)  # Make sure it is a standard dict, not a hdf5 group.
        num_metadata_attrs = len(spec_dt.metadata.keys())
        assert num_metadata_attrs == 6 * num_trials_fooof
        for kv in keys_unique:
            assert (kv) in spec_dt.metadata.keys()
            # Note that the cF-specific unpacking may convert ndarray values into something else. In case of fooof, we convert
            # some ndarrays (all the peak_params__n_m and gaussian_params__n_m) to list, so we accept both types here.
            assert isinstance(spec_dt.metadata.get(kv), list) or isinstance(spec_dt.metadata.get(kv), np.ndarray)

        # check that the cfg is correct (required for replay)
        assert spec_dt.cfg['freqanalysis']['output'] == 'fooof'

    def test_par_with_selections(self):
        """
        Test metadata propagation in with parallel compute and parallel storage,
        and trial selections.

        In the case of trial selections, the computation of absolute trial indices
        from relative ones uses the non-trivial branch. We test for correctly
        reconstructed absolute trial indices in this test.
        """
        cfg = TestMetadataUsingFooof.get_fooof_cfg()
        cfg.pop('fooof_opt', None)
        cfg.parallel = True # enable parallel computation
        cfg.keeptrials = True # enable parallel storage (is turned off when trial averaging is happening)
        fooof_opt = {'peak_width_limits': (1.0, 12.0)}  # Increase lower limit to avoid fooof warning.
        data = self.tfData.copy()

        selected_trials = [3, 5, 7]

        cfg.select = { 'trials': selected_trials }
        #spy.selectdata(data, trials=selected_trials, inplace=True) # TODO: This line should also work,
        #  and be equivalent to the `cfg.select`... line above, but it seems to have no effect. Bug? See #332

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
        keys_unique = [kv + "__0_0" for kv in self.expected_fooof_dict_entries]

        # Now for the metadata. This got attached to the syncopy data instance as the 'metadata' attribute. It is a hdf5 group.
        assert spec_dt.metadata is not None
        assert isinstance(spec_dt.metadata, dict)  # Make sure it is a standard dict, not a hdf5 group.
        num_metadata_attrs = len(spec_dt.metadata.keys())
        assert num_metadata_attrs == 6 * num_trials_fooof_selected
        for kv in keys_unique:
            assert (kv) in spec_dt.metadata.keys()
            assert isinstance(spec_dt.metadata.get(kv), (list, np.ndarray))

        # Check that the metadata keys are absolute.
        # TO DISCUSS: This does not work for fooof, beause the mtmfft comsumes the selection, leaving
        #             fooof with a smaller trials list and no selection (and trial indices in the list
        #             are relative to the mtmfft selection).
        #md_trial_indices = []
        #for k in spec_dt.metadata.keys():
        #    label, trial_idx, chunk_idx = decode_unique_md_label(k)
        #    md_trial_indices.append(int(trial_idx))
        #for ti in md_trial_indices:
        #    assert ti in selected_trials, f"Expected trial index '{ti}' not in selected_trials"

        # check that the cfg is correct (required for replay)
        assert spec_dt.cfg['freqanalysis']['output'] == 'fooof'

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
        calls_per_trial = int(math.ceil(num_channels / chan_per_worker))
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
        for fde in self.expected_fooof_dict_entries:
            for trial_idx in range(num_trials_fooof):
                for call_idx in range(calls_per_trial):
                    keys_unique.append(encode_unique_md_label(fde, trial_idx, call_idx))

        assert spec_dt.metadata is not None
        assert isinstance(spec_dt.metadata, dict)  # Make sure it is a standard dict, not a hdf5 group.
        num_metadata_attrs = len(spec_dt.metadata.keys())
        assert num_metadata_attrs == 6 * num_trials_fooof * calls_per_trial
        for kv in keys_unique:
            assert kv in spec_dt.metadata.keys()
            assert isinstance(spec_dt.metadata.get(kv), (list, np.ndarray))

        # check that the cfg is correct (required for replay)
        assert spec_dt.cfg['freqanalysis']['output'] == 'fooof'

    @skip_without_acme
    def test_metadata_parallel(self, testcluster=None):

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


class TestMetadataUsingMtmfft():
    tfData = _get_fooof_signal()
    expected_metadata_dict_entries = ["freqs_hash"]

    @staticmethod
    def get_mtmfft_cfg():
        cfg = TestMetadataUsingFooof.get_fooof_cfg()
        cfg.output = "pow"
        return cfg

    def test_sequential_mtmfft(self):
        """
        Test metadata propagation with mtmfft in sequential compute mode.
        """
        # These are known from the input data and cfg.
        data_size = 100  # Number of samples (per trial) seen by mtmfftm. The full signal returned by _get_fooof_signal() is
                         # larger, but the cfg.foilim setting (in get_fooof_cfg()) limits to 100 samples.
        num_trials_out = 1 # Because of keeptrials = False in cfg.

        cfg = TestMetadataUsingMtmfft.get_mtmfft_cfg()
        cfg.parallel = False

        freq_axis = np.arange(data_size) # Attach a freq axis to input data, to compare hashes later.
        spec_dt = freqanalysis(cfg, self.tfData)



        assert spec_dt.data.shape == (num_trials_out, 1, data_size, 1)

        keys_unique = [kv + "__0_0" for kv in self.expected_metadata_dict_entries]

        assert spec_dt.metadata is not None
        assert isinstance(spec_dt.metadata, dict)  # Make sure it is a standard dict, not a hdf5 group.
        for kv in keys_unique:
            assert kv in spec_dt.metadata.keys()
            assert isinstance(spec_dt.metadata.get(kv), (np.bytes_))


        # Demo: Use the extra return values to make sure the hashes of the frequency
        #       arrays are identicals across all trials:
        from hmac import compare_digest
        reference_hash = None
        trials_with_mismatches = []
        num_hashes_checked = 0
        for unique_md_label_rel, v in spec_dt.metadata.items():
            label, trial_idx, _ = decode_unique_md_label(unique_md_label_rel)
            if label == "freqs_hash":
                trial_freqs_hash = v
                if reference_hash is None:
                    reference_hash = trial_freqs_hash
                else:
                    if not compare_digest(reference_hash, trial_freqs_hash):
                        trials_with_mismatches.append(trial_idx)
                num_hashes_checked += 1
        if trials_with_mismatches:
            SPYWarning(f"Frequency axes hashes mismatched for {len(trials_with_mismatches)} trials: {trials_with_mismatches} against reference hash from first trial.")
        else:
            print(f"Frequency axes hashes are identical across all {num_hashes_checked} trials.")


if __name__ == "__main__":
    T1 = TestMetadataHelpers()
    T2 = TestMetadataUsingFooof()
    T3 = TestMetadataUsingMtmfft()
    print("=================Testing================")
    T3.test_sequential_mtmfft()
    print("===============Testing done==============")