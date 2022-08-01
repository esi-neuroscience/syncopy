# -*- coding: utf-8 -*-
#
# Test metadata implementation.


import pytest
import numpy as np
import matplotlib.pyplot as plt


# Local imports
from syncopy import freqanalysis
from syncopy.shared.tools import get_defaults
from syncopy.tests.synth_data import AR2_network, phase_diffusion
import syncopy as spy
from syncopy import __acme__
if __acme__:
    import dask.distributed as dd

# Decorator to decide whether or not to run dask-related tests
skip_without_acme = pytest.mark.skipif(not __acme__, reason="acme not available")


def _get_fooof_signal(nTrials=100):
    """
    Produce suitable test signal for fooof, with peaks at 30 and 50 Hz.

    Note: One must perform trial averaging during the FFT to get realistic
    data out of it (and reduce noise). Then work with the averaged data.

    Returns AnalogData instance.
    """
    nSamples = 1000
    nChannels = 1
    samplerate = 1000
    ar1_part = AR2_network(AdjMat=np.zeros(1), nSamples=nSamples, alphas=[0.9, 0], nTrials=nTrials)
    pd1 = phase_diffusion(freq=30., eps=.1, fs=samplerate, nChannels=nChannels, nSamples=nSamples, nTrials=nTrials)
    pd2 = phase_diffusion(freq=50., eps=.1, fs=samplerate, nChannels=nChannels, nSamples=nSamples, nTrials=nTrials)
    signal = ar1_part + .8 * pd1 + 0.6 * pd2
    return signal


class TestMetadataUsingFooof():
    """
    Test passing on 2nd cF function return value in sequential mode.
    This function tests with the sime
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

    def test_metadata_1call(self):
        """
        This tests the intended operation with output type 'fooof': with an input that does not
        include zero, ensured by using the 'foilim' argument/setting when calling freqanalysis.

        This returns the full, fooofed spectrum.
        """
        cfg = TestMetadataUsingFooof.get_fooof_cfg()
        cfg.pop('fooof_opt', None)
        fooof_opt = {'peak_width_limits': (1.0, 12.0)}  # Increase lower limit to avoid fooof warning.
        spec_dt = freqanalysis(cfg, self.tfData, fooof_opt=fooof_opt)

        # check frequency axis
        assert spec_dt.freq.size == 100
        assert spec_dt.freq[0] == 1
        assert spec_dt.freq[99] == 100.

        # check the log
        assert "fooof_method = fooof" in spec_dt._log
        assert "fooof_aperiodic" not in spec_dt._log
        assert "fooof_peaks" not in spec_dt._log
        assert "fooof_opt" in spec_dt._log

        # check the data
        assert spec_dt.data.ndim == 4
        assert spec_dt.data.shape == (1, 1, 100, 1)
        assert not np.isnan(spec_dt.data).any()

        # check metadata from 2nd cF return value, added to the hdf5 dataset as attribute.
        assert len(spec_dt.data.attrs.keys()) == 6
        k_unique = "_0"
        expected_fooof_dict_entries = ["aperiodic_params", "gaussian_params", "peak_params", "n_peaks", "r_squared", "error"]
        assert len(spec_dt.data.attrs.keys()) == len(expected_fooof_dict_entries)
        keys_unique = [kv + k_unique for kv in expected_fooof_dict_entries]
        for kv in keys_unique:
            assert (kv) in spec_dt.data.attrs.keys()
            assert isinstance(spec_dt.data.attrs.get(kv), np.ndarray)
        # Expect one entry in detail.
        n_peaks = spec_dt.data.attrs.get("n_peaks" + k_unique)
        assert isinstance(n_peaks, np.ndarray)
        assert n_peaks.size == 1 # cfg.keeptrials is False, so FOOOF operates on a single trial and we expect only one value here.
        assert spec_dt.data.attrs.get("r_squared" + k_unique).size == 1  # Same, see line above.
        assert spec_dt.data.attrs.get("error" + k_unique).size == 1  # Same, see line above.

        # Now for the metadata. This got attached to the syncopy data instance as the 'metadata' attribute. It is a hdf5 group.
        assert spec_dt.metadata is not None
        num_metadata_dsets = len(spec_dt.metadata.keys())
        num_metadata_attrs = len(spec_dt.metadata.attrs.keys())  # Get keys of hdf5 attribute manager.
        assert num_metadata_dsets == 0
        assert num_metadata_attrs == 6
        for kv in keys_unique:
            assert (kv) in spec_dt.metadata.attrs.keys()
            assert isinstance(spec_dt.metadata.attrs.get(kv), np.ndarray)



        # Now for the separate 'metadata' group in the hdf5 container
        #assert len(spec_dt.metadata.attrs.keys()) == 6
        # We cannot access it like this from the Sycopy data instance, we need to get a handle to the hdf5.


        # check that the cfg is correct (required for replay)
        assert spec_dt.cfg['freqanalysis']['output'] == 'fooof'
        return spec_dt

    @pytest.mark.skip(reason="we only care about sequential for now")
    @skip_without_acme
    def test_fooof_parallel(self, testcluster=None):

        plt.ioff()
        client = dd.Client(testcluster)
        all_tests = [self.test_metadata_1call]

        for test_name in all_tests:
            test_method = getattr(self, test_name)
            test_method()
        client.close()
        plt.ion()

if __name__ == "__main__":
    print("---------------Testing---------------")
    TestMetadataUsingFooof().test_metadata_1call()
    print("------------Testing done------------")
