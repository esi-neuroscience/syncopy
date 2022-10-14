# -*- coding: utf-8 -*-
#
# Ensure synthetic data generator functions used in tests work as expected.
#

# Builtin/3rd party package imports
import numpy as np
import pytest

from syncopy.tests.synth_data import white_noise, AR2_network
from syncopy.shared.errors import SPYValueError
import syncopy as spy

class TestSynthData:

    nTrials=100
    nChannels = 1
    nSamples = 1000
    samplerate = 1000

    def test_white_noise_without_seed(self):
        """Without seed set, the data should not be identical.
           Note: This does not use collect trials.
        """
        wn1 = white_noise(nSamples=self.nSamples, nChannels=self.nChannels, seed=None)
        wn2 = white_noise(nSamples=self.nSamples, nChannels=self.nChannels, seed=None)
        assert isinstance(wn1, np.ndarray)
        assert isinstance(wn2, np.ndarray)

        assert not np.allclose(wn1, wn2)

    def test_white_noise_with_seed(self):
        """With seed set, the data should be identical.
           Note: This does not use @collect_trials.
        """
        seed = 42
        wn1 = white_noise(nSamples=self.nSamples, nChannels=self.nChannels, seed=seed)
        wn2 = white_noise(nSamples=self.nSamples, nChannels=self.nChannels, seed=seed)
        assert isinstance(wn1, np.ndarray)
        assert isinstance(wn2, np.ndarray)

        assert np.allclose(wn1, wn2)

    def test_collect_trials_wn_seed_array(self):
        """Uses @collect_trials."""
        # Trials must differ within an object if seed_per_trial is left at default (true):
        seed = 42
        wn1 = white_noise(nSamples=self.nSamples, nChannels=self.nChannels, nTrials=self.nTrials, seed=seed)
        assert isinstance(wn1, spy.AnalogData)
        assert not np.allclose(wn1.show(trials=0), wn1.show(trials=1))

        # However, using the same seed for a new instance must lead to identical trials between instances:
        wn2 = white_noise(nSamples=self.nSamples, nChannels=self.nChannels, nTrials=self.nTrials, seed=seed)
        assert np.allclose(wn1.show(trials=0), wn2.show(trials=0))
        assert np.allclose(wn1.show(trials=1), wn2.show(trials=1))

    def test_collect_trials_wn_seed_scalar(self):
        """Uses @collect_trials."""
        # Trials must be identical within an object if seed_per_trial is False (and a seed is used).
        seed = 42
        wn1 = white_noise(nSamples=self.nSamples, nChannels=self.nChannels, nTrials=self.nTrials, seed=seed, seed_per_trial=False)
        assert isinstance(wn1, spy.AnalogData)
        assert np.allclose(wn1.show(trials=0), wn1.show(trials=1))

        # And also, using the same scalar seed again should lead to an identical object.
        wn2 = white_noise(nSamples=self.nSamples, nChannels=self.nChannels, nTrials=self.nTrials, seed=seed, seed_per_trial=False)
        assert np.allclose(wn1.show(trials=0), wn2.show(trials=0))
        assert np.allclose(wn1.show(trials=1), wn2.show(trials=1))

    def test_collect_trials_wn_no_seed(self):
        """Uses @collect_trials."""
        # Trials must differ within an object if seed is None:
        seed = None
        wn1 = white_noise(nSamples=self.nSamples, nChannels=self.nChannels, nTrials=self.nTrials, seed=seed)
        assert isinstance(wn1, spy.AnalogData)
        assert not np.allclose(wn1.show(trials=0), wn1.show(trials=1))

        # And instances must also differ:
        wn2 = white_noise(nSamples=self.nSamples, nChannels=self.nChannels, nTrials=self.nTrials, seed=seed)
        assert not np.allclose(wn1.show(trials=0), wn2.show(trials=0))
        assert not np.allclose(wn1.show(trials=1), wn2.show(trials=1))


    #### Tests for AR2_network

    def test_ar2_without_seed(self):
        """Without seed set, the data should not be identical.
           Note: This does not use collect trials.
        """
        num_channels = 2
        arn1 = AR2_network(nSamples=self.nSamples, seed=None)  # 2 channels, via default adj matrix
        arn2 = AR2_network(nSamples=self.nSamples, seed=None)
        assert isinstance(arn1, np.ndarray)
        assert isinstance(arn2, np.ndarray)
        assert arn1.shape == (self.nSamples, num_channels)
        assert arn2.shape == (self.nSamples, num_channels)

        assert not np.allclose(arn1, arn2)

    def test_ar2_with_seed(self):
        """With seed set, the data should be identical.
           Note: This does not use collect trials.
        """
        seed = 42
        arn1 = AR2_network(nSamples=self.nSamples, seed=seed, seed_per_trial=False)
        arn2 = AR2_network(nSamples=self.nSamples, seed=seed, seed_per_trial=False)

        assert np.allclose(arn1, arn2)

if __name__ == '__main__':
    T1 = TestSynthData()
