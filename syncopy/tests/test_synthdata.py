# -*- coding: utf-8 -*-
#
# Ensure synthetic data generator functions used in tests work as expected.
#

# Builtin/3rd party package imports
import numpy as np
import pytest

from syncopy import AnalogData
from syncopy.shared.tools import StructDict
from syncopy.synthdata import collect_trials
from syncopy.synthdata import white_noise, ar2_network
from syncopy.shared.errors import SPYValueError


def test_trial_collection():
    """
    tests the decorator to construct
    multi-trial AnalogData from single trial functions
    """

    # a trivial single trial (np.ndarray) producing function
    @collect_trials
    def ones(nChannels, nSamples):
        return np.ones((nSamples, nChannels))

    cfg = StructDict()
    cfg.nTrials = 10
    cfg.samplerate = 12
    cfg.nChannels = 3
    cfg.nSamples = 10

    adata = ones(cfg)

    assert isinstance(adata, AnalogData)
    assert len(adata.trials) == cfg.nTrials
    assert len(adata.channel) == cfg.nChannels
    assert len(adata.time[0]) == cfg.nSamples
    assert adata.samplerate == cfg.samplerate

    # with nTrials=None, the decorator gets bypassed
    # and returns the single trial array
    cfg["nTrials"] = None
    arr = ones(cfg)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (cfg.nSamples, cfg.nChannels)


class TestSynthData:

    nTrials = 100
    nChannels = 1
    nSamples = 1000
    samplerate = 1000

    def test_white_noise_without_seed(self):
        """Without seed set, the data should not be identical.
        Note: This does not use collect trials.
        """
        cfg = StructDict()
        cfg.nSamples = self.nSamples
        cfg.nChannels = self.nChannels
        cfg.nTrials = None
        # that is the default
        # cfg.seed = None

        wn1 = white_noise(cfg)
        wn2 = white_noise(cfg)
        assert isinstance(wn1, np.ndarray)
        assert isinstance(wn2, np.ndarray)

        assert not np.allclose(wn1, wn2)

    def test_white_noise_with_seed(self):
        """With seed set, the data should be identical.
        Note: This does not use @collect_trials.
        """
        cfg = StructDict()
        cfg.nSamples = self.nSamples
        cfg.nChannels = self.nChannels
        cfg.nTrials = None
        cfg.seed = 42

        wn1 = white_noise(cfg)
        wn2 = white_noise(cfg)

        assert isinstance(wn1, np.ndarray)
        assert isinstance(wn2, np.ndarray)

        assert np.allclose(wn1, wn2)

    def test_collect_trials_wn_seed_array(self):
        """Uses @collect_trials."""
        # Trials must differ within an object if seed_per_trial is left at default (true):
        seed = 42
        wn1 = white_noise(
            nSamples=self.nSamples,
            nChannels=self.nChannels,
            nTrials=self.nTrials,
            seed=seed,
        )
        assert isinstance(wn1, AnalogData)
        assert not np.allclose(wn1.show(trials=0), wn1.show(trials=1))

        # However, using the same seed for a new instance must lead to identical trials between instances:
        wn2 = white_noise(
            nSamples=self.nSamples,
            nChannels=self.nChannels,
            nTrials=self.nTrials,
            seed=seed,
        )
        assert np.allclose(wn1.show(trials=0), wn2.show(trials=0))
        assert np.allclose(wn1.show(trials=1), wn2.show(trials=1))

    def test_collect_trials_wn_seed_scalar(self):
        """Uses @collect_trials."""
        # Trials must be identical within an object if seed_per_trial is False (and a seed is used).
        seed = 42
        wn1 = white_noise(
            nSamples=self.nSamples,
            nChannels=self.nChannels,
            nTrials=self.nTrials,
            seed=seed,
            seed_per_trial=False,
        )
        assert isinstance(wn1, AnalogData)
        assert np.allclose(wn1.show(trials=0), wn1.show(trials=1))

        # And also, using the same scalar seed again should lead to an identical object.
        wn2 = white_noise(
            nSamples=self.nSamples,
            nChannels=self.nChannels,
            nTrials=self.nTrials,
            seed=seed,
            seed_per_trial=False,
        )
        assert np.allclose(wn1.show(trials=0), wn2.show(trials=0))
        assert np.allclose(wn1.show(trials=1), wn2.show(trials=1))

    def test_collect_trials_wn_no_seed(self):
        """Uses @collect_trials."""
        # Trials must differ within an object if seed is None:
        seed = None
        wn1 = white_noise(
            nSamples=self.nSamples,
            nChannels=self.nChannels,
            nTrials=self.nTrials,
            seed=seed,
        )
        assert isinstance(wn1, AnalogData)
        assert not np.allclose(wn1.show(trials=0), wn1.show(trials=1))

        # And instances must also differ:
        wn2 = white_noise(
            nSamples=self.nSamples,
            nChannels=self.nChannels,
            nTrials=self.nTrials,
            seed=seed,
        )
        assert not np.allclose(wn1.show(trials=0), wn2.show(trials=0))
        assert not np.allclose(wn1.show(trials=1), wn2.show(trials=1))

    #### Tests for AR2_network

    def test_ar2_without_seed(self):
        """Without seed set, the data should not be identical.
        Note: This does not use collect trials.
        """
        num_channels = 2
        arn1 = ar2_network(
            nSamples=self.nSamples, seed=None, nTrials=None
        )  # 2 channels, via default adj matrix
        arn2 = ar2_network(nSamples=self.nSamples, seed=None, nTrials=None)
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
        arn1 = ar2_network(nSamples=self.nSamples, seed=seed, seed_per_trial=False, nTrials=None)
        arn2 = ar2_network(nSamples=self.nSamples, seed=seed, seed_per_trial=False, nTrials=None)

        assert np.allclose(arn1, arn2)


if __name__ == "__main__":
    T1 = TestSynthData()
