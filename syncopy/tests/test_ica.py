# -*- coding: utf-8 -*-
#
# Test independent component analysis (ICA).
#

# 3rd party imports
import pytest
import numpy as np

# Local imports
import dask.distributed as dd
from syncopy.tests import synth_data as sd
from syncopy.tests.helpers import test_seed

from syncopy.shared.errors import SPYValueError, SPYTypeError
from syncopy.shared.tools import get_defaults

import syncopy as spy
import sklearn.decomposition as decomposition
from scipy import signal


def get_ica_testdata(n_samples=8000, samplerate=1000, add_noise=True, as_ad=True):
    """
    Create ICA test data from different signals.

    Parameters
    ----------
    n_samples : int, number of samples
    samplerate : int, sampling rate
    add_noise : bool, whether to add noise to the data
    as_ad : bool, whether to return the data as AnalogData object
    """
    duration_sec = n_samples / samplerate
    time = np.linspace(0, duration_sec, n_samples)

    # Create 3 signals
    s1 = np.sin(2 * time)                   # sinusoidal
    s2 = np.sign(np.sin(3 * time))          # square signal
    s3 = signal.sawtooth(2 * np.pi * time)  # saw tooth signal

    # simulate eye blic
    eyeblink_start_times = np.arange(0.5, duration_sec, 1.5)
    eyeblink_duration = 0.1
    for t in eyeblink_start_times:
        s1[int(t * samplerate):int((t + eyeblink_duration) * samplerate)] *= 5
        s2[int(t * samplerate):int((t + eyeblink_duration) * samplerate)] *= 5
        s3[int(t * samplerate):int((t + eyeblink_duration) * samplerate)] *= 5

    # Mix data
    S = np.c_[s1, s2, s3] # The sources
    if add_noise:
        S += 0.2 * np.random.normal(size=S.shape) # Add noise
    S /= S.std(axis=0)  # Normalize
    A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
    X = np.dot(S, A.T)
    X = X.T
    if as_ad:
        X = spy.AnalogData(data=X, samplerate=samplerate)
    return X



class TestFastICA():
    n_trials = 2
    n_samples = 5000
    adata = 100 * sd.white_noise(n_trials, nSamples=n_samples) + 5

    def test_preproc_ica(self):
        cfg = get_defaults(spy.runica)
        cfg.method = 'fastica'
        res = spy.runica(self.adata, cfg)
        assert type(res) == spy.AnalogData

    def test_ica_fails_with_wrong_input(self):
        cfg = get_defaults(spy.runica)
        cfg.method = 'fastica'
        wrong_data = spy.CrossSpectralData(data=np.ones((2, 2, 2, 2)), samplerate=1)
        with pytest.raises(SPYTypeError, match='Syncopy AnalogData object'):
            _ = spy.runica(wrong_data, cfg)


class TestSklearnFastICAAPI():
    """
    Temporary test to evaluate the sklearn API.
    """
    rng = np.random.default_rng(test_seed)
    n_samples = 1000
    n_channels = 16
    data = rng.normal(size=(n_samples, n_channels))

    def test_api_fittransform(self):
        ica = decomposition.FastICA()
        res = ica.fit_transform(self.data)
        assert isinstance(res, np.ndarray)
        assert res.shape == self.data.shape
        #comps = ica.components_ # The linear operator to apply to the data to get the independent sources
        #assert comps.shape == self.data.shape
        #mix = ica.mixing_ # The pseudo-inverse of components_. It is the linear operator that maps independent sources to the data.
        #assert mix.shape == self.data.shape
        assert ica.n_iter_ <= 200

    def test_api_inverse_transform(self):
        ica = decomposition.FastICA()
        res = ica.fit_transform(self.data)
        assert isinstance(res, np.ndarray)
        assert res.shape == self.data.shape
        orig_re = ica.inverse_transform(res)
        assert isinstance(orig_re, np.ndarray)
        assert orig_re.shape == self.data.shape


if __name__ == '__main__':
    T1 = TestFastICA()
    T2 = TestSklearnFastICAAPI()



