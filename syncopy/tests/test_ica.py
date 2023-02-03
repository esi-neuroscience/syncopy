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
from syncopy.shared.tools import get_defaults, best_match

import syncopy as spy
import sklearn.decomposition as decomposition


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


class TestSkleanFastICAAPI():
    rng = np.random.default_rng(test_seed)
    n_samples = 1000
    n_channels = 16
    data = rng.normal(size=(n_samples, n_channels))
    ica = decomposition.FastICA()
    res = ica.fit_transform(data)
    assert isinstance(res, np.ndarray)

if __name__ == '__main__':
    T1 = TestFastICA()
    T2 = TestSkleanFastICAAPI()



