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

from syncopy.shared.errors import SPYValueError, SPYTypeError
from syncopy.shared.tools import get_defaults, best_match

import syncopy as spy


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
            res = spy.runica(wrong_data, cfg)


if __name__ == '__main__':
    T1 = TestFastICA()

