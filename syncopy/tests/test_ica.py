# -*- coding: utf-8 -*-
#
# Test independent component analysis (ICA).
#

# 3rd party imports
import pytest
import numpy as np
import matplotlib.pyplot as ppl

# Local imports
import dask.distributed as dd
from syncopy.tests import synth_data as sd

from syncopy.shared.errors import SPYValueError
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

if __name__ == '__main__':
    T1 = TestFastICA()

