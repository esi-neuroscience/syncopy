# -*- coding: utf-8 -*-
#
# Test Welch's method from user/frontend perspective.


import pytest
import syncopy as spy
import numpy as np
from syncopy.tests.test_specest import TestMTMConvol


class TestWelch():
    """
    Test the frontend (user API) for running Welch's method for estimation of power spectra.
    """

    adata = TestMTMConvol.get_tfdata_mtmconvol()

    def test_welch(self):
        cfg = spy.get_defaults(spy.freqanalysis)
        cfg.method = "welch"
        spec_dt = spy.freqanalysis(cfg, self.adata)
        assert spec_dt.data.ndim == 4


if __name__ == '__main__':
    T = TestWelch()