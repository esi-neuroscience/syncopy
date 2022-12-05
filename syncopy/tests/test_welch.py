# -*- coding: utf-8 -*-
#
# Test Welch's method from user/frontend perspective.


import pytest
import syncopy as spy
import numpy as np
from syncopy.tests.test_specest import TestMTMConvol
from syncopy.shared.errors import SPYValueError
from syncopy.shared.const_def import spectralConversions


class TestWelch():
    """
    Test the frontend (user API) for running Welch's method for estimation of power spectra.
    """

    adata = TestMTMConvol.get_tfdata_mtmconvol()

    def test_welch_simple(self):
        cfg = spy.get_defaults(spy.freqanalysis)
        cfg.method = "welch"
        cfg.t_ftimwin = 0.5
        spec_dt = spy.freqanalysis(cfg, self.adata)
        assert spec_dt.data.ndim == 4

    def test_welch_rejects_multitaper(self):
        cfg = spy.get_defaults(spy.freqanalysis)
        cfg.method = "welch"
        cfg.t_ftimwin = 0.5
        cfg.tapsmofrq = 2  # Activate multi-tapering, which is not allowed.
        with pytest.raises(SPYValueError, match="tapsmofrq"):
            spec_dt = spy.freqanalysis(cfg, self.adata)

    def test_welch_rejects_invalid_tois(self):
        cfg = spy.get_defaults(spy.freqanalysis)
        cfg.method = "welch"
        cfg.t_ftimwin = 0.5

        cfg.toi = 'all'
        with pytest.raises(SPYValueError, match="toi"):
            spec_dt = spy.freqanalysis(cfg, self.adata)

    def test_welch_rejects_invalid_output(self):
        cfg = spy.get_defaults(spy.freqanalysis)
        cfg.method = "welch"
        cfg.t_ftimwin = 0.5
        cfg.toi = 0.0

        for output in spectralConversions.keys():
            if output != "pow":
                cfg.output = output
                with pytest.raises(SPYValueError, match="output"):
                    spec_dt = spy.freqanalysis(cfg, self.adata)



if __name__ == '__main__':
    T1 = TestWelch()
