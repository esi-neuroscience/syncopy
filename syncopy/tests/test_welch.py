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

    @staticmethod
    def get_welch_cfg():
        """
        Get a reasonable Welch cfg for testing purposes.
        """
        cfg = spy.get_defaults(spy.freqanalysis)
        cfg.method = "welch"
        cfg.t_ftimwin = 0.5  # Window length in seconds.
        cfg.toi = 0.0        # Overlap between periodograms (0.5 = 50 percent overlap).
        return cfg

    def _get_mtmconvolv_res(self):
        """Internal function for debuggin only. Ignore, will be deleted."""
        cfg = TestWelch.get_welch_cfg()
        cfg.method = "mtmconvol"
        return spy.freqanalysis(cfg, self.adata)


    def test_welch_simple(self):
        cfg = TestWelch.get_welch_cfg()
        spec_dt = spy.freqanalysis(cfg, self.adata)
        assert spec_dt.data.ndim == 4

    def test_welch_rejects_multitaper(self):
        cfg = TestWelch.get_welch_cfg()
        cfg.tapsmofrq = 2  # Activate multi-tapering, which is not allowed.
        with pytest.raises(SPYValueError, match="tapsmofrq"):
            _ = spy.freqanalysis(cfg, self.adata)

    def test_welch_rejects_invalid_tois(self):
        cfg = TestWelch.get_welch_cfg()
        for toi in ['all', np.linspace(0.0, 1.0, 5)]:
            cfg.toi = toi
            with pytest.raises(SPYValueError, match="toi"):
                _ = spy.freqanalysis(cfg, self.adata)

    def test_welch_rejects_invalid_output(self):
        cfg = TestWelch.get_welch_cfg()

        for output in spectralConversions.keys():
            if output != "pow":
                cfg.output = output
                with pytest.raises(SPYValueError, match="output"):
                    _ = spy.freqanalysis(cfg, self.adata)

    def test_welch_rejects_trial_averaging(self):
        cfg = TestWelch.get_welch_cfg()
        cfg.keeptrials = False
        with pytest.raises(SPYValueError, match="keeptrials"):
                    _ = spy.freqanalysis(cfg, self.adata)


if __name__ == '__main__':
    T1 = TestWelch()
