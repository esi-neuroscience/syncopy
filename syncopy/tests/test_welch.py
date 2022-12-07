# -*- coding: utf-8 -*-
#
# Test Welch's method from user/frontend perspective.


import pytest
import syncopy as spy
import numpy as np
from syncopy.tests.test_specest import TestMTMConvol
from syncopy.shared.errors import SPYValueError
from syncopy.shared.const_def import spectralConversions
import syncopy.tests.synth_data as synth_data


class TestWelch():
    """
    Test the frontend (user API) for running Welch's method for estimation of power spectra.
    """

    # White noise
    adata = synth_data.white_noise(nTrials=2, nChannels=3, nSamples=20000, samplerate=1000)
    do_plot = True

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

    def test_mtmconvolv_res(self):
        """Internal function for interactive debugging purposes only, to better see what we are working with."""
        cfg = TestWelch.get_welch_cfg()
        cfg.method = "mtmconvol"
        res = spy.freqanalysis(cfg, self.adata)

        # Test basic output properties.
        assert len(res.dimord) == 4
        assert len(res.data.shape) == 4
        assert res.dimord.index('time') == 0
        assert res.dimord.index('taper') == 1
        assert res.dimord.index('freq') == 2
        assert res.dimord.index('channel') == 3

        # Test ouput shape:
        # 20.000 samples per trial at 1000 samplerate => 20 sec of data. With window length of
        # 0.5 sec and no overlap, we should get 40 periodograms per trial, so 80 in total.
        assert res.data.shape[res.dimord.index('time')] == 80
        assert res.data.shape[res.dimord.index('taper')] == 1
        assert res.data.shape[res.dimord.index('channel')] == 3

        # Test output trialdefinition
        assert res.trialdefinition.shape[0] == 2  # nTrials

        if self.do_plot:
            fig, ax = res.singlepanelplot(trials=0, channel=0)
            ax.set_title("mtmconvolv result.")
        return res

    def test_welch_basic(self):
        cfg = TestWelch.get_welch_cfg()
        res = spy.freqanalysis(cfg, self.adata)

        # Test basic output properties. Same as for mtmconvolv.
        assert len(res.dimord) == 4
        assert len(res.data.shape) == 4
        assert res.dimord.index('time') == 0
        assert res.dimord.index('taper') == 1
        assert res.dimord.index('freq') == 2
        assert res.dimord.index('channel') == 3

        # Test ouput shape:
        assert res.data.shape[res.dimord.index('time')] == 2  # 1 averaged periodogram per trial left, so 2 periodograms for the 2 trials.
        assert res.data.shape[res.dimord.index('taper')] == 1
        assert res.data.shape[res.dimord.index('channel')] == 3

        # Test output trialdefinition
        assert res.trialdefinition.shape[0] == 2  # nTrials

        if self.do_plot:
            fig, ax = res.singlepanelplot(trials=0)
            ax.set_title("Welch result.")
        return res

    def test_welch_overlap_effect(self):
        cfg_no_overlap = TestWelch.get_welch_cfg()
        cfg_no_overlap.method = "mtmconvol"
        cfg_no_overlap.toi = 0.0
        # TODO: select a suitable foi here?

        cfg_half_overlap = TestWelch.get_welch_cfg()
        cfg_half_overlap.method = "mtmconvol"
        cfg_half_overlap.toi = 0.5
        # TODO: select a suitable foi here?

        wn_short = synth_data.white_noise(nTrials=2, nChannels=3, nSamples=1000, samplerate=1000)
        wn_long = synth_data.white_noise(nTrials=2, nChannels=3, nSamples=40000, samplerate=1000)

        spec_short_no_overlap = spy.freqanalysis(cfg_no_overlap, wn_short)
        spec_short_half_overlap = spy.freqanalysis(cfg_half_overlap, wn_short)
        spec_long_no_overlap = spy.freqanalysis(cfg_no_overlap, wn_long)
        spec_long_half_overlap = spy.freqanalysis(cfg_half_overlap, wn_long)

        var_dim='time'
        var_short_no_overlap = spy.var(spec_short_no_overlap, dim=var_dim)
        var_short_half_overlap = spy.var(spec_short_half_overlap, dim=var_dim)
        var_long_no_overlap = spy.var(spec_long_no_overlap, dim=var_dim)
        var_long_half_overlap = spy.var(spec_long_half_overlap, dim=var_dim)

        if self.do_plot:
            plot_trial=0
            fig0, ax0 = var_short_no_overlap.singlepanelplot(trials=plot_trial)
            ax0.set_title("Short data, no overlap.")
            fig1, ax1 = var_short_half_overlap.singlepanelplot(trials=plot_trial)
            ax0.set_title("Short data, half overlap.")
            fig2, ax2 = var_long_no_overlap.singlepanelplot(trials=plot_trial)
            ax0.set_title("Long data, no overlap.")
            fig3, ax3 = var_long_half_overlap.singlepanelplot(trials=plot_trial)
            ax0.set_title("Long data, half overlap.")


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
