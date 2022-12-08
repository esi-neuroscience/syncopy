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

        # Test ouput shape.
        # The 'time' dimension is the important difference between mtmconvolv and Welch:
        # 20.000 samples per trial at 1000 samplerate => 20 sec of data. With window length of
        # 0.5 sec and no overlap, we should get 40 periodograms per trial, so 80 in total.
        assert res.data.shape[res.dimord.index('time')] == 80
        assert res.data.shape[res.dimord.index('taper')] == 1
        assert res.data.shape[res.dimord.index('channel')] == 3

        # Test output trialdefinition
        assert res.trialdefinition.shape[0] == 2  # nTrials

        if self.do_plot:
            _, ax = res.singlepanelplot(trials=0, channel=0)
            ax.set_title("mtmconvolv result.")
        return res

    def test_welch_basic(self):
        """
        Tests with standard settings, nothing special, no trial averaging.
        """
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
            _, ax = res.singlepanelplot(trials=0)
            ax.set_title("Welch result.")
        return res

    def test_mtmconvolv_overlap_effect(self):
        """Test variance between windows, depending on windows len and overlap."""
        foilim = [10, 70]

        cfg_no_overlap = TestWelch.get_welch_cfg()
        cfg_no_overlap.method = "mtmconvol"
        cfg_no_overlap.toi = 0.0        # overlap [0, 1]
        cfg_no_overlap.t_ftimwin = 0.25   # window length in sec
        cfg_no_overlap.foilim = foilim
        cfg_no_overlap.output = "abs"

        cfg_half_overlap = TestWelch.get_welch_cfg()
        cfg_half_overlap.method = "mtmconvol"
        cfg_half_overlap.toi = 0.8
        cfg_half_overlap.t_ftimwin = 2.0
        cfg_half_overlap.foilim = foilim
        cfg_half_overlap.output = "abs"

        wn_short = synth_data.white_noise(nTrials=2, nChannels=3, nSamples=30000, samplerate=1000)
        #wn_long = synth_data.white_noise(nTrials=2, nChannels=3, nSamples=30000, samplerate=1000)

        spec_short_no_overlap = spy.freqanalysis(cfg_no_overlap, wn_short)
        spec_short_half_overlap = spy.freqanalysis(cfg_half_overlap, wn_short)
        #spec_long_no_overlap = spy.freqanalysis(cfg_no_overlap, wn_long)
        #spec_long_half_overlap = spy.freqanalysis(cfg_half_overlap, wn_long)

        var_dim='time'
        var_short_no_overlap = spy.var(spec_short_no_overlap, dim=var_dim)
        var_short_half_overlap = spy.var(spec_short_half_overlap, dim=var_dim)
        #var_long_no_overlap = spy.var(spec_long_no_overlap, dim=var_dim)
        #var_long_half_overlap = spy.var(spec_long_half_overlap, dim=var_dim)

        if self.do_plot:
            plot_trial=0  # Does not matter.
            _, ax0 = var_short_no_overlap.singlepanelplot(trials=plot_trial)
            ax0.set_title("Var for no overlap.")
            _, ax1 = var_short_half_overlap.singlepanelplot(trials=plot_trial)
            ax1.set_title("Var with overlap.")
            #_, ax2 = var_long_no_overlap.singlepanelplot(trials=plot_trial)
            #ax2.set_title("Var for long data, no overlap.")
            #_, ax3 = var_long_half_overlap.singlepanelplot(trials=plot_trial)
            #ax3.set_title("Var for long data, half overlap.")

    def test_welch_overlap_effect(self):
        """
        Plot variance over different Welch estimations. Variance can be computed along trials.

        Do once with short dataset and once for long dataset.

        1) Vergleichbarkeit: mit langem Signal ohne Overlap, sowie kurzem Signal mit Overlap
        auf gleiche Anzahl Fenster kommen. Dann Varianz des Welch-Estimates berechnen. Sollte
        höher sein für das lange Signal.

        2) Sweet-Spot für overlap in Abhängigkeit von der Signallänge? Evtl später.
        """
        pass

    def test_welch_replay(self):
        """
        TODO: test that replay works with Welch.
        """
        pass

    def test_welch_trial_averaging(self):
        cfg = TestWelch.get_welch_cfg()
        cfg.keeptrials = False  # Activate trial averaging. This happens during mtmfftconvolv, Welch just gets less input.

        res = spy.freqanalysis(cfg, self.adata)
        # Test basic output properties.
        assert len(res.dimord) == 4
        assert len(res.data.shape) == 4
        assert res.dimord.index('time') == 0
        assert res.dimord.index('taper') == 1
        assert res.dimord.index('freq') == 2
        assert res.dimord.index('channel') == 3

        # Test ouput shape:
        # The time dimensions is the important thing, trial averaging of the 2 trials leads to only 1 left:
        # 0.5 sec and no overlap, we should get 40 periodograms per trial, so 80 in total.
        assert res.data.shape[res.dimord.index('time')] == 1
        assert res.data.shape[res.dimord.index('taper')] == 1
        assert res.data.shape[res.dimord.index('channel')] == 3

        # The most relevant test: trialdefinition
        assert res.trialdefinition.shape[0] == 1  # trial averaging has been performed, so only 1 trial left.

        if self.do_plot:
            _, ax = res.singlepanelplot(trials=0, channel=0)
            ax.set_title("Welsh result with trial averaging.")


    def test_welch_rejects_keeptaper_with_multitaper(self):
        cfg = TestWelch.get_welch_cfg()
        cfg.tapsmofrq = 2  # Activate multi-tapering, which is fine.
        cfg.keeptaper = True  # Disable averaging over tapers (taper dimension), which is NOT allowed with Welsh.
        with pytest.raises(SPYValueError, match="keeptaper"):
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


if __name__ == '__main__':
    T1 = TestWelch()
