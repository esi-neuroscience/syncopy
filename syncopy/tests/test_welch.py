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
from syncopy.plotting._helpers import _rewrite_log_output


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
        assert res.trialdefinition.shape[0] == 2  # nTrials should be left intact, as we did not set trial averaging.

        if self.do_plot:
            _rewrite_log_output(res, to="abs")  # Disable log-scale plotting.
            _, ax = res.singlepanelplot(trials=0)
            ax.set_title("Welch result.")
            ax.set_ylabel("Power")
            ax.set_ylabel("Frequency")
        return res

    def test_mtmconvolv_overlap_effect(self):
        """Test variance between windows, depending on windows len and overlap.

        We use the same data for both cases, but run (a) with no overlap and short
        windows, and (b) with overlap but longer windows.

        We select toi and ftimwin in a way that leads to a comparable number of
        windows between the two cases.
        """
        foilim = [10, 70]

        cfg_no_overlap = TestWelch.get_welch_cfg()
        cfg_no_overlap.method = "mtmconvol"
        cfg_no_overlap.toi = 0.0        # overlap [0, 1]
        cfg_no_overlap.t_ftimwin = 0.25   # window length in sec
        cfg_no_overlap.foilim = foilim
        cfg_no_overlap.output = "abs"

        cfg_with_overlap = TestWelch.get_welch_cfg()
        cfg_with_overlap.method = "mtmconvol"
        cfg_with_overlap.toi = 0.8
        cfg_with_overlap.t_ftimwin = 1.2
        cfg_with_overlap.foilim = foilim
        cfg_with_overlap.output = "abs"

        nSamples = 30000
        samplerate = 1000
        wn = synth_data.white_noise(nTrials=1, nChannels=3, nSamples=nSamples, samplerate=samplerate)

        spec_short_windows = spy.freqanalysis(cfg_no_overlap, wn)
        spec_long_windows = spy.freqanalysis(cfg_with_overlap, wn)

        # Check number of windows, we want something similar.
        assert spec_short_windows.dimord.index('time') == spec_long_windows.dimord.index('time')
        ti = spec_short_windows.dimord.index('time')
        assert spec_short_windows.data.shape[ti] == 120, f"Window count without overlap is: {spec_short_windows.data.shape[ti]} (shape: {spec_short_windows.data.shape})"
        assert spec_long_windows.data.shape[ti] == 125, f"Window count with overlap is: {spec_long_windows.data.shape[ti]} (shape: {spec_long_windows.data.shape})"

        # Check windows lengths, these should be different.
        assert spec_short_windows.dimord.index('freq') == spec_long_windows.dimord.index('freq')
        fi = spec_short_windows.dimord.index('freq')
        assert spec_short_windows.data.shape[fi] == 15, f"Window length without overlap is: {spec_short_windows.data.shape[fi]} (shape: {spec_short_windows.data.shape})"
        assert spec_long_windows.data.shape[fi] == 73, f"Window length with overlap is: {spec_long_windows.data.shape[fi]} (shape: {spec_long_windows.data.shape})"

        var_dim='time'
        var_short_windows = spy.var(spec_short_windows, dim=var_dim)
        var_long_windows = spy.var(spec_long_windows, dim=var_dim)

        if self.do_plot:
            plot_trial=0  # Does not matter.
            _, ax0 = var_short_windows.singlepanelplot(trials=plot_trial)
            ax0.set_title("Var for no overlap.")
            _, ax1 = var_long_windows.singlepanelplot(trials=plot_trial)
            ax1.set_title("Var with overlap.")

        chan=0
        assert np.mean(var_short_windows.show(channel=chan)) > np.mean(var_long_windows.show(channel=chan))

    def test_welch_overlap_effect(self):
        """
        Plot variance over different Welch estimates. Variance can be computed along trials.

        Do once with short dataset and once for long dataset.

        1) Vergleichbarkeit: mit langem Signal ohne Overlap, sowie kurzem Signal mit Overlap
        auf gleiche Anzahl Fenster kommen. Dann Varianz des Welch-Estimates berechnen. Sollte
        höher sein für das lange Signal.

        2) Sweet-Spot für overlap in Abhängigkeit von der Signallänge? Evtl später.
        """
        foilim = [10, 70]

        cfg_no_overlap = TestWelch.get_welch_cfg()
        cfg_no_overlap.method = "mtmconvol"
        cfg_no_overlap.toi = 0.0        # overlap [0, 1]
        cfg_no_overlap.t_ftimwin = 0.25   # window length in sec
        cfg_no_overlap.foilim = foilim
        cfg_no_overlap.output = "abs"

        cfg_with_overlap = TestWelch.get_welch_cfg()
        cfg_with_overlap.method = "mtmconvol"
        cfg_with_overlap.toi = 0.8
        cfg_with_overlap.t_ftimwin = 2.0
        cfg_with_overlap.foilim = foilim
        cfg_with_overlap.output = "abs"

        wn_short = synth_data.white_noise(nTrials=2, nChannels=3, nSamples=30000, samplerate=1000)
        #wn_long = synth_data.white_noise(nTrials=2, nChannels=3, nSamples=30000, samplerate=1000)

        spec_no_overlap = spy.freqanalysis(cfg_no_overlap, wn_short)
        spec_with_overlap = spy.freqanalysis(cfg_with_overlap, wn_short)
        #spec_long_no_overlap = spy.freqanalysis(cfg_no_overlap, wn_long)
        #spec_long_half_overlap = spy.freqanalysis(cfg_half_overlap, wn_long)

        assert spec_no_overlap.dimord.index('freq') == spec_with_overlap.dimord.index('freq')
        fi = spec_no_overlap.dimord.index('freq')
        assert spec_no_overlap.data.shape[fi] == 251, f"Window length without overlap is: {spec_no_overlap.data.shape[fi]} (shape: {spec_no_overlap.data.shape})"
        assert spec_with_overlap.data.shape[fi] == 251, f"Window length with overlap is: {spec_with_overlap.data.shape[fi]} (shape: {spec_with_overlap.data.shape})"

        var_dim='time'
        var_no_overlap = spy.var(spec_no_overlap, dim=var_dim)
        var_with_overlap = spy.var(spec_with_overlap, dim=var_dim)
        #var_long_no_overlap = spy.var(spec_long_no_overlap, dim=var_dim)
        #var_long_half_overlap = spy.var(spec_long_half_overlap, dim=var_dim)


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
        assert res.data.shape[res.dimord.index('time')] == 1
        assert res.data.shape[res.dimord.index('taper')] == 1  # Nothing special expected here.
        assert res.data.shape[res.dimord.index('channel')] == 3  # Nothing special expected here.

        # Another relevant assertion specific to this test case: trialdefinition
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
    if TestWelch.do_plot:
        import matplotlib.pyplot as plt
        plt.ion()
    T1 = TestWelch()
