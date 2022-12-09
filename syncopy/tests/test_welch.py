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
            ax.set_title("Welch result")
            ax.set_ylabel("Power")
            ax.set_ylabel("Frequency")
        return res

    def test_mtmconvolv_overlap_effect(self):
        """Test variance between windows of different length.

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

        # Check number of windows, we want something similar/comparable.
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
            plot_trial=0  # Which one does not matter, they are all white noise.
            _rewrite_log_output(var_short_windows, to="abs")  # Disable log-scale plotting.
            _rewrite_log_output(var_long_windows, to="abs")  # Disable log-scale plotting.
            _, ax0 = var_short_windows.singlepanelplot(trials=plot_trial)
            ax0.set_title(f"mtmconvolv overlap effect: Windows without overlap\n(toi={cfg_no_overlap.toi}, f_timwin={cfg_no_overlap.t_ftimwin}).")
            ax0.set_ylabel("Variance")
            _, ax1 = var_long_windows.singlepanelplot(trials=plot_trial)
            ax1.set_title(f"mtmconvolv overlap effect: Windows with overlap\n(toi={cfg_with_overlap.toi}, f_timwin={cfg_with_overlap.t_ftimwin}).")
            ax1.set_ylabel("Variance")

        chan=0
        assert np.mean(var_short_windows.show(channel=chan)) > np.mean(var_long_windows.show(channel=chan))

    def test_welch_overlap_effect(self):
        """
        Comparre variance over different Welch estimates based on signal length and overlap.

        (Variance can be computed along trials.)

        Compare a long signal without overlap versus a short signal with overlap, that result in the
        same window count. We expect to see higher variance for the shorter signal.

        Potential nice-to-have for later: investigate sweet spot for the overlap parameter as a function of signal length.
        """
        wn_long = synth_data.white_noise(nTrials=20, nChannels=1, nSamples=10000, samplerate=1000) # 10 seconds of signal
        wn_short = synth_data.white_noise(nTrials=20, nChannels=1, nSamples=1000, samplerate=1000) # 1  second of signal

        foilim = [10, 70]  # Shared between cases.

        cfg_long_no_overlap = TestWelch.get_welch_cfg()  # Results in 100 windows of length 100.
        cfg_long_no_overlap.toi = 0.0         # overlap [0, 1]
        cfg_long_no_overlap.t_ftimwin = 0.1   # window length in sec
        cfg_long_no_overlap.foilim = foilim

        cfg_short_with_overlap = TestWelch.get_welch_cfg()  # Results in 100 windows of length 20, with 50% overlap.
        cfg_short_with_overlap.toi = 0.5
        cfg_short_with_overlap.t_ftimwin = 0.02
        cfg_short_with_overlap.foilim = foilim


        spec_long_no_overlap = spy.freqanalysis(cfg_long_no_overlap, wn_long)
        spec_short_with_overlap = spy.freqanalysis(cfg_short_with_overlap, wn_short)

        var_dim='trials'
        var_no_overlap = spy.var(spec_long_no_overlap, dim=var_dim)
        var_with_overlap = spy.var(spec_short_with_overlap, dim=var_dim)

        if self.do_plot:
            _rewrite_log_output(var_no_overlap, to="abs")  # Disable log-scale plotting.
            _rewrite_log_output(var_with_overlap, to="abs")  # Disable log-scale plotting.
            plot_trial=0  # Does not matter.
            _, ax0 = var_no_overlap.singlepanelplot(trials=plot_trial)
            ax0.set_title(f"Welch overlap effect: Long signal, no overlap.\n(toi={cfg_long_no_overlap.toi}, f_timwin={cfg_long_no_overlap.t_ftimwin})")
            ax0.set_ylabel("Variance")

            _, ax1 = var_with_overlap.singlepanelplot(trials=plot_trial)
            ax1.set_title(f"Welch overlap effect: Short signal, with overlap.\n(toi={cfg_short_with_overlap.toi}, f_timwin={cfg_short_with_overlap.t_ftimwin})")
            ax1.set_ylabel("Variance")


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


    def test_welch_rejects_keeptaper(self):
        cfg = TestWelch.get_welch_cfg()
        cfg.tapsmofrq = 2  # Activate multi-tapering, which is fine.
        cfg.keeptapers = True  # Disable averaging over tapers (taper dimension), which is NOT allowed with Welsh.
        with pytest.raises(SPYValueError, match="keeptapers"):
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
