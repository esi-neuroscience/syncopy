# -*- coding: utf-8 -*-
#
# Test Welch's method from user/frontend perspective.


import pytest
import syncopy as spy
import numpy as np
import inspect
import dask.distributed as dd
import matplotlib as mpl
import matplotlib.pyplot as plt
from syncopy.shared.errors import SPYValueError
from syncopy.shared.const_def import spectralConversions
import syncopy.tests.synth_data as synth_data
from syncopy.tests.helpers import teardown, test_seed


class TestWelch():
    """
    Test the frontend (user API) for running Welch's method for estimation of power spectra.
    """

    # White noise
    adata = synth_data.white_noise(nTrials=2, nChannels=3, nSamples=20000, samplerate=1000,
                                   seed=test_seed)
    do_plot = True

    def setup_class(cls):
        plt.close('all')    # Close plots that are still open.

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
        """Internal function mainly for interactive debugging purposes,
           to better see what we are working with.

           Welch is implemented as a post-processing of mtmfftconvolv, so it
           is helpful to be sure about its input.
        """
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
            _, ax = res.singlepanelplot(trials=0, logscale=False)
            ax.set_title("Welch result")
            # ax.set_ylabel("Power")
            ax.set_xlabel("Frequency")

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
            _, ax0 = var_short_windows.singlepanelplot(trials=plot_trial, logscale=False)
            ax0.set_title(f"mtmconvolv overlap effect: Windows without overlap\n(toi={cfg_no_overlap.toi}, f_timwin={cfg_no_overlap.t_ftimwin}).")
            ax0.set_ylabel("Variance")
            _, ax1 = var_long_windows.singlepanelplot(trials=plot_trial, logscale=False)
            ax1.set_title(f"mtmconvolv overlap effect: Windows with overlap\n(toi={cfg_with_overlap.toi}, f_timwin={cfg_with_overlap.t_ftimwin}).")
            ax1.set_ylabel("Variance")

        chan=0
        assert np.mean(var_short_windows.show(channel=chan)) > np.mean(var_long_windows.show(channel=chan))

    def test_welch_size_effect(self):
        """
        Compare variance over different Welch estimates based on signal length and overlap.

        (Variance can be computed along trials.)

        Compare a long signal without overlap versus a short signal with overlap, that result in the
        same window count. We expect to see higher variance for the shorter signal.

        Potential nice-to-have for later: investigate sweet spot for the overlap parameter as a function of signal length.
        """
        wn_long = synth_data.white_noise(nTrials=20, nChannels=1, nSamples=10000, samplerate=1000, seed=42)  # 10 seconds of signal
        wn_short = synth_data.white_noise(nTrials=20, nChannels=1, nSamples=1000, samplerate=1000, seed=42)  # 1  second of signal

        foilim = [5, 200]  # Shared between cases.

        cfg_long_no_overlap = TestWelch.get_welch_cfg()  # Results in 100 windows of length 100.
        cfg_long_no_overlap.toi = 0.0         # overlap [0, 1[
        cfg_long_no_overlap.t_ftimwin = 0.1   # window length in sec
        cfg_long_no_overlap.foilim = foilim

        cfg_short_with_overlap = TestWelch.get_welch_cfg()  # Results in 100 windows of length 20, with 50% overlap.
        cfg_short_with_overlap.toi = 0.5
        cfg_short_with_overlap.t_ftimwin = 0.02
        cfg_short_with_overlap.foilim = foilim

        # Check the number of windows that Welch will average over.
        # To do this, we run mtmconvol and check the output size.
        # This is to verify that the number of windows is equal, and as expected.
        cfg_mtm_long = cfg_long_no_overlap.copy()
        cfg_mtm_long.method = "mtmconvol"
        cfg_mtm_short = cfg_short_with_overlap.copy()
        cfg_mtm_short.method = "mtmconvol"

        spec_long_no_overlap = spy.freqanalysis(cfg_long_no_overlap, wn_long)
        spec_short_with_overlap = spy.freqanalysis(cfg_short_with_overlap, wn_short)

        # We got one Welch estimate per trial so far. Now compute the variance over trials:
        var_dim='trials'
        var_longsig_no_overlap = spy.var(spec_long_no_overlap, dim=var_dim)
        var_shortsig_with_overlap = spy.var(spec_short_with_overlap, dim=var_dim)

        assert var_longsig_no_overlap.dimord.index('time') == 0
        assert var_longsig_no_overlap.data.shape[0] == 1
        assert var_shortsig_with_overlap.data.shape[0] == 1

        if self.do_plot:
            mn_long, var_long = np.mean(var_longsig_no_overlap.show(trials=0)), np.var(var_longsig_no_overlap.show(trials=0))

            mn_short, var_short = np.mean(var_shortsig_with_overlap.show(trials=0)), np.var(var_shortsig_with_overlap.show(trials=0))
            _, ax = plt.subplots()
            title = f"Long signal: (toi={cfg_long_no_overlap.toi}, f_timwin={cfg_long_no_overlap.t_ftimwin})\n"
            title += f"Short signal: (toi={cfg_short_with_overlap.toi}, f_timwin={cfg_short_with_overlap.t_ftimwin})"
            ax.bar([1, 2], [mn_long, mn_short], yerr=[var_long, var_short], width=0.5, capsize=2)
            ax.set_title(title)
            ax.set_ylabel("Variance")
            ax.set_xlabel('')
            ax.set_xticklabels(['long', 'short'])
        chan=0
        assert np.mean(var_longsig_no_overlap.show(channel=chan)) < np.mean(var_shortsig_with_overlap.show(channel=chan))

    def test_welch_overlap_effect(self):

        sig_lengths = np.linspace(1000, 4000, num=4, dtype=int)
        overlaps = np.linspace(0.0, 0.99, num=10)
        variances = np.zeros((sig_lengths.size, overlaps.size), dtype=float)  # Filled in loop below.

        foilim = [5, 200]  # Shared between cases.
        f_timwin = 0.2

        for sigl_idx, sig_len in enumerate(sig_lengths):
            for overl_idx, overlap in enumerate(overlaps):
                wn = synth_data.white_noise(nTrials=20, nChannels=1, nSamples=sig_len, samplerate=1000, seed=test_seed)

                cfg = TestWelch.get_welch_cfg()  # Results in 100 windows of length 100.
                cfg.toi = overlap
                cfg.t_ftimwin = f_timwin
                cfg.foilim = foilim

                spec = spy.freqanalysis(cfg, wn)

                # We got one Welch estimate per trial so far. Now compute the variance over trials:
                spec_var = spy.var(spec, dim='trials')
                mvar = np.mean(spec_var.show(channel=0))
                variances[sigl_idx, overl_idx] = mvar

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for row_idx in range(variances.shape[0]):
            ax.scatter(np.tile(sig_lengths[row_idx], overlaps.size), overlaps, variances[row_idx, :], label=f"Signal len {sig_lengths[row_idx]}")
        ax.set_xlabel('Signal length (number of samples)')
        ax.set_ylabel('Window overlap')
        ax.set_zlabel('Mean variance of the Welch estimate')
        ax.set_title('Variance of Welsh estimate as a function of signal length and overlap.\nColors represent different signal lengths.')
        # plt.show()  # We could run 'plt.legend()' before this line, but it's a bit large.

        # Now for the tests.
        # For a fixed overlap, the variance should decrease with signal length:
        for overlap_idx in range(variances.shape[1]):
            for siglen_idx in range(1, variances.shape[0]):
                assert variances[siglen_idx, overlap_idx] < variances[siglen_idx - 1, overlap_idx]

        # For short signals, there is a benefit in using medium overlap:
        assert np.argmin(variances[0, :]) == overlaps.size // 2, f"Expected {overlaps.size // 2}, got {np.argmin(variances[0, :])}."
        # Note: For humans, looking at the plot above will illustrate this a lot better.


    def test_welch_replay(self):
        """Test replay with settings from output cfg."""
        # only preprocessing makes sense to chain atm
        first_cfg = TestWelch.get_welch_cfg()
        first_res = spy.freqanalysis(self.adata, cfg=first_cfg)

        # Now replay with cfg from preceding frontend call:
        replay_res = spy.freqanalysis(self.adata, cfg=first_res.cfg)

        # same results
        assert np.allclose(first_res.data[:], replay_res.data[:])
        assert first_res.cfg == replay_res.cfg


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

    def test_welch_with_multitaper(self):
        cfg = TestWelch.get_welch_cfg()
        cfg.tapsmofrq = 2  # Activate multi-tapering, which is fine.
        cfg.keeptapers = False  # Disable averaging over tapers (taper dimension), which is NOT allowed with Welsh.

        res = spy.freqanalysis(cfg, self.adata)
        assert res.data.shape[res.dimord.index('taper')] == 1  # Averaging over tapers expected.
        assert res.data.shape[res.dimord.index('channel')] == 3  # Nothing special expected here.

    def test_parallel(self, testcluster):
        plt.ioff()
        self.do_plot = False
        client = dd.Client(testcluster)
        all_tests = [attr for attr in self.__dir__()
                     if (inspect.ismethod(getattr(self, attr)) and 'parallel' not in attr and attr.startswith('test'))]

        for test in all_tests:
            test_method = getattr(self, test)
            test_method()
        client.close()
        self.do_plot = True
        plt.ion()


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

    def teardown_class(cls):
        teardown()

if __name__ == '__main__':
    if TestWelch.do_plot:
        import matplotlib.pyplot as plt
        plt.ion()
    T1 = TestWelch()
