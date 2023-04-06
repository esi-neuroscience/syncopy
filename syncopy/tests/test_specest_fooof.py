# -*- coding: utf-8 -*-
#
# Test FOOOF integration from user/frontend perspective.


import pytest
import numpy as np
import inspect
import matplotlib.pyplot as plt
import dask.distributed as dd

# Local imports
from syncopy import freqanalysis
from syncopy.shared.tools import get_defaults
from syncopy.shared.errors import SPYValueError
from syncopy.tests.test_metadata import _get_fooof_signal
import syncopy as spy


def _plot_powerspec_linear(freqs, powers, title="Power spectrum"):
    """Simple, internal plotting function to plot x versus y. Uses linear scale.

    Parameters
    ----------
    powers: can be a vector or a dict with keys being labels and values being vectors
    save: str interpreted as file name if you want to save the figure, None if you do not want to save to disk.

    Called for plotting side effect.
    """
    plt.ion()
    plt.figure()
    if isinstance(powers, dict):
        for label, power in powers.items():
            plt.plot(freqs, power, label=label)
    else:
        plt.plot(freqs, powers)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (a.u.)')
    plt.legend()
    plt.title(title)


def _fft(analog_data, select={"channel": 0}, foilim=[1.0, 100]):
    """Run standard mtmfft with trial averaging on AnalogData instance.
    """
    if not isinstance(analog_data, spy.datatype.continuous_data.AnalogData):
        raise ValueError("Parameter 'analog_data' must be a syncopy.datatype.continuous_data.AnalogData instance.")
    cfg = get_defaults(freqanalysis)
    cfg.method = "mtmfft"
    cfg.taper = "hann"
    cfg.select = select
    cfg.keeptrials = False  # Averages signal over all (selected) trials.
    cfg.output = "pow"
    cfg.foilim = foilim
    return freqanalysis(cfg, analog_data)


def _show_spec_log(analog_data, title=None):
    """Plot the power spectrum for an AnalogData object. Uses singlepanelplot, so data are shown on a log scale.

       Performs mtmfft with `_fft()` to do that. Use `matplotlib.pyplot.ion()` if you dont see the plot.
    """
    if not isinstance(analog_data, spy.datatype.continuous_data.AnalogData):
        raise ValueError("Parameter 'analog_data' must be a syncopy.datatype.continuous_data.AnalogData instance.")
    spp(_fft(analog_data), title=title)


def spp(dt, title=None):
    """Single panet plot with a title."""
    if not isinstance(dt, spy.datatype.base_data.BaseData):
        raise ValueError("Parameter 'dt' must be a syncopy.datatype instance.")
    fig, ax = dt.singlepanelplot()
    if title is not None:
        ax.set_title(title)
    return fig, ax


class TestFooofSpy():
    """
    Test the frontend (user API) for running FOOOF. FOOOF is a post-processing of an FFT, and
    to request the post-processing, the user sets the method to "mtmfft", and the output to
    one of the available FOOOF output types.
    """

    seed = 42
    tfData = _get_fooof_signal(seed=seed)

    @staticmethod
    def get_fooof_cfg():
        cfg = get_defaults(freqanalysis)
        cfg.method = "mtmfft"
        cfg.taper = "hann"
        cfg.select = {"channel": 0}
        cfg.keeptrials = False
        cfg.output = "fooof"
        cfg.foilim = [1., 100.]
        return cfg

    def test_output_fooof_fails_with_freq_zero(self):
        """ The fooof package ignores input values of zero frequency, and shortens the output array
            in that case with a warning. This is not acceptable for us, as the expected output dimension
            will not off by one. Also it is questionable whether users would want that. We therefore use
            consider it an error to pass an input frequency axis that contains the zero, and throw an
            error in the frontend to stop before any expensive computations happen. This test checks for
            that error.
        """
        cfg = TestFooofSpy.get_fooof_cfg()
        cfg['foilim'] = [0., 100.]    # Include the zero in tfData.
        with pytest.raises(SPYValueError) as err:
            _ = freqanalysis(cfg, _get_fooof_signal(seed=self.seed))  # tfData contains zero.
        assert "a frequency range that does not include zero" in str(err.value)

    def test_output_fooof_works_with_freq_zero_in_data_after_setting_frequency(self):
        """
        This tests the intended operation with output type 'fooof': with an input that does not
        include zero, ensured by using the 'frequency' argument/setting when calling freqanalysis.

        This returns the full, fooofed spectrum.
        """
        cfg = TestFooofSpy.get_fooof_cfg()
        cfg.pop('fooof_opt', None)
        fooof_opt = {'peak_width_limits': (1.0, 12.0)}  # Increase lower limit to avoid fooof warning.
        spec_dt = freqanalysis(cfg, _get_fooof_signal(seed=self.seed), fooof_opt=fooof_opt)

        # check frequency axis
        assert spec_dt.freq.size == 100
        assert spec_dt.freq[0] == 1
        assert spec_dt.freq[99] == 100.

        # check the log
        assert "fooof_method = fooof" in spec_dt._log
        assert "fooof_aperiodic" not in spec_dt._log
        assert "fooof_peaks" not in spec_dt._log
        assert "fooof_opt" in spec_dt._log

        # check the data
        assert spec_dt.data.ndim == 4
        assert spec_dt.data.shape == (1, 1, 100, 1)
        assert not np.isnan(spec_dt.data).any()

        # check that the cfg is correct (required for replay)
        assert spec_dt.cfg['freqanalysis']['output'] == 'fooof'

        # Plot it.
        # _plot_powerspec_linear(freqs=spec_dt.freq, powers=spec_dt.data[0, 0, :, 0], title="fooof full model, for ar1 data (linear scale)")
        # spp(spec_dt, "FOOOF full model")
        # plt.savefig("spp.png")

    def test_output_fooof_aperiodic(self):
        """Test fooof with output type 'fooof_aperiodic'. A spectrum containing only the aperiodic part is returned."""

        cfg = TestFooofSpy.get_fooof_cfg()
        cfg.output = "fooof_aperiodic"
        cfg.pop('fooof_opt', None)
        fooof_opt = {'peak_width_limits': (1.0, 12.0)}  # Increase lower limit to avoid fooof warning.
        spec_dt = freqanalysis(cfg, _get_fooof_signal(seed=self.seed), fooof_opt=fooof_opt)

        # log
        assert "fooof" in spec_dt._log  # from the method
        assert "fooof_method = fooof_aperiodic" in spec_dt._log
        assert "fooof_peaks" not in spec_dt._log

        # check the data
        assert spec_dt.data.ndim == 4
        assert spec_dt.data.shape == (1, 1, 100, 1)
        assert not np.isnan(spec_dt.data).any()

    def test_output_fooof_peaks(self):
        """Test fooof with output type 'fooof_peaks'. A spectrum containing only the peaks (actually, the Gaussians fit to the peaks) is returned."""
        cfg = TestFooofSpy.get_fooof_cfg()
        cfg.output = "fooof_peaks"
        cfg.pop('fooof_opt', None)
        fooof_opt = {'peak_width_limits': (1.0, 12.0)}  # Increase lower limit to avoid fooof warning.
        spec_dt = freqanalysis(cfg, _get_fooof_signal(seed=self.seed), fooof_opt=fooof_opt)
        assert spec_dt.data.ndim == 4
        assert "fooof" in spec_dt._log
        assert "fooof_method = fooof_peaks" in spec_dt._log
        assert "fooof_aperiodic" not in spec_dt._log

    def test_different_fooof_outputs_are_consistent(self):
        """Test fooof with all output types plotted into a single plot and ensure consistent output."""
        cfg = TestFooofSpy.get_fooof_cfg()
        cfg['output'] = "pow"
        cfg['foilim'] = [10, 70]
        cfg.pop('fooof_opt', None)
        fooof_opt = {'peak_width_limits': (6.0, 12.0),
                     'min_peak_height': 0.2}  # Increase lower limit to avoid fooof warning.

        out_fft = freqanalysis(cfg, _get_fooof_signal(seed=self.seed))
        cfg['output'] = "fooof"
        out_fooof = freqanalysis(cfg, _get_fooof_signal(seed=self.seed), fooof_opt=fooof_opt)
        cfg['output'] = "fooof_aperiodic"
        out_fooof_aperiodic = freqanalysis(cfg, _get_fooof_signal(seed=self.seed), fooof_opt=fooof_opt)
        cfg['output'] = "fooof_peaks"
        out_fooof_peaks = freqanalysis(cfg, _get_fooof_signal(seed=self.seed), fooof_opt=fooof_opt)

        assert (out_fooof.freq == out_fooof_aperiodic.freq).all()
        assert (out_fooof.freq == out_fooof_peaks.freq).all()

        assert out_fooof.data.shape == out_fooof_aperiodic.data.shape
        assert out_fooof.data.shape == out_fooof_peaks.data.shape

        # biggest peak is at 30Hz
        f1_ind = out_fooof_peaks.show(channel=0).argmax()
        assert 27 < out_fooof_peaks.freq[f1_ind] < 33

        plot_data = {"Raw input data": np.ravel(out_fft.data), "Fooofed spectrum": np.ravel(out_fooof.data), "Fooof aperiodic fit": np.ravel(out_fooof_aperiodic.data), "Fooof peaks fit": np.ravel(out_fooof_peaks.data)}
        #_plot_powerspec_linear(out_fooof.freq, powers=plot_data, title="Outputs from different fooof methods for ar1 data (linear scale)")

    def test_frontend_settings_are_merged_with_defaults_used_in_backend(self):
        cfg = TestFooofSpy.get_fooof_cfg()
        cfg.output = "fooof_peaks"
        cfg.pop('fooof_opt', None)
        fooof_opt = {'max_n_peaks': 8, 'peak_width_limits': (1.0, 12.0)}
        spec_dt = freqanalysis(cfg, _get_fooof_signal(seed=self.seed), fooof_opt=fooof_opt)

        assert spec_dt.data.ndim == 4

    def test_parallel(self, testcluster):

        plt.ioff()
        client = dd.Client(testcluster)
        all_tests = [attr for attr in self.__dir__()
                     if (inspect.ismethod(getattr(self, attr)) and 'parallel' not in attr)]

        for test in all_tests:
            test_method = getattr(self, test)
            test_method()
        client.close()
        plt.ion()


if __name__ == '__main__':
    T = TestFooofSpy()
