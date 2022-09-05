# -*- coding: utf-8 -*-
#
# Test FOOOF integration from user/frontend perspective.


import pytest
import numpy as np
import inspect
import matplotlib.pyplot as plt

# Local imports
from syncopy import freqanalysis
from syncopy.shared.tools import get_defaults
from syncopy.shared.errors import SPYValueError
from syncopy.tests.synth_data import AR2_network, phase_diffusion
import syncopy as spy
from syncopy import __acme__
if __acme__:
    import dask.distributed as dd

# Decorator to decide whether or not to run dask-related tests
skip_without_acme = pytest.mark.skipif(not __acme__, reason="acme not available")


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

def spp(dt, title=None):
    """Single panet plot with a title."""
    if not isinstance(dt, spy.datatype.base_data.BaseData):
        raise ValueError("Parameter 'dt' must be a syncopy.datatype instance.")
    fig, ax = dt.singlepanelplot()
    if title is not None:
        ax.set_title(title)
    return fig, ax


def _get_fooof_signal(nTrials=100):
    """
    Produce suitable test signal for fooof, with peaks at 30 and 50 Hz.

    Note: One must perform trial averaging during the FFT to get realistic
    data out of it (and reduce noise). Then work with the averaged data.

    Returns AnalogData instance.
    """
    nSamples = 1000
    nChannels = 1
    samplerate = 1000
    ar1_part = AR2_network(AdjMat=np.zeros(1), nSamples=nSamples, alphas=[0.9, 0], nTrials=nTrials)
    pd1 = phase_diffusion(freq=30., eps=.1, fs=samplerate, nChannels=nChannels, nSamples=nSamples, nTrials=nTrials)
    pd2 = phase_diffusion(freq=50., eps=.1, fs=samplerate, nChannels=nChannels, nSamples=nSamples, nTrials=nTrials)
    signal = ar1_part + .8 * pd1 + 0.6 * pd2
    return signal


class TestFooofSpy():
    """
    Test the frontend (user API) for running FOOOF. FOOOF is a post-processing of an FFT, and
    to request the post-processing, the user sets the method to "mtmfft", and the output to
    one of the available FOOOF output types.
    """

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
            _ = freqanalysis(cfg, _get_fooof_signal())  # tfData contains zero.
        assert "a frequency range that does not include zero" in str(err.value)

    def test_output_fooof_works_with_freq_zero_and_foilim(self):
        """
        This tests the intended operation with output type 'fooof': with an input that does not
        include zero, ensured by using the 'foilim' argument/setting when calling freqanalysis.

        This returns the full, fooofed spectrum.
        """
        cfg = TestFooofSpy.get_fooof_cfg()
        cfg.pop('fooof_opt', None)
        fooof_opt = {'peak_width_limits': (1.0, 12.0)}  # Increase lower limit to avoid fooof warning.
        spec_dt = freqanalysis(cfg, _get_fooof_signal(), fooof_opt=fooof_opt)

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

    def test_output_fooof_aperiodic(self):
        """Test fooof with output type 'fooof_aperiodic'. A spectrum containing only the aperiodic part is returned."""

        cfg = TestFooofSpy.get_fooof_cfg()
        cfg.output = "fooof_aperiodic"
        cfg.pop('fooof_opt', None)
        fooof_opt = {'peak_width_limits': (1.0, 12.0)}  # Increase lower limit to avoid fooof warning.
        spec_dt = freqanalysis(cfg, _get_fooof_signal(), fooof_opt=fooof_opt)

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
        spec_dt = freqanalysis(cfg, _get_fooof_signal(), fooof_opt=fooof_opt)
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

        out_fft = freqanalysis(cfg, _get_fooof_signal())
        cfg['output'] = "fooof"
        out_fooof = freqanalysis(cfg, _get_fooof_signal(), fooof_opt=fooof_opt)
        cfg['output'] = "fooof_aperiodic"
        out_fooof_aperiodic = freqanalysis(cfg, _get_fooof_signal(), fooof_opt=fooof_opt)
        cfg['output'] = "fooof_peaks"
        out_fooof_peaks = freqanalysis(cfg, _get_fooof_signal(), fooof_opt=fooof_opt)

        assert (out_fooof.freq == out_fooof_aperiodic.freq).all()
        assert (out_fooof.freq == out_fooof_peaks.freq).all()

        freqs = out_fooof.freq

        assert out_fooof.data.shape == out_fooof_aperiodic.data.shape
        assert out_fooof.data.shape == out_fooof_peaks.data.shape

        # biggest peak is at 30Hz
        f1_ind = out_fooof_peaks.show(channel=0).argmax()
        assert 27 < out_fooof_peaks.freq[f1_ind] < 33

        do_plot = False
        if do_plot:
            plot_data = {"Raw input data": np.ravel(out_fft.data), "Fooofed spectrum": np.ravel(out_fooof.data), "Fooof aperiodic fit": np.ravel(out_fooof_aperiodic.data), "Fooof peaks fit": np.ravel(out_fooof_peaks.data)}
            _plot_powerspec_linear(freqs, powers=plot_data, title="Outputs from different fooof methods for ar1 data (linear scale)")

    def test_frontend_settings_are_merged_with_defaults_used_in_backend(self):
        cfg = TestFooofSpy.get_fooof_cfg()
        cfg.output = "fooof_peaks"
        cfg.pop('fooof_opt', None)
        fooof_opt = {'max_n_peaks': 8, 'peak_width_limits': (1.0, 12.0)}
        spec_dt = freqanalysis(cfg, _get_fooof_signal(), fooof_opt=fooof_opt)

        assert spec_dt.data.ndim == 4

    @skip_without_acme
    def test_parallel(self, testcluster=None):

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
