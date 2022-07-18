# -*- coding: utf-8 -*-
#
# Test FOOOF integration from user/frontend perspective.


import pytest
import numpy as np

# Local imports
from syncopy import freqanalysis
from syncopy.shared.tools import get_defaults
from syncopy.datatype import SpectralData
from syncopy.shared.errors import SPYValueError
from syncopy.tests.test_specest import _make_tf_signal

import matplotlib.pyplot as plt


def _plot_powerspec(freqs, powers):
    """Simple, internal plotting function to plot x versus y."""
    plt.plot(freqs, powers)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (db)')
    plt.show()


class TestFooofSpy():
    """
    Test the frontend (user API) for running FOOOF. FOOOF is a post-processing of an FFT, and
    to request the post-prcocesing, the user sets the method to "mtmfft", and the output to
    one of the available FOOOF output types.
    """

    # Construct input signal
    nChannels = 2
    nChan2 = int(nChannels / 2)
    nTrials = 1
    seed = 151120
    fadeIn = None
    fadeOut = None
    tfData, modulators, even, odd, fader = _make_tf_signal(nChannels, nTrials, seed,
                                                           fadeIn=fadeIn, fadeOut=fadeOut, short=True)
    cfg = get_defaults(freqanalysis)
    cfg.method = "mtmfft"
    cfg.taper = "hann"
    cfg.select = {"trials": 0, "channel": 1}
    cfg.output = "fooof"

    def test_fooof_output_fooof_fails_with_freq_zero(self):
        """ The fooof package ignores input values of zero frequency, and shortens the output array
            in that case with a warning. This is not acceptable for us, as the expected output dimension
            will not off by one. Also it is questionable whether users would want that. We therefore use
            consider it an error to pass an input frequency axis that contains the zero, and throw an
            error in the frontend to stop before any expensive computations happen. This test checks for
            that error.
        """
        self.cfg['output'] = "fooof"
        self.cfg['foilim'] = [0., 250.]    # Include the zero in tfData.
        self.cfg['out'] = None
        with pytest.raises(SPYValueError) as err:
            _ = freqanalysis(self.cfg, self.tfData)  # tfData contains zero.
        assert "a frequency range that does not include zero" in str(err.value)

    def test_fooof_output_fooof_works_with_freq_zero_in_data_after_setting_foilim(self):
        """
        This tests the intended operation with output type 'fooof': with an input that does not
        include zero, ensured by using the 'foilim' argument/setting when calling freqanalysis.

        This returns the full, fooofed spectrum.
        """
        self.cfg['output'] = "fooof"
        self.cfg['foilim'] = [0.5, 250.]    # Exclude the zero in tfData.
        self.cfg['out'] = None
        spec_dt = freqanalysis(self.cfg, self.tfData)

        # check frequency axis
        assert spec_dt.freq.size == 500
        assert spec_dt.freq[0] == 0.5
        assert spec_dt.freq[499] == 250.

        # check the log
        assert "fooof_method = fooof" in spec_dt._log
        assert "fooof_aperiodic" not in spec_dt._log
        assert "fooof_peaks" not in spec_dt._log
        assert "fooof_opt" in spec_dt._log

        # check the data
        assert spec_dt.data.ndim == 4
        assert spec_dt.data.shape == (1, 1, 500, 1)
        assert not np.isnan(spec_dt.data).any()

        # check that the cfg is correct (required for replay)
        assert spec_dt.cfg['freqanalysis']['output'] == 'fooof'

        # Plot it.
        #  _plot_powerspec(freqs=spec_dt.freq, powers=spec_dt.data[0, 0, :, 0])
        #spec_dt.singlepanelplot()

    def test_spfooof_output_fooof_aperiodic(self):
        """Test fooof with output type 'fooof_aperiodic'. A spectrum containing only the aperiodic part is returned."""
        self.cfg['output'] = "fooof_aperiodic"
        self.cfg['foilim'] = [0.5, 250.]
        self.cfg['out'] = None
        spec_dt = freqanalysis(self.cfg, self.tfData)

        # log
        assert "fooof" in spec_dt._log  # from the method
        assert "fooof_method = fooof_aperiodic" in spec_dt._log
        assert "fooof_peaks" not in spec_dt._log

        # check the data
        assert spec_dt.data.ndim == 4
        assert spec_dt.data.shape == (1, 1, 500, 1)
        assert not np.isnan(spec_dt.data).any()
        _plot_powerspec(freqs=spec_dt.freq, powers=np.ravel(spec_dt.data))

    def test_spfooof_output_fooof_peaks(self):
        """Test fooof with output type 'fooof_peaks'. A spectrum containing only the peaks (actually, the Gaussians fit to the peaks) is returned."""
        self.cfg['foilim'] = [0.5, 250.]    # Exclude the zero in tfData.
        self.cfg['output'] = "fooof_peaks"
        self.cfg['out'] = None
        spec_dt = freqanalysis(self.cfg, self.tfData)
        assert spec_dt.data.ndim == 4
        assert "fooof" in spec_dt._log
        assert "fooof_method = fooof_peaks" in spec_dt._log
        assert "fooof_aperiodic" not in spec_dt._log
        _plot_powerspec(freqs=spec_dt.freq, powers=np.ravel(spec_dt.data))

    def test_spfooof_outputs_from_different_fooof_methods_are_consistent(self):
        """Test fooof with all output types plotted into a single plot and ensure consistent output."""
        self.cfg['foilim'] = [0.5, 250.]    # Exclude the zero in tfData.
        self.cfg['output'] = "pow"
        self.cfg['out'] = None
        out_fft = freqanalysis(self.cfg, self.tfData)
        self.cfg['output'] = "fooof"
        out_fooof = freqanalysis(self.cfg, self.tfData)
        self.cfg['output'] = "fooof_aperiodic"
        out_fooof_aperiodic = freqanalysis(self.cfg, self.tfData)
        self.cfg['output'] = "fooof_peaks"
        out_fooof_peaks = freqanalysis(self.cfg, self.tfData)

        assert (out_fooof.freq == out_fooof_aperiodic.freq).all()
        assert (out_fooof.freq == out_fooof_peaks.freq).all()

        freqs = out_fooof.freq

        assert out_fooof.data.shape == out_fooof_aperiodic.data.shape
        assert out_fooof.data.shape == out_fooof_peaks.data.shape

        plt.figure()
        plt.plot(freqs, np.ravel(out_fft.data), label="Raw input data")
        plt.plot(freqs, np.ravel(out_fooof.data), label="Fooofed spectrum")
        plt.plot(freqs, np.ravel(out_fooof_aperiodic.data), label="Fooof aperiodic fit")
        plt.plot(freqs, np.ravel(out_fooof_peaks.data), label="Fooof peaks fit")
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (db)')
        plt.legend()
        plt.show()

    def test_spfooof_frontend_settings_are_merged_with_defaults_used_in_backend(self):
        self.cfg['foilim'] = [0.5, 250.]    # Exclude the zero in tfData.
        self.cfg['output'] = "fooof_peaks"
        self.cfg['out'] = None
        self.cfg.pop('fooof_opt', None)  # Remove from cfg to avoid passing twice. We could also modify it (and then leave out the fooof_opt kw below).
        fooof_opt = {'max_n_peaks': 8}
        spec_dt = freqanalysis(self.cfg, self.tfData, fooof_opt=fooof_opt)

        assert spec_dt.data.ndim == 4

        # TODO later: test whether the settings returned as 2nd return value include
        #  our custom value for fooof_opt['max_n_peaks']. Not possible yet on
        #  this level as we have no way to get the 'details' return value.
        #  This is verified in backend tests though.

    def test_fooofspy_rejects_preallocated_output(self):
        """ We do not support a pre-allocated out SpectralData object with output = 'fooof*'.
            Ensure an error is thrown if the user tries it.
        """
        out = SpectralData(dimord=SpectralData._defaultDimord)
        with pytest.raises(SPYValueError) as err:
            self.cfg['out'] = out
            _ = freqanalysis(self.cfg, self.tfData)
        assert "pre-allocated output object not supported" in str(err.value)

