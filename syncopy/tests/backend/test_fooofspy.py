# -*- coding: utf-8 -*-
#
# syncopy.specest fooof backend tests
#
import numpy as np
import pytest


from syncopy.specest.fooofspy import fooofspy
from syncopy.tests.backend.test_resampling import trl_av_power
from syncopy.tests import synth_data as sd
from fooof.sim.gen import gen_power_spectrum

import matplotlib.pyplot as plt


def _power_spectrum(freq_range=[3, 40],
                    freq_res=0.5):

    # use len 2 for fixed, 3 for knee. order is: offset, (knee), exponent.
    aperiodic_params = [1, 1]

    # the Gaussians: Mean (Center Frequency), height (Power), and standard deviation (Bandwidth).
    periodic_params = [[10, 0.2, 1.25], [30, 0.15, 2]]

    noise_level = 0.001
    freqs, powers = gen_power_spectrum(freq_range, aperiodic_params,
                                       periodic_params, nlv=noise_level, freq_res=freq_res)
    return freqs, powers


def AR1_plus_harm_spec(nTrials=30, hfreq=30, ratio=0.7):

    """
    Create AR(1) background + ratio * (harmonic + phase diffusion)
    and take the mtmfft with 1Hz spectral smoothing
    """
    fs = 400
    nSamples = 1000
    # single channel and alpha2 = 0 <-> single AR(1)
    signals = [sd.AR2_network(AdjMat=np.zeros(1),
                              alphas=[0.8, 0],
                              nSamples=nSamples) + ratio * sd.phase_diffusion(freq=hfreq,
                                                                              fs=fs, eps=0.1,
                                                                              nChannels=1)
               for i in range(nTrials)]

    power, freqs = trl_av_power(signals, nSamples, fs, tapsmofrq=1)

    return freqs, power


class TestSpfooof():

    freqs, powers = _power_spectrum()

    def test_output_fooof_single_channel(self, freqs=freqs, powers=powers):
        """
        Tests spfooof with output 'fooof' and a single input spectrum/channel.
        This will return the full, fooofed spectrum.
        """
        spectra, details = fooofspy(powers, freqs, out_type='fooof', fooof_opt={'peak_width_limits': (1.0, 12.0)})

        assert spectra.shape == (freqs.size, 1)
        assert details['settings_used']['out_type'] == 'fooof'
        assert all(key in details for key in ("aperiodic_params", "gaussian_params",
                                              "peak_params", "n_peaks", "r_squared",
                                              "error", "settings_used"))

        assert details['settings_used']['fooof_opt']['peak_threshold'] == 2.0  # Should be in and at default value.

        # Ensure the results resemble the params used to generate the artificial data
        # See the _power_spectrum() function above for the origins of these values.
        assert np.allclose(details['gaussian_params'][0][0], [10, 0.2, 1.25], atol=0.5)  # The first peak
        assert np.allclose(details['gaussian_params'][0][1], [30, 0.15, 2], atol=2.0)  # The second peak

    def test_output_fooof_several_channels(self, freqs=freqs, powers=powers):
        """
        Tests spfooof with output 'fooof' and several input signals/channels.
        This will return the full, fooofed spectrum.
        """
        num_channels = 3
        # Copy signal to create channels.
        powers = np.tile(powers, num_channels).reshape(powers.size, num_channels)
        spectra, details = fooofspy(powers, freqs, out_type='fooof', fooof_opt={'peak_width_limits': (1.0, 12.0)})

        assert spectra.shape == (freqs.size, num_channels)
        assert details['settings_used']['out_type'] == 'fooof'
        assert all(key in details for key in ("aperiodic_params",
                                              "gaussian_params",
                                              "peak_params",
                                              "n_peaks",
                                              "r_squared",
                                              "error",
                                              "settings_used"))

        # Should be in and at default value.
        assert details['settings_used']['fooof_opt']['peak_threshold'] == 2.0

    def test_output_fooof_aperiodic(self, freqs=freqs, powers=powers):
        """
        Tests spfooof with output 'fooof_aperiodic' and a single input signal.
        This will return the aperiodic part of the fit.
        """
        spectra, details = fooofspy(powers, freqs, out_type='fooof_aperiodic', fooof_opt={'peak_width_limits': (1.0, 12.0)})

        assert spectra.shape == (freqs.size, 1)
        assert details['settings_used']['out_type'] == 'fooof_aperiodic'
        assert all(key in details for key in ("aperiodic_params",
                                              "gaussian_params",
                                              "peak_params",
                                              "n_peaks",
                                              "r_squared",
                                              "error",
                                              "settings_used"))
        # Should be in and at default value.
        assert details['settings_used']['fooof_opt']['peak_threshold'] == 2.0

    def test_output_fooof_peaks(self, freqs=freqs, powers=powers):
        """
        Tests spfooof with output 'fooof_peaks' and a single input signal.
        This will return the Gaussian fit of the periodic part of the spectrum.
        """
        spectra, details = fooofspy(powers, freqs, out_type='fooof_peaks', fooof_opt={'peak_width_limits': (1.0, 12.0)})

        assert spectra.shape == (freqs.size, 1)
        assert details['settings_used']['out_type'] == 'fooof_peaks'
        assert all(key in details for key in ("aperiodic_params", "gaussian_params", "peak_params", "n_peaks", "r_squared", "error", "settings_used"))
        # Should be in and at default value.
        assert details['settings_used']['fooof_opt']['peak_threshold'] == 2.0

    def test_together(self, freqs=freqs, powers=powers):
        fooof_opt = {'peak_width_limits': (1.0, 12.0)}
        spec_fooof, det_fooof = fooofspy(powers, freqs, out_type='fooof', fooof_opt=fooof_opt)
        spec_fooof_aperiodic, det_fooof_aperiodic = fooofspy(powers, freqs, out_type='fooof_aperiodic', fooof_opt=fooof_opt)
        spec_fooof_peaks, det_fooof_peaks = fooofspy(powers, freqs, out_type='fooof_peaks', fooof_opt=fooof_opt)

        # Ensure details are correct
        assert det_fooof['settings_used']['out_type'] == 'fooof'
        assert det_fooof_aperiodic['settings_used']['out_type'] == 'fooof_aperiodic'
        assert det_fooof_peaks['settings_used']['out_type'] == 'fooof_peaks'

        # Ensure output shapes are as expected.
        assert spec_fooof.shape == spec_fooof_aperiodic.shape
        assert spec_fooof.shape == spec_fooof_peaks.shape
        assert spec_fooof.shape == (powers.size, 1)
        assert spec_fooof.shape == (freqs.size, 1)

        fooofed_spectrum = spec_fooof.squeeze()
        fooof_aperiodic = spec_fooof_aperiodic.squeeze()
        fooof_peaks = spec_fooof_peaks.squeeze()

        assert np.max(fooof_peaks) < np.max(fooofed_spectrum)

        # Visually compare data and fits.
        plt.figure()
        plt.plot(freqs, powers, label="Raw input data")
        plt.plot(freqs, fooofed_spectrum, label="Fooofed spectrum")
        plt.plot(freqs, fooof_aperiodic, label="Fooof aperiodic fit")
        plt.plot(freqs, fooof_peaks, label="Fooof peaks fit")
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (Db)')
        plt.legend()
        plt.title("Comparison of raw data and fooof results, linear scale.")
        # plt.show()

    def test_the_fooof_opt_settings_are_used(self, freqs=freqs, powers=powers):
        """
        Tests spfooof with output 'fooof_peaks' and a single input signal.
        This will return the Gaussian fit of the periodic part of the spectrum.
        """
        fooof_opt = {'peak_threshold': 3.0, 'peak_width_limits': (1.0, 12.0)}
        spectra, details = fooofspy(powers, freqs, out_type='fooof_peaks', fooof_opt=fooof_opt)

        assert spectra.shape == (freqs.size, 1)
        assert details['settings_used']['out_type'] == 'fooof_peaks'
        assert all(key in details for key in ("aperiodic_params", "gaussian_params", "peak_params", "n_peaks", "r_squared", "error", "settings_used"))
        # Should reflect our custom value.
        assert details['settings_used']['fooof_opt']['peak_threshold'] == 3.0
        # No custom value => should be at default.
        assert details['settings_used']['fooof_opt']['min_peak_height'] == 0.0

    def test_exception_empty_freqs(self):
        # The input frequencies must not be None.
        with pytest.raises(ValueError) as err:
            spectra, details = fooofspy(self.powers, None)
        assert "input frequencies are required and must not be None" in str(err.value)

    def test_exception_freq_length_does_not_match_spectrum_length(self):
        # The input frequencies must have the same length as the spectrum.
        with pytest.raises(ValueError) as err:
            self.test_output_fooof_single_channel(freqs=np.arange(self.powers.size + 1),
                                                  powers=self.powers)
        assert "signal length" in str(err.value)
        assert "must match the number of frequency labels" in str(err.value)

    def test_exception_on_invalid_output_type(self):
        # Invalid out_type is rejected.
        with pytest.raises(ValueError) as err:
            spectra, details = fooofspy(self.powers, self.freqs, out_type='fooof_invalidout')
        assert "out_type" in str(err.value)

    def test_exception_on_invalid_fooof_opt_entry(self):
        # Invalid fooof_opt entry is rejected.
        with pytest.raises(ValueError) as err:
            fooof_opt = {'peak_threshold': 2.0, 'invalid_key': 42}
            spectra, details = fooofspy(self.powers, self.freqs, fooof_opt=fooof_opt)
        assert "fooof_opt" in str(err.value)
