# -*- coding: utf-8 -*-
#
# syncopy.specest fooof backend tests
#
import numpy as np
import pytest

from syncopy.specest.fooofspy import fooofspy
from fooof.sim.gen import gen_power_spectrum
from fooof.sim.utils import set_random_seed

from syncopy.shared.errors import SPYValueError
import matplotlib.pyplot as plt


def _power_spectrum(freq_range=[3, 40], freq_res=0.5, periodic_params=[[10, 0.2, 1.25], [30, 0.15, 2]], aperiodic_params=[1, 1]):
    """
     aperiodic_params = [1, 1]  # use len 2 for fixed, 3 for knee. order is: offset, (knee), exponent.
     periodic_params = [[10, 0.2, 1.25], [30, 0.15, 2]] # the Gaussians: Mean (Center Frequency), height (Power), and standard deviation (Bandwidth).
    """
    set_random_seed(21)
    noise_level = 0.005
    freqs, powers = gen_power_spectrum(freq_range, aperiodic_params,
                                       periodic_params, nlv=noise_level, freq_res=freq_res)
    return freqs, powers


class TestSpfooof():

    freqs, powers = _power_spectrum()

    def test_spfooof_output_fooof_single_channel(self, freqs=freqs, powers=powers):
        """
        Tests spfooof with output 'fooof' and a single input signal/channel. This will return the full, fooofed spectrum.
        """
        spectra, details = fooofspy(powers, freqs, out_type='fooof')

        assert spectra.shape == (freqs.size, 1)
        assert details['settings_used']['out_type'] == 'fooof'
        assert all(key in details for key in ("aperiodic_params", "gaussian_params", "peak_params", "n_peaks", "r_squared", "error", "settings_used"))
        assert details['settings_used']['fooof_opt']['peak_threshold'] == 2.0  # Should be in and at default value.

    def test_spfooof_output_fooof_several_channels(self, freqs=freqs, powers=powers):
        """
        Tests spfooof with output 'fooof' and several input signals/channels. This will return the full, fooofed spectrum.
        """
        num_channels = 3
        powers = np.tile(powers, num_channels).reshape(powers.size, num_channels)  # Copy signal to create channels.
        spectra, details = fooofspy(powers, freqs, out_type='fooof')

        assert spectra.shape == (freqs.size, num_channels)
        assert details['settings_used']['out_type'] == 'fooof'
        assert all(key in details for key in ("aperiodic_params", "gaussian_params", "peak_params", "n_peaks", "r_squared", "error", "settings_used"))
        assert details['settings_used']['fooof_opt']['peak_threshold'] == 2.0  # Should be in and at default value.

    def test_spfooof_output_fooof_aperiodic(self, freqs=freqs, powers=powers):
        """
        Tests spfooof with output 'fooof_aperiodic' and a single input signal. This will return the aperiodic part of the fit.
        """
        spectra, details = fooofspy(powers, freqs, out_type='fooof_aperiodic')

        assert spectra.shape == (freqs.size, 1)
        assert details['settings_used']['out_type'] == 'fooof_aperiodic'
        assert all(key in details for key in ("aperiodic_params", "gaussian_params", "peak_params", "n_peaks", "r_squared", "error", "settings_used"))
        assert details['settings_used']['fooof_opt']['peak_threshold'] == 2.0  # Should be in and at default value.

    def test_spfooof_output_fooof_peaks(self, freqs=freqs, powers=powers):
        """
        Tests spfooof with output 'fooof_peaks' and a single input signal. This will return the Gaussian fit of the periodic part of the spectrum.
        """
        spectra, details = fooofspy(powers, freqs, out_type='fooof_peaks')

        assert spectra.shape == (freqs.size, 1)
        assert details['settings_used']['out_type'] == 'fooof_peaks'
        assert all(key in details for key in ("aperiodic_params", "gaussian_params", "peak_params", "n_peaks", "r_squared", "error", "settings_used"))
        assert details['settings_used']['fooof_opt']['peak_threshold'] == 2.0  # Should be in and at default value.

    def test_spfooof_together(self, freqs=freqs, powers=powers):
        spec_fooof, det_fooof = fooofspy(powers, freqs, out_type='fooof')
        spec_fooof_aperiodic, det_fooof_aperiodic = fooofspy(powers, freqs, out_type='fooof_aperiodic')
        spec_fooof_peaks, det_fooof_peaks = fooofspy(powers, freqs, out_type='fooof_peaks')

        # Ensure output shapes are as expected.
        assert spec_fooof.shape == spec_fooof_aperiodic.shape
        assert spec_fooof.shape == spec_fooof_peaks.shape
        assert spec_fooof.shape == (powers.size, 1)
        assert spec_fooof.shape == (freqs.size, 1)

        fooofed_spectrum = 10 ** spec_fooof.squeeze()
        fooof_aperiodic = 10 ** spec_fooof_aperiodic.squeeze()
        fooof_peaks = spec_fooof_peaks.squeeze()
        fooof_peaks_and_aperiodic = 10 ** (spec_fooof_peaks.squeeze() + spec_fooof_aperiodic.squeeze())

        # Visually compare data and fits.
        plt.figure()
        plt.plot(freqs, powers, label="Raw input data")
        plt.plot(freqs, fooofed_spectrum, label="Fooofed spectrum")
        plt.plot(freqs, fooof_aperiodic, label="Fooof aperiodic fit")
        plt.plot(freqs, fooof_peaks, label="Fooof peaks fit")
        plt.plot(freqs, fooof_peaks_and_aperiodic, label="Fooof peaks fit + aperiodic")
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.legend()
        plt.show()

    def test_spfooof_the_fooof_opt_settings_are_used(self, freqs=freqs, powers=powers):
        """
        Tests spfooof with output 'fooof_peaks' and a single input signal. This will return the Gaussian fit of the periodic part of the spectrum.
        """
        fooof_opt = {'peak_threshold': 3.0}
        spectra, details = fooofspy(powers, freqs, out_type='fooof_peaks', fooof_opt=fooof_opt)

        assert spectra.shape == (freqs.size, 1)
        assert details['settings_used']['out_type'] == 'fooof_peaks'
        assert all(key in details for key in ("aperiodic_params", "gaussian_params", "peak_params", "n_peaks", "r_squared", "error", "settings_used"))
        assert details['settings_used']['fooof_opt']['peak_threshold'] == 3.0  # Should reflect our custom value.
        assert details['settings_used']['fooof_opt']['min_peak_height'] == 0.0  # No custom value => should be at default.

    def test_spfooof_exception_empty_freqs(self):
        # The input frequencies must not be None.
        with pytest.raises(SPYValueError) as err:
            spectra, details = fooofspy(self.powers, None)
        assert "input frequencies are required and must not be None" in str(err.value)

    def test_spfooof_exception_freq_length_does_not_match_spectrum_length(self):
        # The input frequencies must have the same length as the spectrum.
        with pytest.raises(SPYValueError) as err:
            self.test_spfooof_output_fooof_single_channel(freqs=np.arange(self.powers.size + 1), powers=self.powers)
        assert "signal length" in str(err.value)
        assert "must match the number of frequency labels" in str(err.value)

    def test_spfooof_exception_on_invalid_output_type(self):
        # Invalid out_type is rejected.
        with pytest.raises(SPYValueError) as err:
            spectra, details = fooofspy(self.powers, self.freqs, out_type='fooof_invalidout')
        assert "out_type" in str(err.value)

    def test_spfooof_exception_on_invalid_fooof_opt_entry(self):
        # Invalid fooof_opt entry is rejected.
        with pytest.raises(SPYValueError) as err:
            fooof_opt = {'peak_threshold': 2.0, 'invalid_key': 42}
            spectra, details = fooofspy(self.powers, self.freqs, fooof_opt=fooof_opt)
        assert "fooof_opt" in str(err.value)
