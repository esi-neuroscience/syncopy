# -*- coding: utf-8 -*-
#
# syncopy.specest fooof backend tests
#
import numpy as np
import pytest

from syncopy.specest.spfooof import spfooof
from fooof.sim.gen import gen_power_spectrum
from fooof.sim.utils import set_random_seed

from syncopy.shared.errors import SPYValueError


def _power_spectrum():
    set_random_seed(21)
    freqs, powers = gen_power_spectrum([3, 40], [1, 1],
                                       [[10, 0.2, 1.25], [30, 0.15, 2]])
    return (freqs, powers)


class TestSpfooof():

    freqs, powers = _power_spectrum()

    def test_spfooof_output_fooof_single_channel(self, freqs=freqs, powers=powers):
        """
        Tests spfooof with output 'fooof' and a single input signal. This will return the full, foofed spectrum.
        """
        spectra, details = spfooof(powers, fooof_settings={'in_freqs': freqs, 'freq_range': None}, out_type='fooof')

        assert spectra.shape == (freqs.size, 1)
        assert details['settings_used']['out_type'] == 'fooof'
        assert all(key in details for key in ("aperiodic_params", "n_peaks", "r_squared", "error", "settings_used"))

    def test_spfooof_output_fooof_several_channels(self, freqs=freqs, powers=powers):
        """
        Tests spfooof with output 'fooof' and several input signal. This will return the full, foofed spectrum.
        """

        num_channels = 3
        powers = np.tile(powers, num_channels).reshape(powers.size, num_channels)  # Copy signal to create channels.
        spectra, details = spfooof(powers, fooof_settings={'in_freqs': freqs, 'freq_range': None}, out_type='fooof')

        assert spectra.shape == (freqs.size, num_channels)
        assert details['settings_used']['out_type'] == 'fooof'
        assert all(key in details for key in ("aperiodic_params", "n_peaks", "r_squared", "error", "settings_used"))

    def test_spfooof_output_fooof_aperiodic(self, freqs=freqs, powers=powers):
        """
        Tests spfooof with output 'fooof_aperiodic' and a single input signal. This will return the aperiodic part of the fit.
        """
        spectra, details = spfooof(powers, fooof_settings={'in_freqs': freqs, 'freq_range': None}, out_type='fooof_aperiodic')

        assert spectra.shape == (freqs.size, 1)
        assert details['settings_used']['out_type'] == 'fooof_aperiodic'
        assert all(key in details for key in ("aperiodic_params", "n_peaks", "r_squared", "error", "settings_used"))

    def test_spfooof_output_fooof_peaks(self, freqs=freqs, powers=powers):
        """
        Tests spfooof with output 'fooof_peaks' and a single input signal. This will return the Gaussian fit of the periodic part of the spectrum.
        """

        # _plotspec(freqs1, powers)
        spectra, details = spfooof(powers, fooof_settings={'in_freqs': freqs, 'freq_range': None}, out_type='fooof_peaks')

        assert spectra.shape == (freqs.size, 1)
        assert details['settings_used']['out_type'] == 'fooof_peaks'
        assert all(key in details for key in ("aperiodic_params", "n_peaks", "r_squared", "error", "settings_used"))

    def test_spfooof_exceptions(self):
        """
        Tests that spfooof throws the expected error if incomplete data is passed to it.
        """

        # The input frequencies must not be None.
        with pytest.raises(SPYValueError) as err:
            self.test_spfooof_output_fooof_single_channel(freqs=None, powers=self.powers)
            assert "input frequencies are required and must not be None" in str(err)

        # The input frequencies must have the same length as the channel data.
        with pytest.raises(SPYValueError) as err:
            self.test_spfooof_output_fooof_single_channel(freqs=np.arange(self.powers.size + 1), powers=self.powers)
            assert "signal length" in str(err)
            assert "must match the number of frequency labels" in str(err)

        # Invalid out_type is rejected.
        with pytest.raises(SPYValueError) as err:
            spectra, details = spfooof(self.powers, fooof_settings={'in_freqs': self.freqs}, out_type='fooof_invalidout')
            assert "out_type" in str(err)
