# -*- coding: utf-8 -*-
#
# Test the external fooof package, which is one of our dependencies.
#
import numpy as np

from fooof import FOOOF
from syncopy.tests.backend.test_fooofspy import _power_spectrum

class TestFooof():

    freqs, powers = _power_spectrum()
    fooof_opt = {'peak_width_limits': (0.7, 12.0), 'max_n_peaks': np.inf,
                         'min_peak_height': 0.0, 'peak_threshold': 2.0,
                         'aperiodic_mode': 'fixed', 'verbose': True}

    def test_fooof_output_len_equals_in_length(self, freqs=freqs, powers=powers, fooof_opt=fooof_opt):
        """
        Tests FOOOF.fit() to check when output length is not equal to input freq length, which we observe for some example data.
        """
        assert freqs.size == powers.size
        fm = FOOOF(**fooof_opt)
        fm.fit(freqs, powers)
        assert fm.fooofed_spectrum_.size == freqs.size

    def test_fooof_freq_res(self, fooof_opt=fooof_opt):
        """
        Check whether the issue is related to frequency resolution
        """
        self.test_fooof_output_len_equals_in_length(*_power_spectrum(freq_range=[3, 40], freq_res=0.6))
        self.test_fooof_output_len_equals_in_length(*_power_spectrum(freq_range=[3, 40], freq_res=0.62))
        self.test_fooof_output_len_equals_in_length(*_power_spectrum(freq_range=[3, 40], freq_res=0.7))
        self.test_fooof_output_len_equals_in_length(*_power_spectrum(freq_range=[3, 40], freq_res=0.75))
        self.test_fooof_output_len_equals_in_length(*_power_spectrum(freq_range=[3, 40], freq_res=0.2))


