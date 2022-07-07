# -*- coding: utf-8 -*-
#
# Test the external fooof package, which is one of our dependencies.
#
import numpy as np

from fooof import FOOOF
from fooof.sim.gen import gen_power_spectrum
from fooof.sim.utils import set_random_seed


from syncopy.tests.backend.test_fooofspy import _power_spectrum

class TestFooof():

    freqs, powers = _power_spectrum()
    default_fooof_opt = {'peak_width_limits': (0.5, 12.0), 'max_n_peaks': np.inf,
                     'min_peak_height': 0.0, 'peak_threshold': 2.0,
                     'aperiodic_mode': 'fixed', 'verbose': True}


    def test_spfooof_output_fooof_single_channel(self, freqs=freqs, powers=powers, fooof_opt=default_fooof_opt):
        """
        Tests FOOOF.fit() to check when output length is not equal to input freq length, which we observe for some example data.
        """
        assert freqs.size == powers.size
        fm = FOOOF(**fooof_opt)
        fm.fit(freqs, powers)
        assert fm.fooofed_spectrum_.size == freqs.size


