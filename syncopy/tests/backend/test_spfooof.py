# -*- coding: utf-8 -*-
#
# syncopy.specest fooof backend tests
#
import numpy as np
import scipy.signal as sci_sig
import pytest

from syncopy.preproc import resampling, firws
from syncopy.specest.spfooof import spfooof
from fooof.sim.gen import gen_power_spectrum
from fooof.sim.utils import set_random_seed

from syncopy.shared.errors import SPYValueError

import matplotlib.pyplot as plt


def _plotspec(f, p):
    plt.plot(f, p)
    plt.show()


def _power_spectrum():
    set_random_seed(21)    
    freqs, powers = gen_power_spectrum([3, 40], [1, 1],
                                       [[10, 0.2, 1.25], [30, 0.15, 2]])
    return (freqs, powers)


class TestSpfooof():

    freqs, powers = _power_spectrum()

    def test_spfooof_ouput_fooof(self, freqs=freqs, powers=powers):
        """
        Tests spfooof with output 'fooof'. This will return the full, foofed spectrum.
        """                

        # _plotspec(freqs1, powers)
        spectra, details = spfooof(powers, fooof_settings={'in_freqs': freqs, 'freq_range': None}, out_type = 'fooof')

        assert spectra.shape == (freqs.size, 1)
        assert all (key in details for key in ("aperiodic_params", "n_peaks", "r_squared", "error", "settings_used"))


    def test_spfooof_exceptions(self):

        # The input frequencies must not be None.
        with pytest.raises(SPYValueError) as err:
            self.test_spfooof_ouput_fooof(freqs=None, powers=self.powers)
            assert "input frequencies are required and must not be None" in str(err)



