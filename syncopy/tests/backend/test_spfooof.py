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

import matplotlib.pyplot as plt


def _plotspec(f, p):
    plt.plot(f, p)
    plt.show()

@pytest.fixture
def spectrum():
    set_random_seed(21)    
    freqs, powers = gen_power_spectrum([3, 40], [1, 1],
                                       [[10, 0.2, 1.25], [30, 0.15, 2]])
    return (freqs, powers)


def test_fooof_ouput_fooof(spectrum):
    """
    Tests fooof with output 'fooof'. This will return the full, foofed spectrum.
    """        
    freqs, powers = spectrum    

    # _plotspec(freqs1, powers)
    spectra, details = spfooof(powers, fooof_settings={'in_freqs': freqs, 'freq_range': None}, out_type = 'fooof')

    assert spectra.shape == (freqs.size, )
    assert all (key in details for key in ("aperiodic_params", "n_peaks", "r_squared", "error", "settings_used"))

