# -*- coding: utf-8 -*-
#
# syncopy.specest fooof backend tests
#
import numpy as np
import scipy.signal as sci_sig

from syncopy.preproc import resampling, firws
from syncopy.specest import spfooof
from fooof.sim.gen import gen_power_spectrum
from fooof.sim.utils import set_random_seed

import matplotlib.pyplot as plt

def _plotspec(f, p):
    plt.plot(f, p)
    plt.show()


def test_fooof_ouput_fooof():

    """
    Tests fooof with output 'fooof'.
    """
    set_random_seed(21)
    # Simulate example power spectra
    freqs, powers = gen_power_spectrum([3, 40], [1, 1],
                                     [[10, 0.2, 1.25], [30, 0.15, 2]])

    # _plotspec(freqs1, powers)
    res = spfoof()

