# -*- coding: utf-8 -*-
#
# Welch's method for the estimation of power spectra, see doi:10.1109/TAU.1967.1161901.
#

import numpy as np


def welch(data_arr):
    """
    Welch method backeend function.
    """
    return data_arr