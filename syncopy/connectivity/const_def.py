# -*- coding: utf-8 -*-
# 
# Constant definitions and helper functions for spectral estimations
#

# Builtin/3rd party package imports
import numpy as np

# Module-wide output specs
spectralDTypes = {"complex": np.complex64,
                  "real": np.float32}

#: available tapers of :func:`~syncopy.connectivity_analysis`
availableTapers = ("hann", "dpss")

#: available spectral estimation methods of :func:`~syncopy.connectivity_analysis`
availableMethods = ("csd", "corr")

#: general, method agnostic, parameters of :func:`~syncopy.connectivity_analysis`
generalParameters = ("method", "output", "keeptrials",
                     "foi", "foilim", "polyremoval", "out")


# auxiliary functions
def nextpow2(number):
    n = 1
    while n < number:
        n *= 2
    return n
