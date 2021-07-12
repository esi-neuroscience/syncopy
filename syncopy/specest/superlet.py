# -*- coding: utf-8 -*-
# 
# Time-frequency analysis with superlets
# Based on 'Time-frequency super-resolution with superlets'
# by Moca et al., 2021 Nature Communications
# 

# Builtin/3rd party package imports
import numpy as np

# Local imports
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.specest.wavelets import cwt
from syncopy.shared.kwarg_decorators import unwrap_io
from syncopy.datatype import padding
import syncopy.specest.freqanalysis as spyfreq


@unwrap_io
def superlet():
    pass


class SuperletTransform(ComputationalRoutine):

    computeFunction = staticmethod(superlet)


class MorletSL:

    def __init__(self, c_i=3):
        
        """ The Morlet formulation according to
        Vale et al. shifts the admissability criterion from
        the central frequency to the number of cycles c_i
        within the Gaussian envelope with a constant 
        standard deviation of k_sd = 5.

        """ 

        self.c_i = c_i
        self.k_sd = 5

    def __call__(self, *args, **kwargs):
        return self.time(*args, **kwargs)

    def time(self, t, s=1.0):

        """
        Complext Morlet wavelet in the SL formulation.
        
        Parameters
        ----------
        t : float
            Time. If s is not specified, this can be used as the
            non-dimensional time t/s.
        s : float
            Scaling factor. Default is 1.

        Returns
        -------
        out : complex
            Value of the Morlet wavelet at the given time
        
        """

        ts = t / s
        # scaled time spread parameter
        B_c = 5 / (s * self.c_i * (2 * np.pi)**1.5)
        output = B_c * np.exp(1j * ts)                 
        output *= np.exp(-0.5 * (5 * ts / (2 * np.pi * self.c_i))**2)

        return output

    def scale_from_period(self, period):

        return period / (2 * np.pi) 

    def fourier_period(self, scale):

        """
        This is the approximate Morlet fourier period
        as used in the source publication of Moca et al. 2021
        """
        
        return 2 * np.pi * scale
