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
from syncopy.shared.kwarg_decorators import unwrap_io




@unwrap_io
def superlet():
    pass


class SuperletTransform(ComputationalRoutine):

    computeFunction = staticmethod(superlet)


class MorletSL:

    def __init__(self, c_i=3, k_sd=5):
        
        """ The Morlet formulation according to
        Moca et al. shifts the admissability criterion from
        the central frequency to the number of cycles c_i
        within the Gaussian envelope which has a constant 
        standard deviation of k_sd.
        """ 

        self.c_i = c_i
        self.k_sd = k_sd

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
        B_c = self.k_sd / (s * self.c_i * (2 * np.pi)**1.5)
        
        output = B_c * np.exp(1j * ts)                 
        output *= np.exp(-0.5 * (self.k_sd * ts / (2 * np.pi * self.c_i))**2)
        
        return output

    def scale_from_period(self, period):

        return period / (2 * np.pi)

    def fourier_period(self, scale):

        """
        This is the approximate Morlet fourier period
        as used in the source publication of Moca et al. 2021

        Note that w0 (central frequency) is always 1 in this 
        Morlet formulation, hence the scales are not compatible
        to the standard Wavelet definitions!
        """
        
        return 2 * np.pi * scale 


def cwt(data, wavelet, scales, dt):

    '''
    The continuous Wavelet transform specifically
    for Morlets with the Superlet formulation
    of Moca et al. 2021.

    Differences to :func:`~syncopy.specest.wavelets.transform.cwt_time`:

    - Morlet support gets adjusted by number of cycles
    - normalisation is with 1/scale as also suggested by Moca et al. 2021
    '''
    
    # wavelets can be complex so output is complex
    output = np.zeros((len(scales),) + data.shape, dtype=np.complex64)

    # this checks if really a Superlet Wavelet is being used
    if not isinstance(wavelet, MorletSL):
        raise ValueError("Wavelet is not of MorletSL type!")
    
    # compute in time
    for ind, scale in enumerate(scales):

        t = _get_superlet_support(scale, dt, wavelet.c_i)
        
        # sample wavelet and normalise
        norm = dt ** .5 / scale
        wavelet_data = norm * wavelet(t, scale) # this is an 1d array for sure!

        # np.convolve is also via fft..
        output[ind, :] = np.convolve(data,
                                     wavelet_data,
                                     mode='same')
    return output


def _get_superlet_support(scale, dt, cycles):

    '''
    Effective support for the convolution is here not only 
    scale but also cycle dependent.
    '''

    # number of points needed to capture wavelet
    M = 10 * scale * cycles  / dt
    # times to use, centred at zero
    t = np.arange((-M + 1) / 2., (M + 1) / 2.) * dt

    return t
