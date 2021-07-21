# -*- coding: utf-8 -*-
# 
# Time-frequency analysis with superlets
# Based on 'Time-frequency super-resolution with superlets'
# by Moca et al., 2021 Nature Communications
# 

# Builtin/3rd party package imports
import numpy as np
from scipy.signal import fftconvolve

# Local imports
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.shared.kwarg_decorators import unwrap_io
from syncopy.specest.freqanalysis import spectralConversions


@unwrap_io
def superlet(signal, 
             samplerate, scales, 
             order_max, order_min=1, c_1=3, adaptive=False,
             output_fmt="pow",
             noCompute=False,
             chunkShape=None):

    """
    Performs Superlet Transform (SLT) according to Moca et al. 2021
    """

    dt = 1 / samplerate

    # multiplicative SLT
    if not adaptive:

        # create the complete multiplicative set spanning
        # order_min - order_max
        cycles = c_1 * np.arange(order_min, order_max + 1)
        SLs = [MorletSL(c) for c in cycles]
        
        # lowest order
        gmean_spec = cwtSL(signal,
                           SLs[0],
                           scales,
                           dt)
        gmean_spec = np.power(gmean_spec, 1 / order_max)
        
        for wavelet in SLs[1:]:

            spec = cwtSL(signal,
                         wavelet,
                         scales,
                         dt)

            gmean_spec *= np.power(spec, 1 / order_max)
    
    # Adaptive SLT
    else:
        
        # frequencies of interest
        # from the scales for the SL Morlet
        # for len(orders) < len(scales)
        # multiple scales have the same order/SL set (discrete banding)
        fois = 1 / (2 * np.pi * scales)
        orders = compute_adaptive_order(fois, order_min, order_max, fois[0], fois[-1])

        cycles = c_1 * np.unique(orders)
        SLs = [MorletSL(c) for c in cycles]
        
        # potentially every scale needs a different exponent
        # for the geometric mean
        exponents = 1 / orders

        # which frequencies/scales use the same SL set
        order_jumps = np.where(np.diff(orders))[0]
        # if len(orders) >= len(scales) this is just
        # a continuous index array [0, 1, ..., len(scales) - 2]
        # as every scale has it's own order
        # otherwise it provides the mapping scales -> order
        assert len(SLs) == len(order_jumps) + 1 == np.unique(orders).size

        # 1st order
        # lowest order is needed for all scales/frequencies
        gmean_spec = cwtSL(signal,
                           SLs[0], # 1st order <-> order_min
                           scales,
                           dt)
        # Geometric normalization according to scale dependent order
        gmean_spec = np.power(gmean_spec.T, exponents).T
                
        # each frequency/scale can have its own multiplicative SL set
        # which overlap -> higher orders have all the lower orders
        for i, jump in enumerate(order_jumps):
            
            # relevant scales for that order
            scales_o = scales[jump + 1:]
            wavelet = SLs[i + 1]

            spec = cwtSL(signal,
                         wavelet,
                         scales_o,
                         dt)

            # normalize according to scale dependent order
            spec = np.power(spec.T, exponents[jump + 1:]).T            
            gmean_spec[jump + 1:] *= spec

    return spectralConversions[output_fmt](gmean_spec)


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
        # also includes scale normalisation!
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


def cwtSL(data, wavelet, scales, dt):

    '''
    The continuous Wavelet transform specifically
    for Morlets with the Superlet formulation
    of Moca et al. 2021.

    Differences to :func:`~syncopy.specest.wavelets.transform.cwt_time`:

    - Morlet support gets adjusted by number of cycles
    - normalisation is with 1/(scale * 4pi)
    - this way the absolute value of the spectrum (modulus) 
      at the corresponding harmonic frequency is the 
      harmonic signal's amplitude
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
        norm = dt ** .5 / (4 * np.pi)
        wavelet_data = norm * wavelet(t, scale) # this is an 1d array for sure!

        # np.convolve only works if support is capped
        # at signal lengths, as its output has shape
        # max(len(data), len(wavelet_data)
        output[ind, :] = fftconvolve(data,
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


def compute_adaptive_order(freq, order_min, order_max, f_min, f_max):

    '''
    Computes the superlet order for a given frequency of interest 
    for the adaptive SLT (ASLT) according to 
    equation 7 of Moca et al. 2021.
    
    This is a simple linear mapping between the minimal
    and maximal order onto the respective minimal and maximal
    frequencies. As the order strictly is of integer type, this can lead
    to discrete jumps.
    '''

    order = (order_max - order_min) * (freq - f_min) / (f_max - f_min)
    
    return np.int32(order_min + np.rint(order))
