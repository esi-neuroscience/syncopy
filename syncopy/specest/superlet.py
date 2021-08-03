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
from syncopy.datatype import padding
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.shared.kwarg_decorators import unwrap_io
from syncopy.specest.const_def import (
    spectralConversions,
    spectralDTypes,
    _make_trialdef,
)


def superlet(
    data_arr,
    samplerate,
    scales,
    order_max,
    order_min=1,
    c_1=3,
    adaptive=False,
):

    """
    Performs Superlet Transform (SLT) according to Moca et al. [1]_
    Both multiplicative SLT and fractional adaptive SLT are available. 
    The former is recommended for a narrow frequency band of interest, 
    whereas the  is better suited for the analysis of a broad range 
    of frequencies.

    A superlet (SL) is a set of Morlet wavelets with increasing number
    of cycles within the Gaussian envelope. Hence the bandwith 
    is constrained more and more with more cycles yielding a sharper
    frequency resolution. Complementary the low cycle numbers will give a
    high time resolution. The SLT then is the geometric mean 
    of the set of individual wavelet transforms, combining both wide
    and narrow-bandwidth wavelets into a super-resolution estimate.

    Parameters
    ----------
    data_arr : nD :class:`numpy.ndarray`
        Uniformly sampled time-series data
        The 1st dimension is interpreted as the time axis
    samplerate : float
        Samplerate of the time-series in Hz
    scales : 1D :class:`numpy.ndarray`
        Set of scales to use in wavelet transform. 
        Note that for the SL Morlet the relationship
        between scale and frequency simply is s(f) = 1/(2*pi*f)
        Need to be ordered high to low for `adaptive=True`
    order_max : int
        Maximal order of the superlet set. Controls the maximum
        number of cycles within a SL together
        with the `c_1` parameter: c_max = c_1 * order_max
    order_min : int
        Minimal order of the superlet set. Controls 
        the minimal number of cycles within a SL together
        with the `c_1` parameter: c_min = c_1 * order_min
        Note that for admissability reasons c_min should be at least 3!
    c_1 : int
        Number of cycles of the base Morlet wavelet. If set to lower
        than 3 increase `order_min` as to never have less than 3 cycles
        in a wavelet!
    adaptive : bool
        Wether to perform multiplicative SLT or fractional adaptive SLT.
        If set to True, the order of the wavelet set will increase
        linearly with the frequencies of interest from `order_min` 
        to `order_max`. If set to False the same SL will be used for
        all frequencies.
    
    Returns
    -------
    gmean_spec : :class:`numpy.ndarray`
        Complex time-frequency representation of the input data. 
        Shape is (len(scales), data_arr.shape[0], data_arr.shape[1]).

    Notes
    -----
    .. [1] Moca, Vasile V., et al. "Time-frequency super-resolution with superlets." 
       Nature communications 12.1 (2021): 1-18.
 
 
    """

    # adaptive SLT    
    if adaptive:

        gmean_spec = FASLT(data_arr,
                           samplerate,
                           scales,
                           order_max,
                           order_min,
                           c_1)
        
    # multiplicative SLT    
    else:
        
        gmean_spec = multiplicativeSLT(data_arr,
                                       samplerate,
                                       scales,
                                       order_max,
                                       order_min,
                                       c_1)

    return gmean_spec


def multiplicativeSLT(data_arr,
                      samplerate,
                      scales,
                      order_max,
                      order_min=1,
                      c_1=3):

    dt = 1 / samplerate    
    # create the complete multiplicative set spanning
    # order_min - order_max
    cycles = c_1 * np.arange(order_min, order_max + 1)
    SL = [MorletSL(c) for c in cycles]

    # lowest order
    gmean_spec = cwtSL(data_arr, SL[0], scales, dt)
    gmean_spec = np.power(gmean_spec, 1 / order_max)

    for wavelet in SL[1:]:
        
        spec = cwtSL(data_arr, wavelet, scales, dt)
        gmean_spec *= np.power(spec, 1 / order_max)

    return gmean_spec


def FASLT(data_arr,
          samplerate,
          scales,
          order_max,
          order_min=1,          
          c_1=3):

    ''' Fractional adaptive SL transform

    For non-integer orders fractional SLTs are
    calculated in the interval [order, order+1) via:
    
    R(o_f) = R_1 * R_2 * ... * R_i * R_i+1 ** alpha 
    with o_f = o_i + alpha
    '''

    dt = 1 / samplerate    
    # frequencies of interest
    # from the scales for the SL Morlet
    fois = 1 / (2 * np.pi * scales)
    orders = compute_adaptive_order(fois, order_min, order_max)

    # create the complete superlet set from
    # all enclosed integer orders
    orders_int = np.int32(np.floor(orders))
    cycles = c_1 * np.unique(orders_int)
    SL = [MorletSL(c) for c in cycles]

    # every scale needs a different exponent
    # for the geometric mean
    exponents = 1 / orders

    # which frequencies/scales use the same integer orders
    order_jumps = np.where(np.diff(orders_int))[0]
    # each frequency/scale will have its own multiplicative SL
    # which overlap -> higher orders enclose all the lower orders

    assert len(SL) == len(order_jumps) + 1
    
    # the fractions
    alphas = orders % orders_int
    
    # 1st order
    # lowest order is needed for all scales/frequencies
    gmean_spec = cwtSL(data_arr, SL[0], scales, dt)  # 1st order <-> order_min
    # Geometric normalization according to scale dependent order
    gmean_spec = np.power(gmean_spec.T, exponents).T
    
    # we go to the next scale and order in any case..
    # but for order_max == 1 for which order_jumps is empty
    last_jump = 1

    for i, jump in enumerate(order_jumps):

        # relevant scales for the next order
        scales_o = scales[last_jump:]
        # order + 1 spec 
        next_spec = cwtSL(data_arr, SL[i + 1], scales_o, dt)
        
        # which fractions for the current next_spec
        # in the interval [order, order+1)
        scale_span = slice(last_jump, jump + 1)
        gmean_spec[scale_span, :] *= np.power(
            next_spec[:jump - last_jump + 1].T,
            alphas[scale_span] * exponents[scale_span]).T
        
        # multiply non-fractional next_spec for
        # all remaining scales/frequencies          
        gmean_spec[jump + 1 :] *= np.power(
            next_spec[jump - last_jump + 1:].T,
            exponents[jump + 1 :]).T

        # go to the next [order, order+1) interval 
        last_jump = jump + 1
        
    return gmean_spec


def adaptiveSLT(data_arr,
                samplerate,
                scales,
                order_max,
                order_min=1,
                c_1=3):

    '''This function is not used atm, it implements
    the non-fractional adaptive SLT. Kept here for 
    reference/comparisons if ever needed'''

    dt = 1 / samplerate    
    # frequencies of interest
    # from the scales for the SL Morlet
    # for len(orders) < len(scales)
    # multiple scales have the same order/wavelet set (discrete banding)
    fois = 1 / (2 * np.pi * scales)
    orders = compute_adaptive_order(fois, order_min, order_max)
    orders = np.int32(np.rint(orders))

    # create the complete superlet
    cycles = c_1 * np.unique(orders)
    SL = [MorletSL(c) for c in cycles]

    # potentially every scale needs a different exponent
    # for the geometric mean
    exponents = 1 / orders

    # which frequencies/scales use the same SL
    order_jumps = np.where(np.diff(orders))[0]
    # if len(orders) >= len(scales) this is just
    # a continuous index array [0, 1, ..., len(scales) - 2]
    # as every scale has it's own order
    # otherwise it provides the mapping scales -> order
    assert len(SL) == len(order_jumps) + 1 == np.unique(orders).size

    # 1st order
    # lowest order is needed for all scales/frequencies
    gmean_spec = cwtSL(data_arr, SL[0], scales, dt)  # 1st order <-> order_min
    # Geometric normalization according to scale dependent order
    gmean_spec = np.power(gmean_spec.T, exponents).T

    # each frequency/scale can have its own multiplicative SL
    # which overlap -> higher orders have all the lower orders
    for i, jump in enumerate(order_jumps):

        # relevant scales for that order
        scales_o = scales[jump + 1 :]
        wavelet = SL[i + 1]

        spec = cwtSL(data_arr, wavelet, scales_o, dt)

        # normalize according to scale dependent order
        spec = np.power(spec.T, exponents[jump + 1 :]).T
        gmean_spec[jump + 1 :] *= spec
    
    return gmean_spec


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
        B_c = self.k_sd / (s * self.c_i * (2 * np.pi) ** 1.5)

        output = B_c * np.exp(1j * ts)
        output *= np.exp(-0.5 * (self.k_sd * ts / (2 * np.pi * self.c_i)) ** 2)

        return output


def fourier_period(scale):

    """
    This is the approximate Morlet fourier period
    as used in the source publication of Moca et al. 2021

    Note that w0 (central frequency) is always 1 in this 
    Morlet formulation, hence the scales are not compatible
    to the standard Wavelet definitions!
    """

    return 2 * np.pi * scale

    
def scale_from_period(period):

    return period / (2 * np.pi)

    
def cwtSL(data, wavelet, scales, dt):

    """
    The continuous Wavelet transform specifically
    for Morlets with the Superlet formulation
    of Moca et al. 2021.

    Differences to :func:`~syncopy.specest.wavelets.transform.cwt_time`:

    - Morlet support gets adjusted by number of cycles
    - normalisation is with 1/(scale * 4pi)
    - this way the absolute value of the spectrum (modulus) 
      at the corresponding harmonic frequency is the 
      harmonic signal's amplitude

    Notes
    -----
    
    The time axis is expected to be along the 1st dimension.
    """

    # wavelets can be complex so output is complex
    output = np.zeros((len(scales),) + data.shape, dtype=np.complex64)

    # this checks if really a Superlet Wavelet is being used
    if not isinstance(wavelet, MorletSL):
        raise ValueError("Wavelet is not of MorletSL type!")

    # 1st axis is time
    slices = [None for _ in data.shape]
    slices[0] = slice(None)

    # compute in time
    for ind, scale in enumerate(scales):

        t = _get_superlet_support(scale, dt, wavelet.c_i)
        # sample wavelet and normalise
        norm = dt ** 0.5 / (4 * np.pi)
        wavelet_data = norm * wavelet(t, scale)  # this is an 1d array for sure!

        # np.convolve only works if support is capped
        # at signal lengths, as its output has shape
        # max(len(data), len(wavelet_data)
        output[ind, :] = fftconvolve(data, wavelet_data[tuple(slices)], mode="same")
    return output


def _get_superlet_support(scale, dt, cycles):

    """
    Effective support for the convolution is here not only 
    scale but also cycle dependent.
    """

    # number of points needed to capture wavelet
    M = 10 * scale * cycles / dt
    # times to use, centred at zero
    t = np.arange((-M + 1) / 2.0, (M + 1) / 2.0) * dt

    return t


def compute_adaptive_order(freq, order_min, order_max):

    """
    Computes the superlet order for a given frequency of interest 
    for the fractional adaptive SLT (FASLT) according to 
    equation 7 of Moca et al. 2021.
    
    This is a simple linear mapping between the minimal
    and maximal order onto the respective minimal and maximal
    frequencies. 

    Note that `freq` should be ordered low to high.
    """

    f_min, f_max = freq[0], freq[-1]
    order = (order_max - order_min) * (freq - f_min) / (f_max - f_min)

    # return np.int32(order_min + np.rint(order))
    return order_min + order


@unwrap_io
def _computeFunction(
        trl_dat,
        preselect,
        postselect,
        padbegin,
        padend,
        toi=None,
        timeAxis=0,
        output_fmt="pow",
        noCompute=False,
        chunkShape=None,
        method_kwargs=None        
):

    """
    This is the glue function for the 
        :func:`~syncopy.specest.superlet.superlet
    spectral estimation method. 


    Parameters
    ----------
    trl_dat : 2D :class:`numpy.ndarray`
        Uniformly sampled multi-channel time-series
    method_kwargs : dict
        Keyword arguments for :func:`~syncopy.specest.superlet.superlet
        controlling the spectral estimation method
    preselect : slice
        Begin- to end-samples to perform analysis on (trim data to interval). 
        See Notes for details. 
    postselect : list of slices or list of 1D NumPy arrays
        Actual time-points of interest within interval defined by `preselect`
        See Notes for details. 
    padbegin : int
        Number of samples to pre-pend to `trl_dat`
    padend : int
        Number of samples to append to `trl_dat`
    toi : 1D :class:`numpy.ndarray` or str
        Either array of equidistant time-points 
        or `"all"` to perform analysis on all samples in `trl_dat`. Please refer to 
        :func:`~syncopy.freqanalysis` for further details. **Note**: The value 
        of `toi` has to agree with provided padding values. See Notes for more 
        information. 
    output_fmt : str
        Output of spectral estimation; one of 
        :data:`~syncopy.specest.freqanalysis.availableOutputs`
    noCompute : bool
        Preprocessing flag. If `True`, do not perform actual calculation but
        instead return expected shape and :class:`numpy.dtype` of output
        array.
    chunkShape : None or tuple
        If not `None`, represents shape of output object `gmean_spec` 
        (respecting provided values of `scales`, `preselect`, `postselect` etc.)    

    
    Returns
    -------
    gmean_spec : :class:`numpy.ndarray`
        Complex or real time-frequency representation of the input data. 
        Shape is (nTime, 1, nScales, nChannels).

    Notes
    -----
    This method is intended to be used as 
    :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
    inside a :class:`~syncopy.shared.computational_routine.ComputationalRoutine`. 
    Thus, input parameters are presumed to be forwarded from a parent metafunction. 
    Consequently, this function does **not** perform any error checking and operates 
    under the assumption that all inputs have been externally validated and cross-checked. 

    See also
    --------
    syncopy.freqanalysis : parent metafunction
    SuperletTransform : :class:`~syncopy.shared.computational_routine.ComputationalRoutine`
                       instance that calls this method as 
                       :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`

    """

    # Re-arrange array if necessary and get dimensional information
    if timeAxis != 0:
        dat = trl_dat.T       # does not copy but creates view of `trl_dat`
    else:
        dat = trl_dat

    # Pad input array if wanted/necessary
    if padbegin > 0 or padend > 0:
        dat = padding(dat, "zero", pad="relative", padlength=None, 
                      prepadlength=padbegin, postpadlength=padend)

    # Get shape of output for dry-run phase
    nChannels = trl_dat.shape[1]
    if isinstance(toi, np.ndarray):     # `toi` is an array of time-points
        nTime = toi.size
    else:                               # `toi` is 'all'
        nTime = dat.shape[0]
    nScales = method_kwargs['scales'].size
    outShape = (nTime, 1, nScales, nChannels)
    if noCompute:
        return outShape, spectralDTypes[output_fmt]

    # ------------------
    # actual method call
    # ------------------
    gmean_spec = superlet(trl_dat[preselect, :],
                          **method_kwargs)
        
    # the cwtSL stacks the scales on the 1st axis
    gmean_spec = gmean_spec.transpose(1, 0, 2)[postselect, :, :]
    
    return spectralConversions[output_fmt](gmean_spec[:, np.newaxis, :, :])


class SuperletTransform(ComputationalRoutine):
    """
    Compute class that performs time-frequency analysis of :class:`~syncopy.AnalogData` objects
    
    Sub-class of :class:`~syncopy.shared.computational_routine.ComputationalRoutine`, 
    see :doc:`/developer/compute_kernels` for technical details on Syncopy's compute 
    classes and metafunctions. 
    
    See also
    --------
    syncopy.freqanalysis : parent metafunction
    """

    computeFunction = staticmethod(_computeFunction)

    def process_metadata(self, data, out):

        # Get trialdef array + channels from source
        if data._selection is not None:
            chanSec = data._selection.channel
            trl = data._selection.trialdefinition
        else:
            chanSec = slice(None)
            trl = data.trialdefinition

        # Construct trialdef array and compute new sampling rate
        trl, srate = _make_trialdef(self.cfg, trl, data.samplerate)

        # Construct trialdef array and compute new sampling rate
        trl, srate = _make_trialdef(self.cfg, trl, data.samplerate)

        # If trial-averaging was requested, use the first trial as reference
        # (all trials had to have identical lengths), and average onset timings
        if not self.keeptrials:
            t0 = trl[:, 2].mean()
            trl = trl[[0], :]
            trl[:, 2] = t0

        # Attach meta-data
        out.trialdefinition = trl
        out.samplerate = srate
        out.channel = np.array(data.channel[chanSec])
        # for the SL Morlets the conversion is straightforward
        out.freq = 1 / (2 * np.pi * self.cfg["method_kwargs"]["scales"])
