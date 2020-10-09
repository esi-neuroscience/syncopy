# -*- coding: utf-8 -*-
# 
# Time-frequency analysis with wavelets
# 

# Builtin/3rd party package imports
import numpy as np
from numbers import Number

# Local imports
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.specest.wavelets import cwt
from syncopy.shared.kwarg_decorators import unwrap_io
from syncopy.datatype import padding
import syncopy.specest.freqanalysis as spyfreq
from .mtmconvol import _make_trialdef

@unwrap_io
def wavelet(
    trl_dat, preselect, postselect, padbegin, padend,
    samplerate=None, toi=None, scales=None, timeAxis=0, wav=None, 
    polyremoval=None, output_fmt="pow",
    noCompute=False, chunkShape=None):
    """ 
    Perform time-frequency analysis on multi-channel time series data using a wavelet transform
    
    Parameters
    ----------
    trl_dat : 2D :class:`numpy.ndarray`
        Uniformly sampled multi-channel time-series 
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
    samplerate : float
        Samplerate of `trl_dat` in Hz
    toi : 1D :class:`numpy.ndarray` or str
        Either time-points to center wavelets on if `toi` is a :class:`numpy.ndarray`,
        or `"all"` to center wavelets on all samples in `trl_dat`. Please refer to 
        :func:`~syncopy.freqanalysis` for further details. **Note**: The value 
        of `toi` has to agree with provided padding values. See Notes for more 
        information. 
    scales : 1D :class:`numpy.ndarray`
        Set of scales to use in wavelet transform. 
    timeAxis : int
        Index of running time axis in `trl_dat` (0 or 1)
    wav : callable 
        Wavelet function to use, one of :data:`~syncopy.specest.freqanalysis.availableWavelets`
    polyremoval : int
        **FIXME: Not implemented yet**
        Order of polynomial used for de-trending. A value of 0 corresponds to 
        subtracting the mean ("de-meaning"), ``polyremoval = 1`` removes linear 
        trends (subtracting the least squares fit of a linear function), 
        ``polyremoval = N`` for `N > 1` subtracts a polynomial of order `N` (``N = 2`` 
        quadratic, ``N = 3`` cubic etc.). If `polyremoval` is `None`, no de-trending
        is performed. 
    output_fmt : str
        Output of spectral estimation; one of :data:`~syncopy.specest.freqanalysis.availableOutputs`
    noCompute : bool
        Preprocessing flag. If `True`, do not perform actual calculation but
        instead return expected shape and :class:`numpy.dtype` of output
        array.
    chunkShape : None or tuple
        If not `None`, represents shape of output object `spec` (respecting provided 
        values of `scales`, `preselect`, `postselect` etc.)
    
    Returns
    -------
    spec : :class:`numpy.ndarray`
        Complex or real time-frequency representation of (padded) input data. 
            
    Notes
    -----
    This method is intended to be used as 
    :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
    inside a :class:`~syncopy.shared.computational_routine.ComputationalRoutine`. 
    Thus, input parameters are presumed to be forwarded from a parent metafunction. 
    Consequently, this function does **not** perform any error checking and operates 
    under the assumption that all inputs have been externally validated and cross-checked. 

    For wavelets, data concatenation is performed by first trimming `trl_dat` to
    an interval of interest (via `preselect`), then performing the actual wavelet
    transform, and subsequently extracting the actually wanted time-points 
    (via `postselect`). 
    
    See also
    --------
    syncopy.freqanalysis : parent metafunction
    WaveletTransform : :class:`~syncopy.shared.computational_routine.ComputationalRoutine`
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
    nChannels = dat.shape[1]
    if isinstance(toi, np.ndarray):     # `toi` is an array of time-points
        nTime = toi.size
    else:                               # `toi` is 'all'
        nTime = dat.shape[0]
    nScales = scales.size
    outShape = (nTime, 1, nScales, nChannels)
    if noCompute:
        return outShape, spyfreq.spectralDTypes[output_fmt]

    # Compute wavelet transform with given data/time-selection
    spec = cwt(dat[preselect, :], 
               axis=0, 
               wavelet=wav, 
               widths=scales, 
               dt=1/samplerate).transpose(1, 0, 2)[postselect, :, :]
    
    return spyfreq.spectralConversions[output_fmt](spec[:, np.newaxis, :, :])


class WaveletTransform(ComputationalRoutine):
    """
    Compute class that performs time-frequency analysis of :class:`~syncopy.AnalogData` objects
    
    Sub-class of :class:`~syncopy.shared.computational_routine.ComputationalRoutine`, 
    see :doc:`/developer/compute_kernels` for technical details on Syncopy's compute 
    classes and metafunctions. 
    
    See also
    --------
    syncopy.freqanalysis : parent metafunction
    """

    computeFunction = staticmethod(wavelet)

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
        out.freq = 1 / self.cfg["wav"].fourier_period(self.cfg["scales"][::-1])


def _get_optimal_wavelet_scales(self, nSamples, dt, dj=0.25, s0=None):
    """
    Local helper to compute an "optimally spaced" set of scales for wavelet analysis 
    
    Parameters
    ----------
    nSamples : int
        Sample-count (i.e., length) of time-series that is analyzed
    dt : float
        Time-series step-size; temporal spacing between consecutive samples 
        (1 / sampling rate)
    dj : float
        Spectral resolution of scales. The choice of `dj` depends on the spectral 
        width of the employed wavelet function. For instance, ``dj = 0.5`` is the 
        largest value that still yields adequate sampling in scale for the Morlet
        wavelet. Other wavelets allow larger values of `dj` while still providing 
        sufficient spectral resolution. Small values of `dj` yield finer scale 
        resolution. 
    s0 : float or None
        Smallest resolvable scale; should be chosen such that the equivalent 
        Fourier period is approximately ``2 * dt``. If `None`, `s0` is computed
        to satisfy this criterion. 
        
    Returns
    -------
    scales : 1D :class:`numpy.ndarray`
        Set of scales to use in the wavelet transform

    Notes
    -----
    The calculation of an "optimal" set of scales follows [ToCo98]_. 
    This routine is a local auxiliary method that is purely intended for internal
    use. Thus, no error checking is performed. 
    
    .. [ToCo98] C. Torrence and G. P. Compo. A Practical Guide to Wavelet Analysis. 
       Bulletin of the American Meteorological Society. Vol. 79, No. 1, January 1998. 

    See also
    --------
    syncopy.specest.wavelet.wavelet : :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
                                      performing time-frequency analysis using non-orthogonal continuous wavelet transform
    """
    
    # Compute `s0` so that the equivalent Fourier period is approximately ``2 * dt```
    if s0 is None:
        s0 = self.scale_from_period(2*dt)
        
    # Largest scale
    J = int((1 / dj) * np.log2(nSamples * dt / s0))
    return s0 * 2 ** (dj * np.arange(0, J + 1))
