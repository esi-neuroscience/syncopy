# -*- coding: utf-8 -*-
# 
# Time-frequency analysis with wavelets
# 
# Created: 2019-09-02 14:44:41
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2020-07-13 17:21:02>

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
    dat = samples x channel
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
            
        # Construct trialdef array and compute new sampling rate (if necessary)
        if self.keeptrials:
            trl, srate = _make_trialdef(self.cfg, trl, data.samplerate)
        else:
            trl = np.array([[0, 1, 0]])
            srate = 1.0
            
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
