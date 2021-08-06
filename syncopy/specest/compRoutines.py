# -*- coding: utf-8 -*-
#
# Definition of the respective ComputationalRoutines
# for the `freqanalysis` frontend.
#
# Builtin/3rd party package imports
import numpy as np
from numbers import Number

# backend imports
from .superlet import superlet

# Local imports
from syncopy.shared.errors import SPYWarning
from syncopy.datatype import padding
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.shared.kwarg_decorators import unwrap_io
from syncopy.specest.const_def import (
    spectralConversions,
    spectralDTypes,
)


@unwrap_io
def superlet_cF(
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

    computeFunction = staticmethod(superlet_cF)

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

        
def _make_trialdef(cfg, trialdefinition, samplerate):
    """
    Local helper to construct trialdefinition arrays for time-frequency :class:`~syncopy.SpectralData` objects
    
    Parameters
    ----------
    cfg : dict
        Config dictionary attribute of `ComputationalRoutine` subclass 
    trialdefinition : 2D :class:`numpy.ndarray`
        Provisional trialdefnition array either directly copied from the 
        :class:`~syncopy.AnalogData` input object or computed by the
        :class:`~syncopy.datatype.base_data.Selector` class. 
    samplerate : float
        Original sampling rate of :class:`~syncopy.AnalogData` input object
        
    Returns
    -------
    trialdefinition : 2D :class:`numpy.ndarray`
        Updated trialdefinition array reflecting provided `toi`/`toilim` selection
    samplerate : float
        Sampling rate accouting for potentially new spacing b/w time-points (accouting 
        for provided `toi`/`toilim` selection)
    
    Notes
    -----
    This routine is a local auxiliary method that is purely intended for internal
    use. Thus, no error checking is performed. 
    
    See also
    --------
    syncopy.specest.mtmconvol.mtmconvol : :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
                                          performing time-frequency analysis using (multi-)tapered sliding window Fourier transform
    syncopy.specest.wavelet.wavelet : :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
                                      performing time-frequency analysis using non-orthogonal continuous wavelet transform
    syncopy.specest.superlet.superlet : :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
                                      performing time-frequency analysis using super-resolution superlet transform

    """
    
    # If `toi` is array, use it to construct timing info
    toi = cfg["toi"]
    if isinstance(toi, np.ndarray):
        
        # Some index gymnastics to get trial begin/end samples
        nToi = toi.size
        time = np.cumsum([nToi] * trialdefinition.shape[0])
        trialdefinition[:, 0] = time - nToi
        trialdefinition[:, 1] = time
        
        # Important: differentiate b/w equidistant time ranges and disjoint points
        tSteps = np.diff(toi)
        if np.allclose(tSteps, [tSteps[0]] * tSteps.size):
            samplerate = 1 / (toi[1] - toi[0])
        else:
            msg = "`SpectralData`'s `time` property currently does not support " +\
                "unevenly spaced `toi` selections!"
            SPYWarning(msg, caller="freqanalysis")
            samplerate = 1.0
            trialdefinition[:, 2] = 0

        # Reconstruct trigger-onset based on provided time-point array            
        trialdefinition[:, 2] = toi[0] * samplerate
            
    # If `toi` was a percentage, some cumsum/winSize algebra is required
    # Note: if `toi` was "all", simply use provided `trialdefinition` and `samplerate`
    elif isinstance(toi, Number):
        winSize = cfg['nperseg'] - cfg['noverlap']
        trialdefinitionLens = np.ceil(np.diff(trialdefinition[:, :2]) / winSize)
        sumLens = np.cumsum(trialdefinitionLens).reshape(trialdefinitionLens.shape)
        trialdefinition[:, 0] = np.ravel(sumLens - trialdefinitionLens)
        trialdefinition[:, 1] = sumLens.ravel()
        trialdefinition[:, 2] = trialdefinition[:, 2] / winSize
        samplerate = np.round(samplerate / winSize, 2) 
    
    # If `toi` was "all", do **not** simply use provided `trialdefinition`: overlapping
    # trials require thie below `cumsum` gymnastics
    else:
        bounds = np.cumsum(np.diff(trialdefinition[:, :2]))
        trialdefinition[1:, 0] = bounds[:-1]
        trialdefinition[:, 1] = bounds
        
    return trialdefinition, samplerate
        
