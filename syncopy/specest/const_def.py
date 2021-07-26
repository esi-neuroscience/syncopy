# -*- coding: utf-8 -*-
# 
# Constant definitions and helper functions for spectral estimations
#

# Builtin/3rd party package imports
from numbers import Number
import numpy as np

from syncopy.shared.errors import SPYWarning

# Module-wide output specs
spectralDTypes = {"pow": np.float32,
                  "fourier": np.complex128,
                  "abs": np.float32}

#: output conversion of complex fourier coefficients
spectralConversions = {"pow": lambda x: (x * np.conj(x)).real.astype(np.float32),
                       "fourier": lambda x: x.astype(np.complex128),
                       "abs": lambda x: (np.absolute(x)).real.astype(np.float32)}

#: available outputs of :func:`~syncopy.freqanalysis`
availableOutputs = tuple(spectralConversions.keys())

#: available tapers of :func:`~syncopy.freqanalysis`
availableTapers = ("hann", "dpss")

#: available wavelet functions of :func:`~syncopy.freqanalysis`
availableWavelets = ("Morlet", "Paul", "DOG", "Ricker", "Marr", "Mexican_hat")

#: available spectral estimation methods of :func:`~syncopy.freqanalysis`
availableMethods = ("mtmfft", "mtmconvol", "wavelet", "superlet")


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
