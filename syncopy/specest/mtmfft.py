# -*- coding: utf-8 -*-
# 
# Spectral estimation with (multi-)tapered FFT
# 
# Created: 2019-09-02 14:25:34
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2020-07-14 11:19:03>

# Builtin/3rd party package imports
import numpy as np
import scipy.signal.windows as spwin

# Local imports
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.datatype import padding
import syncopy.specest.freqanalysis as freq
from syncopy.shared.kwarg_decorators import unwrap_io
from syncopy.shared.tools import best_match


# Local workhorse that performs the computational heavy lifting
@unwrap_io
def mtmfft(trl_dat, samplerate=None, foi=None, nTaper=1, timeAxis=0,
           taper=spwin.hann, taperopt={}, 
           pad="nextpow2", padtype="zero", padlength=None,
           keeptapers=True, polyremoval=None, output_fmt="pow",
           noCompute=False, chunkShape=None):
    """
    Compute (multi-)tapered Fourier transform of multi-channel time series data
    
    Parameters
    ----------
    trl_dat : 2D :class:`numpy.ndarray`
        Uniformly sampled multi-channel time-series 
    samplerate : float
        Samplerate of `trl_dat` in Hz
    foi : 1D :class:`numpy.ndarray`
        Frequencies of interest  (Hz) for output. If desired frequencies
        cannot be matched exactly the closest possible frequencies (respecting 
        data length and padding) are used.
    nTaper : int
        Number of filter windows to use
    timeAxis : int
        Index of running time axis in `trl_dat` (0 or 1)
    taper : callable 
        Taper function to use, one of :data:`~syncopy.specest.freqanalysis.availableTapers`
    taperopt : dict
        Additional keyword arguments passed to the `taper` function. For further 
        details, please refer to the 
        `SciPy docs <https://docs.scipy.org/doc/scipy/reference/signal.windows.html>`_
    pad : str
        Padding mode; one of `'absolute'`, `'relative'`, `'maxlen'`, or `'nextpow2'`.
        See :func:`syncopy.padding` for more information.
    padtype : str
        Values to be used for padding. Can be 'zero', 'nan', 'mean', 
        'localmean', 'edge' or 'mirror'. See :func:`syncopy.padding` for 
        more information.
    padlength : None, bool or positive scalar
        Number of samples to pad to data (if `pad` is 'absolute' or 'relative'). 
        See :func:`syncopy.padding` for more information.
    keeptapers : bool
        If `True`, results of Fourier transform are preserved for each taper, 
        otherwise spectrum is averaged across tapers. 
    polyremoval : int or None
        **FIXME: Not implemented yet**
        Order of polynomial used for de-trending data in the time domain prior 
        to spectral analysis. A value of 0 corresponds to subtracting the mean 
        ("de-meaning"), ``polyremoval = 1`` removes linear trends (subtracting the 
        least squares fit of a linear polynomial), ``polyremoval = N`` for `N > 1` 
        subtracts a polynomial of order `N` (``N = 2`` quadratic, ``N = 3`` cubic 
        etc.). If `polyremoval` is `None`, no de-trending is performed. 
    output_fmt : str
        Output of spectral estimation; one of :data:`~syncopy.specest.freqanalysis.availableOutputs`
    noCompute : bool
        Preprocessing flag. If `True`, do not perform actual calculation but
        instead return expected shape and :class:`numpy.dtype` of output
        array.
    chunkShape : None or tuple
        If not `None`, represents shape of output `spec` (respecting provided 
        values of `nTaper`, `keeptapers` etc.)
        
    Returns
    -------
    spec : :class:`numpy.ndarray`
        Complex or real spectrum of (padded) input data. 

    Notes
    -----
    This method is intended to be used as 
    :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
    inside a :class:`~syncopy.shared.computational_routine.ComputationalRoutine`. 
    Thus, input parameters are presumed to be forwarded from a parent metafunction. 
    Consequently, this function does **not** perform any error checking and operates 
    under the assumption that all inputs have been externally validated and cross-checked. 
    
    The computational heavy lifting in this code is performed by NumPy's reference
    implementation of the Fast Fourier Transform :func:`numpy.fft.fft`. 
    
    See also
    --------
    syncopy.freqanalysis : parent metafunction
    MultiTaperFFT : :class:`~syncopy.shared.computational_routine.ComputationalRoutine`
                    instance that calls this method as 
                    :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
    numpy.fft.fft : NumPy's FFT implementation
    """
    
    # Re-arrange array if necessary and get dimensional information
    if timeAxis != 0:
        dat = trl_dat.T       # does not copy but creates view of `trl_dat`
    else:
        dat = trl_dat

    # Padding (updates no. of samples)
    if pad is not None:
        dat = padding(dat, padtype, pad=pad, padlength=padlength, prepadlength=True)
    nSamples = dat.shape[0]
    nChannels = dat.shape[1]
    
    # Determine frequency band and shape of output (time=1 x taper x freq x channel)
    nFreq = int(np.floor(nSamples / 2) + 1)
    freqs = np.linspace(0, samplerate / 2, nFreq)
    _, fidx = best_match(freqs, foi, squash_duplicates=True)
    nFreq = fidx.size
    outShape = (1, max(1, nTaper * keeptapers), nFreq, nChannels)
    
    # For initialization of computational routine, just return output shape and dtype
    if noCompute:
        return outShape, freq.spectralDTypes[output_fmt]

    # In case tapers aren't preserved allocate `spec` "too big" and average afterwards
    spec = np.full((1, nTaper, nFreq, nChannels), np.nan, dtype=freq.spectralDTypes[output_fmt])
    fill_idx = tuple([slice(None, dim) for dim in outShape[2:]])

    # Actual computation
    win = np.atleast_2d(taper(nSamples, **taperopt))
    for taperIdx, taper in enumerate(win):
        if dat.ndim > 1:
            taper = np.tile(taper, (nChannels, 1)).T
        spec[(0, taperIdx,) + fill_idx] = freq.spectralConversions[output_fmt](np.fft.rfft(dat * taper, axis=0)[fidx, :])

    # Average across tapers if wanted
    if not keeptapers:
        return spec.mean(axis=1, keepdims=True)
    return spec


class MultiTaperFFT(ComputationalRoutine):
    """
    Compute class that calculates (multi-)tapered Fourier transfrom of :class:`~syncopy.AnalogData` objects
    
    Sub-class of :class:`~syncopy.shared.computational_routine.ComputationalRoutine`, 
    see :doc:`/developer/compute_kernels` for technical details on Syncopy's compute 
    classes and metafunctions. 
    
    See also
    --------
    syncopy.freqanalysis : parent metafunction
    """

    computeFunction = staticmethod(mtmfft)

    def process_metadata(self, data, out):
        
        # Some index gymnastics to get trial begin/end "samples"
        if data._selection is not None:
            chanSec = data._selection.channel
            trl = data._selection.trialdefinition
            for row in range(trl.shape[0]):
                trl[row, :2] = [row, row + 1]
        else:
            chanSec = slice(None)
            time = np.arange(len(data.trials))
            time = time.reshape((time.size, 1))
            trl = np.hstack((time, time + 1, 
                             np.zeros((len(data.trials), 1)), 
                             np.array(data.trialinfo)))

        # Attach constructed trialdef-array (if even necessary)
        if self.keeptrials:
            out.trialdefinition = trl
        else:
            out.trialdefinition = np.array([[0, 1, 0]])

        # Attach remaining meta-data
        out.samplerate = data.samplerate
        out.channel = np.array(data.channel[chanSec])
        out.taper = np.array([self.cfg["taper"].__name__] * self.outputShape[out.dimord.index("taper")])
        out.freq = self.cfg["foi"]
