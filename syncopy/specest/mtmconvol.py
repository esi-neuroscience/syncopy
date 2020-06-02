# -*- coding: utf-8 -*-
# 
# Time-frequency analysis based on a short-time Fourier transform
# 
# Created: 2020-02-05 09:36:38
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2020-06-02 19:24:30>

# Builtin/3rd party package imports
import numpy as np
from scipy import signal

# Local imports
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.shared.kwarg_decorators import unwrap_io
from syncopy.datatype import padding
import syncopy.specest.freqanalysis as spyfreq
from syncopy.shared.errors import SPYWarning
from syncopy.shared.tools import best_match


# Local workhorse that performs the computational heavy lifting
@unwrap_io
def mtmconvol(
    trl_dat, soi, padbegin, padend,
    samplerate=None, noverlap=None, nperseg=None, equidistant=True, toi=None, foi=None,
    nTaper=1, timeAxis=0, taper=signal.windows.hann, taperopt={}, 
    keeptapers=True, polyorder=None, output_fmt="pow",
    noCompute=False, chunkShape=None):
    """
    Perform time-frequency analysis on multi-channel time series data using a sliding window FFT
    
    Parameters
    ----------
    trl_dat : 2D :class:`numpy.ndarray`
        Uniformly sampled multi-channel time-series 
    soi : list of slices or slice
        Samples of interest; either a single slice encoding begin- to end-samples 
        to perform analysis on (if sliding window centroids are equidistant)
        or list of slices with each slice corresponding to coverage of a single
        analysis window (if spacing between windows is not constant)
    padbegin : int
        Number of samples to pre-pend to `trl_dat`
    padbegin : int
        Number of samples to append to `trl_dat`
    samplerate : int
        Samplerate of `trl_dat` in Hz
    noverlap : int
        Number of samples covered by two adjacent analysis windows
    nperseg : int
        Size of analysis windows (in samples)
    equidistant : bool
        If `True`, spacing of window-centroids is equidistant. 
    toi : 1D :class:`numpy.ndarray` or float or str
        Either sample-indices of window centroids if `toi` is a :class:`numpy.ndarray`,
        or percentage of overlap between windows if `toi` is a scalar or `"all"`
        to center windows on all samples in `trl_dat`. Please refer to 
        :func:`~syncopy.freqanalysis` for further details. **Note**: The value 
        of `toi` has to agree with provided padding and window settings. See 
        Notes for more information. 
    foi : 1D :class:`numpy.ndarray`
        Frequencies of interest  (Hz) for output. If desired frequencies
        cannot be matched exactly the closest possible frequencies (respecting 
        data length and padding) are used.
    nTaper : int
        Number of tapers to use
    timeAxis : int
        Index of running time axis in `trl_dat` (0 or 1)
    taper : callable 
        Taper function to use, one of :mod:`scipy.signal.windows`. Internal call
        signature is ``taper(nSamples, **taperopt)``
    taperopt : dict
        Additional keyword arguments passed to `taper` (see above). For further 
        details, please refer to the 
        `SciPy docs <https://docs.scipy.org/doc/scipy/reference/signal.windows.html>`_
    keeptapers : bool
        If `True`, results of Fourier transform are preserved for each taper, 
        otherwise spectrum is averaged across tapers. 
    polyorder : int
        **FIXME: Not implemented yet**
        Order of polynomial used for de-trending. A value of 0 corresponds to 
        subtracting the mean ("de-meaning"), ``polyorder = 1`` removes linear 
        trends (subtracting the least squares fit of a linear function), 
        ``polyorder = N`` for `N > 1` subtracts a polynomial of order `N` (``N = 2`` 
        quadratic, ``N = 3`` cubic etc.). If `polyorder` is `None`, no de-trending
        is performed. 
    output_fmt : str               
        Output of spectral estimation; use `'pow'` for power spectrum 
        (:obj:`numpy.float32`), `'fourier'` for complex Fourier coefficients 
        (:obj:`numpy.complex128`) or `'abs'` for absolute values (:obj:`numpy.float32`).
    noCompute : bool
        Preprocessing flag. If `True`, do not perform actual calculation but
        instead return expected shape and :class:`numpy.dtype` of output
        array.
    chunkShape : None or tuple
        If not `None`, represents shape of output object `spec` (respecting provided 
        values of `nTaper`, `keeptapers` etc.)
    
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
    
    The computational heavy lifting in this code is performed by SciPy's Short Time 
    Fourier Transform (STFT) implementation :func:`scipy.signal.stft`. 
    
    See also
    --------
    syncopy.freqanalysis : parent metafunction
    MultiTaperFFTConvol : :class:`~syncopy.shared.computational_routine.ComputationalRoutine`
                          instance that calls this method as 
                          :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
    scipy.signal.stft : SciPy's STFT implementation
    """
    
    # Re-arrange array if necessary and get dimensional information
    if timeAxis != 0:
        dat = trl_dat.T       # does not copy but creates view of `trl_dat`
    else:
        dat = trl_dat
        
    # Pad input array if necessary
    if padbegin > 0 or padend > 0:
        dat = padding(dat, "zero", pad="relative", padlength=None, 
                      prepadlength=padbegin, postpadlength=padend)

    # Get shape of output for dry-run phase
    nChannels = dat.shape[1]
    if isinstance(toi, np.ndarray):     # `toi` is an array of time-points
        nTime = toi.size
        stftBdry = None
        stftPad = False
    else:                               # `toi` is either 'all' or a percentage
        nTime = np.ceil(dat.shape[0] / (nperseg - noverlap)).astype(np.intp)
        stftBdry = "zeros"
        stftPad = True
    nFreq = foi.size
    outShape = (nTime, max(1, nTaper * keeptapers), nFreq, nChannels)
    if noCompute:
        return outShape, spyfreq.spectralDTypes[output_fmt]
    
    # In case tapers aren't preserved allocate `spec` "too big" and average afterwards
    spec = np.full((nTime, nTaper, nFreq, nChannels), np.nan, dtype=spyfreq.spectralDTypes[output_fmt])
    
    # Collect keyword args for `stft` in dictionary
    stftKw = {"fs": samplerate,
              "nperseg": nperseg,
              "noverlap": noverlap,
              "return_onesided": True,
              "boundary": stftBdry,
              "padded": stftPad,
              "axis": 0}
    
    # Call `stft` w/first taper to get freq/time indices: transpose resulting `pxx`
    # to have a time x freq x channel array
    win = np.atleast_2d(taper(nperseg, **taperopt))
    stftKw["window"] = win[0, :]
    if equidistant:
        freq, _, pxx = signal.stft(dat[soi, :], **stftKw)
        _, fIdx = best_match(freq, foi, squash_duplicates=True)
        spec[:, 0, ...] = \
            spyfreq.spectralConversions[output_fmt](
                pxx.transpose(2, 0, 1))[:nTime, fIdx, :]
    else:
        freq, _, pxx = signal.stft(dat[soi[0], :], **stftKw)
        _, fIdx = best_match(freq, foi, squash_duplicates=True)
        spec[0, 0, ...] = \
            spyfreq.spectralConversions[output_fmt](
                pxx.transpose(2, 0, 1).squeeze())[fIdx, :]
        for tk in range(1, len(soi)):
            spec[tk, 0, ...] = \
                spyfreq.spectralConversions[output_fmt](
                    signal.stft(
                        dat[soi[tk], :], 
                        **stftKw)[2].transpose(2, 0, 1).squeeze())[fIdx, :]

    # Compute FT using determined indices above for the remaining tapers (if any)
    for taperIdx in range(1, win.shape[0]):
        stftKw["window"] = win[taperIdx, :]
        if equidistant:
            spec[:, taperIdx, ...] = \
                spyfreq.spectralConversions[output_fmt](
                    signal.stft(
                        dat[soi, :],
                        **stftKw)[2].transpose(2, 0, 1))[:nTime, fIdx, :]
        else:
            for tk, sample in enumerate(soi):
                spec[tk, taperIdx, ...] = \
                    spyfreq.spectralConversions[output_fmt](
                        signal.stft(
                            dat[sample, :],
                            **stftKw)[2].transpose(2, 0, 1).squeeze())[fIdx, :]

    # Average across tapers if wanted
    if not keeptapers:
        return np.nanmean(spec, axis=1, keepdims=True)
    return spec
    

class MultiTaperFFTConvol(ComputationalRoutine):
    """
    Compute class that performs time-frequency analysis of :class:`~syncopy.AnalogData` objects
    
    Sub-class of :class:`~syncopy.shared.computational_routine.ComputationalRoutine`, 
    see :doc:`/developer/compute_kernels` for technical details on Syncopy's compute 
    classes and metafunctions. 
    
    See also
    --------
    syncopy.freqanalysis : parent metafunction
    """

    computeFunction = staticmethod(mtmconvol)

    def process_metadata(self, data, out):

        # Get trialdef array + channels from source        
        if data._selection is not None:
            chanSec = data._selection.channel
            trl = data._selection.trialdefinition
        else:
            chanSec = slice(None)
            trl = data.trialdefinition

        # Construct trialdef array (if necessary)
        if self.keeptrials:
            
            # If `toi` is array, use it to construct timing info
            toi = self.cfg["toi"]
            if isinstance(toi, np.ndarray):
                
                # Some index gymnastics to get trial begin/end samples
                nToi = toi.size
                time = np.cumsum([nToi] * trl.shape[0])
                trl[:, 0] = time - nToi
                trl[:, 1] = time
                
                # If trigger onset was part of `toi`, get its relative position wrt 
                # to other elements, otherwise use first element as "onset"
                t0Idx = np.where(toi == 0)[0]
                if t0Idx:
                    trl[:, 2] = -t0Idx[0]
                else:
                    trl[:, 2] = 0
                    
                # Important: differentiate b/w equidistant time ranges and disjoint points        
                if self.cfg["equidistant"]:
                    out.samplerate = 1 / (toi[1] - toi[0])
                else:
                    msg = "`SpectralData`'s `time` property currently does not support " +\
                        "unevenly spaced `toi` selections!"
                    SPYWarning(msg, caller="freqanalysis")
                    out.samplerate = 1.0
                    trl[:, 2] = 0
                    
            # If all samples have been used, simply copy relevant info from source
            elif toi == "all":
                out.samplerate = data.samplerate
                    
            # If `toi` was a percentage, some cumsum/winSize algebra is required
            else:
                winSize = self.cfg['nperseg'] - self.cfg['noverlap']
                trlLens = np.ceil(np.diff(trl[:, :2]) / winSize)
                sumLens = np.cumsum(trlLens).reshape(trlLens.shape)
                trl[:, 0] = np.ravel(sumLens - trlLens)
                trl[:, 1] = sumLens.ravel()
                trl[:, 2] = trl[:, 2] / winSize
                out.samplerate = np.round(data.samplerate / winSize, 2) 
            
            # Assign (calculated) trialdef array     
            out.trialdefinition = trl
            
        else:
            out.trialdefinition = np.array([[0, 1, 0]])
            
        # Attach remaining meta-data
        out.channel = np.array(data.channel[chanSec])
        out.taper = np.array([self.cfg["taper"].__name__] * self.outputShape[out.dimord.index("taper")])
        out.freq = self.cfg["foi"]
