# -*- coding: utf-8 -*-
#
# computeFunctions and -Routines for parallel calculation
# of single trial measures needed for the averaged
# measures like cross spectral densities
#

# Builtin/3rd party package imports
import numpy as np
from scipy.signal import fftconvolve, detrend
from inspect import signature

# backend method imports
from .csd import csd

# syncopy imports
from syncopy.shared.const_def import spectralDTypes
from syncopy.shared.tools import best_match
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.shared.kwarg_decorators import unwrap_io


@unwrap_io
def cross_spectra_cF(trl_dat,
                     samplerate=1,
                     nSamples=None,
                     foi=None,
                     taper="hann",
                     taper_opt=None,
                     polyremoval=False,
                     timeAxis=0,
                     chunkShape=None,
                     noCompute=False):

    """
    Single trial Fourier cross spectral estimates between all channels
    of the input data. First all the individual Fourier transforms
    are calculated via a (multi-)tapered FFT, then the pairwise
    cross-spectra are computed.

    Averaging over tapers is done implicitly
    for multi-taper analysis with `taper="dpss"`.

    Output consists of all (nChannels x nChannels+1)/2 different complex
    estimates arranged in a symmetric fashion (``CS_ij == CS_ji*``). The
    elements on the main diagonal (`CS_ii`) are the (real) auto-spectra.

    This is NOT the same as what is commonly referred to as
    "cross spectral density" as there is no (time) averaging!!
    Multi-tapering alone is not necessarily sufficient to get enough
    statitstical power for a robust csd estimate. Yet for completeness
    and testing the option `norm=True` will output a single-trial
    coherence estimate.

    Parameters
    ----------
    trl_dat : (K, N) :class:`numpy.ndarray`
        Uniformly sampled multi-channel time-series data
        The 1st dimension is interpreted as the time axis,
        columns represent individual channels.
        Dimensions can be transposed to `(N, K)` with the `timeAxis` parameter.
    samplerate : float
        Samplerate in Hz
    nSamples : int or None
        Absolute length of the (potentially to be padded) signal or
        `None` for no padding
    foi : 1D :class:`numpy.ndarray` or None, optional
        Frequencies of interest  (Hz) for output. If desired frequencies
        cannot be matched exactly the closest possible frequencies (respecting
        data length and padding) are used.
    taper : str or None
        Taper function to use, one of scipy.signal.windows
        Set to `None` for no tapering.
    taper_opt : dict, optional
        Additional keyword arguments passed to the `taper` function.
        For multi-tapering with `taper='dpss'` set the keys
        `'Kmax'` and `'NW'`.
        For further details, please refer to the
        `SciPy docs <https://docs.scipy.org/doc/scipy/reference/signal.windows.html>`_
    polyremoval : int or None
        Order of polynomial used for de-trending data in the time domain prior
        to spectral analysis. A value of 0 corresponds to subtracting the mean
        ("de-meaning"), ``polyremoval = 1`` removes linear trends (subtracting the
        least squares fit of a linear polynomial).
        If `polyremoval` is `None`, no de-trending is performed.
    timeAxis : int, optional
        Index of running time axis in `trl_dat` (0 or 1)
    noCompute : bool
        Preprocessing flag. If `True`, do not perform actual calculation but
        instead return expected shape and :class:`numpy.dtype` of output
        array.

    Returns
    -------
    CS_ij : (1, nFreq, N, N) :class:`numpy.ndarray`
        Complex cross spectra for all channel combinations ``i,j``.
        `N` corresponds to number of input channels.

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
    csd : :func:`~syncopy.connectivity.csd.csd`
             Cross-spectra backend function
    normalize_csd : :func:`~syncopy.connectivity.csd.normalize_csd`
             Coherence from trial averages
    mtmfft : :func:`~syncopy.specest.mtmfft.mtmfft`
             (Multi-)tapered Fourier analysis

    """

    # Re-arrange array if necessary and get dimensional information
    if timeAxis != 0:
        dat = trl_dat.T       # does not copy but creates view of `trl_dat`
    else:
        dat = trl_dat

    if nSamples is None:
        nSamples = dat.shape[0]

    nChannels = dat.shape[1]

    freqs = np.fft.rfftfreq(nSamples, 1 / samplerate)

    if foi is not None:
        _, freq_idx = best_match(freqs, foi, squash_duplicates=True)
        nFreq = freq_idx.size
    else:
        freq_idx = slice(None)
        nFreq = freqs.size

    # we always average over tapers here
    outShape = (1, nFreq, nChannels, nChannels)

    # For initialization of computational routine,
    # just return output shape and dtype
    # cross spectra are complex!
    if noCompute:
        return outShape, spectralDTypes["fourier"]

    # detrend
    if polyremoval == 0:
        # SciPy's overwrite_data not working for type='constant' :/
        dat = detrend(dat, type='constant', axis=0, overwrite_data=True)
    elif polyremoval == 1:
        dat = detrend(dat, type='linear', axis=0, overwrite_data=True)

    CS_ij = csd(dat, samplerate, nSamples, taper=taper, taper_opt=taper_opt)
    
    # where does freqs go/come from -
    # we will eventually solve this issue..
    return CS_ij[None, freq_idx, ...]


class ST_CrossSpectra(ComputationalRoutine):

    """
    Compute class that calculates single-trial (multi-)tapered cross spectra
    of :class:`~syncopy.AnalogData` objects

    Sub-class of :class:`~syncopy.shared.computational_routine.ComputationalRoutine`,
    see :doc:`/developer/compute_kernels` for technical details on Syncopy's compute
    classes and metafunctions.

    See also
    --------
    syncopy.connectivityanalysis : parent metafunction
    """

    # the hard wired dimord of the cF
    dimord = ['time', 'freq', 'channel_i', 'channel_j']

    computeFunction = staticmethod(cross_spectra_cF)

    backends = [csd]
    # 1st argument,the data, gets omitted
    valid_kws = list(signature(cross_spectra_cF).parameters.keys())[1:]
    # hardcode some parameter names which got digested from the frontend
    valid_kws += ['tapsmofrq', 'nTaper', 'pad_to_length']

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
        out.channel_i = np.array(data.channel[chanSec])
        out.channel_j = np.array(data.channel[chanSec])
        out.freq = self.cfg['foi']


@unwrap_io
def cross_covariance_cF(trl_dat,
                        samplerate=1,
                        polyremoval=0,
                        timeAxis=0,
                        norm=False,
                        chunkShape=None,
                        noCompute=False,
                        fullOutput=False):

    """
    Single trial covariance estimates between all channels
    of the input data. Output consists of all ``(nChannels x nChannels+1)/2``
    different estimates arranged in a symmetric fashion
    (``COV_ij == COV_ji``). The elements on the
    main diagonal (`CS_ii`) are the channel variances.

    Parameters
    ----------
    trl_dat : (K, N) :class:`numpy.ndarray`
        Uniformly sampled multi-channel time-series data
        The 1st dimension is interpreted as the time axis,
        columns represent individual channels.
        Dimensions can be transposed to `(N, K)` with the `timeAxis` parameter.
    samplerate : float
        Samplerate in Hz
    polyremoval : int or None
        Order of polynomial used for de-trending data in the time domain prior
        to spectral analysis. A value of 0 corresponds to subtracting the mean
        ("de-meaning"), ``polyremoval = 1`` removes linear trends (subtracting the
        least squares fit of a linear polynomial).
        If `polyremoval` is `None`, no de-trending is performed.
    timeAxis : int, optional
        Index of running time axis in `trl_dat` (0 or 1)
    norm : bool, optional
        Set to `True` to normalize for single-trial cross-correlation.
    noCompute : bool
        Preprocessing flag. If `True`, do not perform actual calculation but
        instead return expected shape and :class:`numpy.dtype` of output
        array.
    fullOutput : bool
        For backend testing or stand-alone applications, set to `True`
        to return also the `lags` array.

    Returns
    -------
    CC_ij : (K, 1, N, N) :class:`numpy.ndarray`
        Cross covariance for all channel combinations ``i,j``.
        `N` corresponds to number of input channels.

    lags : (M,) :class:`numpy.ndarray`
        The lag times if `fullOutput=True`

    Notes
    -----
    This method is intended to be used as
    :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
    inside a :class:`~syncopy.shared.computational_routine.ComputationalRoutine`.
    Thus, input parameters are presumed to be forwarded from a parent metafunction.
    Consequently, this function does **not** perform any error checking and operates
    under the assumption that all inputs have been externally validated and cross-checked.

    """

    # Re-arrange array if necessary and get dimensional information
    if timeAxis != 0:
        dat = trl_dat.T       # does not copy but creates view of `trl_dat`
    else:
        dat = trl_dat

    nSamples = dat.shape[0]
    nChannels = dat.shape[1]

    # positive lags in time units
    if nSamples % 2 == 0:
        lags = np.arange(0, nSamples // 2)
    else:
        lags = np.arange(0, nSamples // 2 + 1)
    lags = lags * 1 / samplerate

    outShape = (len(lags), 1, nChannels, nChannels)

    # For initialization of computational routine,
    # just return output shape and dtype
    # cross covariances are real!
    if noCompute:
        return outShape, spectralDTypes["abs"]

    # detrend, has to be done after noCompute!
    if polyremoval == 0:
        # SciPy's overwrite_data not working for type='constant' :/
        dat = detrend(dat, type='constant', axis=0, overwrite_data=True)
    elif polyremoval == 1:
        detrend(dat, type='linear', axis=0, overwrite_data=True)
    
    # re-normalize output for different effective overlaps
    norm_overlap = np.arange(nSamples, nSamples // 2, step = -1)

    CC = np.empty(outShape)
    for i in range(nChannels):
        for j in range(i + 1):
            cc12 = fftconvolve(dat[:, i], dat[::-1, j], mode='same')
            CC[:, 0, i, j] = cc12[nSamples // 2:] / norm_overlap
            if i != j:
                # cross-correlation is symmetric with C(tau) = C(-tau)^T
                cc21 = cc12[::-1]
                CC[:, 0, j, i] = cc21[nSamples // 2:] / norm_overlap

    # normalize with products of std
    if norm:
        STDs = np.std(dat, axis=0)
        N = STDs[:, None] * STDs[None, :]
        CC = CC / N

    if not fullOutput:
        return CC
    else:
        return CC, lags


class ST_CrossCovariance(ComputationalRoutine):

    """
    Compute class that calculates single-trial cross-covariances
    of :class:`~syncopy.AnalogData` objects

    Sub-class of :class:`~syncopy.shared.computational_routine.ComputationalRoutine`,
    see :doc:`/developer/compute_kernels` for technical details on Syncopy's compute
    classes and metafunctions.

    See also
    --------
    syncopy.connectivityanalysis : parent metafunction
    """

    # the hard wired dimord of the cF
    dimord = ['time', 'freq', 'channel_i', 'channel_j']

    computeFunction = staticmethod(cross_covariance_cF)

    # 1st argument,the data, gets omitted
    valid_kws = list(signature(cross_covariance_cF).parameters.keys())[1:]

    def process_metadata(self, data, out):

        # Get trialdef array + channels from source: note, since lags are encoded
        # in time-axis, trial offsets etc. are bogus anyway: simply take max-sample
        # counts / 2 to fit lags
        if data._selection is not None:
            chanSec = data._selection.channel
            trl = np.ceil(data._selection.trialdefinition / 2)
        else:
            chanSec = slice(None)
            trl = np.ceil(data.trialdefinition / 2)

        # If trial-averaging was requested, use the first trial as reference
        # (all trials had to have identical lengths), and average onset timings

        if not self.keeptrials:
            trl = trl[[0], :]
            
        # set 1st entry of time axis to the 0-lag
        trl[:, 2] = 0
        out.trialdefinition = trl

        # Attach remaining meta-data
        out.samplerate = data.samplerate
        out.channel_i = np.array(data.channel[chanSec])
        out.channel_j = np.array(data.channel[chanSec])
