# -*- coding: utf-8 -*-
#
# computeFunctions and -Routines for parallel calculation
# of single trial measures needed for the averaged
# measures like cross spectral densities
#

# Builtin/3rd party package imports
import numpy as np
import h5py
from scipy.signal import fftconvolve, detrend
from inspect import signature
from hashlib import blake2b

# backend method imports
from .csd import csd

# syncopy imports
from syncopy.shared.const_def import spectralDTypes
from syncopy.shared.tools import best_match
from syncopy.shared.computational_routine import ComputationalRoutine, propagate_properties
from syncopy.shared.metadata import metadata_from_hdf5_file, check_freq_hashes
from syncopy.shared.kwarg_decorators import process_io


@process_io
def spectral_dyadic_product_cF(specs,
                               chunkShape=None,
                               noCompute=False):
    """
    Single trial cross spectra directly from complex power spectra,
    hence no Fourier transforms are needed and all what is
    left to do is to take the outer product along the channel axis.

    In case the spectral input has a taper axis, those get averaged
    out after the cross products are calculated.

    Parameters
    ----------
    specs: (K, M, L, N) :class:`numpy.ndarray`
        Complex and time and frequency aligned multi-channel single-trial spectral data.
        The 3rd dimension is interpreted as the frequency axis, `M` is the number
        of tapers used. `N` columns represent individual channels.
    noCompute : bool
        Preprocessing flag. If `True`, do not perform actual calculation but
        instead return expected shape and :class:`numpy.dtype` of output
        array.

    Returns
    -------
    CS_ij : (K, L, N, N) :class:`numpy.ndarray` with complex dtype
        Complex cross spectra for all channel combinations ``i,j``.
        `K` is the number  of samples, `L` is number of frequencies
        and `N` corresponds to number of input channels.

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
    normalize_csd : :func:`~syncopy.connectivity.csd.normalize_csd`
             Coherence from trial averages

    """

    # default dimord for SpectralData is ['time', 'taper', 'freq', 'channel']
    nTime = specs.shape[0]
    nFreq = specs.shape[2]
    nChannels = specs.shape[3]

    # we always average over tapers here
    outShape = (nTime, nFreq, nChannels, nChannels)

    # cross spectra are complex, input gets checked in frontend!
    if noCompute:
        return outShape, spectralDTypes["fourier"]

    # dyadic product along channel axes
    # result has shape (nTime, nTapers x nFreq x nChannels x nChannels)
    CS_ij = specs[..., np.newaxis] * specs[..., np.newaxis, :].conj()

    # now average tapers
    # result has shape (nTime x nFreq x nChannels x nChannels)
    CS_ij = CS_ij.mean(axis=1)
    return CS_ij


class SpectralDyadicProduct(ComputationalRoutine):

    """
    Compute class that calculates single-trial cross spectra
    from :class:`~syncopy.SpectralData` objects, which in the end
    is just the dyadic (outer) product between channels.
    For coherence computation, `keeptrials` is set to `False` to right away
    average the single-trial cross-spectra

    Sub-class of :class:`~syncopy.shared.computational_routine.ComputationalRoutine`,
    see :doc:`/developer/compute_kernels` for technical details on Syncopy's compute
    classes and metafunctions.

    See also
    --------
    syncopy.connectivityanalysis : parent metafunction
    """

    # The hard wired dimord of the cF
    dimord = ['time', 'freq', 'channel_i', 'channel_j']

    computeFunction = staticmethod(spectral_dyadic_product_cF)

    # 1st argument, the data, gets omitted.
    valid_kws = list(signature(spectral_dyadic_product_cF).parameters.keys())[1:]
    valid_kws += ['output']

    def process_metadata(self, data, out):

        time_axis = np.any(np.diff(data.trialdefinition)[:,0] != 1)
        propagate_properties(data, out, self.keeptrials, time_axis)
        out.freq = data.freq


@process_io
def ppc_column_cF(cross_spectrum,
                  trl2_idx=None,
                  hdf5_path=None,
                  chunkShape=None,
                  noCompute=False):
    """
    The PPC involves computations on all nTrials(nTrials-1) pairs. This compute function
    in combination with the PPC CR can be used to compute one column, consisting
    of nTrials-1 pairs (and single diagonal entry), of the implicit
    nTrials x nTrials matrix of trial pairs.

    We express the dot product of equation 14 of the source publication
    (Vinck 2010) between the two unit vectors associated to the relative phases
    via the cosine and the conjugate product of the complex
    cross-spectral entries of two trials j and k:

    .. math::

          z_j = r_j e^(i \theta_j)

    with

    .. math::

        arg(z_j) = \theta_j

    we can write :

    .. math::

          f(\theta_j, \theta_k) = cos(arg(z_j z_k^*))

    Where `f()` denotes the dot product as in the source publication, equation 14.
    Note that no normalization to unit vectors is needed,
    as `arg` directly extracts the angular distance from the conjugate product,
    irrespective of the norms of the z's.

    Parameters
    ----------
    cross_spectrum : (K, L, N, N) :class:`numpy.ndarray` with complex dtype
        A single trial cross spectrum, the 1st trial of the pair
    trl2_idx : tuple
        A tuple of indices pointing to the 2nd trial of the pair
    hdf5_path : str
        Path to the backing hdf5 dataset, providing the 2nd trial via `trl2_idx`
    noCompute : bool
        Preprocessing flag. If `True`, do not perform actual calculation but
        instead return expected shape and :class:`numpy.dtype` of output
        array.

    Returns
    -------
    ppc_ij : (K, L, N, N) :class:`numpy.ndarray` with complex dtype
        Dot product for a single cross-spectral trial pair i, j.
        `K` is the number  of samples, `L` is number of frequencies
        and `N` corresponds to number of channels.

    """

    # the shape does not change
    outShape = cross_spectrum.shape

    # ppc spectra are real
    if noCompute:
        return outShape, spectralDTypes["abs"]

    # get the 2nd trial as array
    with h5py.File(hdf5_path, "r") as h5file:
        cross_spectrum2 = h5file['data'][trl2_idx]

    # first compute the conjugate product for the complete cross spectra
    # (all channels, all freqs and even time if applicable)
    ppc_ij = cross_spectrum * cross_spectrum2.conj()

    # now all what remains is to extract the angular distance
    # and take the cosine
    ppc_ij = np.cos(np.angle(ppc_ij))

    return ppc_ij


class PPC_column(ComputationalRoutine):

    """
    Compute class that computes the dotproduct between nTrials-1 cross-spectral trial pairs
    by streaming as usual through all trials available, and pairing it with one
    fixed 2nd trial (controlled by `trl2_idx`).

    Sub-class of :class:`~syncopy.shared.computational_routine.ComputationalRoutine`,
    see :doc:`/developer/compute_kernels` for technical details on Syncopy's compute
    classes and metafunctions.

    See also
    --------
    syncopy.connectivityanalysis : parent metafunction
    """

    computeFunction = staticmethod(ppc_column_cF)

    # 1st argument,the data, gets omitted
    valid_kws = list(signature(ppc_column_cF).parameters.keys())[1:]

    def process_metadata(self, data, out):

        time_axis = np.any(np.diff(data.trialdefinition)[:, 0] != 1)

        propagate_properties(data, out, self.keeptrials, time_axis)
        out.freq = data.freq

        # same data type, so trivial propagation
        propagate_properties(data, out, self.keeptrials, time_axis)



@process_io
def cross_spectra_cF(trl_dat,
                     samplerate=1,
                     nSamples=None,
                     foi=None,
                     taper="hann",
                     taper_opt=None,
                     demean_taper=False,
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
    "cross spectral density" as there is no (time) averaging!
    Multi-tapering alone is not necessarily sufficient to get enough
    statistical power for a robust csd estimate. Still, for completeness
    and testing the option `norm=True`, this function does output a single-trial
    coherence estimate.

    Parameters
    ----------
    trl_dat : (K, N) :class:`numpy.ndarray`
        Uniformly sampled multi-channel time-series data.
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
        Taper function to use, one of `scipy.signal.windows`.
        Set to `None` for no tapering.
    taper_opt : dict, optional
        Additional keyword arguments passed to the `taper` function.
        For multi-tapering with `taper='dpss'` set the keys
        `'Kmax'` and `'NW'`.
        For further details, please refer to the
        `SciPy docs <https://docs.scipy.org/doc/scipy/reference/signal.windows.html>`_
    demean_taper : bool
        Set to `True` to perform de-meaning after tapering
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
    For time dependent cross-spectra, first compute a time-frequency spectrum
    via :func:`~syncopy.freqanalysis` and then directly use
    :func:`~syncopy.connectivity.ST_compRoutines.spectral_dyadic_product_cF`

    This method is intended to be used as
    :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
    inside a :class:`~syncopy.shared.computational_routine.ComputationalRoutine`.
    Thus, input parameters are presumed to be forwarded from a parent metafunction.
    Consequently, this function does **not** perform any error checking and operates
    under the assumption that all inputs have been externally validated and cross-checked.

    See also
    --------
    spectral_dyadic_product_cF : :func:`~syncopy.connectivity.ST_compRoutines.spectral_dyadic_product_cF`
             Cross spectra from spectral input
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

    CS_ij, freqs = csd(dat,
                       samplerate,
                       nSamples,
                       taper=taper,
                       taper_opt=taper_opt,
                       demean_taper=demean_taper)

    # Hash the freqs and add to second return value.
    freqs_hash = blake2b(freqs).hexdigest().encode('utf-8')
    metadata = {'freqs_hash': np.array(freqs_hash)}  # Will have dtype='|S128'

    return CS_ij[np.newaxis, freq_idx, ...], metadata


class CrossSpectra(ComputationalRoutine):

    """
    Compute class that calculates single-trial (multi-)tapered cross spectra
    of :class:`~syncopy.AnalogData` objects.
    For coherence computation, `keeptrials` is set to `False` to right away
    average the single-trial cross-spectra.

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

    # 1st argument,the data, gets omitted
    valid_kws = list(signature(cross_spectra_cF).parameters.keys())[1:]
    # hardcode some parameter names which got digested from the frontend
    valid_kws += ['tapsmofrq', 'nTaper', 'pad', 'output']

    def process_metadata(self, data, out):

        propagate_properties(data, out, self.keeptrials)

        # General-purpose loading of metadata.
        metadata = metadata_from_hdf5_file(out.filename)
        check_freq_hashes(metadata, out)

        out.freq = self.cfg['foi']


@process_io
def cross_covariance_cF(trl_dat,
                        samplerate=1,
                        polyremoval=0,
                        timeAxis=0,
                        norm=False,
                        fullOutput=False,
                        chunkShape=None,
                        noCompute=False):

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


class CrossCovariance(ComputationalRoutine):

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
        # in time-axis, trial offsets etc. are bogus anyways: simply take max-sample
        # counts / 2 to fit lags
        if data.selection is not None:
            chanSec = data.selection.channel
            old_trldef = data.selection.trialdefinition
        else:
            chanSec = slice(None)
            old_trldef = data.trialdefinition

        # these are the new trial sizes:
        # max lag is half the available sample number
        trl_sizes = np.ceil(np.diff(old_trldef, axis=1)[:, 0] / 2)
        # create sampleinfo
        si = np.r_[0, np.cumsum(trl_sizes)]
        sampleinfo = np.column_stack([si[:-1], si[1:]])
        # set offset to 0 (the 0-lag)
        trl_def = np.column_stack([sampleinfo, np.zeros(len(trl_sizes), dtype=np.float32)])

        # If trial-averaging was requested, use the first trial as reference
        # trials had to have the same shape
        if not self.keeptrials:
            trl_def = trl_def[[0], :]

        out.trialdefinition = trl_def

        # Attach remaining meta-data
        out.samplerate = data.samplerate
        out.channel_i = np.array(data.channel[chanSec])
        out.channel_j = np.array(data.channel[chanSec])
