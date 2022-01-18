# -*- coding: utf-8 -*-
#
# computeFunctions and -Routines for parallel calculation
# of single trial measures needed for the averaged
# measures like cross spectral densities
#

# Builtin/3rd party package imports
import numpy as np

# syncopy imports
from syncopy.specest.mtmfft import mtmfft
from syncopy.shared.errors import SPYValueError
from syncopy.shared.const_def import spectralConversions


def csd(trl_dat,
        samplerate=1,
        nSamples=None,
        taper="hann",
        taper_opt=None,
        norm=False,
        fullOutput=False):

    """
    Single trial Fourier cross spectral estimates between all channels
    of the input data. First all the individual Fourier transforms
    are calculated via a (multi-)tapered FFT, then the pairwise
    cross-spectra are computed.

    Averaging over tapers is done implicitly
    for multi-taper analysis with `taper="dpss"`.

    Output consists of all (``nChannels x nChannels + 1) / 2`` different complex
    estimates arranged in a symmetric fashion (``CS_ij == CS_ji*``). The
    elements on the main diagonal (`CS_ii`) are the (real) auto-spectra.

    This is NOT the same as what is commonly referred to as
    "cross spectral density" as there is no (time) averaging!!
    Multi-tapering alone is not necessarily sufficient to get enough
    statitstical power for a robust csd estimate. Yet for completeness
    and testing the option ``norm = True`` returns a single-trial
    coherence estimate for ``taper = "dpss"``.

    Parameters
    ----------
    trl_dat : (N, K) :class:`numpy.ndarray`
        Uniformly sampled multi-channel time-series data
        The 1st dimension is interpreted as the time axis,
        columns represent individual channels.
    samplerate : float
        Samplerate in Hz
    nSamples : int or None
        Absolute length of the (potentially to be padded) signals
        or `None` for no padding (`N` is the number of samples)
    taper : str or None
        Taper function to use, one of :module:`scipy.signal.windows`
        Set to `None` for no tapering.
    taper_opt : dict, optional
        Additional keyword arguments passed to the `taper` function.
        For multi-tapering with ``taper = 'dpss'`` set the keys
        `'Kmax'` and `'NW'`.
        For further details, please refer to the
        `SciPy docs <https://docs.scipy.org/doc/scipy/reference/signal.windows.html>`_
    norm : bool, optional
        Set to `True` to normalize for a single-trial coherence measure.
        Only meaningful in a multi-taper (``taper = "dpss"``) setup and if no
        additional (trial-)averaging is performed afterwards.
    fullOutput : bool
        For backend testing or stand-alone applications, set to `True`
        to return also the `freqs` array.

    Returns
    -------
    CS_ij : (nFreq, K, K) :class:`numpy.ndarray`
        Complex cross spectra for all channel combinations ``i,j``.
        `K` corresponds to number of input channels.

    freqs : (nFreq,) :class:`numpy.ndarray`
        The Fourier frequencies if ``fullOutput = True``

    See also
    --------
    normalize_csd : :func:`~syncopy.connectivity.csd.normalize_csd`
             Coherence from trial averages

    mtmfft : :func:`~syncopy.specest.mtmfft.mtmfft`
             (Multi-)tapered Fourier analysis

    """

    # compute the individual spectra
    # specs have shape (nTapers x nFreq x nChannels)
    specs, freqs = mtmfft(trl_dat, samplerate, nSamples, taper, taper_opt)

    # outer product along channel axes
    # has shape (nTapers x nFreq x nChannels x nChannels)
    CS_ij = specs[:, :, np.newaxis, :] * specs[:, :, :, np.newaxis].conj()

    # average tapers and transpose:
    # now has shape (nChannels x nChannels x nFreq)
    CS_ij = CS_ij.mean(axis=0).T

    if norm:
        # only meaningful for multi-tapering
        if taper != 'dpss':
            msg = "Normalization of single trial csd only possible with taper='dpss'"
            raise SPYValueError(legal=msg, varname="taper", actual=taper)
        # main diagonal has shape (nChannels x nFreq): the auto spectra
        diag = CS_ij.diagonal()
        # get the needed product pairs of the autospectra
        Ciijj = np.sqrt(diag[:, :, None] * diag[:, None, :]).T
        CS_ij = CS_ij / Ciijj

    if fullOutput:
        return CS_ij.transpose(2, 0, 1), freqs
    else:
        return CS_ij.transpose(2, 0, 1)


def normalize_csd(csd_av_dat,
                  output='abs'):

    """
    Given the trial averaged cross spectral densities,
    calculates the normalizations to arrive at the
    channel x channel coherencies. If ``S_ij(f)`` is the
    averaged cross-spectrum between channel `i` and `j`, the
    coherency [1]_ is defined as:

    .. math::

          C_{ij} = S_{ij}(f) / (|S_{ii}| |S_{jj}|)

    The coherence is now defined as either ``|C_ij|``
    or ``|C_ij|^2``, this can be controlled with the `output`
    parameter.

    Parameters
    ----------
    csd_av_dat : (nFreq, N, N) :class:`numpy.ndarray`
        Averaged cross-spectral densities for `N` x `N` channels
        and `nFreq` frequencies averaged over trials.
    output : {'abs', 'pow', 'fourier'}, default: 'abs'
        After normalization the coherency is still complex (`'fourier'`);
        to get the real valued coherence ``0 < C_ij(f) < 1`` one can either take the
        absolute (`'abs'`) or the absolute squared (`'pow'`) values of the
        coherencies. The definitions are not uniform in the literature,
        hence multiple output types are supported.

    Returns
    -------
    CS_ij : (nFreq, N, N) :class:`numpy.ndarray`
        Coherence for all channel combinations ``i,j``.
        `N` corresponds to number of input channels.

    Notes
    -----
    .. [1] Nolte, Guido, et al. "Identifying true brain interaction from EEG
          data using the imaginary part of coherency."
          Clinical neurophysiology 115.10 (2004): 2292-2307.
    """
    # re-shape to (nChannels x nChannels x nFreq)
    CS_ij = csd_av_dat.transpose(1, 2, 0)

    # main diagonal has shape (nFreq x nChannels): the auto spectra
    diag = CS_ij.diagonal()

    # get the needed product pairs of the autospectra
    Ciijj = np.sqrt(diag[:, :, None] * diag[:, None, :]).T
    CS_ij = CS_ij / Ciijj

    CS_ij = spectralConversions[output](CS_ij)

    # re-shape to original form
    return CS_ij.transpose(2, 0, 1)
