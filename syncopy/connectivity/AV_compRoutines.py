# -*- coding: utf-8 -*-
#
# computeFunctions and -Routines to post-process
# the parallel single trial computations to be found in ST_compRoutines.py
# The standard use case involves computations on the
# trial average, meaning that the SyNCoPy input to these routines
# consists of only '1 trial' and parallelising over channels
# is non trivial and atm also not supported. Pre-processing
# like padding or detrending already happened in the single trial
# compute functions.
#

# Builtin/3rd party package imports
import numpy as np
from inspect import signature

# backend method imports
from .csd import normalize_csd
from .wilson_sf import wilson_sf, regularize_csd
from .granger import granger

# syncopy imports
from syncopy.shared.const_def import spectralDTypes
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.shared.kwarg_decorators import unwrap_io
from syncopy.shared.errors import (
    SPYValueError,
)


@unwrap_io
def normalize_csd_cF(csd_av_dat,
                     output='abs',
                     chunkShape=None,
                     noCompute=False):

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
    csd_av_dat : (1, nFreq, N, N) :class:`numpy.ndarray`
        Cross-spectral densities for `N` x `N` channels
        and `nFreq` frequencies averaged over trials.
    output : {'abs', 'pow', 'fourier'}, default: 'abs'
        Also after normalization the coherency is still complex (`'fourier'`),
        to get the real valued coherence ``0 < C_ij(f) < 1`` one can either take the
        absolute (`'abs'`) or the absolute squared (`'pow'`) values of the
        coherencies. The definitions are not uniform in the literature,
        hence multiple output types are supported.
    noCompute : bool
        Preprocessing flag. If `True`, do not perform actual calculation but
        instead return expected shape and :class:`numpy.dtype` of output
        array.

    Returns
    -------
    CS_ij : (1, nFreq, N, N) :class:`numpy.ndarray`
        Coherence for all channel combinations ``i,j``.
        `N` corresponds to number of input channels.

    Notes
    -----

    This method is intended to be used as
    :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
    inside a :class:`~syncopy.shared.computational_routine.ComputationalRoutine`.
    Thus, input parameters are presumed to be forwarded from a parent metafunction.
    Consequently, this function does **not** perform any error checking and operates
    under the assumption that all inputs have been externally validated and cross-checked.

    .. [1] Nolte, Guido, et al. "Identifying true brain interaction from EEG
          data using the imaginary part of coherency."
          Clinical neurophysiology 115.10 (2004): 2292-2307.


    See also
    --------
    cross_spectra_cF : :func:`~syncopy.connectivity.ST_compRoutines.cross_spectra_cF`
             Single trial (Multi-)tapered cross spectral densities.

    """

    # it's the same as the input shape!
    outShape = csd_av_dat.shape

    # For initialization of computational routine,
    # just return output shape and dtype
    if noCompute:
        return outShape, spectralDTypes[output]

    CS_ij = normalize_csd(csd_av_dat[0], output)

    # re-attach dummy time axis
    return CS_ij[None, ...]


class NormalizeCrossSpectra(ComputationalRoutine):

    """
    Compute class that normalizes trial averaged csd's
    of :class:`~syncopy.CrossSpectralData` objects
    to arrive at the respective coherencies.

    Sub-class of :class:`~syncopy.shared.computational_routine.ComputationalRoutine`,
    see :doc:`/developer/compute_kernels` for technical details on Syncopy's compute
    classes and metafunctions.

    See also
    --------
    syncopy.connectivityanalysis : parent metafunction
    """

    # the hard wired dimord of the cF
    dimord = ['time', 'freq', 'channel_i', 'channel_j']

    computeFunction = staticmethod(normalize_csd_cF)

    method = "" # there is no backend
    # 1st argument,the data, gets omitted
    valid_kws = list(signature(normalize_csd_cF).parameters.keys())[1:]

    def pre_check(self):
        '''
        Make sure we have a trial average,
        so the input data only consists of `1 trial`.
        Can only be performed after initialization!
        '''

        if self.numTrials is None:
            lgl = 'Initialize the computational Routine first!'
            act = 'ComputationalRoutine not initialized!'
            raise SPYValueError(legal=lgl, varname=self.__class__.__name__, actual=act)

        if self.numTrials != 1:
            lgl = "1 trial: normalizations can only be done on averaged quantities!"
            act = f"DataSet contains {self.numTrials} trials"
            raise SPYValueError(legal=lgl, varname="data", actual=act)

    def process_metadata(self, data, out):

        # Some index gymnastics to get trial begin/end "samples"
        if data._selection is not None:
            chanSec_i = data._selection.channel_i
            chanSec_j = data._selection.channel_j
            trl = data._selection.trialdefinition
            for row in range(trl.shape[0]):
                trl[row, :2] = [row, row + 1]
        else:
            chanSec_i = slice(None)
            chanSec_j = slice(None)
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
        out.channel_i = np.array(data.channel_i[chanSec_i])
        out.channel_j = np.array(data.channel_j[chanSec_j])
        out.freq = data.freq


@unwrap_io
def normalize_ccov_cF(trl_av_dat,
                      chunkShape=None,
                      noCompute=False):

    """
    Given the trial averaged cross-covariances,
    we normalize with the 0-lag auto-covariances
    (~averaged single trial variances)
    to arrive at the cross-correlations.

    Parameters
    ----------
    trl_av_dat : (nLag, 1, N, N) :class:`numpy.ndarray`
        Cross-covariances for `N` x `N` channels
        and `nLag` epochs averaged over trials.
    noCompute : bool
        Preprocessing flag. If `True`, do not perform actual calculation but
        instead return expected shape and :class:`numpy.dtype` of output
        array.

    Returns
    -------
    Corr_ij : (nLag, 1, N, N) :class:`numpy.ndarray`
        Cross-correlations for all channel combinations ``i,j``.
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
    cross_covariance_cF : :func:`~syncopy.connectivity.ST_compRoutines.cross_covariance_cF`
             Single trial cross covariances.

    """

    # it's the same as the input shape!
    outShape = trl_av_dat.shape

    # For initialization of computational routine,
    # just return output shape and dtype
    # cross spectra are complex!
    if noCompute:
        return outShape, spectralDTypes['abs']

    # re-shape to (nLag x nChannels x nChannels)
    CCov_ij = trl_av_dat[:, 0, ...]

    # main diagonal has shape (nChannels x nChannels):
    # the auto-covariances at 0-lag (~stds)
    diag = trl_av_dat[0, 0, ...].diagonal()

    # get the needed product pairs
    Ciijj = np.sqrt(diag[:, None] * diag[None, :]).T
    CCov_ij = CCov_ij / Ciijj

    # re-attach dummy freq axis
    return CCov_ij[:, None, ...]


class NormalizeCrossCov(ComputationalRoutine):

    """
    Compute class that normalizes trial averaged
    cross-covariances of :class:`~syncopy.CrossSpectralData` objects
    to arrive at the respective correlations

    Sub-class of :class:`~syncopy.shared.computational_routine.ComputationalRoutine`,
    see :doc:`/developer/compute_kernels` for technical details on Syncopy's compute
    classes and metafunctions.

    See also
    --------
    syncopy.connectivityanalysis : parent metafunction
    """

    # the hard wired dimord of the cF
    dimord = ['time', 'freq', 'channel_i', 'channel_j']

    computeFunction = staticmethod(normalize_ccov_cF)

    method = "" # there is no backend
    # 1st argument,the data, gets omitted
    valid_kws = list(signature(normalize_ccov_cF).parameters.keys())[1:]

    def pre_check(self):
        '''
        Make sure we have a trial average,
        so the input data only consists of `1 trial`.
        Can only be performed after initialization!
        '''

        if self.numTrials is None:
            lgl = 'Initialize the computational Routine first!'
            act = 'ComputationalRoutine not initialized!'
            raise SPYValueError(legal=lgl, varname=self.__class__.__name__, actual=act)

        if self.numTrials != 1:
            lgl = "1 trial: normalizations can only be done on averaged quantities!"
            act = f"DataSet contains {self.numTrials} trials"
            raise SPYValueError(legal=lgl, varname="data", actual=act)

    def process_metadata(self, data, out):

        # Get trialdef array + channels from source
        if data._selection is not None:
            chanSec_i = data._selection.channel_i
            chanSec_j = data._selection.channel_j
            trl = data._selection.trialdefinition
        else:
            chanSec_i = slice(None)
            chanSec_j = slice(None)
            trl = data.trialdefinition

        out.trialdefinition = trl
        # Attach remaining meta-data
        out.samplerate = data.samplerate
        out.channel_i = np.array(data.channel_i[chanSec_i])
        out.channel_j = np.array(data.channel_j[chanSec_j])


@unwrap_io
def granger_cF(csd_av_dat,
               rtol=1e-8,
               nIter=100,
               cond_max=1e4,
               chunkShape=None,
               noCompute=False):

    """
    Given the trial averaged cross spectral densities,
    calculates the pairwise Granger-Geweke causalities
    for all (non-symmetric!) channel combinations
    following the algorithm proposed in [1]_.

    First the CSD matrix is factorized using Wilson's
    algorithm, the resulting transfer functions and
    noise covariance matrix is then used to calculate
    Granger causality according to Eq. 8 in [1]_.

    Selection of channels and frequencies of interest
    can and should be done beforehand when calculating the CSDs.

    Critical numerical parameters for Wilson's algorithm
    (`rtol`, `nIter`, `cond_max`) have sensitive defaults,
    which were tested for datasets with up to
    5000 samples and 256 channels. Changing them is
    recommended for expert users only.

    Parameters
    ----------
    csd_av_dat : (1, nFreq, N, N) :class:`numpy.ndarray`
        Cross-spectral densities for `N` x `N` channels
        and `nFreq` frequencies averaged over trials.
    rtol : float
        Relative error tolerance for Wilson's algorithm
        for spectral matrix factorization. Default should
        be fine for most cases, handle with care!
    nIter : int
        Maximum number of iterations for CSD factorization. A result
        is returned if exhausted also if error tolerance was not met.
    cond_max : float
        The maximal condition number of the spectral matrix.
        The CSD matrix can be almost singular in cases of many channels and
        low sample number. In these cases Wilson's factorization fails
        to converge, as it relies on positive definiteness of the CSD matrix.
        If the condition number is above `cond_max`, a brute force
        regularization is performed until the regularized CSD matrix has a
        condition number below `cond_max`.
    noCompute : bool
        Preprocessing flag. If `True`, do not perform actual calculation but
        instead return expected shape and :class:`numpy.dtype` of output
        array.

    Returns
    -------
    Granger : (1, nFreq, N, N) :class:`numpy.ndarray`
        Spectral Granger-Geweke causality between all channel
        combinations. Directionality follows array
        notation: causality from ``i -> j`` is ``Granger[0,:,i,j]``,
        causality from ``j -> i`` is ``Granger[0,:,j,i]``

    Notes
    -----

    This method is intended to be used as
    :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
    inside a :class:`~syncopy.shared.computational_routine.ComputationalRoutine`.
    Thus, input parameters are presumed to be forwarded from a parent metafunction.
    Consequently, this function does **not** perform any error checking and operates
    under the assumption that all inputs have been externally validated and cross-checked.

    .. [1] Dhamala, Mukeshwar, Govindan Rangarajan, and Mingzhou Ding.
       "Estimating Granger causality from Fourier and wavelet transforms
        of time series data." Physical review letters 100.1 (2008): 018701.

    See also
    --------
    cross_spectra_cF : :func:`~syncopy.connectivity.ST_compRoutines.cross_spectra_cF`
             Single trial (Multi-)tapered cross spectral densities. Trial averages
             can be obtained by calling the respective computational routine
             with `keeptrials=False`.
    wilson_sf : :func:`~syncopy.connectivity.wilson_sf.wilson_sf
             Spectral matrix factorization that yields the
             transfer functions and noise covariances
             from a cross spectral density.
    regularize_csd : :func:`~syncopy.connectivity.wilson_sf.regularize_csd
             Brute force regularization scheme for the CSD matrix
    granger : :func:`~syncopy.connectivity.granger.granger
            Given the results of the spectral matrix
            factorization, calculates the granger causalities
    """

    # it's the same as the input shape!
    outShape = csd_av_dat.shape

    # For initialization of computational routine,
    # just return output shape and dtype
    # Granger causalities are real
    if noCompute:
        return outShape, spectralDTypes['abs']

    # strip off singleton time dimension
    # for the backend calls
    CSD = csd_av_dat[0]

    # auto-regularize to `cond_max` condition number
    # maximal regularization factor is 1e-3, raises a ValueError
    # if this is not enough!
    CSDreg, factor = regularize_csd(CSD, cond_max=cond_max, eps_max=1e-3)
    # call Wilson
    H, Sigma, conv = wilson_sf(CSDreg, nIter=nIter, rtol=rtol)

    # calculate G-causality
    Granger = granger(CSDreg, H, Sigma)

    # reattach dummy time axis
    return Granger[None, ...]


class GrangerCausality(ComputationalRoutine):

    """
    Compute class that computes pairwise Granger causalities
    of :class:`~syncopy.CrossSpectralData` objects.

    Sub-class of :class:`~syncopy.shared.computational_routine.ComputationalRoutine`,
    see :doc:`/developer/compute_kernels` for technical details on Syncopy's compute
    classes and metafunctions.

    See also
    --------
    syncopy.connectivityanalysis : parent metafunction
    """

    # the hard wired dimord of the cF
    dimord = ['time', 'freq', 'channel_i', 'channel_j']

    computeFunction = staticmethod(granger_cF)

    method = "" # there is no backend
    # 1st argument,the data, gets omitted
    valid_kws = list(signature(granger_cF).parameters.keys())[1:]

    def pre_check(self):
        '''
        Make sure we have a trial average,
        so the input data only consists of `1 trial`.
        Can only be performed after initialization!
        '''

        if self.numTrials is None:
            lgl = 'Initialize the computational Routine first!'
            act = 'ComputationalRoutine not initialized!'
            raise SPYValueError(legal=lgl, varname=self.__class__.__name__, actual=act)

        if self.numTrials != 1:
            lgl = "1 trial: Granger causality can only be computed on trial averages!"
            act = f"DataSet contains {self.numTrials} trials"
            raise SPYValueError(legal=lgl, varname="data", actual=act)

    def process_metadata(self, data, out):

        # Some index gymnastics to get trial begin/end "samples"
        if data._selection is not None:
            chanSec_i = data._selection.channel_i
            chanSec_j = data._selection.channel_j
            trl = data._selection.trialdefinition
            for row in range(trl.shape[0]):
                trl[row, :2] = [row, row + 1]
        else:
            chanSec_i = slice(None)
            chanSec_j = slice(None)
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
        out.channel_i = np.array(data.channel_i[chanSec_i])
        out.channel_j = np.array(data.channel_j[chanSec_j])
        out.freq = data.freq
