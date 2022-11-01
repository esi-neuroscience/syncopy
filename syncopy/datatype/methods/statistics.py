# -*- coding: utf-8 -*-
#
# Syncopy object simple statistics (mean, std, ...)
#


# Builtin/3rd party package imports
import numpy as np
import h5py

# Local imports
# from .selectdata import _get_selection_size
from syncopy.shared.parsers import data_parser
from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYWarning, SPYInfo
from syncopy.shared.computational_routine import ComputationalRoutine, propagate_properties
from syncopy.shared.kwarg_decorators import process_io, unwrap_select, detect_parallel_client

__all__ = ['average', 'std', 'var', 'median']

@unwrap_select
@detect_parallel_client
def average(spy_data, dim, keeptrials=True, **kwargs):
    """
    Calculates the average along arbitrary
    dimensions as long as the selected ``dim`` is contained
    within the ``dimord`` of the Syncopy object ``spy_data``.

    Additional trial averaging can be performed with ``keeptrials=False``.
    Standalone (only) trial averaging can only be done sequentially
    and requires ``dim='trials'``.

    Parameters
    ----------
    spy_data : Syncopy data object
        The object where an average is to be computed
    dim : str
        Dimension label over which to calculate the statistic.
        Must be present in the ``spy_data`` object,
        e.g. 'channel' or 'trials'
    keeptrials : bool
        Set to ``False`` to trigger additional trial averaging

    Returns
    -------
    res : Syncopy data object
        New object with the desired dimension averaged out

    """

    # call general backend function with desired operation
    return _statistics(spy_data,
                       operation='mean',
                       dim=dim,
                       keeptrials=keeptrials,
                       **kwargs)


@unwrap_select
@detect_parallel_client
def std(spy_data, dim, keeptrials=True, **kwargs):
    """
    Calculates the standard deviation along arbitrary
    dimensions as long as the selected ``dim`` is contained
    within the ``dimord`` of the Syncopy object ``spy_data``.

    Additional trial averaging can be performed with ``keeptrials=False``
    after the standard deviation got calculated.

    Parameters
    ----------
    spy_data : Syncopy data object
        The object where a standard deviation is to be computed
    dim : str
        Dimension label over which to calculate the statistic.
        Must be present in the ``spy_data`` object,
        e.g. 'channel' or 'trials'
    keeptrials : bool
        Set to ``False`` to trigger additional trial averaging

    Returns
    -------
    res : Syncopy data object
        New object with the desired dimension averaged out

    """

    # call general backend function with desired operation
    return _statistics(spy_data,
                       operation='std',
                       dim=dim,
                       keeptrials=keeptrials,
                       **kwargs)


@unwrap_select
@detect_parallel_client
def var(spy_data, dim, keeptrials=True, **kwargs):
    """
    Calculates the variance along arbitrary
    dimensions as long as the selected ``dim`` is contained
    within the ``dimord`` of the Syncopy object ``spy_data``.

    Additional trial averaging can be performed with ``keeptrials=False``
    after the variance got calculated.

    Parameters
    ----------
    spy_data : Syncopy data object
        The object where a variance is to be computed
    dim : str
        Dimension label over which to calculate the statistic.
        Must be present in the ``spy_data`` object,
        e.g. 'channel' or 'trials'
    keeptrials : bool
        Set to ``False`` to trigger additional trial averaging

    Returns
    -------
    res : Syncopy data object
        New object with the desired dimension averaged out

    """

    # call general backend function with desired operation
    return _statistics(spy_data,
                       operation='var',
                       dim=dim,
                       keeptrials=keeptrials,
                       **kwargs)


@unwrap_select
@detect_parallel_client
def median(spy_data, dim, keeptrials=True, **kwargs):
    """
    Calculates the median along arbitrary
    dimensions as long as the selected ``dim`` is contained
    within the ``dimord`` of the Syncopy object ``spy_data``.

    Additional trial averaging can be performed with ``keeptrials=False``
    after the median got calculated.

    Parameters
    ----------
    spy_data : Syncopy data object
        The object where a median is to be computed
    dim : str
        Dimension label over which to calculate the statistic.
        Must be present in the ``spy_data`` object,
        e.g. 'channel' or 'trials'
    keeptrials : bool
        Set to ``False`` to trigger additional trial averaging

    Returns
    -------
    res : Syncopy data object
        New object with the desired dimension averaged out

    """

    # call general backend function with desired operation
    return _statistics(spy_data,
                       operation='median',
                       dim=dim,
                       keeptrials=keeptrials,
                       **kwargs)

def _statistics(spy_data, operation, dim, keeptrials=True, **kwargs):

    """
    Entry point to calculate simple statistics (mean, std, ...) along arbitrary
    dimensions as long as the selected ``dim`` is contained
    within the ``dimord`` of the ``spy_data``.

    Additional trial averaging can be performed with ``keeptrials=False``.
    Standalone (only) trial averaging can only be done sequentially
    and requires ``dim='trials'``.

    Parameters
    ----------
    spy_data : Syncopy data object
        The object where an average is to be computed
    operation : {'mean', 'std', 'var', 'median'}
        The statistical operation to perform
    dim : str
        Dimension label over which to calculate the statistic.
        Must be present in the ``spy_data`` object,
        e.g. 'channel' or 'trials'

    Returns
    -------
    res : Syncopy data object
        New object with the desired dimension averaged out

    """

    log_dict = {'input': spy_data.filename,
                'operation': operation,
                'dim': dim,
                'keeptrials': keeptrials}

    # If no active selection is present, create a "fake" all-to-all selection
    # to harmonize processing down the road (and attach `_cleanup` attribute for later removal)
    if spy_data.selection is None:
        spy_data.selectdata(inplace=True)
        spy_data.selection._cleanup = True

    # trial statistics
    if dim == 'trials':
        raise NotImplementedError

    # any other average
    else:
        if dim not in spy_data.dimord:
            lgl = f"one of {spy_data.dimord}"
            act = dim
            raise SPYValueError(lgl, 'dim', act)

        chan_per_worker = kwargs.get('chan_per_worker')
        if chan_per_worker is not None and 'channel' in dim:
            msg = "Parallelization over channels not possible for channel averages"
            SPYWarning(msg)
            chan_per_worker=None

        # log also possible selections
        log_dict['selection'] = getattr(spy_data.selection, dim)

        axis = spy_data.dimord.index(dim)
        avCR = NumpyStatDim(operation=operation, axis=axis)

    # ---------------------------------
    # Initialize output and call the CR
    # ---------------------------------

    # initialize output object of same datatype
    out = spy_data.__class__(dimord=spy_data.dimord)

    avCR.initialize(spy_data, spy_data._stackingDim,
                    keeptrials=keeptrials, chan_per_worker=chan_per_worker)
    avCR.compute(spy_data, out, parallel=kwargs.get("parallel"), log_dict=log_dict)

    # revert helper all-to-all selection
    if hasattr(spy_data.selection, '_cleanup'):
        spy_data.selection = None

    return out

@process_io
def npstats_cF(trl_dat, operation='mean', axis=0, noCompute=False, chunkShape=None):

    """
    Numpy simple statistics on single-trial arrays along indicated `axis`.

    Parameters
    ----------
    trl_dat : :class:`numpy.ndarray`
        Single trial data of arbitrary dimension
    operation : {'mean', 'std', 'var', 'median'}
        The statistical operation to perform
    axis : int
        The axis over which to calulate the average

    Returns
    -------
    res : :class:`numpy.ndarray`
        Result of the average

    Notes
    -----
    This method is intended to be used as :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
    inside a :class:`~syncopy.shared.computational_routine.ComputationalRoutine`.
    Thus, input parameters are presumed to be forwarded from a parent metafunction.
    Consequently, this function does **not** perform any error checking and operates
    under the assumption that all inputs have been externally validated and cross-checked.
    """

    if noCompute:
        # initialize result array
        out_shape = list(trl_dat.shape)
        out_shape[axis] = 1   # this axis will be summed over

        return out_shape, trl_dat.dtype

    return NumpyStatDim.methods[operation](trl_dat, axis=axis, keepdims=True)


class NumpyStatDim(ComputationalRoutine):

    """
    Simple compute class which applies basic numpy statistical funtions
    along  a ``dim`` of a Syncopy data object. The resulting Syncopy object
    is of the same type as the input object, having one of its
    dimensions reduced to a singleton due to the operation.

    Notes
    -----
    If ``keeptrials`` is set to False, a trial average is additionally computed
    as in any other CR. For a standalone trial average use the AverageTrials CR.
    """

    methods = {'mean': np.mean,
               'std': np.std,
               'var': np.var,
               'median': np.median
               }

    computeFunction = staticmethod(npstats_cF)

    def process_metadata(self, in_data, out_data):

        # the dimension over which the statistic got computed
        dim = in_data.dimord[self.cfg['axis']]

        out_data.samplerate = in_data.samplerate

        # Get/set timing-related selection modifiers
        # We've set a fallback all-to-all selection in any case

        # time axis really gone, only one trial and time got averaged out
        if dim == 'time' and not self.keeptrials:
            trldef = np.array([[0, 1, 0]])

        # trial average, needs equal trial lengths.. just copy from 1st
        elif dim != 'time' and not self.keeptrials:
            trldef = in_data.trialdefinition[0, :][None, :]

        # each trial has empty time axis, so we attach trivial trialdefinition:
        # 1 sample per trial for stacking
        elif dim == 'time' and self.keeptrials:
            nTrials = len(in_data.selection.trials)
            stacking_time = np.arange(nTrials)[:, None]
            trldef = np.hstack((stacking_time, stacking_time + 1,
                               np.zeros((nTrials, 1))))

        # nothing happened on the time axis
        else:
            trldef = in_data.selection.trialdefinition

        out_data.trialdefinition = trldef

        # Get/set dimensional attributes changed by selection
        for prop in in_data.selection._dimProps:
            selection = getattr(in_data.selection, prop)
            if selection is not None:
                # propagate without change
                if not dim in prop:
                    if np.issubdtype(type(selection), np.number):
                        selection = [selection]
                    setattr(out_data, prop, getattr(in_data, prop)[selection])
                # set to singleton or None
                else:
                    # numerical freq axis is gone after averaging
                    if dim == 'freq':
                        out_data.freq = None
                    else:
                        setattr(out_data, prop, [self.cfg['operation']])


def _attach_stat_doc(orig_doc):
    """
    This is a helper to attach the full doc to the statistical methods in ContinuousData.
    Including the `select` and `parallel` sections from the kwarg decorators,
    which can/should only be applied once.
    """

    # the wrapper
    def _attach_doc(func):
        # delete the `spy_data` entries which
        # are not needed (got self)
        doc = orig_doc.replace(' ``spy_data``', '')
        idx1 = doc.find('spy_data')
        idx2 = doc.find('dim', idx1)
        doc = doc[:idx1] + doc[idx2:]

        func.__doc__ = doc
        return func

    return _attach_doc


def seq_contraction(dataset, axis, component_op = None):
    """

    THIS IS A STUB - would be needed if more memory efficiency is needed (all-to-all trialdef)

    Sequential tensor contraction of D dimensional hdf5 datasets along the indicated axis.
    Working principle is to stream sequentially each required D-1 dimensional component,
    apply ``component_op`` (e.i. squaring for std) and adding up slice by slice..

    Note that due to the mixing of trial and time axis within syncopy's datasets,
    this function assumes to operate on single trials. Trial averaging itself
    uses the `AverageTrials` CR.
    """

    # initialize result array
    shape_idx = np.ones(dataset.ndim, dtype=bool)
    shape_idx[axis] = False   # this axis will be summed over
    res = np.zeros(np.array(dataset.shape[shape_idx]))

    # the number of summations along the contracting axis
    nSum =dataset.shape[axis]

    # identity operation
    if component_op is None:
        component_op = lambda x: x

    for idx in range(nSum):
        res += dataset[()]   # correct slcing is missing
