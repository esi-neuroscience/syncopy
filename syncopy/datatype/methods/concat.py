# -*- coding: utf-8 -*-
#
# Syncopy object concatenation
#

# Builtin/3rd party package imports
import numpy as np
import h5py
import dask.distributed as dd

# Local imports
from syncopy import __acme__
from syncopy.datatype.continuous_data import ContinuousData
from syncopy.shared.parsers import data_parser
from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYWarning, SPYInfo
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.shared.kwarg_decorators import process_io, detect_parallel_client

import dask.distributed as dd

__all__ = ["concat"]


def concat(spy_obj1, spy_obj2, dim="channel"):
    """
    Concatenate two Syncopy data objects trial-by-trial along `dim` axis.

    Only `channel` axis supported at the moment!

    Parameters
    ----------
    spy_obj1 : Syncopy data object
    spy_obj1 : Syncopy data object
    dim : str
       The concatenation dimension
    """

    # -- sanity checks --

    if not issubclass(spy_obj1.__class__, ContinuousData):
        raise SPYTypeError(spy_obj1, "spy_obj1", "continuous data type")

    data_parser(spy_obj1, empty=False)
    data_parser(spy_obj2, empty=False)

    # catches also differing classes
    if spy_obj1.dimord != spy_obj2.dimord:
        raise SPYValueError(
            "objects with equal dimensional layout",
            "spy object dimensions",
            f"{spy_obj1.dimord} and {spy_obj2.dimord}",
        )

    if dim not in spy_obj1.dimord:
        raise SPYValueError(
            f"object which has a `{dim}` dimension",
            "spy_obj1.dimord",
            f"{spy_obj1.dimord}",
        )

    if dim != "channel":
        raise NotImplementedError("Only `channel` concatenation supported atm")

    concat_axis = spy_obj1.dimord.index(dim)

    # check shapes
    shape_zip = zip(spy_obj1.data.shape, spy_obj2.data.shape)
    for axis, (s1, s2) in enumerate(shape_zip):
        if axis != concat_axis and s1 != s2:
            raise SPYValueError(
                "objects with matching shapes",
                "spy objects",
                f"{spy_obj1.data.shape}  and {spy_obj2.data.shape}",
            )

    # new size of the concatenation axis/dimension
    # this works for `time`, `channel`, `freq` and even `trials`
    new_size = len(getattr(spy_obj1, dim)) + len(getattr(spy_obj2, dim))

    # these get distributed over the cF calls (one for each) to index the trials of the 2nd object
    obj2_idxs = [spy_obj2._preview_trial(trlno).idx for trlno in range(len(spy_obj2.trials))]

    # Create output object
    result = spy_obj1.__class__(dimord=spy_obj1.dimord)

    parallel = False

    try:
        dd.get_client()
        parallel = True
    except ValueError:
        parallel = False

    CR = SpyConcat(
        obj2_idxs,  # get unpacked to `trl2_idx` for each concat_cF
        hdf5_path=spy_obj2.filename,
        axis=concat_axis,
        new_size=new_size,
    )

    CR.initialize(spy_obj1, result._stackingDim, chan_per_worker=None, keeptrials=True)
    log_dict = {"obj1": spy_obj1.filename, "obj2": spy_obj2.filename, "dim": dim}

    CR.compute(spy_obj1, result, parallel=parallel, log_dict=log_dict)

    return result


@process_io
def concat_cF(
    trl1_dat,
    trl2_idx,
    hdf5_path=None,
    axis=-1,
    new_size=None,
    noCompute=False,
    chunkShape=None,
):
    """
    Concatenate 2 trials along `axis`

    Parameters
    ----------
    trl1_dat : :class:`numpy.ndarray`
        Trial data
    trl2_idx : tuple
        A tuple of indices pointing to the 2nd trial
    hdf5_path : str
        Path to the backing hdf5 dataset, providing the 2nd trial via `trl2_idx`
    new_size : int
        New size along the concatenation axis
    noCompute : bool
        Preprocessing flag. If `True`, do not perform actual calculation but
        instead return expected shape and :class:`numpy.dtype` of output
        array.
    chunkShape : None or tuple
        If not `None`, represents shape of output

    Returns
    -------
    res : :class:`numpy.ndarray`
        Concatenate result

    Notes
    -----
    This method is intended to be used as :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
    inside a :class:`~syncopy.shared.computational_routine.ComputationalRoutine`.
    Thus, input parameters are presumed to be forwarded from a parent metafunction.
    Consequently, this function does **not** perform any error checking and operates
    under the assumption that all inputs have been externally validated and cross-checked.

    See also
    --------
    _perform_computation : execute arithmetic operation
    SpyArithmetic : :class:`~syncopy.shared.computational_routine.ComputationalRoutine` subclass
    """

    if noCompute:
        new_shape = np.array(trl1_dat.shape)
        new_shape[axis] = new_size
        return tuple(new_shape), trl1_dat.dtype

    # get the 2nd trial as array
    with h5py.File(hdf5_path, "r") as h5file:
        trl2_dat = h5file["data"][trl2_idx]

    return np.concatenate([trl1_dat, trl2_dat], axis=axis)


class SpyConcat(ComputationalRoutine):
    """
    Compute class for performing concatenation of Syncopy objects

    Sub-class of :class:`~syncopy.shared.computational_routine.ComputationalRoutine`,
    see :doc:`/developer/compute_kernels` for technical details on Syncopy's compute
    classes and metafunctions.
    """

    computeFunction = staticmethod(concat_cF)

    def process_metadata(self, baseObj, out):

        # -- only channel concat atm --

        # Get/set dimensional attributes with the help of a selection
        if baseObj.selection is None:
            revert = True
            baseObj.selectdata(inplace=True)
        else:
            revert = False

        concat_dim = baseObj.dimord[self.cfg["axis"]]

        # Get/set timing-related selection modifiers
        out.trialdefinition = baseObj.selection.trialdefinition
        if baseObj.selection._samplerate:
            out.samplerate = baseObj.samplerate

        for prop in [prop for prop in baseObj.selection._dimProps if prop != concat_dim]:
            selection = getattr(baseObj.selection, prop)
            if selection is not None:
                if np.issubdtype(type(selection), np.number):
                    selection = [selection]
                setattr(out, prop, getattr(baseObj, prop)[selection])

        if revert:
            baseObj.selection = None
