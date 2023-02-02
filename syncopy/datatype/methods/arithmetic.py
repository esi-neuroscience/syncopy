# -*- coding: utf-8 -*-
#
# Syncopy object arithmetics
#

# Builtin/3rd party package imports
import numpy as np
import h5py

# Local imports
from syncopy import __acme__
from .selectdata import _get_selection_size
from syncopy.shared.parsers import data_parser
from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYWarning, SPYInfo
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.shared.kwarg_decorators import process_io, detect_parallel_client

if __acme__:
    import dask.distributed as dd

__all__ = []


# Main entry point for overloaded operators
def _process_operator(obj1, obj2, operator):
    """
    Perform binary arithmetic operation on Syncopy data object

    Parameters
    ----------
    obj1 : Syncopy data class or Python object
        Depending on left/right application of arithmetic operator, `obj1` may be
        either a Syncopy class or any Python object
    obj2 : Syncopy data class or Python object
        Depending on left/right application of arithmetic operator, `obj2` may be
        either a Syncopy class or any Python object
    operator : str
        Operation to be performed encoded as string. Currently supported operators
        are `'+'`, `'-'`, `'*'`, `'/'` and `'**'` (i.e., `'pow'`).

    Returns
    -------
    res : Syncopy object
        Result of arithmetic operation

    Notes
    -----
    All arithmetic operations are performed on a per-trial basis. This means,
    any data not covered by a Syncopy object's `trialdefinition` will not be
    affected by the arithmetic operation.
    Note further, that error checking is only performed on a very basic level, i.e.,
    the code ensures that instances of different classes are not mashed together
    (e.g., ``AnalogData + SpectralData``) and that objects have compatible trial
    counts and dtypes (no mixing of complex/real data). However, as long as trial
    shapes align, it is possible to process objects w/diverging `samplerate`,
    `channels`, `freqs` etc. The reason for this object parsing leniency is that
    it might be interesting/necessary to manipulate objects arising from different
    configurations (e.g., subtract channel `x` in `obj1` from channel `y` in `obj2`).

    See also
    --------
    _parse_input : prepare objects for arithmetic operations
    _perform_computation : execute arithmetic operation
    """
    baseObj, operand, operand_dat, opres_type, operand_idxs = _parse_input(obj1, obj2, operator)
    return _perform_computation(baseObj, operand, operand_dat, operand_idxs, opres_type, operator)


# Error checking and input preparation
def _parse_input(obj1, obj2, operator):
    """
    Prepare objects for performing binary arithmetics

    Parameters
    ----------
    obj1 : Syncopy data class or Python object
        See :func:`_process_operator` for details.
    obj2 : Syncopy data class or Python object
        See :func:`_process_operator` for details.
    operator : str
        See :func:`_process_operator` for details.

    Returns
    -------
    baseObj : Syncopy data object
        The "base" object to perform arithmetics on. By default, the left object
        is considered as base (if possible), i.e., in the expression ``data1 + data2``,
        `data1` is defined as base object
    operand : Syncopy data object, scalar or array-like
        Term to perform arithmetic operation with.
    operand_dat : dict or scalar or array-like
        If `operand` is a scalar, list or NumPy ndarray then ``operand_dat == operand``.
        If `operand` is a Syncopy object, then `operand_dat` is a dictionary with
        keys `"filename"` (pointing to the HDF5 backing device of `operand`) and
        `"dsetname"`(name of the corresponding dataset(s) of `operand`).
    opres_type : dtype
        Numerical type of the Syncopy object resulting from applying the arithmetic
        operation.
    operand_idxs : None or list
        If `operand` is a scalar, list or NumPy ndarray then `operand_idxs` is
        `None`. If `operand` is a Syncopy object, then `operand_idxs` is a list
        containing the array indices of `operands` data(subset) for each (selected)
        trial.

    Note
    ----
    The distinction between `baseObj` and `operand` is not only syntactic sugar
    but has consequences if both `baseObj` and `operand` are Syncopy objects:
    the `baseObj` is allowed to come with any valid subset selection (may require
    advanced indexing involving multiple slice/list combinations, might include
    repetitions and be unordered). Conversely, the `operand` object can only
    contain `simple` selections (no fancy indexing allowed, no repetitions or
    unordered selections). This restriction simplifies the required HDF dataset
    indexing considerably.
    """

    # Determine which input is a Syncopy object (depending on lef/right application of
    # operator, i.e., `data + 1` or `1 + data`). Can be both as well, but we just need
    # one `baseObj` to get going
    if "BaseData" in str(obj1.__class__.__mro__):
        baseObj = obj1
        operand = obj2
    elif "BaseData" in str(obj2.__class__.__mro__):
        baseObj = obj2
        operand = obj1

    # Ensure base object is not discrete
    if "DiscreteData" in str(baseObj.__class__.__mro__):
        lgl = "`AnalogData`, `SpectralData` or `CrossSpectralData`"
        raise SPYTypeError(baseObj, varname="base", expected=lgl)

    # Ensure our base object is not empty
    try:
        data_parser(baseObj, varname="base", empty=False)
    except Exception as exc:
        raise exc

    # If no active selection is present, create a "fake" all-to-all selection
    # to harmonize processing down the road (and attach `_cleanup` attribute for later removal)
    if baseObj.selection is None:
        baseObj.selectdata(inplace=True)
        baseObj.selection._cleanup = True
    baseTrialList = baseObj.selection.trial_ids

    # Use the `_preview_trial` functionality of Syncopy objects to get each trial's
    # shape and dtype (existing selections are taken care of automatically)
    baseTrials = [baseObj._preview_trial(trlno) for trlno in baseTrialList]

    # Depending on the what is thrown at `baseObj` perform more or less extensive parsing
    # First up: operand is a scalar
    if np.issubdtype(type(operand), np.number):

        # Don't allow `np.inf` manipulations and catch zero-divisions
        if np.isinf(operand):
            raise SPYValueError("finite scalar", varname="operand", actual=str(operand))
        if operator == "/" and operand == 0:
            raise SPYValueError("non-zero scalar for division", varname="operand", actual=str(operand))

        # Ensure complex and real values are not mashed up
        _check_complex_operand(baseTrials, operand, "scalar", operator)

        # Determine exact numeric type of operation's result
        opres_type = np.result_type(*(trl.dtype for trl in baseTrials), operand)

        # That's it set output vars
        operand_dat = operand
        operand_idxs = None

    # Operand is array-like
    elif isinstance(operand, (np.ndarray, list)):

        # First, ensure operand is a NumPy array to make things easier
        operand = np.array(operand)

        # Ensure complex and real values are not mashed up
        _check_complex_operand(baseTrials, operand, "array", operator)

        # Determine exact numeric type of the operation's result
        opres_type = np.result_type(*(trl.dtype for trl in baseTrials), operand.dtype)

        # Ensure shapes match up
        if not all(trl.shape == operand.shape for trl in baseTrials):
            lgl = "array of compatible shape"
            act = "array with shape {}"
            raise SPYValueError(lgl, varname="operand", actual=act.format(operand.shape))

        # No more info needed, the array is the only quantity we need
        operand_dat = operand
        operand_idxs = None

        # All good, nevertheless warn of potential havoc this operation may cause...
        msg = "Performing arithmetic with NumPy arrays may cause inconsistency " +\
            "in Syncopy objects (channels, samplerate, trialdefintions etc.)"
        SPYWarning(msg, caller=operator)

    # Operand is another Syncopy object
    elif "BaseData" in str(operand.__class__.__mro__):

        # Ensure operand object class, and `dimord` match up (and it's non-empty)
        try:
            data_parser(operand, varname="operand", dimord=baseObj.dimord,
                        dataclass=baseObj.__class__.__name__, empty=False)
        except Exception as exc:
            raise exc

        # Make sure samplerates are identical (if present)
        baseSr = getattr(baseObj, "samplerate")
        opndSr = getattr(operand, "samplerate")
        if baseSr  != opndSr:
            lgl = "Syncopy objects with identical samplerate"
            act = "Syncopy object with samplerates {} and {}, respectively"
            raise SPYValueError(lgl, varname="operand",
                                actual=act.format(baseSr, opndSr))

        # If only a subset of `operand` is selected, adjust for this (and warn
        # that arbitrarily ugly things might happen with mis-matched selections)
        if operand.selection is not None:
            wrng = "Found existing in-place selection in operand. " +\
                "Shapes and trial counts of base and operand objects have to match up!"
            SPYWarning(wrng, caller=operator)
            opndTrialList = operand.selection.trial_ids
        else:
            opndTrialList = list(range(len(operand.trials)))

        # Ensure the same number of trials is about to be processed
        opndTrials = [operand._preview_trial(trlno) for trlno in opndTrialList]
        if len(opndTrials) != len(baseTrials):
            lgl = "Syncopy object with same number of trials (selected)"
            act = "Syncopy object with {} trials (selected)"
            raise SPYValueError(lgl, varname="operand", actual=act.format(len(opndTrials)))

        # Ensure complex and real values are not mashed up
        baseIsComplex = ["complex" in trl.dtype.name for trl in baseTrials]
        opndIsComplex = ["complex" in trl.dtype.name for trl in opndTrials]
        if baseIsComplex != opndIsComplex:
            lgl = "Syncopy data object of same numerical type (real/complex)"
            raise SPYTypeError(operand, varname="operand", expected=lgl)

        # Determine the numeric type of the operation's result
        opres_type = np.result_type(*(trl.dtype for trl in baseTrials),
                                    *(trl.dtype for trl in opndTrials))

        # Ensure shapes align
        if not all(baseTrials[k].shape == opndTrials[k].shape for k in range(len(baseTrials))):
            lgl = "Syncopy object (selection) of compatible shapes {}"
            act = "Syncopy object (selection) with shapes {}"
            baseShapes = [trl.shape for trl in baseTrials]
            opndShapes = [trl.shape for trl in opndTrials]
            raise SPYValueError(lgl.format(baseShapes), varname="operand",
                                actual=act.format(opndShapes))

        # Avoid things becoming too nasty: if `operand`` contains wild selections
        # (unordered lists or index repetitions) or selections requiring advanced
        # (aka fancy) indexing (multiple slices mixed with lists), abort
        for trl in opndTrials:
            if any(np.diff(sel).min() <= 0 if isinstance(sel, list) and len(sel) > 1 \
                else False for sel in trl.idx):
                lgl = "Syncopy object with ordered unreverberated subset selection"
                act = "Syncopy object with selection {}"
                raise SPYValueError(lgl, varname="operand", actual=act.format(operand.selection))
            if sum(isinstance(sel, slice) for sel in trl.idx) > 1 and \
                sum(isinstance(sel, list) for sel in trl.idx) > 1:
                lgl = "Syncopy object without selections requiring advanced indexing"
                act = "Syncopy object with selection {}"
                raise SPYValueError(lgl, varname="operand", actual=act.format(operand.selection))

        # Propagate indices for fetching data from operand
        operand_idxs = [trl.idx for trl in opndTrials]

        # Assemble dict with relevant info for performing operation
        operand_dat = {"filename" : operand.filename,
                       "dsetname" : operand._hdfFileDatasetProperties[0]}

    # If `operand` is anything else it's invalid for performing arithmetic on
    else:
        lgl = "Syncopy object, scalar or array-like"
        raise SPYTypeError(operand, varname="operand", expected=lgl)

    return baseObj, operand, operand_dat, opres_type, operand_idxs

# Check for complexity in `operand` vs. `baseObj`
def _check_complex_operand(baseTrials, operand, opDimType, operator):
    """
    Local helper to determine if provided scalar/array and `baseObj` are both real/complex
    """

    # Ensure complex and real values are not mashed up
    if np.iscomplexobj(operand):
        sameType = lambda dt : "complex" in dt.name
    else:
        sameType = lambda dt : "complex" not in dt.name
    if not all(sameType(trl.dtype) for trl in baseTrials):
        wrng = "Operand is {} of different mathematical type (real/complex)"
        SPYWarning(wrng.format(opDimType), caller=operator)

    return


# Invoke `ComputationalRoutine` to compute arithmetic operation
def _perform_computation(baseObj,
                         operand,
                         operand_dat,
                         operand_idxs,
                         opres_type,
                         operator):
    """
    Leverage `ComputationalRoutine` to process arithmetic operation

    Parameters
    ----------
    baseObj : Syncopy data object
        See :func:`_parse_input` for details.
    operand : Syncopy data object, scalar or array-like
        See :func:`_parse_input` for details.
    operand_dat : dict or scalar or array-like
        See :func:`_parse_input` for details.
    opres_type : dtype
        See :func:`_parse_input` for details.
    operator : str
        See :func:`_process_operator` for details.

    Returns
    -------
    out : Syncopy data object
        Result of performing arithmetic operation on `baseObj` and `operand`

    Note
    ----
    This method instantiates a subclass of
    :class:`~syncopy.shared.computational_routine.ComputationalRoutine`
    to perform arithmetic operations on Syncopy objects either sequentially or
    in parallel. Note that due to this code being only invoked via operator
    overloading the `@detect_parallel_client` decorator is *not* invoked, since
    the user cannot supply any keyword arguments. Instead, the code scans for
    running dask distributed computing clients (if ACME is available) and uses
    concurrent processing if a client is found.

    See also
    --------
    arithmetic_cF : `computeFunction` performing arithmetics
    SpyArithmetic : :class:`~syncopy.shared.computational_routine.ComputationalRoutine` subclass
    """

    # Prepare logging info in dictionary: we know that `baseObj` is definitely
    # a Syncopy data object, operand may or may not be; account for this
    if "BaseData" in str(operand.__class__.__mro__):
        opSel = operand.selection
    else:
        opSel = None
    log_dct = {"operator": operator,
               "base": baseObj.__class__.__name__,
               "base selection": baseObj.selection,
               "operand": operand.__class__.__name__,
               "operand selection": opSel}

    # Create output object
    out = baseObj.__class__(dimord=baseObj.dimord)

    # Now create actual functional operations: wrap operator in lambda
    if operator == "+":
        operation = lambda x, y : x + y
    elif operator == "-":
        operation = lambda x, y : x - y
    elif operator == "*":
        operation = lambda x, y : x * y
    elif operator == "/":
        operation = lambda x, y : x / y
    elif operator == "**":
        operation = lambda x, y : x ** y
    else:
        raise SPYValueError("supported arithmetic operator", actual=operator)

    # If ACME is available, try to attach (already running) parallel computing client
    parallel = False
    if __acme__:
        try:
            dd.get_client()
            parallel = True
        except ValueError:
            parallel = False

    # Perform actual computation: instantiate `ComputationalRoutine` w/extracted info
    opMethod = SpyArithmetic(operand_dat, operand_idxs, operation=operation,
                             opres_type=opres_type)
    opMethod.initialize(baseObj,
                        out._stackingDim,
                        chan_per_worker=None,
                        keeptrials=True)

    # In case of parallel execution, be careful: use a distributed lock to prevent
    # ACME from performing chained operations (`x + y + 3``) simultaneously (thereby
    # wrecking the underlying HDF5 datasets). Similarly, if `operand` is a Syncopy
    # object, close its corresponding dataset(s) before starting to concurrently read
    # from them (triggering locking errors)
    if parallel:
        lock = dd.lock.Lock(name='arithmetic_ops')
        lock.acquire()
        if "BaseData" in str(operand.__class__.__mro__):
            for dsetName in operand._hdfFileDatasetProperties:
                dset = getattr(operand, dsetName)
                dset.file.close()

    opMethod.compute(baseObj, out, parallel=parallel, log_dict=log_dct)

    # Re-open `operand`'s dataset(s) and release distributed lock
    if parallel:
        if "BaseData" in str(operand.__class__.__mro__):
            for dsetName in operand._hdfFileDatasetProperties:
                setattr(operand, dsetName, operand.filename)
        lock.release()

    # Delete any created subset selections
    if hasattr(baseObj.selection, "_cleanup"):
        baseObj.selection = None

    return out


@process_io
def arithmetic_cF(base_dat, operand_dat, operand_idx, operation=None, opres_type=None,
                  noCompute=False, chunkShape=None):
    """
    Perform arithmetic operation

    Parameters
    ----------
    base_dat : :class:`numpy.ndarray`
        Trial data
    operand_dat : dict or scalar or array-like
        If two Syncopy objects are processed, then `operand_dat` is a dictionary
        containing information about the operand's HDF5 backing device (see
        :func:`_parse_input` for details). Otherwise, `operand_dat` is either a
        scalar or array-like quantity.
    operand_idx : tuple
        If `operand` is a scalar, list or NumPy ndarray then `operand_idx` is
        `None`. If `operand` is a Syncopy object, then `operand_idx` is an indexing
        tuple.
    operation : lambda object
        A lambda expression encapsulating the requested arithmetic operation.
    opres_type : dtype
        Numerical type of applying ``operation(base_dat, operand)``
    noCompute : bool
        Preprocessing flag. If `True`, do not perform actual calculation but
        instead return expected shape and :class:`numpy.dtype` of output
        array.
    chunkShape : None or tuple
        If not `None`, represents shape of output

    Returns
    -------
    res : :class:`numpy.ndarray`
        Result of ``operation(base_dat, operand)``

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
        return base_dat.shape, opres_type

    if isinstance(operand_dat, dict):
        with h5py.File(operand_dat["filename"], "r") as h5f:
            operand = h5f[operand_dat["dsetname"]][operand_idx]
            # enforce original shape in case `operand_idx` contained scalar
            # selections that squeezed the array
            operand.shape = chunkShape
    else:
        operand = operand_dat

    return operation(base_dat, operand)

class SpyArithmetic(ComputationalRoutine):
    """
    Compute class for performing arithmetic operations with Syncopy objects

    Sub-class of :class:`~syncopy.shared.computational_routine.ComputationalRoutine`,
    see :doc:`/developer/compute_kernels` for technical details on Syncopy's compute
    classes and metafunctions.

    See also
    --------
    _perform_computation : execute arithmetic operation
    """

    computeFunction = staticmethod(arithmetic_cF)

    def process_metadata(self, baseObj, out):

        # Get/set timing-related selection modifiers
        out.trialdefinition = baseObj.selection.trialdefinition
        # if baseObj.selection._timeShuffle: # FIXME: should be implemented done the road
        #     out.time = baseObj.selection.timepoints
        if baseObj.selection._samplerate:
            out.samplerate = baseObj.samplerate

        # Get/set dimensional attributes changed by selection
        for prop in baseObj.selection._dimProps:
            selection = getattr(baseObj.selection, prop)
            if selection is not None:
                if np.issubdtype(type(selection), np.number):
                    selection = [selection]
                setattr(out, prop, getattr(baseObj, prop)[selection])
