# -*- coding: utf-8 -*-
#
# Syncopy object arithmetics
#

# Builtin/3rd party package imports
import numbers
import numpy as np
import h5py

# Local imports
from syncopy import __acme__
from syncopy.shared.parsers import data_parser
from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYWarning
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.shared.kwarg_decorators import unwrap_io
from syncopy.shared.computational_routine import ComputationalRoutine
if __acme__:
    import dask.distributed as dd

__all__ = []

def _process_operator(obj1, obj2, operator):
    """
    Coming soon...
    """
    baseObj, operand, operand_dat, opres_type, operand_idxs = _parse_input(obj1, obj2, operator)
    return _perform_computation(baseObj, operand, operand_dat, operand_idxs, opres_type, operator)


def _parse_input(obj1, obj2, operator):

    # Determine which input is a Syncopy object (depending on lef/right application of
    # operator, i.e., `data + 1` or `1 + data`). Can be both as well, but we just need
    # one `baseObj` to get going
    if "BaseData" in str(obj1.__class__.__mro__):
        baseObj = obj1
        operand = obj2
    elif "BaseData" in str(obj2.__class__.__mro__):
        baseObj = obj2
        operand = obj1

    # Ensure our base object is not empty
    try:
        data_parser(baseObj, varname="base", empty=False)
    except Exception as exc:
        raise exc

    # If no active selection is present, create a "fake" all-to-all selection
    # to harmonize processing down the road (and attach `_cleanup` attribute for later removal)
    if baseObj._selection is None:
        baseObj.selectdata(inplace=True)
        baseObj._selection._cleanup = True
    baseTrialList = baseObj._selection.trials

    # Use the `_preview_trial` functionality of Syncopy objects to get each trial's
    # shape and dtype (existing selections are taken care of automatically)
    baseTrials = [baseObj._preview_trial(trlno) for trlno in baseTrialList]

    # Depending on the what is thrown at `baseObj` perform more or less extensive parsing
    # First up: operand is a scalar
    if isinstance(operand, numbers.Number):

        # Don't allow `np.inf` manipulations and catch zero-divisions
        if np.isinf(operand):
            raise SPYValueError("finite scalar", varname="operand", actual=str(operand))
        if operator == "/" and operand == 0:
            raise SPYValueError("non-zero scalar for division", varname="operand", actual=str(operand))

        # Ensure complex and real values are not mashed up
        _check_complex_operand(baseTrials, operand, "scalar")

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
        _check_complex_operand(baseTrials, operand, "array")

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

        # If only a subset of `operand` is selected, adjust for this
        if operand._selection is not None:
            opndTrialList = operand._selection.trials
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

        # Avoid things becoming too nasty: if operand contains wild selections
        # (unordered lists or index repetitions), abort
        for trl in opndTrials:
            if any(np.diff(sel).min() <= 0 if isinstance(sel, list) and len(sel) > 1 \
                else False for sel in trl.idx):
                lgl = "Syncopy object with ordered unreverberated subset selection"
                act = "Syncopy object with selection {}"
                raise SPYValueError(lgl, varname="operand", actual=act.format(operand._selection))

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

def _check_complex_operand(baseTrials, operand, opDimType):
    """
    Coming soon...
    """

    # Ensure complex and real values are not mashed up
    if np.iscomplexobj(operand):
        sameType = lambda dt : "complex" in dt.name
    else:
        sameType = lambda dt : "complex" not in dt.name
    if not all(sameType(trl.dtype) for trl in baseTrials):
        lgl = "{} of same mathematical type (real/complex)"
        raise SPYTypeError(operand, varname="operand", expected=lgl.format(opDimType))

    return


def _perform_computation(baseObj,
                         operand,
                         operand_dat,
                         operand_idxs,
                         opres_type,
                         operator):
    """
    Coming soon...
    """

    # Prepare logging info in dictionary: we know that `baseObj` is definitely
    # a Syncopy data object, operand may or may not be; account for this
    if "BaseData" in str(operand.__class__.__mro__):
        opSel = operand._selection
    else:
        opSel = None
    log_dct = {"operator": operator,
               "base": baseObj.__class__.__name__,
               "base selection": baseObj._selection,
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
    if hasattr(baseObj._selection, "_cleanup"):
        baseObj._selection = None

    return out


@unwrap_io
def arithmetic_cF(base_dat, operand_dat, operand_idx, operation=None, opres_type=None,
                  noCompute=False, chunkShape=None):
    """
    Coming soon...
    """

    if noCompute:
        return base_dat.shape, opres_type

    if isinstance(operand_dat, dict):
        with h5py.File(operand_dat["filename"], "r") as h5f:
            operand = h5f[operand_dat["dsetname"]][operand_idx]
    else:
        operand = operand_dat

    return operation(base_dat, operand)

class SpyArithmetic(ComputationalRoutine):

    computeFunction = staticmethod(arithmetic_cF)

    def process_metadata(self, baseObj, out):

        # Get/set timing-related selection modifiers
        out.trialdefinition = baseObj._selection.trialdefinition
        # if baseObj._selection._timeShuffle: # FIXME: should be implemented done the road
        #     out.time = baseObj._selection.timepoints
        if baseObj._selection._samplerate:
            out.samplerate = baseObj.samplerate

        # Get/set dimensional attributes changed by selection
        for prop in baseObj._selection._dimProps:
            selection = getattr(baseObj._selection, prop)
            if selection is not None:
                setattr(out, prop, getattr(baseObj, prop)[selection])
