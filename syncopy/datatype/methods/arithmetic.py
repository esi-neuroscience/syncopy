# -*- coding: utf-8 -*-
#
# Syncopy object arithmetics
#

# Builtin/3rd party package imports
import numbers
import numpy as np
import h5py

# Local imports
from syncopy.shared.parsers import data_parser
from syncopy.shared.tools import get_defaults
from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYInfo, SPYWarning
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.shared.kwarg_decorators import unwrap_cfg, unwrap_io, detect_parallel_client
from syncopy.shared.computational_routine import ComputationalRoutine

__all__ = []

def _add(obj1, obj2):

    operand_dat, opres_type, operand_idxs = _parse_input(obj1, obj2, "+")
    operation = lambda x, y : x + y

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
        data_parser(baseObj, varname="data", empty=False)
    except Exception as exc:
        raise exc

    # If no active selection is present, create a "fake" all-to-all selection
    # to harmonize processing down the road
    if baseObj._selection is None:
        baseObj.selectdata(inplace=True)
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

        # Determine numeric type of operation's result
        opres_type = np.result_type(*(trl.dtype for trl in baseTrials), operand)

        # That's it set output vars
        operand_dat = operand
        operand_idxs = None

    # Operand is array-like
    elif isinstance(operand, (np.ndarray, list)):

        # First, ensure operand is a NumPy array to make things easier
        operand = np.array(operand)

        # Ensure complex and real values are not mashed up
        if np.all(np.iscomplex(operand)):
            sameType = lambda dt : "complex" in dt.name
        else:
            sameType = lambda dt : "complex" not in dt.name
        if not all(sameType(trl.dtype) for trl in baseTrials):
            lgl = "array of same numerical type (real/complex)"
            raise SPYTypeError(operand, varname="operand", expected=lgl)

        # Determine the numeric type of the operation's result
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

        # Finally, ensure shapes align
        if not all(baseTrials[k].shape == opndTrials[k].shape for k in range(len(baseTrials))):
            lgl = "Syncopy object (selection) of compatible shapes {}"
            act = "Syncopy object (selection) with shapes {}"
            baseShapes = [trl.shape for trl in baseTrials]
            opndShapes = [trl.shape for trl in opndTrials]
            raise SPYValueError(lgl.format(baseShapes), varname="operand",
                                actual=act.format(opndShapes))

        # Propagate indices for fetching data from operand
        operand_idxs = [trl.idx for trl in opndTrials]

        # Assemble dict with relevant info for performing operation
        operand_dat = {"filename" : operand.filename,
                       "dsetname" : operand._hdfFileDatasetProperties[0]}

    # If `operand` is anything else it's invalid for performing arithmetic on
    else:
        lgl = "Syncopy object, scalar or array-like"
        raise SPYTypeError(operand, varname="operand", expected=lgl)

    return operand_dat, opres_type, operand_idxs

@unwrap_io
def arithmetic_cF(base_dat, operand_dat, operand_idx, operation=None, opres_type = None,
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

    def process_metadata(self, data, out):

        # Get/set timing-related selection modifiers
        out.trialdefinition = data._selection.trialdefinition
        # if data._selection._timeShuffle: # FIXME: should be implemented done the road
        #     out.time = data._selection.timepoints
        if data._selection._samplerate:
            out.samplerate = data.samplerate

        # Get/set dimensional attributes changed by selection
        for prop in data._selection._dimProps:
            selection = getattr(data._selection, prop)
            if selection is not None:
                setattr(out, prop, getattr(data, prop)[selection])
