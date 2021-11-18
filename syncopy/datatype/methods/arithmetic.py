# -*- coding: utf-8 -*-
#
# Syncopy object arithmetics
#

# Builtin/3rd party package imports
import numbers
import numpy as np

# Local imports
from syncopy.shared.parsers import data_parser
from syncopy.shared.tools import get_defaults
from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYInfo, SPYWarning
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.shared.kwarg_decorators import unwrap_cfg, unwrap_io, detect_parallel_client
from syncopy.shared.computational_routine import ComputationalRoutine

__all__ = []

def _add(obj1, obj2):

    _parse_input(obj1, obj2, "+")
    pass

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

    # If only a subset of `data` is worked on, adjust for this
    if baseObj._selection is not None:
        trialList = baseObj._selection.trials
    else:
        trialList = list(range(len(baseObj.trials)))

    # Use the `_preview_trial` functionality of Syncopy objects to get each trial's
    # shape and dtype (existing selections are taken care of automatically)
    baseTrials = [baseObj._preview_trial(trlno) for trlno in trialList]

    # Depending on the what is thrown at `baseObj` perform more or less extensive parsing
    # First up: operand is a scalar
    if isinstance(operand, numbers.Number):
        if np.isinf(operand):
            raise SPYValueError("finite scalar", varname="operand", actual=str(operand))
        if operator == "/" and operand == 0:
            raise SPYValueError("non-zero scalar for division", varname="operand", actual=str(operand))

    # Operand is array-like
    elif isinstance(operand, (np.ndarray, list)):

        # First, ensure operand is a NumPy array to make things easier
        operand = np.array(operand)

        # Ensure complex and real values are not mashed together
        if np.all(np.iscomplex(operand)):
            sameType = lambda dt : "complex" in dt.name
        else:
            sameType = lambda dt : "complex" not in dt.name
        if not all(sameType(trl.dtype) for trl in baseTrials):
            lgl = "array of same numerical type (real/complex)"
            raise SPYTypeError(operand, varname="operand", expected=lgl)

        # Ensure shapes match up
        if not all(trl.shape == operand.shape for trl in baseTrials):
            lgl = "array of compatible shape"
            act = "array with shape {}"
            raise SPYValueError(lgl, varname="operand", actual=act.format(operand.shape))

        # All good, nevertheless warn of potential havoc this operation may cause...
        msg = "Performing arithmetic with NumPy arrays may cause inconsistency " +\
            "in Syncopy objects (channels, samplerate, trialdefintions etc.)"
        SPYWarning(msg, caller=operator)

    # Operand is another Syncopy object
    elif "BaseData" in str(operand.__class__.__mro__):

        # First, ensure operand is same object class and has same `dimord` as `baseObj``
        try:
            data_parser(operand, varname="operand",
                        dataclass=baseObj.__class__.__name__, empty=False)
        except Exception as exc:
            raise exc

        opndTrials = [operand._preview_trial(trlno) for trlno in trialList]

    else:
        typeerror

@unwrap_io
def arithmetic_cF(trl_dat, noCompute=False, chunkShape=None):
    """
    Coming soon...
    """
    pass

class SpyArithmetic(ComputationalRoutine):

    computeFunction = staticmethod(arithmetic_cF)