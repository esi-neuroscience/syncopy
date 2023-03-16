# -*- coding: utf-8 -*-
#
# Set/update trial settings of Syncopy data objects
#

# Builtin/3rd party package imports
import sys
import numpy as np
from numbers import Number

# Local imports
import syncopy as spy
from syncopy.shared.kwarg_decorators import unwrap_cfg
from syncopy.shared.parsers import data_parser, array_parser, scalar_parser
from syncopy.shared.errors import SPYTypeError, SPYValueError, SPYError
from syncopy.shared.tools import get_defaults, get_frontend_cfg
from syncopy.shared.input_processors import check_passed_kwargs

__all__ = ["redefinetrial"]


@unwrap_cfg
def redefinetrial(data_obj, trials=None, minlength=None,
                  offset=None, toilim=None,
                  begsample=None, endsample=None, trl=None):
    """
    This function allows you to adjust the time axis of your data, i.e. to
    change from stimulus-locked to response-locked. Furthermore, it allows
    you to select a time window of interest, or to resegment your long trials
    into shorter fragments.

    Parameters
    ----------
    data_obj : Syncopy data object (:class:`BaseData`-like)
    trials : list or ndarray
        List of integers representing trial numbers to be selected
    minlength : float or str
        Minimum length of trials in seconds, can be 'maxperlen'. Trials which
        are shorter get thrown out.
    toilim : [begin, end]
        specify latency window in seconds
    offset : int or Mx1 ndarray
        Expressed in samples relative to current t=0
    begsample : int or Mx1 ndarray of ints
        Expressed in samples relative to the start of the input trial
    endsample : int or Mx1 ndarray of ints
        Expressed in samples relative to the start of the input trial
    trl : Mx3 ndarray
        [start, stop, trigger_offset] sample indices for `M` trials

    Returns
    -------
    ret_obj : Syncopy data object (:class:`BaseData`-like))

    Notes
    -----
    This is a compatibility function, mimicking ``ft_redefinetrial``. However, 
    if selected sample ranges (via ``toilim`` or ``trl``) are (partially) outside 
    of the available data, an error is thrown. This is different to FieldTrips'implementation,
    where missing data is filled up with NaNs.

    See also
    --------
    definetrial : :func:`~syncopy.definetrial`
        Manipulate the trial definition
    selectdata : :func:`~syncopy.selectdata`
        Select along general attributes like channel, frequency, etc.
    `FieldTrip's redefinetrial <https://https://github.com/fieldtrip/fieldtrip/blob/master/ft_redefinetrial.m>`_    
    """

    # Start by vetting input object
    data_parser(data_obj, varname="data_obj", empty=False)

    defaults = get_defaults(redefinetrial)
    lcls = locals()
    # check for ineffective additional kwargs
    check_passed_kwargs(lcls, defaults, frontend_name="redefinetrial")
    new_cfg = get_frontend_cfg(defaults, lcls, kwargs={})

    # sort out mutually exclusive parameters
    vals = [new_cfg[par] for par in ['minlength', 'toilim', 'begsample', 'trl']]
    if vals.count(None) < 3:
        msg = "either `minlength` or `begsample`/`endsample` or `trl` or `toilim`"
        raise SPYError("Incompatible input arguments, " + msg)

    # total number of samples
    scount = data_obj.data.shape[data_obj._stackingDim]

    if new_cfg['trl'] is not None:
        vals = [new_cfg[par] for par in ['begsample', 'endsample', 'offset']]
        if vals.count(None) < 3:
            msg = ("either complete trialdefinition `trl` or  "
                   "`begsample`/`endsample` and `offset`")
            raise SPYError("Incompatible input arguments, " + msg)

        new_trldef = trl

    elif begsample is not None or endsample is not None:
        vals = [new_cfg[par] for par in ['begsample', 'endsample']]
        if vals.count(None) != 0:            
            lgl = "both `begsample` and `endsample`"
            act = f"got [{begsample}, {endsample}]"
            raise SPYValueError(lgl, 'begsample/endsample', act)

        try:
            begsample = np.array(begsample, dtype=int)
        except ValueError:
            raise SPYTypeError(begsample, 'begsample', "integer number or array")

        try:
            endsample = np.array(endsample, dtype=int)
        except ValueError:
            raise SPYTypeError(endsample, 'endsample', "number or array")
        
        if np.any(begsample < 0):
            lgl = "integers > 0"
            act = "relative `begsample` < 0"
            raise SPYValueError(lgl, 'begsample', act)

        if np.any(endsample > scount):
            lgl = f"integers < {scount}"
            act = "out of range `endsample`"
            raise SPYValueError(lgl, 'endsample', act)

        if np.any(endsample - begsample < 0):
            raise SPYValueError("endsample > begsample", "begsample/endsample",
                                "endsample < begsample")
        
        if begsample.size != endsample.size:
            raise SPYValueError("same sizes for `begsample/endsample`",'',
                                "different sizes")
        
        # construct new trialdefinition
        new_trldef = data_obj.trialdefinition
        new_trldef[:, 0] += begsample
        new_trldef[:, 1] -= endsample

    if isinstance(offset, Number):
        pass
    
    array_parser(new_trldef, varname="trl", dims=2)

    array_parser(new_trldef[:, :2], varname="trl", dims=(None, 2),
                 hasnan=False, hasinf=False, ntype="int_like", lims=[0, scount])

    # just apply new trialdefinition and be done with it
    spy.definetrial(data_obj, trialdefinition=trl)
    return
        
    return new_cfg

    # Check array holding trial specifications
    if trl is not None:

        trl = np.array(trialdefinition, dtype="float")

