# -*- coding: utf-8 -*-
#
# Syncopy preprocessing frontend
#

# Builtin/3rd party package imports
import numpy as np

# Syncopy imports
from syncopy import AnalogData
from syncopy.shared.parsers import data_parser, scalar_parser, array_parser
from syncopy.shared.tools import get_defaults
from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYWarning, SPYInfo
from syncopy.shared.kwarg_decorators import (unwrap_cfg, unwrap_select,
                                             detect_parallel_client)
from syncopy.shared.input_processors import (
    check_effective_parameters,
    check_passed_kwargs
)

from .compRoutines import Downsample

availableMethods = ('downsample', 'resample')


@unwrap_cfg
@unwrap_select
@detect_parallel_client
def resampledata(data,
                 resamplefs=1,
                 method='downsample',
                 **kwargs
                 ):
    """
    Performs resampling or downsampling of :class:`~syncopy.AnalogData`
    """

    # -- Basic input parsing --

    # Make sure our one mandatory input object can be processed
    try:
        data_parser(data, varname="data", dataclass="AnalogData",
                    writable=None, empty=False)
    except Exception as exc:
        raise exc
    timeAxis = data.dimord.index("time")

    # Get everything of interest in local namespace
    defaults = get_defaults(resampledata)
    lcls = locals()
    # check for ineffective additional kwargs
    check_passed_kwargs(lcls, defaults, frontend_name="preprocessing")

    # check resampling frequency
    scalar_parser(resamplefs, varname='resamplefs', lims=[1, np.inf])

    # -- downsampling --
    if method == 'downsample':

        if data.samplerate % resamplefs != 0:
            lgl = ("integeger division of the original sampling rate "
                   "for `method='downsample'`")
            raise SPYValueError(lgl, varname='resamplefs', actual=resamplefs)
            
        resampleMethod = Downsample(samplerate=data.samplerate,
                                    new_samplerate=resamplefs,
                                    timeAxis=timeAxis)

    # keyword dict for logging
    log_dict = {'method': method,
                'resamplefs': resamplefs
                }
    # ------------------------------------
    # Call the chosen ComputationalRoutine
    # ------------------------------------

    resampled = AnalogData(dimord=data.dimord)
    resampleMethod.initialize(data,
                              resampled._stackingDim,
                              chan_per_worker=kwargs.get("chan_per_worker"),
                              keeptrials=True)
    resampleMethod.compute(data, resampled, parallel=kwargs.get("parallel"), log_dict=log_dict)

    return resampled
