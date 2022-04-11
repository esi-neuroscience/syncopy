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



@unwrap_cfg
@unwrap_select
@detect_parallel_client
def resampledata(data,
                 resamplefs,
                  **kwargs
                  ):
    """
    Performs resampling or downsampling of :class:`~syncopy.AnalogData`
    """

    pass
