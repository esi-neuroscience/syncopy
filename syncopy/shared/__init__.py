# -*- coding: utf-8 -*-
#
#
# 
# Created: 2019-01-15 11:04:33
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-06-26 10:15:49>
"""
SynCoPy Convenience Utilities (:mod:`syncopy.utils`)
====================================================
Some profoundly insightful text here...

Tools for SynCoPy development
-----------------------------
We built some tools...

.. autosummary::
   :toctree: _stubs 
   
   io_parser
   scalar_parser
   array_parser
   get_defaults   

"""

# Import __all__ routines from local modules
from .queries import *
from .errors import *
from .parsers import *
from .computational_routine import *
from .dask_helpers import *

# Populate local __all__ namespace
__all__ = []
__all__.extend(computational_routine.__all__)
__all__.extend(dask_helpers.__all__)
__all__.extend(errors.__all__)
__all__.extend(parsers.__all__)
__all__.extend(queries.__all__)
