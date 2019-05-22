# -*- coding: utf-8 -*-
#
#
# 
# Created: 2019-01-15 11:04:33
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-05-22 16:14:28>
"""
SynCoPy Convenience Utilities (:mod:`syncopy.utils`)
====================================================
Some profoundly insightful text here...

Tools for SynCoPy development
-----------------------------
We built some tools...

.. autosummary::
   :toctree: _stubs 
   
   spy_io_parser
   spy_scalar_parser
   spy_array_parser
   spy_get_defaults
   spy_print

"""

# Import __all__ routines from local modules
from .queries import *
from .errors import *
from .parsers import *
from .computational_routine import *

# Populate local __all__ namespace
__all__ = []
__all__.extend(computational_routine.__all__)
__all__.extend(errors.__all__)
__all__.extend(parsers.__all__)
__all__.extend(queries.__all__)
