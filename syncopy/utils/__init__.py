# -*- coding: utf-8 -*-
#
#
# 
# Created: 2019-01-15 11:04:33
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-03-08 13:30:23>
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
from .misc import *
from .parsers import *

# Populate local __all__ namespace
__all__ = []
__all__.extend(misc.__all__)
__all__.extend(parsers.__all__)
