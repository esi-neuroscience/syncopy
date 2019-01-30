# __init__.py - Initialize utilities package
# 
# Created: Januar 15 2019
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-01-30 13:49:58>
"""
SpykeWave Convenience Utilities (:mod:`spykewave.utils`)
========================================================
Some profoundly insightful text here...

Tools for SpykeWave development
-------------------------------
We built some tools...

.. autosummary::
   :toctree: _stubs 
   
   spw_io_parser
   spw_scalar_parser
   spw_array_parser
   spw_get_defaults
   spw_print

"""

# Import __all__ routines from local modules
from .misc import *
from .spw_parsers import *

# Populate local __all__ namespace
__all__ = []
__all__.extend(misc.__all__)
__all__.extend(spw_parsers.__all__)
