# -*- coding: utf-8 -*-
#
#
# 
# Created: 2019-01-15 11:04:33
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-05-07 17:32:23>
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
from .misc import *
from .parsers import *

# Populate local __all__ namespace
__all__ = []
__all__.extend(misc.__all__)
__all__.extend(parsers.__all__)
