# __init__.py - Initialize datatype package
# 
# Author: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Created: Januar 15 2019
# Last modified: <2019-01-17 16:23:14>
"""
SpykeWave Data Containers (:mod:`spykewave.datatype`)
=====================================================
Some profoundly insightful text here...

The SpykeWave `BaseData` Data Container
---------------------------------------
Some info highlighting the boundless wisdom underlying the class design...

.. autosummary::
   :toctree: _stubs 
   
   BaseData

"""

# Import __all__ routines from local modules
from .core import *

# Populate local __all__ namespace
__all__ = []
__all__.extend(core.__all__)
