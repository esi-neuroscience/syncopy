# __init__.py - Initialize datatype package
# 
# Created: January 15 2019
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-02-25 15:40:52>
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
from .data_classes import *
from .data_methods import *

# Populate local __all__ namespace
__all__ = []
__all__.extend(data_classes.__all__)
__all__.extend(data_methods.__all__)
