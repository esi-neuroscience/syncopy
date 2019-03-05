# -*- coding: utf-8 -*-
#
#
#
# Created: 2019-01-15 10:03:44
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-03-04 14:34:00>
"""
SynCoPy Data Containers (:mod:`syncopy.datatype`)
=====================================================
Some profoundly insightful text here...

The SynCoPy `BaseData` Data Container
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
