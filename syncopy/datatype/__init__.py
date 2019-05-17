# -*- coding: utf-8 -*-
#
#
#
# Created: 2019-01-15 10:03:44
# Last modified by: Joscha Schmiedt [joscha.schmiedt@esi-frankfurt.de]
# Last modification time: <2019-05-17 16:32:23>
"""
SyNCoPy Data Classes (:mod:`syncopy.datatype`)
==============================================
Some profoundly insightful text here...

.. inheritance-diagram:: AnalogData SpectralData SpikeData EventData
   :top-classes: BaseData
   :parts: 1

The usable SyNCoPy data classes
---------------------------------------
Some info highlighting the boundless wisdom underlying the class design...

.. autosummary::
   :toctree: _stubs 
   
   AnalogData
   SpectralData
   SpikeData
   EventData





"""

# Import __all__ routines from local modules
from .base_data import *
from .continuous_data import *
from .discrete_data import *
from .data_methods import *

# Populate local __all__ namespace
__all__ = []
__all__.extend(base_data.__all__)
__all__.extend(continuous_data.__all__)
__all__.extend(discrete_data.__all__)
__all__.extend(data_methods.__all__)
