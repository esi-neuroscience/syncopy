# -*- coding: utf-8 -*-
#
#
#
# Created: 2019-01-15 10:03:44
# Last modified by: Joscha Schmiedt [joscha.schmiedt@esi-frankfurt.de]
# Last modification time: <2019-05-21 15:00:07>
"""
Syncopy Data Classes (:mod:`syncopy.datatype`)
==============================================
Some profoundly insightful text here...

.. inheritance-diagram:: AnalogData SpectralData SpikeData EventData
   :top-classes: BaseData
   :parts: 1

The usable Syncopy data classes
-------------------------------
Some info highlighting the boundless wisdom underlying the class design...

.. autosummary::
   :toctree: _stubs 
   
   AnalogData
   SpectralData
   SpikeData
   EventData
   ContinuousData
   DiscreteData


Functions for handling data
---------------------------
These functions are useful for editing and slicing data:

.. autosummary::      
    :toctree: _stubs
    
    selectdata
    definetrial

"""

# Import __all__ routines from local modules
from .base_data import *
from .continuous_data import *
from .discrete_data import *
from .data_methods import *
from .continuous_data import ContinuousData
from .discrete_data import DiscreteData

# Populate local __all__ namespace
__all__ = []
__all__.extend(base_data.__all__)
__all__.extend(continuous_data.__all__)
__all__.extend(discrete_data.__all__)
__all__.extend(data_methods.__all__)
