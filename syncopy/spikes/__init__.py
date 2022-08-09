# -*- coding: utf-8 -*-
# 
# Populate namespace with spike analysis
# 

# Import __all__ routines from local modules
from .spike_psth import spike_psth

# Populate local __all__ namespace
# with the user-exposed frontend
__all__ = ['spike_psth']
