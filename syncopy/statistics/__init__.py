# -*- coding: utf-8 -*-
# 
# Populate namespace with statistics routines and classes
# 

# Import __all__ routines from local modules
from . import summary_stats
from .spike_psth import spike_psth
from .timelockanalysis import timelockanalysis
from .summary_stats import (
    mean,
    var,
    std,
    median,
    itc,
)

# Populate local __all__ namespace
__all__ = ['spike_psth', 'timelockanalysis']
__all__.extend(summary_stats.__all__)
