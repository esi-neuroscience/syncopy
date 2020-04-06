# -*- coding: utf-8 -*-
# 
# 
# 
# Created: 2020-03-17 17:29:36
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2020-04-06 12:33:21>

# If matplotlib is available, turn on interactive mode and use its 'fast' style
# to automagically set simplification and chunking parameters to speed up plotting
from syncopy import __plt__

# Import __all__ routines from local modules
from . import (spy_plotting)
from .spy_plotting import *

# Populate local __all__ namespace
__all__ = []
__all__.extend(spy_plotting.__all__)
