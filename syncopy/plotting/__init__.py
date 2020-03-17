# -*- coding: utf-8 -*-
# 
# 
# 
# Created: 2020-03-17 17:29:36
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2020-03-17 18:17:19>

# If matplotlib is available, turn on interactive mode and use its 'fast' style
# to automagically set simplification and chunking parameters to speed up plotting
from syncopy import __plt__
if __plt__:
    import matplotlib.pyplot as plt 
    import matplotlib.style as mplstyle
    import matplotlib as mpl
    plt.ion()
    mplstyle.use("fast")
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['lines.linestyle'] = '--'    
    
# Import __all__ routines from local modules
from . import (spy_plotting)
from .spy_plotting import *

# Populate local __all__ namespace
__all__ = []
__all__.extend(spy_plotting.__all__)
