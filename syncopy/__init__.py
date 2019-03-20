# -*- coding: utf-8 -*-
#
# 
# 
# Created: 2019-01-15 09:03:46
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-03-20 10:56:31>

# Builtin/3rd party package imports
import os
import numpy as np

# Global version number
__version__ = "0.1a"

# Set up sensible printing options for NumPy arrays
np.set_printoptions(suppress=True, precision=4, linewidth=80)

# Set up dask configuration
try:
    import dask
    __dask__ = True
except ImportError:
    __dask__ = False

# Define package-wide temp directory (and create it if not already present)
if os.environ.get("SPYTMPDIR"):
    __storage__ = os.path.abspath(os.path.expanduser(os.environ["SPYTMPDIR"]))
else:
    __storage__ = os.path.join(os.path.expanduser("~"), ".spy")
if not os.path.exists(__storage__):
    try:
        os.mkdir(__storage__)
    except:
        raise IOError("Cannot create SyNCoPy storage directory `{}`".format( __storage__))
    
# Fill up namespace
from .utils import *
from .io import *
from .datatype import *
from .specest import *

# Take care of `from spykewave import *` statements
__all__ = []
__all__.extend(datatype.__all__)
__all__.extend(io.__all__)
__all__.extend(utils.__all__)
__all__.extend(specest.__all__)
__all__.extend([__version__, __dask__, __storage__])
