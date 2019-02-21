# __init__.py - Import SpykeWave sub-packages
# 
# Created: January 15 2019
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-02-20 18:07:24>

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

# Define package-wide temp directory
__storage__ = os.path.join(os.path.expanduser("~"), ".spy")
    
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
