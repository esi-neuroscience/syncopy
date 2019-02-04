# __init__.py - Import SpykeWave sub-packages
# 
# Created: January 15 2019
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-02-04 16:42:35>

# Global version number
__version__ = "0.1a"

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
__all__.extend([__version__])

# Set up sensible printing options for NumPy arrays
import numpy as np
np.set_printoptions(suppress=True, precision=4, linewidth=80)
