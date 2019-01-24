# __init__.py - Import SpykeWave sub-packages
# 
# Author: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Created: Januar 15 2019
# Last modified: <2019-01-23 16:59:58>

# Global version number
__version__ = "0.1a"

# Fill up namespace
from .utils import *
from .io import *
from .datatype import *

# Take care of `from spykewave import *` statements
__all__ = []
__all__.extend(datatype.__all__)
__all__.extend(io.__all__)
__all__.extend(utils.__all__)
__all__.extend([__version__])
