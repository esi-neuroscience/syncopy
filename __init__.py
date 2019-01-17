# __init__.py - Import SpykeWave sub-packages
# 
# Author: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Created: Januar 15 2019
# Last modified: <2019-01-17 14:29:07>

# Global version number
__version__ = "0.1a"

# Fill up namespace
from .datatype import *
from .utils import *

# Take care of `from spykewave import *` statements
__all__ = []
__all__.extend(datatype.__all__)
__all__.extend(utils.__all__)
__all__.extend([__version__])
