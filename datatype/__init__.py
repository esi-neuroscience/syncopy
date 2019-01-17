# __init__.py - Initialize datatype package
# 
# Author: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Created: Januar 15 2019
# Last modified: <2019-01-15 11:20:33>

# Import __all__ routines from local modules
from .core import *

# Populate local __all__ namespace
__all__ = []
__all__.extend(core.__all__)
