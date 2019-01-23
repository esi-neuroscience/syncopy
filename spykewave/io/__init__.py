# __init__.py - Initialize I/O package
# 
# Author: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Created: Januar 23 2019
# Last modified: <2019-01-23 17:01:49>
"""
Coming soon...
"""

# Import __all__ routines from local modules
from .read_raw_binary import *
from .reader import *

# Populate local __all__ namespace
__all__ = []
__all__.extend(reader.__all__)
__all__.extend(read_raw_binary.__all__)
