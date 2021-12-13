# -*- coding: utf-8 -*-
#
# Populate namespace with user exposed
# connectivity methods
# 

from .connectivity_analysis import connectivity
from .connectivity_analysis import __all__ as _all_

# Populate local __all__ namespace
__all__ = []
__all__.extend(_all_)
