# -*- coding: utf-8 -*-
# 
# Populate namespace with plotting routines
# 

# Importlocal modules, but only import routines from spy_plotting.py
from . import (spy_plotting, _plot_analog)
from .spy_plotting import *

# Populate local __all__ namespace
__all__ = []
__all__.extend(spy_plotting.__all__)
