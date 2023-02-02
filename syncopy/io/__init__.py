# -*- coding: utf-8 -*-
#
# Populate namespace with io routines
#

# Import __all__ routines from local modules
from . import (
    utils,
    load_spy_container,
    save_spy_container,
    load_ft,
    load_tdt,
    load_nwb
)
from .utils import *
from .load_spy_container import *
from .save_spy_container import *
from .load_ft import *
from .load_tdt import *
from .load_nwb import *

# Populate local __all__ namespace
__all__ = ['load_ft_raw', 'load_tdt', 'load_nwb']
__all__.extend(utils.__all__)
__all__.extend(load_spy_container.__all__)
__all__.extend(save_spy_container.__all__)
