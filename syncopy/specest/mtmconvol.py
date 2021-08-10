# -*- coding: utf-8 -*-
# 
# Time-frequency analysis based on a short-time Fourier transform
# 

# Builtin/3rd party package imports
import numpy as np
from scipy import signal

# Local imports
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.shared.kwarg_decorators import unwrap_io
from syncopy.datatype import padding
from syncopy.shared.tools import best_match
from syncopy.specest.const_def import (
    spectralConversions,
    spectralDTypes,
)
# this is temporary!
from .compRoutines import _make_trialdef




    
