# -*- coding: utf-8 -*-
# 
# Time-frequency analysis with superlets
# Based on 'Time-frequency super-resolution with superlets'
# by Moca et al., 2021 Nature Communications
# 

# Builtin/3rd party package imports
import numpy as np

# Local imports
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.specest.wavelets import cwt
from syncopy.shared.kwarg_decorators import unwrap_io
from syncopy.datatype import padding
import syncopy.specest.freqanalysis as spyfreq
