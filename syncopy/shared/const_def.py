# -*- coding: utf-8 -*-
# 
# Constant definitions used throughout SyNCoPy
#

# Builtin/3rd party package imports
import numpy as np
from scipy.signal import windows

# Module-wide output specs
spectralDTypes = {"pow": np.float32,
                  "fourier": np.complex64,
                  "abs": np.float32}

#: output conversion of complex fourier coefficients
spectralConversions = {"pow": lambda x: (x * np.conj(x)).real.astype(np.float32),
                       "fourier": lambda x: x.astype(np.complex64),
                       "abs": lambda x: (np.absolute(x)).real.astype(np.float32)}


#: available tapers of :func:`~syncopy.freqanalysis` and  :func:`~syncopy.connectivity`
all_windows = windows.__all__
all_windows.remove("exponential") # not symmetric
all_windows.remove("hanning") # deprecated
all_windows.remove("gaussian") # we don't support taper with args
all_windows.remove("kaiser") # we don't support taper with args

availableTapers = all_windows
availablePaddingOpt = [None, 'nextpow2']

#: general, method agnostic, parameters for our CRs
generalParameters = ("method", "output", "keeptrials", "samplerate",
                     "foi", "foilim", "polyremoval", "out", "pad_to_length")
