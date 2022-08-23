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
spectralConversions = {
    'abs': lambda x: (np.absolute(x)).real.astype(np.float32),
    'pow': lambda x: (x * np.conj(x)).real.astype(np.float32),
    'fourier': lambda x: x.astype(np.complex64),
    'real': lambda x: np.real(x),
    'imag': lambda x: np.imag(x),
    'angle': lambda x: np.angle(x),
    'absreal': lambda x: np.abs(np.real(x)),
    'absimag': lambda x: np.abs(np.imag(x)),
}

# FT compat
spectralConversions["complex"] = spectralConversions["fourier"]


#: available tapers of :func:`~syncopy.freqanalysis` and  :func:`~syncopy.connectivity`
all_windows = windows.__all__
all_windows.remove("get_window")  # aux. function
all_windows.remove("exponential")  # not symmetric
all_windows.remove("dpss")  # activated via `tapsmofrq`

availableTapers = all_windows
availablePaddingOpt = ['maxperlen', 'nextpow2']

#: general, method agnostic, parameters for our CRs
generalParameters = ("method", "output", "keeptrials", "samplerate",
                     "foi", "foilim", "polyremoval", "out", "pad_to_length")
