# -*- coding: utf-8 -*-
#
# Constant definitions used throughout SyNCoPy
#

# Builtin/3rd party package imports
import numpy as np
from scipy.signal import windows


# Module-wide output specs
spectralDTypes = {"pow": np.float32,
                  "abs": np.float32,
                  "real": np.float32,
                  "imag": np.float32,
                  "angle": np.float32,
                  "absreal": np.float32,
                  "absimag": np.float32,
                  "fourier": np.complex64,
                  "complex": np.complex64
                  }

#: output conversion of complex fourier coefficients
spectralConversions = {
    'pow': lambda x: (x * np.conj(x)).real.astype(spectralDTypes['pow']),
    'abs': lambda x: (np.absolute(x)).real.astype(spectralDTypes['abs']),
    'fourier': lambda x: x.astype(spectralDTypes['fourier']),
    'real': lambda x: np.real(x).astype(spectralDTypes['real']),
    'imag': lambda x: np.imag(x).astype(spectralDTypes['imag']),
    'angle': lambda x: np.angle(x).astype(spectralDTypes['angle']),
    'absreal': lambda x: np.abs(np.real(x)).astype(spectralDTypes['absreal']),
    'absimag': lambda x: np.abs(np.imag(x)).astype(spectralDTypes['absimag'])
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
generalParameters = ("method", "keeptrials", "samplerate",
                     "foi", "foilim", "polyremoval", "out", "pad")
