# -*- coding: utf-8 -*-
# 
# Constant definitions and helper functions for spectral estimations
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

#: available outputs of :func:`~syncopy.freqanalysis`
availableOutputs = tuple(spectralConversions.keys())

#: available tapers of :func:`~syncopy.freqanalysis`
all_windows = windows.__all__
all_windows.remove("exponential") # not symmetric
all_windows.remove("hanning") # deprecated
availableTapers = all_windows

#: available wavelet functions of :func:`~syncopy.freqanalysis`
availableWavelets = ("Morlet", "Paul", "DOG", "Ricker", "Marr", "Mexican_hat")

#: available spectral estimation methods of :func:`~syncopy.freqanalysis`
availableMethods = ("mtmfft", "mtmconvol", "wavelet", "superlet")

#: general, method agnostic, parameters of :func:`~syncopy.freqanalysis`
generalParameters = ("method", "output", "keeptrials",
                     "foi", "foilim", "polyremoval", "out")
