# -*- coding: utf-8 -*-
# 
# Constant definitions specific for spectral estimations
#

from syncopy.shared.const_def import spectralConversions

#: available outputs of :func:`~syncopy.freqanalysis`
availableOutputs = tuple(spectralConversions.keys())

#: available wavelet functions of :func:`~syncopy.freqanalysis`
availableWavelets = ("Morlet", "Paul", "DOG", "Ricker", "Marr", "Mexican_hat")

#: available spectral estimation methods of :func:`~syncopy.freqanalysis`
availableMethods = ("mtmfft", "mtmconvol", "wavelet", "superlet")

