# -*- coding: utf-8 -*-
#
# Test spectral estimation methods
#
# Created: 2019-06-17 09:45:47
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-06-18 09:36:59>

import os
import tempfile
import pytest
import time
import numpy as np
from numpy.lib.format import open_memmap
from syncopy.datatype import AnalogData, SpectralData
from syncopy.datatype.base_data import VirtualData
from syncopy.shared.errors import SPYValueError, SPYTypeError
from syncopy.specest import freqanalysis

class TestMTMFFT(object):

    # Constructe simple trigonometric signal to check FFT consistency: each
    # channel is a sine wave of frequency `freqs[nchan]` with single unique
    # amplitude `amp` and sampling frequency `fs`
    nChannels = 32
    nTrials = 8
    fs = 1024
    fband = np.linspace(0, fs/2, int(np.floor(fs/2) + 1))
    freqs = np.random.choice(fband[1:-1], size=nChannels, replace=False)
    amp = np.pi
    phases = np.random.permutation(np.linspace(0, 2*np.pi, nChannels))
    t = np.linspace(0, nTrials, nTrials*fs)
    sig = np.zeros((t.size, nChannels), dtype="float32")
    for nchan in range(nChannels):
        sig[:, nchan] = amp*np.sin(2*np.pi*freqs[nchan]*t + phases[nchan])

    trialdefinition = np.zeros((nTrials, 3), dtype="int")
    for ntrial in range(nTrials):
        trialdefinition[ntrial, :] = np.array([ntrial*fs, (ntrial + 1)*fs, 0])

    adata = AnalogData(data=sig, samplerate=fs,
                       trialdefinition=trialdefinition)

    def test_padding(self):
        asdff

    # TODO: check padding
    # TODO: check allocation of output object
    # TODO: check specification of `foi`

    def test_vdata(self):
        pass

    def test_output(self):
        # ensure that output type specification is respected
        spec = freqanalysis(self.adata, method="mtmfft", taper="hann",
                            output="fourier")
        assert "complex" in spec.data.dtype.name
        spec = freqanalysis(self.adata, method="mtmfft", taper="hann",
                            output="abs")
        assert "float" in spec.data.dtype.name
        spec = freqanalysis(self.adata, method="mtmfft", taper="hann",
                            output="pow")
        assert "float" in spec.data.dtype.name

        # ensure consistency of output shape
        spec = freqanalysis(self.adata, method="mtmfft", taper="hann",
                            keeptrials=False)
        assert spec.data.shape[0] == 1
        spec = freqanalysis(self.adata, method="mtmfft", taper="dpss",
                            keeptapers=False)
        assert spec.data.shape[1] == 1
        spec = freqanalysis(self.adata, method="mtmfft", taper="dpss",
                            keeptapers=False, keeptrials=False)
        assert spec.data.shape[0] == 1
        assert spec.data.shape[1] == 1

    def test_solution(self):
        # ensure channel-specific frequencies are identified correctly
        spec = freqanalysis(self.adata, method="mtmfft", taper="hann",
                            output="pow")
        amps = np.empty((self.nTrials*self.nChannels,))
        k = 0
        for nchan in range(self.nChannels):
            for ntrial in range(self.nTrials):
                amps[k] = spec.data[ntrial, :, :, nchan].max()/self.t.size
                assert np.argmax(spec.data[ntrial, :, :, nchan]) == self.freqs[nchan]
                k += 1

        # ensure amplitude is consistent across all channels/trials
        assert np.all(np.diff(amps) < 1)
