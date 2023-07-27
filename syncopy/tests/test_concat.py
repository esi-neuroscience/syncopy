# -*- coding: utf-8 -*-
#
# Test spy.concat function
#

import pytest
import numpy as np

# Local imports
import syncopy as spy
from syncopy.shared.errors import SPYTypeError, SPYValueError


class TestConcat:

    nTrials = 10
    nSamples = 100
    nChannels = 2

    nFreq = 5
    nTaper = 4

    def test_ad_concat(self):

        arr = np.zeros((self.nSamples, self.nChannels))
        adata = spy.AnalogData(data=[arr for _ in range(self.nTrials)], samplerate=10)

        # create 3 channel 2nd data object

        adata2 = spy.AnalogData(
            data=[np.zeros((self.nSamples, 3)) for _ in range(self.nTrials)],
            samplerate=10,
        )

        res = spy.concat(adata, adata2)

        assert isinstance(res, spy.AnalogData)
        assert len(res.trials) == len(adata.trials)
        assert len(res.channel) == len(adata.channel) + len(adata2.channel)
        # check total size
        assert res.data.size == adata.data.size + 3 * self.nSamples * self.nTrials

    def test_sd_concat(self):

        # -- SpectralData with non-standard dimord --

        arr = np.zeros((self.nSamples, self.nChannels, self.nTaper, self.nFreq))
        sdata = spy.SpectralData(
            data=[arr for _ in range(self.nTrials)],
            samplerate=10,
            dimord=["time", "channel", "taper", "freq"],
        )

        # create 3 channel 2nd data object

        arr = np.zeros((self.nSamples, 3, self.nTaper, self.nFreq))
        sdata2 = spy.SpectralData(
            data=[arr for _ in range(self.nTrials)],
            samplerate=10,
            dimord=["time", "channel", "taper", "freq"],
        )

        res = spy.concat(sdata, sdata2)

        assert isinstance(res, spy.SpectralData)
        assert len(res.trials) == len(sdata.trials)
        assert len(res.channel) == len(sdata.channel) + len(sdata2.channel)
        # check total size
        assert res.data.size == sdata.data.size + 3 * self.nSamples * self.nTrials * self.nTaper * self.nFreq

    def test_exceptions(self):

        # non matching data types
        adata = spy.AnalogData(data=np.zeros((10, 2)), samplerate=2)
        sdata = spy.SpectralData(data=np.zeros((10, 2, 2, 2)), samplerate=2)

        with pytest.raises(SPYValueError, match="expected objects with equal dimensional layout"):
            spy.concat(adata, sdata)

        # non matching dimord
        adata2 = spy.AnalogData(data=np.zeros((10, 2)), samplerate=2, dimord=["channel", "time"])

        with pytest.raises(SPYValueError, match="expected objects with equal dimensional layout"):
            spy.concat(adata, adata2)

        # dim not in dimord
        with pytest.raises(SPYValueError, match="object which has a `sth` dimension"):
            spy.concat(adata, adata, dim="sth")

        # only channel supported atm
        with pytest.raises(NotImplementedError, match="Only `channel`"):
            spy.concat(adata, adata, dim="time")

        # objects don't have the same size along remaining axes
        adata3 = spy.AnalogData(data=np.zeros((12, 2)), samplerate=3)
        with pytest.raises(SPYValueError, match="matching shapes"):
            spy.concat(adata, adata3, dim="channel")


if __name__ == "__main__":

    T1 = TestConcat()
