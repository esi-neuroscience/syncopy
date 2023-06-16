

import syncopy as spy
from syncopy.synthdata.analog import white_noise
import numpy as np
import pytest

has_mne = False
try:
    import mne
    has_mne = True
except ImportError:
    pass

skip_no_mne = pytest.mark.skipif(not has_mne, reason="MNE Python not installed")

class TestSpyToMNE():

    numChannels = 64
    numTrials = 5
    adata = white_noise(nTrials = numTrials, nChannels=numChannels, nSamples= 1000)

    @skip_no_mne
    def test_spy_analog_raw_to_mne(self):
        """
        Test conversion of spy.AnalogData to MNE RawArray.

        This uses raw data, i.e., data without trial definition.
        """

        adata = self.adata
        # Convert to MNE RawArray
        ar = spy.io.mne_conv.raw_adata_to_mne(adata)

        assert type(ar) == mne.io.RawArray
        # Check that the data is the same
        assert np.allclose((adata.data[()]).T, ar.get_data())
        # Check that the channel names are the same
        assert all(adata.channel == ar.ch_names)
        # Check that the sampling rate is the same
        assert adata.samplerate == ar.info['sfreq']

    @skip_no_mne
    def test_mne_rawarray_to_spy_analog(self):
        """
        Test conversion of mne.io.RawArray to spy.AnalogData.

        This uses raw data, i.e., data without trial definition.
        """

        adata = self.adata
        # Convert to MNE RawArray
        ar = spy.io.mne_conv.raw_adata_to_mne(adata)
        # Now convert back to AnalogData
        adata2 = spy.io.mne_conv.raw_mne_to_adata(ar)
        assert type(adata2) == spy.AnalogData
        assert all(adata.channel == adata2.channel)
        assert np.allclose(adata.data, adata2.data)
        assert adata.samplerate == adata2.samplerate

if __name__ == '__main__':
    T0 = TestSpyToMNE()


