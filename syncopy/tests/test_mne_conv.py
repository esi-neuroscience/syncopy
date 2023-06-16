

import syncopy as spy
from syncopy.synthdata.analog import white_noise
from syncopy.synthdata.spikes import poisson_noise
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

    @skip_no_mne
    def test_tldata_to_mne_with_TimeLockData(self):
        """
        Test conversion of spy.TimeLockData to mne.EpochsArray.

        This uses epoched data, i.e., data with trial definition and trials of identical length (and offset), i.e., timelocked data.
        """
        adata = self.adata
        assert type(adata) == spy.AnalogData
        tldata = spy.timelockanalysis(adata, latency="maxperiod")
        assert type(tldata) == spy.TimeLockData
        # Convert to MNE EpochsArray
        epoched = spy.io.mne_conv.tldata_to_mne(tldata)
        assert type(epoched) == mne.EpochsArray

    @skip_no_mne
    def test_tldata_to_mne_with_AnalogData(self):
        """
        Test conversion of spy.AnalogData that is time locked to mne.EpochsArray.

        This uses epoched data, i.e., data with trial definition and trials of identical length (and offset), i.e., timelocked data.
        """
        adata = self.adata
        assert type(adata) == spy.AnalogData
        assert adata.is_time_locked == True
        # Convert to MNE EpochsArray
        epoched = spy.io.mne_conv.tldata_to_mne(adata)
        assert type(epoched) == mne.EpochsArray

    @skip_no_mne
    def test_mne_epoched_to_AnalogData(self):
        """
        Test conversion of spy.AnalogData that is time locked to mne.EpochsArray.

        This uses epoched data, i.e., data with trial definition and trials of identical length (and offset), i.e., timelocked data.
        """
        adata = self.adata
        assert type(adata) == spy.AnalogData
        assert adata.is_time_locked == True
        # Convert to MNE EpochsArray
        epoched = spy.io.mne_conv.tldata_to_mne(adata)
        assert type(epoched) == mne.EpochsArray
        adata2 = spy.io.mne_conv.mne_epochs_to_tldata(epoched)
        
        assert type(adata2) == spy.AnalogData
        assert adata2.is_time_locked == True
        


if __name__ == '__main__':
    T0 = TestSpyToMNE()


