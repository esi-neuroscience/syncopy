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


class TestSpyToMNE:

    numChannels = 64
    numTrials = 5
    numSamples = 1000
    adata = white_noise(nTrials=numTrials, nChannels=numChannels, nSamples=numSamples)
    adata_notrials = white_noise(nTrials=1, nChannels=numChannels, nSamples=numSamples)

    @skip_no_mne
    def test_spy_analog_raw_to_mne(self):
        """
        Test conversion of spy.AnalogData to MNE RawArray.

        This uses raw data, i.e., data without trial definition.
        """

        adata_notrials = self.adata_notrials
        # Convert to MNE RawArray
        ar = spy.io.mne_conv.raw_adata_to_mne_raw(adata_notrials)

        assert type(ar) == mne.io.RawArray
        # Check that the data is the same
        assert np.allclose((adata_notrials.data[()]).T, ar.get_data())
        # Check that the channel names are the same
        assert all(adata_notrials.channel == ar.ch_names)
        # Check that the sampling rate is the same
        assert adata_notrials.samplerate == ar.info["sfreq"]

    @skip_no_mne
    def test_mne_rawarray_to_spy_analog(self):
        """
        Test conversion of mne.io.RawArray to spy.AnalogData.

        This uses raw data, i.e., data without trial definition.
        """

        adata = self.adata_notrials
        # Convert to MNE RawArray
        ar = spy.io.mne_conv.raw_adata_to_mne_raw(adata)
        # Now convert back to AnalogData
        adata2 = spy.io.mne_conv.raw_mne_to_adata(ar)
        assert type(adata2) == spy.AnalogData
        assert all(adata.channel == adata2.channel)
        assert np.allclose(adata.data, adata2.data)
        assert np.allclose(adata.time[0], adata2.time[0])
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
        epoched = spy.io.mne_conv.tldata_to_mne_epochs(tldata)
        assert type(epoched) == mne.EpochsArray

        # Check dimensions
        n_times = epoched.get_data().shape[2]
        assert n_times == self.numSamples
        n_epochs = epoched.get_data().shape[0]
        assert n_epochs == self.numTrials
        n_channels = epoched.get_data().shape[1]

        assert n_times == tldata.trials[0].shape[0]
        assert n_epochs == len(tldata.trials)
        assert n_channels == len(tldata.channel)

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
        epoched = spy.io.mne_conv.tldata_to_mne_epochs(adata)
        assert epoched.get_data().shape == (
            self.numTrials,
            self.numChannels,
            self.numSamples,
        )
        for ea in epoched.iter_evoked():  # ea is an mne.EvokedArray
            assert type(ea) == mne.EvokedArray
            assert ea.get_data().shape == (self.numChannels, self.numSamples)
        assert type(epoched) == mne.EpochsArray

        # Check dimensions
        n_times = epoched.get_data().shape[2]
        assert n_times == self.numSamples
        n_epochs = epoched.get_data().shape[0]
        n_channels = epoched.get_data().shape[1]

        assert n_times == adata.trials[0].shape[0]
        assert n_epochs == len(adata.trials)
        assert n_channels == len(adata.channel)

    @skip_no_mne
    def test_mne_epoched_to_AnalogData(self):
        """
        Test conversion of mne.EpochsArray to spy.AnalogData that is time locked.
        """
        adata = self.adata
        assert type(adata) == spy.AnalogData
        assert adata.is_time_locked == True
        # Convert to MNE EpochsArray
        epoched = spy.io.mne_conv.tldata_to_mne_epochs(adata)
        assert type(epoched) == mne.EpochsArray
        for ea in epoched.iter_evoked():  # ea is an mne.EvokedArray
            assert type(ea) == mne.EvokedArray
            assert ea.get_data().shape == (self.numChannels, self.numSamples)
        adata2 = spy.io.mne_conv.mne_epochs_to_tldata(epoched)

        # Check dimensions
        n_times = epoched.get_data().shape[2]
        assert n_times == self.numSamples
        n_epochs = epoched.get_data().shape[0]
        n_channels = epoched.get_data().shape[1]

        assert n_times == adata.trials[0].shape[0]
        assert n_epochs == len(adata.trials)
        assert n_channels == len(adata.channel)

        # assert n_times == adata2.trials[0].shape[0]
        assert n_epochs == len(adata2.trials)
        assert n_channels == len(adata2.channel)

        # check data
        assert type(adata2) == spy.AnalogData
        assert adata2.is_time_locked == True
        assert np.allclose(adata.data[()], adata2.data[()])

        # check time axis
        assert np.allclose(adata.time, adata2.time)


if __name__ == "__main__":
    T0 = TestSpyToMNE()
