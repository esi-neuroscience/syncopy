# Syncopy benchmark suite.
# See "Writing benchmarks" in the asv docs for more information.

import syncopy as spy
from syncopy.synthdata.analog import white_noise

class SelectionSuite:
    """
    Benchmark selections on AnalogData objects.
    """
    def setup(self):
        self.adata = white_noise(nSamples=25000, nChannels=32, nTrials=500, samplerate=1000)

    def time_external_channel_selection(self):
        _ = spy.selectdata(self.adata, channel=[0, 1, 7], inplace=False)

    def time_inplace_channel_selection(self):
        spy.selectdata(self.adata, channel=[0, 1, 7], inplace=True)



class MemSuite:
    """Test memory usage of data classes.
    Note that this is intented to test memory usage of python objects, not of a function call.
    Use the mem_peak prefix for that.
    """

    def setup(self):
        self.adata = white_noise(nSamples=25000, nChannels=32, nTrials=500, samplerate=1000)

    def mem_analogdata(self):
        """Test memory usage of AnalogData object."""
        return self.adata
