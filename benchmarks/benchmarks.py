# Syncopy benchmark suite.
# See "Writing benchmarks" in the asv docs for more information.

import syncopy as spy
from syncopy.synthdata.analog import white_noise


class SelectionSuite:
    """
    Benchmark selections on AnalogData objects.
    """
    def setup(self):
        self.adata = white_noise(nSamples=25000, nChannels=32, nTrials=250, samplerate=1000)

    def teardown(self):
        del self.adata

    def time_external_channel_selection(self):
        _ = spy.selectdata(self.adata, channel=[0, 1, 7], inplace=False)

    def time_inplace_channel_selection(self):
        spy.selectdata(self.adata, channel=[0, 1, 7], inplace=True)


class MTMFFT:
    """
    Benchmark multi-tapered fft
    """
    def setup(self):
        self.adata = white_noise(nSamples=5000, nChannels=32, nTrials=250, samplerate=1000)

    def teardown(self):
        del self.adata

    def time_mtmfft_untapered(self):
        _ = spy.freqanalysis(self.adata, taper=None)

    def time_mtmfft_multitaper(self):
        _ = spy.freqanalysis(self.adata, tapsmofrq=2)


class Arithmetic:
    """
    Benchmark Syncopy's arithmetic
    """

    def setup(self):
        self.adata = white_noise(nSamples=25000, nChannels=32, nTrials=250, samplerate=1000)
        self.adata2 = self.adata.copy()

    def teardown(self):
        del self.adata
        del self.adata2

    def time_scalar_mult(self):
        _ = 3 * self.adata

    def time_scalar_add(self):
        _ = 3 + self.adata

    def time_dset_add(self):
        _ = self.adata + self.adata2


class MemSuite:
    """Test memory usage of data classes.
    Note that this is intented to test memory usage of python objects, not of a function call.
    Use the mem_peak prefix for that.
    """

    def setup(self):
        self.adata = white_noise(nSamples=10_000, nChannels=32, nTrials=250, samplerate=1000)

    def teardown(self):
        del self.adata

    def mem_analogdata(self):
        """Test memory usage of AnalogData object."""
        return self.adata

    def peakmem_mtmfft(self):
        """Test memory usage of mtmfft"""
        _ =  spy.freqanalysis(self.adata, tapsmofrq=2)
