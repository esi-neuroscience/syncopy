# -*- coding: utf-8 -*-
#
# Test Peri-Stimulus Time Histogram
#

import numpy as np

# syncopy imports
import syncopy as spy
from syncopy.tests import synth_data as sd
from syncopy.spikes.spike_psth import available_outputs, available_latencies

class TestPSTH:

    # synthetic spike data
    nTrials = 10
    spd = sd.poisson_noise(nTrials,
                           nUnits=4,
                           nChannels=2,
                           nSpikes=10000,
                           samplerate=10000)

    def test_psth_binsize(self):

        cfg = spy.StructDict()
        cfg.latency = 'maxperiod'  # default

        # directly in seconds
        cfg.binsize = 0.2
        counts = spy.spike_psth(self.spd,
                                cfg,
                                keeptrials=True)

        # check that all trials have the same 'time locked' time axis
        # as enfored by TimeLockData.trialdefinition setter
        assert len(set([t.size for t in counts.time])) == 1

        # check that time steps correspond to binsize
        assert np.allclose(np.diff(counts.time[0]), cfg.binsize)

        # automatic binsize selection
        cfg.binsize = 'rice'
        counts = spy.spike_psth(self.spd,
                                cfg,
                                keeptrials=True)
        # number of bins is length of time axis
        nBins_rice = counts.time[0].size
        assert len(set([t.size for t in counts.time])) == 1

        cfg.binsize = 'sqrt'
        counts = spy.spike_psth(self.spd,
                                cfg,
                                keeptrials=True)
        # number of bins is length of time axis
        nBins_sqrt = counts.time[0].size
        assert len(set([t.size for t in counts.time])) == 1

        # sqrt rule gives more bins than Rice rule
        assert nBins_sqrt > nBins_rice

    def test_psth_latency(self):

        cfg = spy.StructDict()
        # directly in seconds
        cfg.binsize = 0.1

        sinfo = self.spd.sampleinfo
        trl_def = self.spd.trialdefinition
        # get trial starting and end times
        beg_ends = sinfo + trl_def[:, 2][:, None]
        beg_ends = (beg_ends - sinfo[:, 0][:, None]) / self.spd.samplerate

        trl_starts = beg_ends[:, 0]
        trl_ends = beg_ends[:, 1]

        return beg_ends
        
        # bins stretch over the largest common time window
        cfg.latency = 'maxperiod'  # default


        counts = spy.spike_psth(self.spd,
                                cfg,
                                keeptrials=True)
        


if __name__ == '__main__':
    T1 = TestPSTH()
