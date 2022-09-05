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

        """Test all available `latency` (time window interval) settings"""

        cfg = spy.StructDict()
        # directly in seconds
        cfg.binsize = 0.1

        trl_starts = self.spd.trialintervals[:, 0]
        trl_ends = self.spd.trialintervals[:, 1]

        # -- bins stretch over the largest common time window --
        cfg.latency = 'maxperiod'  # frontend default
        counts = spy.spike_psth(self.spd, cfg, keeptrials=True)

        # sampling interval for histogram output
        delta_t = 1 / counts.samplerate

        # check that histogram time points are less than 1
        # delta_t away from the maximal interval boundaries
        assert np.abs(trl_starts.min() - counts.time[0][0]) < delta_t
        assert np.abs(trl_ends.max() - counts.time[0][-1]) < delta_t

        # check that there are NaNs as not all trials have data
        # in this maximal interval (due to start/end randomization)
        assert np.any(np.isnan(counts.data[:]))

        # -- bins stretch over the minimal interval present in all trials --
        cfg.latency = 'minperiod'
        counts = spy.spike_psth(self.spd, cfg, keeptrials=True)

        # check that histogram time points are less than 1
        # delta_t away from the minimal interval boundaries
        assert np.abs(trl_starts.max() - counts.time[0][0]) < delta_t
        assert np.abs(trl_ends.min() - counts.time[0][-1]) < delta_t

        # check that there are NO NaNs as all trials have data
        # in this minimal interval
        assert not np.any(np.isnan(counts.data[:]))

        # -- prestim --> only events with t < 0
        cfg.latency = 'prestim'
        counts = spy.spike_psth(self.spd, cfg, keeptrials=True)

        assert np.all(counts.time[0] <= 0)

        # -- poststim --> only events with t > 0
        cfg.latency = 'poststim'
        counts = spy.spike_psth(self.spd, cfg, keeptrials=True)

        assert np.all(counts.time[0] >= 0)

        # -- finally the manual latency interval --
        # this is way to big, so we have many NaNs (empty bins)
        cfg.latency = [-.5, 1.5]   # in seconds
        assert cfg.latency[0] < trl_starts.min()
        assert cfg.latency[1] > trl_ends.max()

        counts = spy.spike_psth(self.spd, cfg, keeptrials=True)
        # check that histogram time points are less than 1
        # delta_t away from the manual set interval boundaries
        assert np.abs(cfg.latency[0] - counts.time[0][0]) <= delta_t
        # the midpoint gets rounded down, so the last time point is close to
        # 1 delta_t off actually..
        assert np.allclose(np.abs(cfg.latency[1] - counts.time[0][-1]), delta_t)

        # check that there are NaNs as the interval is way too large
        assert np.any(np.isnan(counts.data[:]))

    def test_psth_vartriallen(self):
        pass 
        # return counts


if __name__ == '__main__':
    T1 = TestPSTH()
