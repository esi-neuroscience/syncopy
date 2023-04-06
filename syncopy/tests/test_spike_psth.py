# -*- coding: utf-8 -*-
#
# Test Peri-Stimulus Time Histogram
#

import matplotlib.pyplot as ppl
import numpy as np
import pytest
import dask.distributed as dd

# syncopy imports
import syncopy as spy
from syncopy.shared.errors import SPYValueError
from syncopy.tests import synth_data as sd
from syncopy.statistics.spike_psth import available_outputs


def get_spike_data(nTrials = 10, seed=None):
    return sd.poisson_noise(nTrials,
                            nUnits=3,
                            nChannels=2,
                            nSpikes=10_000,
                            samplerate=10_000,
                            seed=seed)


def get_spike_cfg():
    cfg = spy.StructDict()
    cfg.binsize = 0.3
    cfg.latency = [-.5, 1.5]
    return cfg


class TestPSTH:

    # synthetic spike data
    spd = get_spike_data(seed=42)

    def test_psth_binsize(self):

        cfg = spy.StructDict()
        cfg.latency = 'maxperiod'  # default

        # directly in seconds
        cfg.binsize = 0.2
        counts = spy.spike_psth(self.spd, cfg)

        assert isinstance(counts, spy.TimeLockData)
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
        counts = spy.spike_psth(self.spd, cfg)

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
        counts = spy.spike_psth(self.spd, cfg)

        # check that histogram time points are less than 1
        # delta_t away from the minimal interval boundaries
        assert np.abs(trl_starts.max() - counts.time[0][0]) < delta_t
        assert np.abs(trl_ends.min() - counts.time[0][-1]) < delta_t

        # check that there are NO NaNs as all trials have data
        # in this minimal interval
        assert not np.any(np.isnan(counts.data[:]))

        # -- prestim --> only events with t < 0
        cfg.latency = 'prestim'
        counts = spy.spike_psth(self.spd, cfg)

        assert np.all(counts.time[0] <= 0)

        # -- poststim --> only events with t > 0
        cfg.latency = 'poststim'
        counts = spy.spike_psth(self.spd, cfg)

        assert np.all(counts.time[0] >= 0)

        # -- finally the manual latency interval --
        # this is way to big, so we have many NaNs (empty bins)
        cfg.latency = [-.5, 1.5]   # in seconds
        assert cfg.latency[0] < trl_starts.min()
        assert cfg.latency[1] > trl_ends.max()

        counts = spy.spike_psth(self.spd, cfg)
        # check that histogram time points are less than 1
        # delta_t away from the manual set interval boundaries
        assert np.abs(cfg.latency[0] - counts.time[0][0]) <= delta_t
        # the midpoint gets rounded down, so the last time point is close to
        # 1 delta_t off actually..
        assert np.allclose(np.abs(cfg.latency[1] - counts.time[0][-1]), delta_t)

        # check that there are NaNs as the interval is way too large
        assert np.any(np.isnan(counts.data[:]))

    def test_psth_vartriallen(self):

        """
        Test setting vartriallen to False excludes trials which
        don't cover the latency time window
        """

        # everything else default
        cfg = spy.StructDict()
        cfg.vartriallen = False

        starts, ends = self.spd.trialintervals[:, 0], self.spd.trialintervals[:, -1]

        # choose latency which excludes 3 trials because
        # of starting time (condition is <= latency)
        cfg.latency = [np.sort(starts)[-4], ends.min()]
        counts = spy.spike_psth(self.spd, cfg)

        # check that 3 trials were excluded
        assert len(self.spd.trials) - len(counts.trials) == 3

        # choose latency which excludes 3 trials because
        # of end time (condition is >= latency)
        cfg.latency = [starts.max(), np.sort(ends)[3]]
        counts = spy.spike_psth(self.spd, cfg)

        # check that 3 trials were excluded
        assert len(self.spd.trials) - len(counts.trials) == 3

        # setting latency to 'maxperiod' with vartriallen=False
        # excludes all trials which raises an error
        cfg.latency = 'maxperiod'
        with pytest.raises(SPYValueError,
                           match='no trial that completely covers the latency window'):
            counts = spy.spike_psth(self.spd, cfg)

        # setting latency to 'minperiod' with vartriallen=False
        # excludes no trials by definition of 'minperiod'
        cfg.latency = 'minperiod'
        counts = spy.spike_psth(self.spd, cfg)
        # check that 0 trials were excluded
        assert len(self.spd.trials) - len(counts.trials) == 0

    def test_psth_outputs(self):

        cfg = spy.StructDict()
        cfg.latency = 'minperiod'  # to avoid NaNs
        cfg.output = 'spikecount'
        cfg.binsize = 0.1  # in seconds

        counts = spy.spike_psth(self.spd, cfg)

        # -- plot single trial statistics --
        # single trials have high variance, see below

        last_data = np.zeros(counts.time[0].size)
        for chan in counts.channel:
            bars = counts.show(trials=5, channel=chan)
            ppl.bar(counts.time[0], bars, alpha=0.7, bottom=last_data,
                    width=0.9 / counts.samplerate, label=chan)
            # for stacking
            last_data += bars
        ppl.legend()
        ppl.xlabel('time (s)')
        ppl.ylabel('spike counts')

        # -- plot mean and variance --

        # shows that each channel-unit combination
        # has a flat distribution as expected for poisson noise
        # however the absolute intensity differs: we have more and less
        # active channels/units by synthetic data costruction

        ppl.figure()
        ppl.title("Trial statistics")
        last_data = np.zeros(len(counts.time[0]))
        for chan in range(len(counts.channel)):
            bars = counts.avg[:, chan]
            yerr = counts.var[:, chan]
            ppl.bar(counts.time[0], bars, alpha=0.7, bottom=last_data,
                    width=0.9 / counts.samplerate, label=chan, yerr=yerr, capsize=2)
            # for stacking
            last_data += bars
        ppl.legend()
        ppl.xlabel('time (s)')
        ppl.ylabel('spike counts')

        cfg.output = 'rate'  # the default
        rates = spy.spike_psth(self.spd, cfg)

        # check that the rates are just the counts times samplerate
        assert counts * counts.samplerate == rates

        # this gives the spike histogram as normalized density
        cfg.output = 'proportion'
        cfg.latency = 'maxperiod'  # to provoke NaNs
        spike_densities = spy.spike_psth(self.spd, cfg)

        # check that there are NaNs as not all trials have data
        # in this maximal interval (due to start/end randomization)
        assert np.any(np.isnan(spike_densities.data[:]))

        # check for one arbitrary trial should be enough
        for chan in spike_densities.channel:
            integral = np.nansum(spike_densities.show(trials=2, channel=chan))
            assert np.allclose(integral, 1, atol=1e-3)

    def test_psth_exceptions(self):

        cfg = spy.StructDict()

        # -- output validation --

        # invalid string
        cfg.output = 'counts'
        with pytest.raises(SPYValueError,
                           match="expected one of"):
            spy.spike_psth(self.spd, cfg)

        # invalid type
        cfg.output = 12
        with pytest.raises(SPYValueError,
                           match="expected one of"):
            spy.spike_psth(self.spd, cfg)

        # -- binsize validation --

        cfg.output = 'rate'
        # no negative binsizes
        cfg.binsize = -0.2
        with pytest.raises(SPYValueError,
                           match="expected value to be greater"):
            spy.spike_psth(self.spd, cfg)

        cfg.latency = [0, 0.2]
        # binsize larger than time interval
        cfg.binsize = 0.3
        with pytest.raises(SPYValueError,
                           match="less or equals 0.2"):
            spy.spike_psth(self.spd, cfg)

        # not available rule
        cfg.binsize = 'sth'
        with pytest.raises(SPYValueError,
                           match="expected one of"):
            spy.spike_psth(self.spd, cfg)

        # -- latency validation --

        cfg.binsize = 0.1
        # not available latency
        cfg.latency = 'sth'
        with pytest.raises(SPYValueError,
                           match="expected one of"):
            spy.spike_psth(self.spd, cfg)

        # latency not ordered
        cfg.latency = [0.1, 0]
        with pytest.raises(SPYValueError,
                           match="expected start < end"):
            spy.spike_psth(self.spd, cfg)

        # latency completely outside of data
        cfg.latency = [-999, -99]
        with pytest.raises(SPYValueError,
                           match="expected end of latency window"):
            spy.spike_psth(self.spd, cfg)
        cfg.latency = [99, 999]
        with pytest.raises(SPYValueError,
                           match="expected start of latency window"):
            spy.spike_psth(self.spd, cfg)

    def test_psth_chan_unit_mapping(self):
        """
        Test that non-existent channel/unit combinations
        are accounted for correctly
        """

        # check that unit 1 really is there
        assert np.any(self.spd.data[:, 2] == 1)
        counts = spy.spike_psth(self.spd, output='spikecount')
        assert 'channel0_unit1' in counts.channel
        assert 'channel1_unit1' in counts.channel

        # get rid of unit 1
        pruned_spd = self.spd.selectdata(unit=[0, 2])
        # check that unit 1 really is gone
        assert np.all(pruned_spd.data[:, 2] != 1)

        pruned_counts = spy.spike_psth(pruned_spd, output='spikecount')
        # check that unit 1 really is gone
        assert len(pruned_counts.channel) < len(counts.channel)
        assert 'channel0_unit1' not in pruned_counts.channel
        assert 'channel1_unit1' not in pruned_counts.channel

        # check that counts for remaining channel/units are unchanged
        for chan in pruned_counts.channel:
            assert np.array_equal(counts.show(trials=4, channel=chan),
                                  pruned_counts.show(trials=4, channel=chan),
                                  equal_nan=True)

        # now the same with an active in-place selection
        # Already fixed: #332
        # get rid of unit 1
        # self.spd.selectdata(unit=[0, 2], inplace=True)

        pruned_counts2 = spy.spike_psth(self.spd, output='spikecount', select={'unit': [0, 2]})

        # check that unit 1 really is gone
        assert len(pruned_counts2.channel) < len(counts.channel)
        assert 'channel0_unit1' not in pruned_counts2.channel
        assert 'channel1_unit1' not in pruned_counts2.channel
        # check that counts for remaining channel/units are unchanged
        for chan in pruned_counts2.channel:
            assert np.array_equal(counts.show(trials=4, channel=chan),
                                  pruned_counts2.show(trials=4, channel=chan),
                                  equal_nan=True)

    def test_parallel_selection(self, testcluster):

        cfg = spy.StructDict()
        cfg.latency = 'minperiod'
        cfg.parallel = True

        client = dd.Client(testcluster)

        # test standard run
        counts = spy.spike_psth(self.spd, cfg)
        # check that there are NO NaNs as all trials
        # have data in `minperiod` by definition
        assert not np.any(np.isnan(counts.data[:]))

        # test channel selection
        cfg.select = {'channel': 0}
        counts = spy.spike_psth(self.spd, cfg)
        assert all(['channel1' not in chan for chan in counts.channel])

        client.close()


if __name__ == '__main__':
    T1 = TestPSTH()
    spd = T1.spd
    trl0 = spd.trials[0]
    spd.selectdata(unit=[0,2], inplace=True)
    arr1 = spd.selection._get_trial(1)
