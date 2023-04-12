# -*- coding: utf-8 -*-
#
# Test Timelockanalysis and latency selection
#

import matplotlib.pyplot as ppl
import numpy as np
import pytest
import dask.distributed as dd
import h5py

# syncopy imports
import syncopy as spy
from syncopy.shared.errors import SPYValueError, SPYTypeError
from syncopy.tests import synth_data


class TestTimelockanalysis:

    nTrials = 10
    nChannels = 3
    nSamples = 500
    fs = 200

    # create simple white noise
    # "real" data gets created for semantic test
    adata = synth_data.white_noise(nTrials, samplerate=fs,
                                   nSamples=nSamples,
                                   nChannels=nChannels,
                                   seed=42)

    # change trial sizes, original interval is [-1, 1.495] seconds
    trldef = adata.trialdefinition
    trldef[1] = [500, 700, -200]  # [-1, -0.005] seconds
    trldef[2] = [680, 960, -20]  # [-0.1, 1.295] seconds
    trldef[3] = [1000, 1200, -100]  # [-0.5, 0.495] seconds
    adata.trialdefinition = trldef

    # in samples
    overlap = (np.min(adata.trialintervals[:, 1]) -\
        np.max(adata.trialintervals[:, 0])) * fs + 1

    def test_timelockanalysis(self):

        # create bigger dataset for statistics only here
        # moderate phase diffusion, same initial phase/value!
        adata = synth_data.phase_diffusion(nTrials=300,
                                           rand_ini=False,
                                           samplerate=self.fs,
                                           nSamples=self.nSamples,
                                           nChannels=self.nChannels,
                                           freq=40,
                                           eps=0.01,
                                           seed=42)

        # change trial sizes, original interval is [-1, 1.495] seconds
        trldef = adata.trialdefinition
        trldef[1] = [500, 700, -200]  # [-1, -0.005] seconds
        trldef[2] = [680, 960, -20]  # [-0.1, 1.295] seconds
        trldef[3] = [1000, 1200, -100]  # [-0.5, 0.495] seconds
        adata.trialdefinition = trldef

        cfg = spy.StructDict()
        cfg.latency = 'maxperiod'  # default
        cfg.covariance = True
        cfg.keeptrials = False

        tld = spy.timelockanalysis(adata, cfg)

        assert isinstance(tld, spy.TimeLockData)
        # check that all trials have the same 'time locked' time axis
        # as enfored by TimeLockData.trialdefinition setter
        assert len(set([t.size for t in tld.time])) == 1

        # check that 3 trials got kicked out
        assert 3 == len(adata.trials) - len(tld.trials)

        assert isinstance(tld.avg, h5py.Dataset)
        assert isinstance(tld.var, h5py.Dataset)
        assert isinstance(tld.cov, h5py.Dataset)

        # check that the results are the same when kicking
        # out the offending trial via the same latency selection
        ad = spy.selectdata(adata, latency='maxperiod')
        avg = spy.mean(ad, dim='trials')
        var = spy.var(ad, dim='trials')

        assert np.allclose(avg.data, tld.avg)
        assert np.allclose(var.data, tld.var)

        # over time the phases will diffuse and be uniform in [-pi, pi],
        # hence by transforming the random variable with x = cos(phases) the
        # signal values are distributed like 1 / sqrt(1 - x**2) * 1 / pi
        # the mean of that distribution is 0 and the variance is 1 / 2

        # in the beginning the mean will be close to 1 as cos(0) = 1
        assert np.all((tld.avg[0, :] > 0.99) & (tld.avg[0, :] <= 1))

        # later on when coherence is lost the mean will be very low
        assert np.all(tld.avg[-1, :] <= 0.1)

        # for the variance it is the opposite
        assert np.all(tld.var[0, :] <= 0.001)

        # converges statistically to 1/2
        assert np.all((tld.var[-1, :] > 0.45) & (tld.var[-1, :] <= 0.55))

        # check that covariance diagonal entries are the variance
        variances = np.mean([np.var(trl, axis=0, ddof=1) for trl in tld.trials], axis=0)
        assert variances.shape == (self.nChannels,)
        assert np.allclose(np.diagonal(tld.cov), variances)

        # here just check that off-diagonals have vastly lower covariance
        assert np.all(np.diagonal(tld.cov)[:-1] > 5 * np.diagonal(tld.cov, offset=1))

        # plot the Syncopy objects which have the same data as .avg and .var
        fig, ax = avg.singlepanelplot()
        ax.set_title('Trial mean')
        fig.tight_layout()
        fig, ax = var.singlepanelplot()
        ax.set_title('Trial variance')
        fig.tight_layout()

    def test_latency(self):

        """Test all available `latency` (time window interval) settings"""

        # first make sure we have unequal trials
        assert np.any(np.diff(self.adata.sampleinfo, n=2, axis=0) != 0)

        # check that now all trial have been cut to the overlap interval
        tld = spy.timelockanalysis(self.adata, latency='minperiod')
        assert np.all(np.diff(tld.sampleinfo) == self.overlap)

        # check that trials got kicked out and all times are smaller 0
        tld = spy.timelockanalysis(self.adata, latency='prestim')
        assert 3 == len(self.adata.trials) - len(tld.trials)
        # only trigger relative negative times ("pre-stimulus")
        assert np.all([tld.time[i] <= 0 for i in range(len(tld.trials))])

        # check that trials got kicked out and all times are larger than 0
        tld = spy.timelockanalysis(self.adata, latency='poststim')
        assert 3 == len(self.adata.trials) - len(tld.trials)
        # only trigger relative positive times ("post-stimulus")
        assert np.all([tld.time[i] >= 0 for i in range(len(tld.trials))])

        # finally manually set a latency window which excludes only 2 trials
        tld = spy.timelockanalysis(self.adata, latency=[-.1, 0.5])
        assert 2 == len(self.adata.trials) - len(tld.trials)

        # and ultimately check that we have no dangling selections
        # after all this
        assert self.adata.selection is None
        assert tld.selection is None

    def test_exceptions(self):

        cfg = spy.StructDict()

        # -- latency validation --

        # not available latency
        cfg.latency = 'sth'
        with pytest.raises(SPYValueError,
                           match="expected one of"):
            spy.timelockanalysis(self.adata, cfg)

        # latency not ordered
        cfg.latency = [0.1, 0]
        with pytest.raises(SPYValueError,
                           match="expected start < end"):
            spy.timelockanalysis(self.adata, cfg)

        # latency completely outside of data
        cfg.latency = [-999, -99]
        with pytest.raises(SPYValueError,
                           match="expected end of latency window"):
            spy.timelockanalysis(self.adata, cfg)

        cfg.latency = [99, 999]
        with pytest.raises(SPYValueError,
                           match="expected start of latency window"):
            spy.timelockanalysis(self.adata, cfg)

        # here we need to manually wipe the selection due to the
        # exceptioned runs above
        self.adata.selection = None

        # -- trial selection with both selection and keyword --
        with pytest.raises(SPYValueError,
                           match="expected either `trials != 'all'`"):
            spy.timelockanalysis(self.adata, trials=0, select={'trials': 8})


        # -- remaining parameters --

        cfg.latency = None
        cfg.covariance = 'fd'
        with pytest.raises(SPYTypeError,
                           match="expected bool"):
            spy.timelockanalysis(self.adata, cfg)

        with pytest.raises(SPYTypeError,
                           match="expected bool"):
            spy.timelockanalysis(self.adata, keeptrials=2)

        with pytest.raises(SPYValueError,
                           match="expected positive integer"):
            spy.timelockanalysis(self.adata, ddof='2')

        # here we need to manually wipe the selection due to the
        # exception runs above
        self.adata.selection = None

    def test_parallel_selection(self, testcluster):

        cfg = spy.StructDict()
        cfg.latency = 'minperiod'
        cfg.parallel = True

        client = dd.Client(testcluster)

        # test standard run
        tld = spy.timelockanalysis(self.adata, cfg)
        # check that there are NO NaNs
        assert not np.any(np.isnan(tld.data[:]))

        # test channel selection
        cfg.select = {'channel': 0}
        tld = spy.timelockanalysis(self.adata, cfg)
        assert all(['channel2' not in chan for chan in tld.channel])
        assert self.adata.selection is None

        # trial selection via FT compat parameter
        cfg.trials = [5, 6]
        cfg.select = None
        tld = spy.timelockanalysis(self.adata, cfg)
        assert len(tld.trials) == 2
        assert self.adata.selection is None

        # and via normal selection
        tld2 = spy.timelockanalysis(self.adata, select={'trials': [5, 6]})
        assert np.all(tld2.data[()] == tld.data[()])
        client.close()


if __name__ == '__main__':
    T1 = TestTimelockanalysis()
