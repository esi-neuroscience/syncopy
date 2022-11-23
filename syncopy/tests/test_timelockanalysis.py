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
from syncopy.shared.errors import SPYValueError
from syncopy.tests import synth_data


class TestTimelockanalysis:

    nTrials = 300
    nChannels = 3
    nSamples = 500
    fs = 200

    # moderate phase diffusion, same initial phase/value!
    adata = synth_data.phase_diffusion(nTrials,
                                       rand_ini=False,
                                       samplerate=fs,
                                       nSamples=nSamples,
                                       nChannels=nChannels,
                                       freq=40,
                                       eps=0.01)

    # change one trial size

    trldef = adata.trialdefinition
    trldef[1] = [500, 700, -200]
    adata.trialdefinition = trldef

    def test_timelockanalysis(self, **kwargs):

        def_test = True if len(kwargs) == 0 else False

        cfg = spy.StructDict()
        cfg.latency = 'maxperiod'  # default
        cfg.covariance = True
        cfg.keeptrials = False

        tld = spy.timelockanalysis(self.adata, cfg)

        assert isinstance(tld, spy.TimeLockData)
        # check that all trials have the same 'time locked' time axis
        # as enfored by TimeLockData.trialdefinition setter
        assert len(set([t.size for t in tld.time])) == 1

        # check that one trial got kicked out
        assert 1 == len(self.adata.trials) - len(tld.trials)

        assert isinstance(tld.avg, h5py.Dataset)
        assert isinstance(tld.var, h5py.Dataset)
        assert isinstance(tld.cov, h5py.Dataset)

        # check that the results are the same when kicking
        # out the offending trial via the same latency selection
        ad = spy.selectdata(self.adata, latency='maxperiod')
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
        if def_test:
            fig, ax = avg.singlepanelplot()
            ax.set_title('Trial mean')
            fig.tight_layout()
            fig, ax = var.singlepanelplot()
            ax.set_title('Trial variance')
            fig.tight_layout()

    def test_timelockanalysis_latency(self):

        """Test all available `latency` (time window interval) settings"""

        # check that now all trial have been cut to the shortest one
        tld = spy.timelockanalysis(self.adata, latency='minperiod')
        assert np.all(np.diff(tld.trialdefinition[:, :2], axis=1))

    def test_exceptions(self):

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

    def test_parallel_selection(self, testcluster=None):

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

        # test toilim selection
        # FIXME: Not supported atm, see #348
        # cfg.select['toilim'] = [0.1, 0.2]
        # counts = spy.spike_psth(self.spd, cfg)
        # assert all(['channel1' not in chan for chan in counts.channel])

        client.close()


if __name__ == '__main__':
    T1 = TestTimelockanalysis()
