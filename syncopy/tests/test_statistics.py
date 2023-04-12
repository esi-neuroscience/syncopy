# -*- coding: utf-8 -*-
#
# Test proper functionality of Syncopy's `ContinuousData` class + subclasses
#

# Builtin/3rd party package imports
import os
import pytest
import numpy as np
import dask.distributed as dd
import matplotlib.pyplot as ppl
import scipy.stats as st

# Local imports
import syncopy as spy
from syncopy.datatype import AnalogData, SpectralData, CrossSpectralData
from syncopy.shared.errors import SPYValueError, SPYTypeError
from syncopy.tests import helpers
from syncopy.tests import synth_data as sd
from syncopy.statistics import jackknifing as jk
from syncopy.connectivity.AV_compRoutines import NormalizeCrossSpectra


class TestSumStatistics:

    # initialize rng instance
    rng = np.random.default_rng(helpers.test_seed)

    # lognormal distribution parameters
    mu, sigma = 2, .5

    nTrials = 4
    nChannels = 3
    nSamples = 10
    nFreq = 10
    nTaper = 2

    ln_samples = rng.lognormal(mu, sigma, size=(nTrials, nSamples, nChannels))
    adata = AnalogData(data=[trl for trl in ln_samples], samplerate=1)

    ln_samples = rng.lognormal(mu, sigma, size=(nTrials, nSamples, nTaper, nFreq, nChannels))
    spec_data = SpectralData(data=[trl for trl in ln_samples], samplerate=1)

    ln_samples = rng.lognormal(mu, sigma, size=(nTrials, nSamples, nFreq, nChannels, nChannels))
    crossspec_data = CrossSpectralData(data=[trl for trl in ln_samples], samplerate=1)

    data_types = [adata, spec_data, crossspec_data]

    def test_dim_statistics(self):
        """
        Tests statistics over dimensions, not trials
        """

        # check only 2nd trial
        test_trial = 1

        for spy_data in self.data_types:
            for dim in spy_data.dimord:

                # get index of dimension
                axis = spy_data.dimord.index(dim)

                # -- test average --

                # top-level function
                spy_res1 = spy.mean(spy_data, dim=dim)
                # check only one trial
                npy_res = np.mean(spy_data.trials[test_trial], axis=axis)

                self._check_trial('mean', dim, spy_res1, npy_res, trial=test_trial)

                # --- test variance ---

                # top-level function
                spy_res1 = spy.var(spy_data, dim=dim)
                # check only one trial
                npy_res = np.var(spy_data.trials[test_trial], axis=axis)

                self._check_trial('var', dim, spy_res1, npy_res, trial=test_trial)

                # --- test standard deviation ---

                # top-level function
                spy_res1 = spy.std(spy_data, dim=dim)
                # check only one trial
                npy_res = np.std(spy_data.trials[test_trial], axis=axis)

                self._check_trial('std', dim, spy_res1, npy_res, trial=test_trial)

                # --- test median ---

                # top-level function
                spy_res1 = spy.median(spy_data, dim=dim)
                # check only one trial
                npy_res = np.median(spy_data.trials[test_trial], axis=axis)

                self._check_trial('median', dim, spy_res1, npy_res, trial=test_trial)

    def _check_trial(self, operation, dim, spy_res, npy_res, trial=1):
        """
        Test that direct numpy stats give the same results
        for a single trial
        """

        # show returns list of trials, pick only one
        # show also squeezes out the singleton dimension
        # which remains after the statistic got computed!
        trial_result_spy = spy_res.show()[trial]
        assert np.allclose(trial_result_spy, npy_res)

        # check the dimension label was set to the statistical operation
        if dim not in ['freq', 'time']:
            assert getattr(spy_res, dim) == operation
        # numerical dimension labels get set to 0 (axis is gone)
        elif dim == 'time':
            assert spy_res.time[trial] == 0
        elif dim == 'freq':
            assert spy_res.freq == 0

    def test_trial_statistics(self):
        """
        Test statistics over trials
        """

        # --- test statistics trial average against CR trial average ---

        spec = spy.freqanalysis(self.adata, keeptrials=True)
        # trigger trial average after spectral estimation
        spec1 = spy.mean(spec, dim='trials')

        # trial average via CR keeptrials
        spec2 = spy.freqanalysis(self.adata, keeptrials=False)

        assert len(spec1.trials) == 1
        assert np.allclose(spec1.data, spec2.data)

        # --- test trial var and std ---
        for data in self.data_types:

            spy_var = spy.var(data, dim='trials')

            # reshape to get rid of trial stacking along time axis
            # array has shape (nTrials, nSamples, ..rest-of-dims..)
            arr = data.data[()].reshape(self.nTrials, self.nSamples, *data.data.shape[1:])
            # now compute directly over the trial axis
            npy_var = np.var(arr, axis=0)

            assert len(spy_var.trials) == 1
            assert np.allclose(npy_var, spy_var.data)

            spy_std = spy.std(data, dim='trials')

            # reshape to get rid of trial stacking along time axis
            # array has shape (nTrials, nSamples, ..rest-of-dims..)
            arr = data.data[()].reshape(self.nTrials, self.nSamples, *data.data.shape[1:])
            # now compute directly over the trial axis
            npy_std = np.std(arr, axis=0)

            assert len(spy_var.trials) == 1
            assert np.allclose(npy_std, spy_std.data)

    def test_selections(self):

        # got 10 samples with 1s samplerate,so time is [-1, ..., 8]
        sdict1 = {'trials': [1, 3], 'latency': [2, 6]}
        res = spy.mean(self.adata, dim='channel', select=sdict1)
        assert len(res.trials) == 2
        assert self.adata.time[0].min() == -1
        assert res.time[0].min() == 2
        assert self.adata.time[0].max() == 8
        assert res.time[0].max() == 6

        # freq axis is [0, ..., 9]
        sdict2 = {'channel': [0, 2], 'frequency': [1, 5]}
        res = spy.var(self.spec_data, dim='trials', select=sdict2)
        assert np.all(res.channel == np.array(['channel1', 'channel3']))
        assert np.all(res.freq == np.arange(1, 6))

        # check at least a few times that the statistics are indeed
        # computed correctly on the trimmed down data
        sdict3 = {'trials': [1, 3]}
        res = spy.mean(self.crossspec_data, dim='trials', select=sdict3)
        # reshape to extract trial separated arrays
        arr = self.crossspec_data.data[()].reshape(self.nTrials, self.nSamples,
                                                   self.nFreq, self.nChannels, self.nChannels)
        # now cut out the same 2 trials and average
        npy_res = arr[1::2].mean(axis=0)
        assert np.allclose(npy_res, res.data)

        sdict4 = {'channel': [0, 2]}
        res = spy.mean(self.spec_data, dim='channel', select=sdict4)
        # now cut out the same 2 channels and average, dimord is (time, taper, freq, channel)
        npy_res = self.spec_data.data[..., ::2].mean(axis=-1)
        # check only 1st trial
        assert np.allclose(npy_res[:self.nSamples], res.show(trials=0))

        # one last time for the freq axis
        sdict5 = {'frequency': [1, 4]}
        res = spy.median(self.spec_data, dim='freq', select=sdict5)
        # cut out same frequencies directly from the dataset array
        npy_res = np.median(self.spec_data.data[..., 1:5, :], axis=2)
        # check only 2nd trial
        assert np.allclose(npy_res[self.nSamples:2 * self.nSamples], res.show(trials=1))

    def test_exceptions(self):

        with pytest.raises(SPYValueError) as err:
            spy.mean(self.adata, dim='sth')
        assert "expected one of ['time', 'channel']" in str(err.value)

        # unequal trials and trial average can't work
        with pytest.raises(SPYValueError) as err:
            # to not screw sth up
            adata_cpy = spy.copy(self.adata)
            trldef = adata_cpy.trialdefinition
            trldef[2] = [21, 25, -1]
            adata_cpy.trialdefinition = trldef
            spy.mean(adata_cpy, dim='trials')
        assert "found trials of different shape" in str(err.value)

    def test_stat_parallel(self, testcluster):
        client = dd.Client(testcluster)
        self.test_selections()
        # should have no effect here
        self.test_trial_statistics()
        client.close()

    def test_itc(self, do_plot=True):

        adata = sd.white_noise(100,
                               nSamples=1000,
                               nChannels=2,
                               samplerate=500,
                               seed=42)

        # add simple 60Hz armonic
        adata += sd.harmonic(100,
                             freq=60,
                             nSamples=1000,
                             nChannels=2,
                             samplerate=500)

        trials = []
        # add frequency drift along trials
        # note same initial phase
        # so ITC will be ~1 at time 0
        freq = 30
        dfreq = 2 / 100  # frequency difference
        for trl in adata.trials:
            # start at 0s
            dat = np.cos(2 * np.pi * freq * (adata.time[0] + 1))
            trials.append(trl + np.c_[dat, dat])
            freq += dfreq
        adata = spy.AnalogData(data=trials, samplerate=500)

        tf_spec = spy.freqanalysis(adata, method='mtmconvol',
                                   t_ftimwin=0.5,
                                   output='fourier',
                                   foilim=[0, 100])

        # test also taper averaging
        spec = spy.freqanalysis(adata, foilim=[0, 100],
                                output='fourier', tapsmofrq=.5, keeptapers=True)

        # -- calculate itc --
        itc = spy.itc(spec)
        tf_itc = spy.itc(tf_spec)

        assert isinstance(tf_itc, spy.SpectralData)

        assert np.all(np.imag(itc.data[()]) == 0)
        assert itc.data[()].max() <= 1
        assert itc.data[()].min() >= 0

        # high itc around the in phase 60Hz
        assert np.all(itc.show(frequency=60) > 0.6)
        # low (time averaged) itc around the drifters
        assert np.all(itc.show(frequency=30) < 0.25)

        assert np.all(np.imag(tf_itc.data[()]) == 0)
        assert tf_itc.data[()].max() <= 1
        assert tf_itc.data[()].min() >= 0
        assert np.allclose(tf_itc.time[0], tf_spec.time[0])

        if do_plot:

            # plot tf power spectrum
            # after 500 samples / 1s
            fig, ax = ppl.subplots()
            for idx in [0, 50, 99]:
                power500 = np.abs(tf_spec.trials[idx][500, 0, :, 0])
                ax.plot(tf_spec.freq, power500, label=f"trial {idx}")
            # note that the 30Hz peak wandered to ~31Hz
            # hence the phases will decorrelate over time
            ax.legend()
            ax.set_xlabel('frequency (Hz)')

            # plot ITCs
            itc.singlepanelplot(channel=1)
            tf_itc.singlepanelplot(channel=1)

            # plot tf ITC time profiles
            fig, ax = ppl.subplots()
            ax.set_xlabel('time (s)')
            for frq in [31, 60, 10]:
                itc_profile = tf_itc.show(frequency=frq, channel=0)
                ax.plot(tf_itc.time[0], itc_profile, label=f"{frq}Hz")
            ax.legend()
            ax.set_title("time dependent ITC")


class TestJackknife:

    def test_jk_avg(self):
        """
        Mean estimation via jackknifing yields exactly the same
        results as the direct estimate (sample mean/variance)
        and hence can directly serve as a straightforward test
        """

        # create test data
        nTrials = 10
        adata = spy.AnalogData(data=[i * np.ones((5, 3)) for i in range(nTrials)],
                               samplerate=7)

        # to test for property propagation
        adata.channel = [f'chV_{i}' for i in range(1, 4)]
        raw_est = spy.mean(adata, dim='time', keeptrials=True)

        # first compute all the leave-one-out (loo) replicates
        replicates = jk.trial_avg_replicates(raw_est)
        # as many replicates as there are trials
        assert len(replicates.trials) == len(adata.trials)

        # direct estimate is just the trial average
        direct_est = spy.mean(raw_est, dim='trials')

        # now compute bias and variance
        bias, variance = jk.bias_var(direct_est, replicates)

        # no bias for mean estimation
        assert np.allclose(bias.data[()], np.zeros(bias.data.shape))

        # jackknife variance is here the same as sample variance over trials
        # yet we have to correct for the denominator (N-1 vs. N)
        direct_var = spy.var(raw_est, dim='trials').trials[0]
        assert np.allclose(nTrials / (nTrials - 1) * direct_var, variance.trials[0])

        # check properties
        assert np.all(bias.channel == adata.channel)
        assert bias.samplerate == adata.samplerate
        assert np.all(variance.channel == adata.channel)
        assert variance.samplerate == adata.samplerate

        # the bias corrected jackknife estimate
        jack_estimate = direct_est - bias

        # as there is no bias, this is the same as the direct estimate
        assert np.allclose(jack_estimate.data, direct_est.data)

    def test_jk_csd(self, **kwargs):
        """
        Jackknife cross-spectral densities, here again the trivial average/variance
        yields the same results.
        """
        nTrials = 10
        nSamples = 500
        adata = 10 * sd.white_noise(nTrials, nSamples=nSamples, seed=helpers.test_seed)
        # to test for property propagation
        adata.channel = [f'chV_{i}' for i in range(1, 3)]

        # -- still trivial CSDs --

        # single trial cross spectra (not densities!)
        cross_spectra = spy.connectivityanalysis(adata,
                                                 method='csd',
                                                 keeptrials=True)

        # direct cross spectral density estimate (must be a trial average)
        # is here just the trial average
        csd = spy.mean(cross_spectra, dim='trials')

        # compute avg replicates
        replicates_avg = jk.trial_avg_replicates(cross_spectra)

        # now compute bias and variance
        bias, variance = jk.bias_var(csd, replicates_avg)

        # check properties
        assert np.all(replicates_avg.channel_i == adata.channel)
        assert np.all(bias.channel_j == adata.channel)
        assert np.all(variance.channel_j == adata.channel)

        # now again as this is still just a simple average
        # there can be no real bias
        assert np.allclose(bias.data[()], np.zeros(bias.data.shape), atol=1e-5)

        # direct variances still coincide,
        # `show` strips of empty time axis
        direct_var = spy.var(cross_spectra, dim='trials').show()
        assert np.allclose(direct_var * (nTrials / (nTrials - 1) + 0j),
                           variance.show())

    def test_jk_coh(self, **kwargs):
        """
        Jackknife a coherence analysis.

        For the coherence confidence intervals see:
        "Tables of the distribution of the coefficient of coherence for
        stationary bivariate Gaussian processes, Sandia monograph by Amos and Koopmans(1963)"
        """

        nTrials = 50
        nSamples = 1000
        # sufficient to check this entry
        show_kwargs = {'channel_i': 0, 'channel_j': 1}
        adata = sd.white_noise(nTrials, nSamples=nSamples, seed=helpers.test_seed)

        # confidence for 100 trials from
        # above mentioned publication for squared coherence
        ci95 = {30: 0.98, 50: 0.06, 100: 0.03}

        # important to match between
        # replicates and direct estimate!
        output = 'pow'

        # direct estimate
        coh = spy.connectivityanalysis(adata,
                                       method='coh',
                                       output=output)

        # first check that we got the right statistics
        # by asserting that less of 5% of the freq. bins are outside
        # the (one-sided) 95% conf. interval
        assert np.sum(coh.show(**show_kwargs) > ci95[nTrials]) / coh.freq.size < 0.05

        # single trial cross spectra (not densities!)
        cross_spectra = spy.connectivityanalysis(adata,
                                                 method='csd',
                                                 keeptrials=True)

        # first create trivial avg/csd replicates
        replicates_avg = jk.trial_avg_replicates(cross_spectra)

        # -- compute coherence replicates --

        # from those compute jackknife replicates of the coherence
        CR = NormalizeCrossSpectra(output=output)
        replicates_coh = CrossSpectralData(dimord=coh.dimord)
        log_dict = {}
        # now fire up CR on all loo averages to get
        # the coherence jackknife replicates
        CR.initialize(replicates_avg, replicates_coh._stackingDim, chan_per_worker=None)
        CR.compute(replicates_avg, replicates_coh, parallel=kwargs.get("parallel"),
                   log_dict=log_dict)

        assert len(replicates_coh.trials) == nTrials
        assert np.all(replicates_coh.channel_i == adata.channel)

        # now compute bias and variance
        bias, variance = jk.bias_var(coh, replicates_coh)

        # here we have some actual bias
        assert not np.allclose(bias.data[()], np.zeros(bias.data.shape), atol=1e-5)


        # look at the 0,1 entry
        b01, v01, c01 = (bias.show(**show_kwargs),
                         variance.show(**show_kwargs),
                         coh.show(**show_kwargs))

        # standard error of the mean
        SEM = np.sqrt(v01 / nTrials)

        fig, ax = ppl.subplots()
        ax.set_title(f"Coherence of white noise with nTrials={nTrials}")
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("Coherence $|C|^2$")
        ax.plot(coh.freq, c01, marker='.')
        ax.fill_between(coh.freq, c01, c01 + 1.96 * SEM, color='k', alpha=0.3, label='95% Jackknife CI')
        # truncate to 0 for negative values
        ci2 = c01 - 1.96 * SEM
        ci2[ci2 < 0] = 0
        ax.fill_between(coh.freq, c01, ci2, color='k', alpha=0.3)
        ax.set_xlim((100, 150))
        ax.legend()

        # calculate the z-scores from the jackknife estimate
        # and jackknife variance for 0 coherence
        # as 0-hypothesis (we have uncorrelated noise)
        Zs = (c01 - b01) / SEM

        # now get p-values from survival function
        pvals = st.norm.sf(Zs)

        fig, ax = ppl.subplots()
        ax.set_title("Jackknife p-values for $H_0$: $|C|^2 = 0$")
        ax.set_xlabel("Coherence $|C|^2$")
        ax.set_ylabel("p-value")
        ax.plot(c01, pvals, 'o', alpha=0.4, c='k', ms=3.5, mec='w')
        ax.plot([0, c01.max()], [0.05, 0.05], 'k--', label='5%')
        ax.legend()

        # turns out, the jackknife confidence intervals are quite conservative
        # and no frequency bin has a coherence high enough to reject the C=0 hypothesis
        # we could still expect to get a 'significant' coherence max 5% of the time
        assert np.sum(pvals < 0.05) / coh.freq.size < 0.05

        # finally fire up the frontend and compare results
        res = spy.connectivityanalysis(adata, method='coh', jackknife=True, output=output)

        assert np.allclose(res.jack_var, variance.data, atol=1e-5)
        assert np.allclose(res.jack_bias, bias.data, atol=1e-5)

    def test_jk_frontend(self):

        # no seed needed here
        adata = spy.AnalogData(data=[i * np.random.randn(5, 3) for i in range(3)],
                               samplerate=7)

        # test check for boolean type
        with pytest.raises(SPYTypeError, match='expected boolean'):
            spy.connectivityanalysis(adata, method='coh', jackknife=3)

        # check that jack attributes are not appended if no jackknifing was done
        res = spy.connectivityanalysis(adata, method='corr')
        assert not hasattr(res, 'jack_var')
        assert not hasattr(res, 'jack_bias')

        res = spy.connectivityanalysis(adata, method='coh', jackknife=False)
        assert not hasattr(res, 'jack_var')
        assert not hasattr(res, 'jack_bias')

        res = spy.connectivityanalysis(adata, method='granger', jackknife=False)
        assert not hasattr(res, 'jack_var')
        assert not hasattr(res, 'jack_bias')

    def test_jk_granger(self):

        AdjMat = np.zeros((2,2))
        # weak coupling 1 -> 0
        AdjMat[1, 0] = 0.025
        nTrials = 35
        adata = sd.AR2_network(nTrials, AdjMat=AdjMat, seed=42)
        # true causality is at 200Hz
        flims = [190, 210]

        # direct estimate
        res = spy.connectivityanalysis(adata, method='granger',
                                       jackknife=True,
                                       tapsmofrq=5)
        # there will be bias
        assert not np.allclose(res.jack_bias, np.zeros(res.data.shape), atol=1e-5)

        b10, v10, g10 = (res.jack_bias[0, :, 1, 0],
                         res.jack_var[0, :, 1, 0],
                         res.show(channel_i=1, channel_j=0)
                         )
        # standard error of the mean
        SEM = np.sqrt(v10 / nTrials)

        # plot confidence intervals
        fig, ax = ppl.subplots()
        ax.set_title(f"Granger causality between weakly coupled AR(2)")
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("Granger")
        ax.plot(res.freq, g10, label=f"nTrials={nTrials}")
        ax.fill_between(res.freq, g10, g10 + 1.96 * SEM, color='k', alpha=0.3, label='95% Jackknife CI')
        ci2 = g10 - 1.96 * SEM
        ci2[ci2 < 0] = 0
        ax.fill_between(res.freq, g10, ci2, color='k', alpha=0.3)
        ax.plot([flims[0], flims[0]], [0, 0.04], '--', alpha=0.5, lw=2, c='red')
        ax.plot([flims[-1], flims[-1]], [0, 0.04], '--', alpha=0.5, lw=2, c='red')
        ax.legend()
        fig.tight_layout()

        # calculate the z-scores from the jackknife estimate
        # and jackknife variance for 0 granger causality
        # as 0-hypothesis
        Zs = (g10 - b10) / SEM

        # now get p-values from survival function
        pvals = st.norm.sf(Zs)

        # boolean indices of frequency interval with true causality
        bi = (res.freq > flims[0]) & (res.freq < flims[-1])

        fig, ax = ppl.subplots()
        ax.set_title("Jackknife p-values for $H_0$: Granger = 0")
        ax.set_xlabel("Granger causality")
        ax.set_ylabel("p-value")
        ax.plot(g10[~bi], pvals[~bi], 'o', alpha=0.4, c='k', ms=5, mec='w')
        ax.plot(g10[bi], pvals[bi], 'o', alpha=0.4, c='red', ms=4, label=f'{flims[0]}Hz-{flims[-1]}Hz')

        ax.plot([0, g10.max()], [0.05, 0.05], 'k--', label='5%')
        ax.legend()

        # make sure most (>95%) frequency bins outside the causality region have high p-value
        assert np.sum(pvals[~bi] > 0.05) / (res.freq.size - np.sum(bi)) > 0.95

        # check that at least 80% of causality values within the freq interval are below
        # the 5% significance interval and hence are deteceted as true positives
        assert np.sum(pvals[bi] < 0.05) / bi[bi].size > 0.8


if __name__ == '__main__':

    T1 = TestSumStatistics()
    T2 = TestJackknife()
