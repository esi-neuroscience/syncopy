# -*- coding: utf-8 -*-
#
# Test proper functionality of Syncopy's `ContinuousData` class + subclasses
#

# Builtin/3rd party package imports
import pytest
import numpy as np
import dask.distributed as dd
import matplotlib.pyplot as ppl

# Local imports
import syncopy as spy
from syncopy.datatype import AnalogData, SpectralData, CrossSpectralData, TimeLockData
from syncopy.shared.errors import SPYValueError, SPYTypeError
from syncopy.shared.tools import StructDict
from syncopy.tests import helpers
from syncopy.tests import synth_data as sd


class TestStatistics:

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

    def test_stat_parallel(self, testcluster=None):
        client = dd.Client(testcluster)
        self.test_selections()
        # should have no effect here
        self.test_trial_statistics()

    def test_itc(self, do_plot=True):

        adata = sd.white_noise(100,
                               nSamples=1000,
                               nChannels=2,
                               samplerate=500)

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

        spec = spy.freqanalysis(adata, foilim=[0, 100], output='fourier')

        # -- calculate itc --
        itc = spy.itc(spec)
        tf_itc = spy.itc(tf_spec)

        assert isinstance(tf_itc, spy.SpectralData)

        assert np.all(np.imag(itc.data[()]) == 0)
        assert itc.data[()].max() <= 1
        assert itc.data[()].min() >= 0

        # high itc around the in phase 60Hz
        assert np.all(itc.show(frequency=60) > 0.99)
        # low (time averaged) itc around the drifters
        assert np.all(itc.show(frequency=30) < 0.4)

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

        return itc, spec


if __name__ == '__main__':

    T1 = TestStatistics()
