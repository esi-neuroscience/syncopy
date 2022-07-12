# -*- coding: utf-8 -*-
#
# Test resampledata
#

# 3rd party imports
import pytest
import inspect
import numpy as np
import matplotlib.pyplot as ppl

# Local imports
from syncopy import __acme__
if __acme__:
    import dask.distributed as dd

from syncopy import resampledata, freqanalysis
import syncopy.tests.synth_data as synth_data
import syncopy.tests.helpers as helpers
from syncopy.shared.errors import SPYValueError
from syncopy.shared.tools import get_defaults

# Decorator to decide whether or not to run dask-related tests
skip_without_acme = pytest.mark.skipif(not __acme__, reason="acme not available")

# availableFilterTypes = ('lp', 'hp', 'bp', 'bs')


class TestDownsampling:

    nSamples = 1000
    nChannels = 4
    nTrials = 100
    fs = 200
    fNy = fs / 2

    # -- use flat white noise as test data --
    adata = synth_data.white_noise(nTrials,
                                   nChannels=nChannels,
                                   nSamples=nSamples,
                                   samplerate=fs)

    # original spectrum
    spec = freqanalysis(adata, tapsmofrq=1, keeptrials=False)
    # mean of the flat spectrum
    pow_orig = spec.show(channel=0).mean()

    # for toi tests, -1s offset
    time_span = [-.8, 4.2]

    def test_downsampling(self, **kwargs):

        """
        We test for remaining power after
        downsampling.
        """
        # check if we run the default test
        def_test = not len(kwargs)

        # write default parameters dict
        if def_test:
            kwargs = {'resamplefs': self.fs // 2}

        ds = resampledata(self.adata, method='downsample', **kwargs)
        spec_ds = freqanalysis(ds, tapsmofrq=1, keeptrials=False)

        # all channels are equal
        pow_ds = spec_ds.show(channel=0).mean()

        if def_test:
            # without anti-aliasing we get double the power per freq. bin
            # as we removed half of the frequencies
            assert np.allclose(2 * self.pow_orig, pow_ds, rtol=1e-2)

            f, ax = mk_spec_ax()
            ax.plot(spec_ds.freq, spec_ds.show(channel=0), label='downsampled')
            ax.plot(self.spec.freq, self.spec.show(channel=0), label='original')
            ax.legend()

            return

        return spec_ds

    def test_aa_filter(self):

        # filter with new Nyquist
        kwargs = {'resamplefs': self.fs // 2,
                  'lpfreq': self.fs // 4}

        spec_ds = self.test_downsampling(**kwargs)
        # all channels are equal
        pow_ds = spec_ds.show(channel=0).mean()

        # now with the anti-alias filter the powers should be equal
        assert np.allclose(self.pow_orig, pow_ds, rtol=.5e-1)

        f, ax = mk_spec_ax()
        ax.plot(spec_ds.freq, spec_ds.show(channel=0), label='downsampled')
        ax.plot(self.spec.freq, self.spec.show(channel=0), label='original')
        ax.legend()

    def test_ds_exceptions(self):

        # test non-integer division
        with pytest.raises(SPYValueError) as err:
            self.test_downsampling(resamplefs=self.fs / 3.142)
        assert "integer division" in str(err.value)

        # test sub-optimal lp freq, needs to be maximally the new Nyquist
        with pytest.raises(SPYValueError) as err:
            self.test_downsampling(resamplefs=self.fs // 2, lpfreq=self.fs / 1.5)
        assert f"less or equals {self.fs / 4}" in str(err.value)

        # test wrong order
        with pytest.raises(SPYValueError) as err:
            self.test_downsampling(resamplefs=self.fs // 2, lpfreq=self.fs / 10, order=-1)
        assert "less or equals inf" in str(err.value)

    def test_ds_selections(self):

        sel_dicts = helpers.mk_selection_dicts(nTrials=20,
                                               nChannels=2,
                                               toi_min=self.time_span[0],
                                               toi_max=self.time_span[1],
                                               min_len=3.5)
        for sd in sel_dicts:
            self.test_downsampling(select=sd, resamplefs=self.fs // 2)

    def test_ds_cfg(self):

        cfg = get_defaults(resampledata)

        cfg.lpfreq = 25
        cfg.order = 200
        cfg.resamplefs = self.fs // 4
        cfg.keeptrials = False

        ds = resampledata(self.adata, cfg)
        spec_ds = freqanalysis(ds, tapsmofrq=1, keeptrials=False)

        # all channels are equal
        pow_ds = spec_ds.show(channel=0).mean()

        # with aa filter power does not change
        assert np.allclose(self.pow_orig, pow_ds, rtol=.5e-1)

    @skip_without_acme
    def test_ds_parallel(self, testcluster=None):

        ppl.ioff()
        client = dd.Client(testcluster)
        all_tests = [attr for attr in self.__dir__()
                     if (inspect.ismethod(getattr(self, attr)) and 'parallel' not in attr)]

        for test_name in all_tests:
            test_method = getattr(self, test_name)
            test_method()
        client.close()
        ppl.ion()


def mk_spec_ax():

    fig, ax = ppl.subplots()
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('power (a.u.)')
    return fig, ax


if __name__ == '__main__':
    T1 = TestDownsampling()
