# -*- coding: utf-8 -*-
#
# Test cfg structure to replay frontend calls
#

import pytest
import numpy as np
import inspect

# Local imports
import syncopy as spy
from syncopy import __acme__
if __acme__:
    import dask.distributed as dd

import syncopy.tests.synth_data as synth_data

# Decorator to decide whether or not to run dask-related tests
skip_without_acme = pytest.mark.skipif(not __acme__, reason="acme not available")

availableFrontend_cfgs = {'freqanalysis': {'method': 'mtmconvol', 't_ftimwin': 0.1},
                          'preprocessing': {'freq': 10, 'filter_class': 'firws', 'filter_type': 'hp'},
                          'resampledata': {'resamplefs': 125, 'lpfreq': 100},
                          'connectivityanalysis': {'method': 'coh', 'tapsmofrq': 5}
                          }


class TestCfg:

    nSamples = 100
    nChannels = 3
    nTrials = 10
    fs = 200
    fNy = fs / 2

    # -- use flat white noise as test data --

    adata = synth_data.white_noise(nTrials,
                                   nSamples=nSamples,
                                   nChannels=nChannels,
                                   samplerate=fs)

    # for toi tests, -1s offset
    time_span = [-.9, -.6]
    flow, fhigh = 0.3 * fNy, 0.4 * fNy

    def test_single_frontends(self):

        for frontend in availableFrontend_cfgs.keys():

            # unwrap cfg into keywords
            res = getattr(spy, frontend)(self.adata, **availableFrontend_cfgs[frontend])
            # now replay with cfg from preceding frontend call
            res2 = getattr(spy, frontend)(self.adata, res.cfg)

            # same results
            assert np.allclose(res.data[:], res2.data[:])
            assert res.cfg == res2.cfg

            # check that it's not just the defaults
            if frontend == 'freqanalysis':
                res3 = getattr(spy, frontend)(self.adata)
                assert np.any(res.data[:] != res3.data[:])
                assert res.cfg != res3.cfg

    def test_selection(self):

        select = {'toilim': self.time_span, 'trials': [1, 2, 3], 'channel': [2, 0]}
        for frontend in availableFrontend_cfgs.keys():
            res = getattr(spy, frontend)(self.adata,
                                         cfg=availableFrontend_cfgs[frontend],
                                         select=select)

            # now replay with cfg from preceding frontend call
            res2 = getattr(spy, frontend)(self.adata, res.cfg)

            # same results
            assert 'select' in res.cfg[frontend]
            assert 'select' in res2.cfg[frontend]
            assert np.allclose(res.data[:], res2.data[:])
            assert res.cfg == res2.cfg

    def test_chaining_frontends(self):

        # only preprocessing makes sense to chain atm
        res_pp = spy.preprocessing(self.adata, cfg=availableFrontend_cfgs['preprocessing'])

        for frontend in availableFrontend_cfgs.keys():
            res = getattr(spy, frontend)(res_pp,
                                         cfg=availableFrontend_cfgs[frontend])

            # now replay with cfg from preceding frontend calls
            # note we can use the final results `res.cfg` for both calls!
            res_pp2 = spy.preprocessing(self.adata, res.cfg)
            res2 = getattr(spy, frontend)(res_pp2, res.cfg)

            # same results
            assert np.allclose(res.data[:], res2.data[:])
            assert res.cfg == res2.cfg

    @skip_without_acme
    def test_parallel(self, testcluster=None):

        client = dd.Client(testcluster)
        all_tests = [attr for attr in self.__dir__()
                     if (inspect.ismethod(getattr(self, attr)) and 'parallel' not in attr)]

        for test_name in all_tests:
            test_method = getattr(self, test_name)
            test_method()
        client.close()


if __name__ == '__main__':
    T1 = TestCfg()
