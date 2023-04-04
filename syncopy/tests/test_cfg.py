# -*- coding: utf-8 -*-
#
# Test cfg structure to replay frontend calls
#

import pytest
import numpy as np
import inspect
import tempfile
import os
import dask.distributed as dd

# Local imports
import syncopy as spy

import syncopy.tests.synth_data as synth_data
from syncopy.shared.tools import StructDict


availableFrontend_cfgs = {'freqanalysis': {'method': 'mtmconvol', 't_ftimwin': 0.1, 'foi': np.arange(1,60)},
                          'preprocessing': {'freq': 10, 'filter_class': 'firws', 'filter_type': 'hp'},
                          'resampledata': {'resamplefs': 125, 'lpfreq': 60},
                          'connectivityanalysis': {'method': 'coh', 'tapsmofrq': 5},
                          'selectdata': {'trials': np.array([1, 7, 3]), 'channel': [np.int64(2), 0]}
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

            # check that it's not just the defaults (mtmfft)
            if frontend == 'freqanalysis':
                res3 = getattr(spy, frontend)(self.adata)
                assert res.data.shape != res3.data.shape
                assert res.cfg != res3.cfg

    def test_io(self):

        for frontend in availableFrontend_cfgs.keys():

            # unwrap cfg into keywords
            res = getattr(spy, frontend)(self.adata, **availableFrontend_cfgs[frontend])
            # make a copy
            cfg = StructDict(res.cfg)

            # test saving and loading
            with tempfile.TemporaryDirectory() as tdir:
                fname = os.path.join(tdir, "res")
                res.save(container=fname)

                res = spy.load(fname)
                assert res.cfg == cfg

                # now replay with cfg from preceding frontend call
                res2 = getattr(spy, frontend)(self.adata, res.cfg)
                # same results
                assert np.allclose(res.data[:], res2.data[:])
                assert res.cfg == res2.cfg

                del res, res2

    def test_selection(self):

        select = {'latency': self.time_span, 'trials': [1, 2, 3], 'channel': [2, 0]}
        for frontend in availableFrontend_cfgs.keys():
            # select kw for selectdata makes no direct sense
            if frontend == 'selectdata':
                continue
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

    def test_chaining_frontends_with_fooof_types(self):

        # only preprocessing makes sense to chain atm
        res_pp = spy.preprocessing(self.adata, cfg=availableFrontend_cfgs['preprocessing'])

        frontend = 'freqanalysis'
        frontend_cfg = {'method': 'mtmfft', 'output': 'fooof', 'foilim': [0.5, 100.]}

        res = getattr(spy, frontend)(res_pp,
                                        cfg=frontend_cfg)

        # now replay with cfg from preceding frontend calls
        # note we can use the final results `res.cfg` for both calls!
        res_pp2 = spy.preprocessing(self.adata, res.cfg)
        res2 = getattr(spy, frontend)(res_pp2, res.cfg)

        # same results
        assert np.allclose(res.data[:], res2.data[:])
        assert res.cfg == res2.cfg

    def test_parallel(self, testcluster):

        client = dd.Client(testcluster)
        all_tests = [attr for attr in self.__dir__()
                     if (inspect.ismethod(getattr(self, attr)) and 'parallel' not in attr)]

        for test_name in all_tests:
            test_method = getattr(self, test_name)
            test_method()
        client.close()


if __name__ == '__main__':
    T1 = TestCfg()
