# -*- coding: utf-8 -*-
#
# Test connectivity measures
#

# 3rd party imports
import pytest
import itertools
import numpy as np
import matplotlib.pyplot as ppl

# Local imports
import syncopy as spy
import syncopy.tests.synth_data as synth_data
import syncopy.tests.helpers as helpers
from syncopy.shared.errors import SPYValueError


class TestAnalogPlotting():

    nTrials = 10
    nChannels = 9
    nSamples = 300
    adata = synth_data.AR2_network(nTrials=nTrials,
                                   AdjMat=np.zeros(nChannels),
                                   nSamples=nSamples)

    adata += 0.3 * synth_data.linear_trend(nTrials=nTrials,
                                           y_max=nSamples / 20,
                                           nSamples=nSamples,
                                           nChannels=nChannels)

    # add an offset
    adata = adata + 5

    # all trials are equal
    toi_min, toi_max = adata.time[0][0], adata.time[0][-1]

    def test_ad_plotting(self, **kwargs):

        # no interactive plotting
        ppl.ioff()

        # check if we run the default test
        def_test = not len(kwargs)

        if def_test:
            # interactive plotting
            ppl.ion()
            def_kwargs = {'trials': 1,
                          'toilim': [self.toi_min, 0.75 * self.toi_max]}
            kwargs = def_kwargs

            fig1, ax1 = self.adata.singlepanelplot(**kwargs)
            fig2, ax2 = self.adata.singlepanelplot(**kwargs, shifted=False)
            fig3, axs = self.adata.multipanelplot(**kwargs)

            # check axes/figure references work
            ax1.set_title('Shifted signals')
            fig1.tight_layout()
            ax2.set_title('Overlayed signals')
            fig2.tight_layout()
            fig3.suptitle("Multipanel plot")
            fig3.tight_layout()
        else:
            ppl.ioff()
            self.adata.singlepanelplot(**kwargs)
            self.adata.singlepanelplot(**kwargs, shifted=False)
            self.adata.multipanelplot(**kwargs)
            ppl.close('all')

    def test_ad_selections(self):

        # trial, channel and toi selections
        selections = helpers.mk_selection_dicts(self.nTrials,
                                                self.nChannels - 1,
                                                toi_min=self.toi_min,
                                                toi_max=self.toi_max)

        # test all combinations
        for sel_dict in selections:
            # only single trial plotting
            # is supported until averaging is availbale
            # take random 1st trial
            sel_dict['trials'] = sel_dict['trials'][0]
            # we have to sort the channels (hdf5 access)
            sel_dict['channel'] = sorted(sel_dict['channel'])
            self.test_ad_plotting(**sel_dict)

    def test_ad_exceptions(self):

        # empty arrays get returned for empty time selection
        with pytest.raises(SPYValueError) as err:
            self.test_ad_plotting(trials=0,
                                  toilim=[self.toi_max + 1, self.toi_max + 2])
            assert "zero size" in str(err)

        # invalid channel selection
        with pytest.raises(SPYValueError) as err:
            self.test_ad_plotting(trials=0, channel=self.nChannels + 1)
            assert "channel existing names" in str(err)

        # invalid trial selection
        with pytest.raises(SPYValueError) as err:
            self.test_ad_plotting(trials=self.nTrials + 1)
            assert "select: trials" in str(err)


class TestSpectralPlotting():

    nTrials = 10
    nChannels = 4
    nSamples = 300
    AdjMat = np.zeros((nChannels, nChannels))
    adata = synth_data.AR2_network(nTrials=nTrials,
                                   AdjMat=AdjMat,
                                   nSamples=nSamples)

    # add AR(1) 'background'
    adata = adata + 1.2 * synth_data.AR2_network(nTrials=nTrials,
                                                 AdjMat=AdjMat,
                                                 nSamples=nSamples,
                                                 alphas=[0.8, 0])

    # some interesting range
    foilim = [1, 400]

    # all trials are equal
    toi_min, toi_max = adata.time[0][0], adata.time[0][-1]

    spec_fft = spy.freqanalysis(adata, tapsmofrq=1)
    spec_wlet = spy.freqanalysis(adata, method='wavelet',
                                 foi=np.arange(0, 400, step=4))

    def test_spectral_plotting(self, **kwargs):

        # no interactive plotting
        ppl.ioff()

        # check if we run the default test
        def_test = not len(kwargs)

        if def_test:
            ppl.ion()
            kwargs = {'trials': self.nTrials - 1, 'foilim': [5, 300]}
            # to visually compare
            self.adata.singlepanelplot(trials=self.nTrials - 1, channel=0)

            # this simulates the interactive plotting
            fig1, ax1 = self.spec_fft.singlepanelplot(**kwargs)
            fig2, axs = self.spec_fft.multipanelplot(**kwargs)

            fig3, ax2 = self.spec_wlet.singlepanelplot(channel=0, **kwargs)
            fig4, axs = self.spec_wlet.multipanelplot(**kwargs)

            ax1.set_title('AR(1) + AR(2)')
            fig2.suptitle('AR(1) + AR(2)')
        else:
            print(kwargs)
            self.spec_fft.singlepanelplot(**kwargs)
            self.spec_fft.multipanelplot(**kwargs)

            self.spec_wlet.multipanelplot(**kwargs)
            # take the 1st random channel for 2d spectra
            if 'channel' in kwargs:
                chan = kwargs.pop('channel')[0]
            self.spec_wlet.singlepanelplot(channel=chan, **kwargs)
            ppl.close('all')

    def test_spectral_selections(self):

        # trial, channel and toi selections
        selections = helpers.mk_selection_dicts(self.nTrials,
                                                self.nChannels - 1,
                                                toi_min=self.toi_min,
                                                toi_max=self.toi_max)

        # add a foi selection
        # FIXME: show() and foi selections   #291
        # selections[0]['foi'] = np.arange(5, 300, step=2)

        # test all combinations
        for sel_dict in selections:

            # only single trial plotting
            # is supported until averaging is availbale
            # take random 1st trial
            sel_dict['trials'] = sel_dict['trials'][0]
            # we have to sort the channels (hdf5 access)
            sel_dict['channel'] = sorted(sel_dict['channel'])
            self.test_spectral_plotting(**sel_dict)

    def test_spectral_exceptions(self):

        # empty arrays get returned for empty time selection
        with pytest.raises(SPYValueError) as err:
            self.test_spectral_plotting(trials=0,
                                        toilim=[self.toi_max + 1, self.toi_max + 2])
            assert "zero size" in str(err)

        # invalid channel selection
        with pytest.raises(SPYValueError) as err:
            self.test_spectral_plotting(trials=0, channel=self.nChannels + 1)
            assert "channel existing names" in str(err)

        # invalid trial selection
        with pytest.raises(SPYValueError) as err:
            self.test_spectral_plotting(trials=self.nTrials + 1)
            assert "select: trials" in str(err)

        # invalid foi selection
        with pytest.raises(SPYValueError) as err:
            self.test_spectral_plotting(trials=0, foilim=[-1, 0])
            assert "foi/foilim" in str(err)
            assert "bounded by" in str(err)

        # invalid foi selection
        with pytest.raises(SPYValueError) as err:
            self.test_spectral_plotting(trials=0, foi=[-1, 0])
            assert "foi/foilim" in str(err)
            assert "bounded by" in str(err)


class TestCrossSpectralPlotting():

    nTrials = 40
    nChannels = 4
    nSamples = 400

    AdjMat = np.zeros((nChannels, nChannels))
    AdjMat[2, 3] = 0.2   # coupling
    adata = synth_data.AR2_network(nTrials=nTrials,
                                   AdjMat=AdjMat,
                                   nSamples=nSamples)

    # add 'background'
    adata = adata + .6 * synth_data.AR2_network(nTrials=nTrials,
                                                AdjMat=np.zeros((nChannels,
                                                                 nChannels)),
                                                nSamples=nSamples,
                                                alphas=[0.8, 0])

    # some interesting range
    foilim = [1, 400]

    # all trials are equal
    toi_min, toi_max = adata.time[0][0], adata.time[0][-1]

    coh = spy.connectivityanalysis(adata, method='coh', tapsmofrq=1)
    corr = spy.connectivityanalysis(adata, method='corr')
    granger = spy.connectivityanalysis(adata, method='granger', tapsmofrq=1)

    def test_cs_plotting(self, **kwargs):

        # no interactive plotting
        ppl.ioff()

        # check if we run the default test
        def_test = not len(kwargs)

        if def_test:
            ppl.ion()

            self.coh.singlepanelplot(channel_i=0, channel_j=1, foilim=[50, 320])
            self.coh.singlepanelplot(channel_i=1, channel_j=2, foilim=[50, 320])
            self.coh.singlepanelplot(channel_i=2, channel_j=3, foilim=[50, 320])

            self.corr.singlepanelplot(channel_i=0, channel_j=1, toilim=[0, .1])
            self.corr.singlepanelplot(channel_i=1, channel_j=0, toilim=[0, .1])
            self.corr.singlepanelplot(channel_i=2, channel_j=3, toilim=[0, .1])

            self.granger.singlepanelplot(channel_i=0, channel_j=1, foilim=[50, 320])
            self.granger.singlepanelplot(channel_i=3, channel_j=2, foilim=[50, 320])
            self.granger.singlepanelplot(channel_i=2, channel_j=3, foilim=[50, 320])

        elif 'toilim' in kwargs:

            self.corr.singlepanelplot(**kwargs)
            self.corr.singlepanelplot(**kwargs)
            self.corr.singlepanelplot(**kwargs)
            ppl.close('all')

        else:

            self.coh.singlepanelplot(**kwargs)
            self.coh.singlepanelplot(**kwargs)
            self.coh.singlepanelplot(**kwargs)

            self.granger.singlepanelplot(**kwargs)
            self.granger.singlepanelplot(**kwargs)
            self.granger.singlepanelplot(**kwargs)

    def test_cs_selections(self):

        # channel combinations
        chans = itertools.product(self.coh.channel_i[:self.nTrials - 1],
                                  self.coh.channel_j[1:])

        # out of range toi selections are allowed..
        toilim = ([0, .1], [-1, 1], 'all')
        toilim_comb = itertools.product(chans, toilim)

        # out of range foi selections are NOT allowed..
        foilim = ([10., 82.31], 'all')
        foilim_comb = itertools.product(chans, foilim)

        for comb in toilim_comb:
            sel_dct = {}
            c1, c2 = comb[0]
            sel_dct['channel_i'] = c1
            sel_dct['channel_j'] = c2
            sel_dct['toilim'] = comb[1]
            self.test_cs_plotting(**sel_dct)

        for comb in foilim_comb:
            sel_dct = {}
            c1, c2 = comb[0]
            sel_dct['channel_i'] = c1
            sel_dct['channel_j'] = c2
            sel_dct['foilim'] = comb[1]
            self.test_cs_plotting(**sel_dct)

    def test_cs_exceptions(self):

        chan_sel = {'channel_i': 0, 'channel_j': 2}
        # empty arrays get returned for empty time selection
        with pytest.raises(SPYValueError) as err:
            self.test_cs_plotting(trials=0,
                                  toilim=[self.toi_max + 1, self.toi_max + 2],
                                  **chan_sel)
            assert "zero size" in str(err)

        # invalid channel selections
        with pytest.raises(SPYValueError) as err:
            self.test_cs_plotting(trials=0, channel_i=self.nChannels + 1, channel_j=0)
            assert "channel existing names" in str(err)

        # invalid trial selection
        with pytest.raises(SPYValueError) as err:
            self.test_cs_plotting(trials=self.nTrials + 1, **chan_sel)
            assert "select: trials" in str(err)

        # invalid foi selection
        with pytest.raises(SPYValueError) as err:
            self.test_cs_plotting(trials=0, foilim=[-1, 0], **chan_sel)
            assert "foi/foilim" in str(err)
            assert "bounded by" in str(err)

        # invalid foi selection
        with pytest.raises(SPYValueError) as err:
            self.test_cs_plotting(trials=0, foi=[-1, 0], **chan_sel)
            assert "foi/foilim" in str(err)
            assert "bounded by" in str(err)


if __name__ == '__main__':
    T1 = TestAnalogPlotting()
    T2 = TestSpectralPlotting()
    T3 = TestCrossSpectralPlotting()
