# -*- coding: utf-8 -*-
# 
# 
# 
# Created: 2020-04-17 08:25:48
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2020-04-20 18:10:19>

import pytest
import numpy as np

from syncopy.tests.misc import generate_artificial_data, compare_figs
from syncopy.shared.errors import SPYValueError
from syncopy.plotting import singleplot, multiplot
from syncopy import __plt__
if __plt__:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams["figure.dpi"] = 50

# If matplotlib is not available, this whole testing module is pointless
skip_without_plt = pytest.mark.skipif(not __plt__, reason="matplotlib not available")


@skip_without_plt
class TestAnalogDataPlotting():
    
    nChannels = 16
    nTrials = 8

    # To use `selectdata` w/``trials = None``-raw plotting, the trials must not 
    # overlap - construct separate set of objects for testing!
    dataReg = generate_artificial_data(nTrials=nTrials,
                                       nChannels=nChannels,
                                       equidistant=True,
                                       overlapping=True)
    dataInv = generate_artificial_data(nTrials=nTrials,
                                       nChannels=nChannels,
                                       equidistant=True,
                                       overlapping=True,
                                       dimord=dataReg.dimord[::-1])
    rawReg = generate_artificial_data(nTrials=nTrials,
                                      nChannels=nChannels,
                                      equidistant=True,
                                      overlapping=False)
    rawInv = generate_artificial_data(nTrials=nTrials,
                                      nChannels=nChannels,
                                      equidistant=True,
                                      overlapping=False,
                                      dimord=rawReg.dimord[::-1])
    
    trials = ["all", [4, 3, 2, 2, 7]]
    channels = ["all", [14, 13, 12, 12, 15]]
    toilim = [None, [1.9, 2.5], [2.1, np.inf]]
    
    # trlToilim = lambda self, trl, tlim: None if trl is None else tlim
    
    def test_singleplot(self):

        # Test everything except "raw" plotting       
        for trials in self.trials:
            for channels in self.channels:
                for toilim in self.toilim:
                    for avg_channels in [True, False]:
                        fig1 = self.dataReg.singleplot(trials=trials,
                                                       channels=channels,
                                                       toilim=toilim,
                                                       avg_channels=avg_channels)
                        selected = self.dataReg.selectdata(trials=trials, 
                                                           channels=channels,
                                                           toilim=toilim)
                        fig2 = selected.singleplot(trials="all", 
                                                   avg_channels=avg_channels)
                        assert compare_figs(fig1, fig2)
                        plt.close("all")
                        # fig3 = singleplot(self.dataReg, self.dataInv,
                        #                   trials=trials,
                        #                   channels=channels,
                        #                   toilim=toilim,
                        #                   avg_channels=avg_channels)
                        # fig4 = singleplot(self.dataInv, 
                        #                   trials=trials,
                        #                   channels=channels,
                        #                   toilim=toilim,
                        #                   avg_channels=avg_channels,
                        #                   fig=fig1)
                        # assert compare_figs(fig3, fig4)

        # The `selectdata(trials="all")` part requires consecutive trials!
        for channels in self.channels:
            for avg_channels in [True, False]:
                fig1 = self.rawReg.singleplot(trials=None,
                                              channels=channels,
                                              avg_channels=avg_channels)
                selected = self.rawReg.selectdata(trials="all",
                                                  channels=channels)
                fig2 = selected.singleplot(trials=None, avg_channels=avg_channels)
                assert compare_figs(fig1, fig2)
                plt.close("all")

        # Do not allow selecting time-intervals w/o trial-specification        
        with pytest.raises(SPYValueError):
            self.dataReg.singleplot(trials=None, toilim=self.toilim[1])
        
        # Do not overlay multi-channel plot w/chan-average and unequal channel-count
        multiChannelFig = self.dataReg.singleplot(avg_channels=False, 
                                                  channels=self.channels[1])
        with pytest.raises(SPYValueError):
            self.dataReg.singleplot(avg_channels=True, fig=multiChannelFig)
        with pytest.raises(SPYValueError):
            self.dataReg.singleplot(channels="all", fig=multiChannelFig)
            
        # Do not overlay multi-panel plot w/figure produced by `singleplot`
        multiChannelFig.nTrialsPanels = 99
        with pytest.raises(SPYValueError):
            self.dataReg.singleplot(fig=multiChannelFig)
        multiChannelFig.nChanPanels = 99
        with pytest.raises(SPYValueError):
            self.dataReg.singleplot(fig=multiChannelFig)
        
        # Ensure grid and title specifications are rendered correctly
        theTitle = "A title"
        gridFig = self.dataReg.singleplot(grid=True)
        assert gridFig.axes[0].get_xgridlines()[0].get_visible() == True
        titleFig = self.dataReg.singleplot(title=theTitle)
        assert titleFig.axes[0].get_title() == theTitle
        
        plt.close("all")

    def test_multiplot(self):

        # Test everything except "raw" plotting       
        for trials in self.trials:
            for channels in self.channels:
                for toilim in self.toilim:
                    for avg_trials in [True, False]:
                        for avg_channels in [True, False]:
                            fig1 = self.dataReg.multiplot(trials=trials,
                                                          channels=channels,
                                                          toilim=toilim,
                                                          avg_trials=avg_trials,
                                                          avg_channels=avg_channels)
                            selected = self.dataReg.selectdata(trials=trials, 
                                                               channels=channels,
                                                               toilim=toilim)
                            fig2 = selected.multiplot(trials="all", 
                                                      avg_trials=avg_trials,
                                                      avg_channels=avg_channels)
                            
                            # `selectdata` preserve trial order but not numbering; ensure
                            # plot titles are correct, but then remove them to allow 
                            # comparison of `selected` and `dataReg` plots
                            if avg_trials is False:
                                if trials != "all":
                                    titleList = []
                                    for ax in fig1.axes:
                                        titleList.append(ax.get_title())
                                        ax.set_title("")
                                    for ax in fig2.axes:
                                        ax.set_title("")
                                    assert titleList == ["Trial #{}".format(trlno) for trlno in trials]
                            else:
                                # ``avg_trials == avg_channels == True`` yields `SPYWarning` to 
                                # use `singleplot` -> no figs to compare here
                                if avg_channels is True:
                                    continue
                                
                            assert compare_figs(fig1, fig2)
                            plt.close("all")
                            
                            # fig3 = multiplot(self.dataReg, self.dataInv,
                            #                  trials=trials,
                            #                  channels=channels,
                            #                  toilim=toilim,
                            #                  avg_trials=avg_trials,
                            #                  avg_channels=avg_channels)
                            # fig4 = multiplot(self.dataInv, 
                            #                  trials=trials,
                            #                  channels=channels,
                            #                  toilim=toilim,
                            #                  avg_trials=avg_trials,
                            #                  avg_channels=avg_channels,
                            #                  fig=fig1)
                            # assert compare_figs(fig3, fig4)

        # # The `selectdata(trials="all")` part requires consecutive trials!
        # for channels in self.channels:
        #     for avg_channels in [True, False]:
        #         fig1 = self.rawReg.multiplot(trials=None,
        #                                      channels=channels,
        #                                      avg_channels=avg_channels,
        #                                      avg_trials=False)
        #         selected = self.rawReg.selectdata(trials="all",
        #                                           channels=channels)
        #         fig2 = selected.multiplot(trials=None, 
        #                                   avg_channels=avg_channels,
        #                                   avg_trials=avg_trials)
        #         assert compare_figs(fig1, fig2)
        #         plt.close("all")

        # Do not allow selecting time-intervals w/o trial-specification        
        with pytest.raises(SPYValueError):
            self.dataReg.multiplot(trials=None, toilim=self.toilim[1])
        
        # Do not overlay multi-trial plot w/averages or multi-channel figure        
        multiTrialSingleChanFig = self.dataReg.multiplot(trials=self.trials[1], 
                                                          avg_trials=False,
                                                          avg_channels=True)
        with pytest.raises(SPYValueError):
            self.dataReg.multiplot(trials="all", 
                                   avg_trials=False,
                                   avg_channels=True,
                                   fig=multiTrialSingleChanFig)
        with pytest.raises(SPYValueError):
            self.dataReg.multiplot(trials=self.trials[1], 
                                   avg_trials=False,
                                   avg_channels=False,
                                   fig=multiTrialSingleChanFig)
        with pytest.raises(SPYValueError):
            self.dataReg.multiplot(trials=None, 
                                    avg_trials=False,
                                    avg_channels=True,
                                    fig=multiTrialSingleChanFig)

        # Do not overlay multi-channel-per-panel plot w/non-matching counts/averages
        multiTrialMultiChanFig = self.dataReg.multiplot(trials=self.trials[1],
                                                         avg_trials=False,
                                                         avg_channels=False)
        with pytest.raises(SPYValueError):
            self.dataReg.multiplot(trials=self.trials[1], 
                                    channels=self.channels[1],
                                    avg_trials=False,
                                    avg_channels=False,
                                    fig=multiTrialMultiChanFig)
        with pytest.raises(SPYValueError):
            self.dataReg.multiplot(trials=self.trials[1], 
                                    avg_trials=False,
                                    avg_channels=True,
                                    fig=multiTrialMultiChanFig)

        # Same for multi-channel trial-averages            
        multiChannelFig = self.dataReg.multiplot(trials=self.trials[1],
                                                  avg_trials=True,
                                                  avg_channels=False)
        with pytest.raises(SPYValueError):
            self.dataReg.multiplot(trials=self.trials[1], 
                                    avg_trials=False,
                                    avg_channels=False,
                                    fig=multiChannelFig)
        with pytest.raises(SPYValueError):
            self.dataReg.multiplot(trials=self.trials[1], 
                                    channels=self.channels[1],
                                    avg_trials=True,
                                    avg_channels=False,
                                    fig=multiChannelFig)
        with pytest.raises(SPYValueError):
            self.dataReg.multiplot(trials=self.trials[1], 
                                    avg_trials=True,
                                    avg_channels=True,
                                    fig=multiChannelFig)
            
        # # Do not overlay multi-panel plot w/figure produced by `singleplot`
        # multiChannelFig.nTrialsPanels = 99
        # with pytest.raises(SPYValueError):
        #     self.dataReg.singleplot(fig=multiChannelFig)
        # multiChannelFig.nChanPanels = 99
        # with pytest.raises(SPYValueError):
        #     self.dataReg.singleplot(fig=multiChannelFig)
        
        # # Ensure grid and title specifications are rendered correctly
        # theTitle = "A title"
        # gridFig = self.dataReg.singleplot(grid=True)
        # assert gridFig.axes[0].get_xgridlines()[0].get_visible() == True
        # titleFig = self.dataReg.singleplot(title=theTitle)
        # assert titleFig.axes[0].get_title() == theTitle

        