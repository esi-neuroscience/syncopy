# -*- coding: utf-8 -*-
# 
# 
# 
# Created: 2020-04-17 08:25:48
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2020-05-14 16:00:40>

import pytest
import numpy as np
from copy import copy

from syncopy.tests.misc import generate_artificial_data, figs_equal
from syncopy.shared.errors import SPYValueError
from syncopy.plotting import singleplot, multiplot
from syncopy import __plt__
if __plt__:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    # mpl.rcParams["figure.dpi"] = 50
    # mpl.rcParams["figure.dpi"] = 75

# If matplotlib is not available, this whole testing module is pointless
skip_without_plt = pytest.mark.skipif(not __plt__, reason="matplotlib not available")


@skip_without_plt
class TestAnalogDataPlotting():
    
    nChannels = 16
    nTrials = 8
    seed = 130810

    # To use `selectdata` w/``trials = None``-raw plotting, the trials must not 
    # overlap - construct separate set of `raw*` AnalogData-objects for testing!
    dataReg = generate_artificial_data(nTrials=nTrials,
                                       nChannels=nChannels,
                                       seed=seed,
                                       equidistant=True,
                                       overlapping=True)
    dataInv = generate_artificial_data(nTrials=nTrials,
                                       nChannels=nChannels,
                                       seed=seed,
                                       equidistant=True,
                                       overlapping=True,
                                       dimord=dataReg.dimord[::-1])
    rawReg = generate_artificial_data(nTrials=nTrials,
                                      nChannels=nChannels,
                                      seed=seed,
                                      equidistant=True,
                                      overlapping=False)
    rawInv = generate_artificial_data(nTrials=nTrials,
                                      nChannels=nChannels,
                                      seed=seed,
                                      equidistant=True,
                                      overlapping=False,
                                      dimord=rawReg.dimord[::-1])
    
    trials = ["all", [4, 3, 2, 2, 7]]
    channels = ["all", [14, 13, 12, 12, 15]]
    toilim = [None, [1.9, 2.5], [2.1, np.inf]]
    
    # trlToilim = lambda self, trl, tlim: None if trl is None else tlim
    
    def test_singleplot(self):

        # Lowest possible dpi setting permitting valid png comparisons in `figs_equal`
        mpl.rcParams["figure.dpi"] = 50
        
        # Test everything except "raw" plotting       
        for trials in self.trials:
            for channels in self.channels:
                for toilim in self.toilim:
                    for avg_channels in [True, False]:
                        
                        # Render figure using singleplot mechanics, recreate w/`selectdata`, 
                        # results must be identical
                        fig1 = self.dataReg.singleplot(trials=trials,
                                                       channels=channels,
                                                       toilim=toilim,
                                                       avg_channels=avg_channels)
                        selected = self.dataReg.selectdata(trials=trials, 
                                                           channels=channels,
                                                           toilim=toilim)
                        fig2 = selected.singleplot(trials="all", 
                                                   avg_channels=avg_channels)
                        # assert figs_equal(fig1, fig2)
                        
                        # Recreate `fig1` and `fig2` in a single sweep by using 
                        # `spy.singleplot` w/multiple input objects 
                        fig1a, fig2a = singleplot(self.dataReg, self.dataInv,
                                                  trials=trials,
                                                  channels=channels,
                                                  toilim=toilim,
                                                  avg_channels=avg_channels,
                                                  overlay=False)

                        # `fig2a` is based on `dataInv` - be more lenient there
                        tol = None
                        if avg_channels:
                            tol = 1e-2
                        assert figs_equal(fig1, fig2)
                        assert figs_equal(fig1, fig1a)
                        assert figs_equal(fig2, fig2a, tol=tol)
                        assert figs_equal(fig1, fig2a, tol=tol)
                        
                        # Create overlay figures: `fig3` combines `dataReg` and 
                        # `dataInv` - must be identical to overlaying `fig1` w/`dataInv`
                        fig3 = singleplot(self.dataReg, self.dataInv,
                                          trials=trials,
                                          channels=channels,
                                          toilim=toilim,
                                          avg_channels=avg_channels,
                                          overlay=True)
                        fig1 = singleplot(self.dataInv, 
                                          trials=trials,
                                          channels=channels,
                                          toilim=toilim,
                                          avg_channels=avg_channels,
                                          fig=fig1)
                        assert figs_equal(fig1, fig3)

                        # Close figures to avoid memory overflow
                        plt.close("all")

        # The `selectdata(trials="all")` part requires consecutive trials!
        for channels in self.channels:
            for avg_channels in [True, False]:
                fig1 = self.rawReg.singleplot(trials=None,
                                              channels=channels,
                                              avg_channels=avg_channels)
                selected = self.rawReg.selectdata(trials="all",
                                                  channels=channels)
                fig2 = selected.singleplot(trials=None, avg_channels=avg_channels)
                assert figs_equal(fig1, fig2)
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

        # Lowest possible dpi setting permitting valid png comparisons in `figs_equal`
        mpl.rcParams["figure.dpi"] = 75
        
        # Test everything except "raw" plotting       
        for trials in self.trials:
            for channels in self.channels:
                for toilim in self.toilim:
                    for avg_trials in [True, False]:
                        for avg_channels in [True, False]:
                            
                            # ``avg_trials == avg_channels == True`` yields `SPYWarning` to 
                            # use `singleplot` -> no figs to compare here
                            if avg_trials is avg_channels is True:
                                continue
                            
                            # Render figure using multiplot mechanics, recreate w/`selectdata`, 
                            # results must be identical
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
                            
                            # Recreate `fig1` and `fig2` in a single sweep by using 
                            # `spy.multiplot` w/multiple input objects 
                            fig1a, fig2a = multiplot(self.dataReg, self.dataInv,
                                                     trials=trials,
                                                     channels=channels,
                                                     toilim=toilim,
                                                     avg_trials=avg_trials,
                                                     avg_channels=avg_channels,
                                                     overlay=False)
                            
                            
                            # `selectdata` preserves trial order but not numbering: ensure
                            # plot titles are correct, but then remove them to allow 
                            # comparison of `selected` and `dataReg` figures
                            figTitleLists = []
                            if avg_trials is False:
                                if trials != "all":
                                    for fig in [fig1, fig1a]:
                                        titleList = []
                                        for ax in fig.axes:
                                            titleList.append(copy(ax.title))
                                            ax.set_title("")
                                        titles = [title.get_text() for title in titleList]
                                        assert titles == ["Trial #{}".format(trlno) for trlno in trials]
                                        figTitleLists.append(titleList)
                                    for fig in [fig2, fig2a]:
                                        for ax in fig.axes:
                                            ax.set_title("")

                            # After (potential) axes title removal, compare figures; 
                            # `fig2a` is based on `dataInv` - be more lenient there
                            tol = None
                            if avg_channels:
                                tol = 1e-2
                            assert figs_equal(fig1, fig2)
                            assert figs_equal(fig1, fig1a)
                            assert figs_equal(fig2, fig2a, tol=tol)
                            assert figs_equal(fig1, fig2a, tol=tol)

                            # If necessary, restore axes title from `figTitleLists`
                            if figTitleLists:
                                for k, ax in enumerate(fig1.axes):
                                    ax.title = figTitleLists[0][k]

                            # Create overlay figures: `fig3` combines `dataReg` and 
                            # `dataInv` - must be identical to overlaying `fig1` w/`dataInv`
                            fig3 = multiplot(self.dataReg, self.dataInv,
                                             trials=trials,
                                             channels=channels,
                                             toilim=toilim,
                                             avg_trials=avg_trials,
                                             avg_channels=avg_channels)
                            fig4 = multiplot(self.dataInv, 
                                             trials=trials,
                                             channels=channels,
                                             toilim=toilim,
                                             avg_trials=avg_trials,
                                             avg_channels=avg_channels,
                                             fig=fig1)
                            assert figs_equal(fig3, fig4)

                            plt.close("all")

        # The `selectdata(trials="all")` part requires consecutive trials! Add'ly, 
        # `avg_channels` must be `False`, otherwise single-panel plot warning is triggered
        for channels in self.channels:
                fig1 = self.rawReg.multiplot(trials=None,
                                             channels=channels,
                                             avg_channels=False,
                                             avg_trials=False)
                selected = self.rawReg.selectdata(trials="all",
                                                  channels=channels)
                fig2 = selected.multiplot(trials=None, 
                                          avg_channels=False)
                assert figs_equal(fig1, fig2)
                plt.close("all")

        # Do not allow selecting time-intervals w/o trial-specification        
        with pytest.raises(SPYValueError):
            self.dataReg.multiplot(trials=None, toilim=self.toilim[1])
        
        # Panels = trials, each panel shows single (averaged) channel
        multiTrialSingleChanFig = self.dataReg.multiplot(trials=self.trials[1], 
                                                         avg_trials=False,
                                                         avg_channels=True)
        with pytest.raises(SPYValueError):  # trial-count does not match up
            self.dataReg.multiplot(trials="all", 
                                   avg_trials=False,
                                   avg_channels=True,
                                   fig=multiTrialSingleChanFig)
        with pytest.raises(SPYValueError):  # multi-channel overlay
            self.dataReg.multiplot(trials=self.trials[1], 
                                   avg_trials=False,
                                   avg_channels=False,
                                   fig=multiTrialSingleChanFig)

        # Panels = trials, each panel shows multiple channels
        multiTrialMultiChanFig = self.dataReg.multiplot(trials=self.trials[1],
                                                         avg_trials=False,
                                                         avg_channels=False)
        with pytest.raises(SPYValueError):  # no trial specification provided
            self.dataReg.multiplot(trials=None, 
                                   avg_trials=False,
                                   avg_channels=False,
                                   fig=multiTrialMultiChanFig)
        with pytest.raises(SPYValueError):  # channel-count does not match up
            self.dataReg.multiplot(trials=self.trials[1], 
                                   channels=self.channels[1],
                                   avg_trials=False,
                                   avg_channels=False,
                                   fig=multiTrialMultiChanFig)
        with pytest.raises(SPYValueError):  # single-channel overlay
            self.dataReg.multiplot(trials=self.trials[1], 
                                   avg_trials=False,
                                   avg_channels=True,
                                   fig=multiTrialMultiChanFig)

        # Panels = channels
        multiChannelFig = self.dataReg.multiplot(trials=self.trials[1],
                                                 avg_trials=True,
                                                 avg_channels=False)
        with pytest.raises(SPYValueError):  # multi-trial overlay
            self.dataReg.multiplot(trials=self.trials[1], 
                                   avg_trials=False,
                                   avg_channels=False,
                                   fig=multiChannelFig)
        with pytest.raises(SPYValueError):  # channel-count does not match up
            self.dataReg.multiplot(trials=self.trials[1], 
                                   channels=self.channels[1],
                                   avg_trials=True,
                                   avg_channels=False,
                                   fig=multiChannelFig)
            
        # Do not overlay single-panel plot w/figure produced by `multiplot`
        singlePanelFig = self.dataReg.singleplot()
        with pytest.raises(SPYValueError):
            self.dataReg.multiplot(fig=singlePanelFig)
        
        # Ensure grid and title specifications are rendered correctly
        theTitle = "A title"
        gridFig = self.dataReg.multiplot(avg_trials=False, 
                                         avg_channels=True, 
                                         grid=True)
        assert all([gridFig.axes[k].get_xgridlines()[0].get_visible() for k in range(self.nTrials)])
        titleFig = self.dataReg.multiplot(avg_trials=False, 
                                          avg_channels=True, 
                                          title=theTitle)
        assert titleFig._suptitle.get_text() == theTitle
