# -*- coding: utf-8 -*-
# 
# 
# 
# Created: 2020-04-17 08:25:48
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2020-04-17 19:15:25>

import pytest
import numpy as np

from syncopy.tests.misc import generate_artificial_data, compare_figs
from syncopy.plotting import singleplot
from syncopy import __plt__
if __plt__:
    import matplotlib.pyplot as plt

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

        # test title, grid
            