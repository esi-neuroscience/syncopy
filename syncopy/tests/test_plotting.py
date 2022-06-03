# -*- coding: utf-8 -*-
#
# Test connectivity measures
#

# 3rd party imports
import pytest
import inspect
import numpy as np
import matplotlib.pyplot as ppl

# Local imports

from syncopy import AnalogData
import syncopy.tests.synth_data as synth_data
import syncopy.tests.helpers as helpers
from syncopy.shared.errors import SPYValueError
from syncopy.shared.tools import get_defaults


class TestAnalogDataPlotting():

    nTrials = 10
    nChannels = 5
    adata = synth_data.AR2_network(nTrials=nTrials,
                                   AdjMat=np.identity(nChannels))
    adata += synth_data.linear_trend(nTrials=nTrials,
                                     y_max=100,
                                     nSamples=1000,
                                     nChannels=nChannels)

    def test_ad_plotting(self, **kwargs):

        # no interactive plotting
        ppl.ioff()        

        # check if we run the default test
        def_test = not len(kwargs)

        if def_test:
            # interactive plotting
            ppl.ion()
            def_kwargs = {'trials': 1, 'toilim': [-.4, -.2]}
            kwargs = def_kwargs

        self.adata.singlepanelplot(**kwargs)


if __name__ == '__main__':
    T1 = TestAnalogDataPlotting()
    dy = T1.adata.show(trials=0, toilim=[-.4, -.2])
