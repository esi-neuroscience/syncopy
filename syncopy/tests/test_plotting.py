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
    nChannels = 9
    nSamples = 250
    adata = synth_data.AR2_network(nTrials=nTrials,
                                   AdjMat=np.identity(nChannels),
                                   nSamples=nSamples)

    adata += 0.3 * synth_data.linear_trend(nTrials=nTrials,
                                           y_max=nSamples / 10,
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
            def_kwargs = {'trials': 1}
            kwargs = def_kwargs

        # all plotting routines accept selection
        # `show_kwargs` directly
        if 'select' in kwargs:
            sd = kwargs.pop('select')
            self.adata.singlepanelplot(**sd, **kwargs)
            self.adata.singlepanelplot(**sd, **kwargs, shifted=False)
            self.adata.multipanelplot(**sd, **kwargs)
        else:
            fig1, ax1 = self.adata.singlepanelplot(**kwargs)
            fig2, ax2 = self.adata.singlepanelplot(**kwargs, shifted=False)
            fig3, axs = self.adata.multipanelplot(**kwargs)

        # check axes/figure references work
        if def_test:
            ax1.set_title('Shifted signals')
            fig1.tight_layout()
            ax2.set_title('Overlayed signals')
            fig2.tight_layout()
            fig3.suptitle("Multipanel plot")
            fig3.tight_layout()
        else:
            ppl.close('all')

    def test_ad_selections(self):

        # trial, channel and toi selections
        selections = helpers.mk_selection_dicts(self.nTrials,
                                                self.nChannels,
                                                toi_min=self.toi_min,
                                                toi_max=self.toi_max)

        # test all combinations
        for sel_dict in selections:
            # only single trial plotting
            # is supported until averaging is availbale
            # take random 1st trial
            sel_dict['trials'] = sel_dict['trials'][0]
            # we have to sort the channels
            # FIXME: see #291
            sel_dict['channel'] = sorted(sel_dict['channel'])
            self.test_ad_plotting(select=sel_dict)

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


if __name__ == '__main__':
    T1 = TestAnalogDataPlotting()
