# -*- coding: utf-8 -*-
#
# Test functionality of data selection features in Syncopy
#

# Builtin/3rd party package imports
import pytest
import numpy as np
import inspect
import dask.distributed as dd
from numbers import Number

# Local imports
import syncopy.datatype as spd
from syncopy.tests.misc import flush_local_cluster
from syncopy.datatype import AnalogData, SpectralData
from syncopy.datatype.base_data import Selector
from syncopy.datatype.methods.selectdata import selectdata
from syncopy.shared.errors import SPYError, SPYValueError, SPYTypeError
from syncopy.tests.test_specest_fooof import _get_fooof_signal
from syncopy.shared.tools import StructDict
from syncopy import freqanalysis

import syncopy as spy

# The procedure here is:
# (1) test if `Selector` instance was constructed correctly (i.e., indexing tuples
#     look as expected, ordered list -> slice conversion works etc.)
# (2) test if data was correctly selected from source object (i.e., compare shapes,
#     property contents and actual numeric data arrays)
# Multi-selections are not tested here but in the respective class tests (e.g.,
# "time" + "channel" + "trial" `AnalogData` selections etc.)


class Test_AD_Selections():

    # Set up "global" parameters for data objects to be tested 
    nChannels = 10
    nSamples = 5  # per trial
    nTrials = 3
    samplerate = 2.0
    data = {}
    trl = {}

    trldef = np.vstack([np.arange(0, nSamples * nTrials, nSamples),
                        np.arange(0, nSamples * nTrials, nSamples) + nSamples,
                        np.ones(nTrials) * -1]).T

    # this is a running array with shape: nSamples*nTrials x nChannels
    # and with data[i, j] = i+1 + j * nSamples*nTrials

    data = np.arange(1, nTrials * nChannels * nSamples + 1).reshape(nChannels, nSamples * nTrials).T
    adata = spy.AnalogData(data=data, samplerate=samplerate,
                           trialdefinition=trldef)


    # map selection keywords to selector attributes (holding the idx to access selected data)
    map_sel_attr = dict(trials = 'trial_ids',
                       channel = 'channel',
                       latency = 'time',
                       )

    def test_ad_selection(self):

        """ 
        Create a simple selection and test the returned data
        """

        selection = {'trials': 1, 'channel': [6, 2], 'latency': [0, 1]}
        
        # pick the data by hand, latency [0, 1] covers 2nd - 4th sample index
        # as time axis is array([-0.5,  0. ,  0.5,  1. ,  1.5])

        solution = T1.adata.data[self.nSamples : self.nSamples * 2]
        solution = np.column_stack([solution[1:4, 6], solution[1:4, 2]])
        res = spy.selectdata(T1.adata, selection)

        assert np.all(solution == res.data)
        

    def test_valid_ad(self):

        """
        Instantiate Selector class and check its only attributes (the idx)
        """
        
        # each selection test is a 2-tuple: (selection kwargs, dict with same kws and the idx "solutions")
        valid_selections = [
            ({
            'channel': ["channel03", "channel01"],
            'latency': [0, 1], 
            'trials': np.arange(2)},
                           {
            # these are the idx used to access the actual data
            'channel': [2, 0],
            'latency': 2 * [slice(1, 4, 1)],
            'trials': [0, 1]
                           }),
            ({
            # with some repetitions
            'channel': [7, 3, 3],
            'trials': [0, 1, 1]},
                           {
            'channel': [7, 3, 3],
            'trials': [0, 1, 1]
                               })

             ]                
        
        for selection in valid_selections:
            # instantiate Selector and check attributes
            sel_kwargs, solution = selection
            selector_object = Selector(self.adata, sel_kwargs)
            for sel_kw in sel_kwargs.keys():
                attr_name = self.map_sel_attr[sel_kw]
                assert getattr(selector_object, attr_name) == solution[sel_kw]

    def test_invalid_ad(self):

        # each selection test is a 3-tuple: (selection kwargs, Error, error message sub-string)
        invalid_selections = [
            ({'channel': ["channel33", "channel01"]},
             SPYValueError, "existing names or indices"),
            ({'channel': "my-non-existing-channel"},
             SPYValueError, "existing names or indices"),                                    
            ({'channel': 99},
             SPYValueError, "existing names or indices"),  
            ({'latency': 1}, SPYTypeError, "expected array_like"),
            ({'latency': [0, 10]}, SPYValueError, "at least one trial covering the latency window"),
            ({'latency': 'sth'}, SPYValueError, "'maxperiod'"),
            ({'trials': [-3]}, SPYValueError, "all array elements to be bound"),
            ({'trials': ['1', '6']}, SPYValueError, "expected dtype = numeric")
        ]

        for selection in invalid_selections:
            sel_kw, error, err_str = selection
            with pytest.raises(error, match=err_str):
                spy.selectdata(self.adata, sel_kw)
                        

if __name__ == '__main__':
    T1 = Test_AD_Selections()
