# -*- coding: utf-8 -*-
#
# Simple script for testing/adjusting the tdt reader 
#

# Builtin/3rd party package imports
import numpy as np

# Add SynCoPy package to Python search path
import os
import sys

# Import package
import syncopy as spy

data_path = '/cs/slurm/syncopy/Tdt_reader'
out_path = data_path
TDTdata = spy.io.load_tdt.ESI_TDTdata(data_path,
                                      out_path,
                                      combined_data_filename='sth',
                                      subtract_median=False)
    
# how to go on from here to end up with a Syncopy Data (spy.AnalogData) object?
# ...
