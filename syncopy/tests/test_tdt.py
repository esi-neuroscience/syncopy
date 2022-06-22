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

data_path = '/mnt/hpc/slurm/syncopy/Tdt_reader/'
out_path = data_path
# how to go on from here to end up with a Syncopy Data (spy.AnalogData) object?
# ...

TDT_Load_Info = spy.io.load_tdt.ESI_TDTinfo(data_path)
DataInfo_loaded = TDT_Load_Info.load_tdt_info()
Files = TDT_Load_Info.get_files('.sev', 'DivAtt_session-25_LFPs')
TDT_Data = spy.io.load_tdt.ESI_TDTdata(data_path, out_path, 'sth', subtract_median=False, channels=None, export=True)
Data_Syncopy = TDT_Data.data_aranging(Files, DataInfo_loaded)