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

data_path = '/mnt/hpc/slurm/syncopy/Tdt_reader/session-25'

adata = spy.io.load_tdt(data_path)
