# -*- coding: utf-8 -*-
#
# Basic SyNCoPy usage example
# 
# Created: 2019-03-28 16:48:31
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-04-23 11:36:37>

# Builtin/3rd party package imports
import h5py
import numpy as np

# Add SynCoPy package to Python search path
import os
import sys
spy_path = os.path.abspath(".." + os.sep + "..")
if spy_path not in sys.path:
    sys.path.insert(0, spy_path)

# Import SynCoPy
import syncopy as spy

# Define location of test data
# datadir = "/mnt/hpx/it/dev/SpykeWave/testdata/"
datadir = ".." + os.sep + ".." + os.sep + ".." + os.sep + ".." + os.sep + "Data"\
          + os.sep + "testdata" + os.sep
basename = "MT_RFmapping_session-168a1"

# Create `AnalogData` object from *.lfp/*.mua files
adataFiles = [os.path.join(datadir, basename + ext) for ext in ["_xWav.lfp", "_xWav.mua"]]
adata = spy.AnalogData(filename=adataFiles, filetype="esis")

# Create `SpikeData` object from *.spk file
sdataFiles = os.path.join(datadir, basename + "_Raws.spk")
sdata = spy.SpikeData(filename=sdataFiles, filetype="esi", dimord=["sample", "unit", "channel"])

# Create `EventData` object from *.evt file
edataFiles = os.path.join(datadir, basename + ".evt")
edata = spy.EventData(filename=edataFiles, filetype="esi", dimord=["eventid", "sample"])

# Create `EventData` object from *.dpd file
ddataFiles = os.path.join(datadir, basename + ".dpd")
ddata = spy.EventData(filename=ddataFiles, filetype="esi", dimord=["sample", "eventid"])

# Define trials for both `EventData` objects
edata.definetrial(pre=40, post=90, trigger=23001)
ddata.definetrial(pre=0.25, post=0.5, trigger=1)

# Apply `ddata` to `adata` to define trials
adata.definetrial(ddata)

