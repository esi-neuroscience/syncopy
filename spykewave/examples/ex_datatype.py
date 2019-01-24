# ex_datatype.py - Example script illustrating usage of `BaseData` class
# 
# Author: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Created: Januar 17 2019
# Last modified: <2019-01-24 09:04:47>

# Builtin/3rd party package imports
import numpy as np

# Add spykewave package to Python search path
import os
import sys
spw_path = os.path.abspath(".." + os.sep + "..")
if spw_path not in sys.path:
    sys.path.insert(0, spw_path)

# Import Spykewave
import spykewave as sw

# Formal defintion of function for Sphinx docstring building
def ex_datatype():
    """
    Docstring...
    """

# This prevents Sphinx from executing the script
if __name__ == "__main__":

    # # Set path to data directory
    # datadir = ".." + os.sep + ".." + os.sep + ".." + os.sep + ".." + os.sep + "Data"\
    #           + os.sep + "Joscha_testdata" + os.sep
    # 
    # # Choose data-set to read
    # dataset = "klecks_20180406_rfmapping-bar_1_xWav.lfp"
    # 
    # # Create SpykeWave data object
    # data = sw.BaseData(datadir + dataset)

    # # ChunkData testing
    # a = np.memmap('a.array', dtype='float64', mode='w+', shape=( 5000,1000)) # 38.1MB
    # a[:,:] = 111
    # b = np.memmap('b.array', dtype='float64', mode='w+', shape=(15000,1000)) # 114 MB
    # b[:,:] = 222
    # c = np.memmap('c.array', dtype='float64', mode='w+', shape=(10000,1000)) # 114 MB
    # c[:,:] = 333
    # d = ChunkData([a,b,c]) 

    # Set path to data directory
    datadir = ".." + os.sep + ".." + os.sep + ".." + os.sep + ".." + os.sep + "Data"\
              + os.sep + "testdata" + os.sep
    # datadir = os.path.join(os.sep, "mnt", "hpx", "it", "dev", "SpykeWave", "testdata")

    files = ["MT_RFmapping_session-168a1_xWav.lfp", "MT_RFmapping_session-168a1_xWav.lfp"]
    trl = np.array([[0, 20000, 2],
                    [20000, 40000, 3],
                    [40000, 150000, 1],
                    [150000, 200000, 5],
                    [200000, 350000, 4],
                    [350000, 400000, 0]])
    data = sw.BaseData(filename=[datadir + file for file in files],
                       filetype="esi",
                       trialdefinition=trl)

    # # Choose data-set to read
    # filename = "klecks_20180406_rfmapping-bar_1_xWav.lfp"
    
