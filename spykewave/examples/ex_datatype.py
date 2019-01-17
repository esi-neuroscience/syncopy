# ex_datatype.py - Example script illustrating usage of `BaseData` class
# 
# Author: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Created: Januar 17 2019
# Last modified: <2019-01-17 15:04:24>

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

    # Set path to data directory
    datadir = ".." + os.sep + ".." + os.sep + ".." + os.sep + ".." + os.sep + "Data"\
              + os.sep + "Joscha_testdata" + os.sep

    # Choose data-set to read
    dataset = "klecks_20180406_rfmapping-bar_1_xWav.lfp"

    # Create SpykeWave data object
    data = sw.BaseData(datadir + dataset)
