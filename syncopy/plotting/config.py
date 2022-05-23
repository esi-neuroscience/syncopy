# -*- coding: utf-8 -*-
#
# Syncopy plotting setup
#

from syncopy import __plt__

if __plt__:
    import matplotlib as mpl
    mpl.style.use('seaborn-colorblind')
    # a hint of gray
    mpl.rcParams['figure.facecolor'] = '#fcfcfc'
    mpl.rcParams['figure.edgecolor'] = '#fcfcfc'
    mpl.rcParams['axes.facecolor'] = '#fcfcfc'

# Global style settings for single-/multi-plots
pltConfig = {"sTitleSize": 15,
             "sLabelSize": 16,
             "sTickSize": 12,
             "sLegendSize": 12,
             "sFigSize": (6.4, 4.8),
             "mTitleSize": 12.5,
             "mLabelSize": 12.5,
             "mTickSize": 11,
             "mLegendSize": 10,
             "mXSize": 3.2,
             "mYSize": 2.4,
             "mMaxAxes": 35,
             "cmap": "magma"}

# Global consistent error message if matplotlib is missing
pltErrMsg = "\nSyncopy <core> WARNING: Could not import 'matplotlib'. \n" +\
          "{} requires a working matplotlib installation. \n" +\
          "Please consider installing 'matplotlib', e.g., via conda: \n" +\
          "\tconda install matplotlib\n" +\
          "or using pip:\n" +\
          "\tpip install matplotlib"
