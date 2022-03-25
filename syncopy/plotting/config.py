# -*- coding: utf-8 -*-
#
# Syncopy plotting setup
#

from syncopy import __plt__

if __plt__:
    import matplotlib as mpl
    mpl.style.use('seaborn-colorblind')
    # a hint of gray
    mpl.rcParams['figure.facecolor'] = '#f5faf6'
    mpl.rcParams['figure.edgecolor'] = '#f5faf6'
    mpl.rcParams['axes.facecolor'] = '#f5faf6'

# Global style settings for single-/multi-plots
pltConfig = {"sTitleSize": 15,
             "sLabelSize": 16,
             "sTickSize": 12,
             "sLegendSize": 12,
             "sFigSize": (6.4, 3.2),
             "mTitleSize": 14,
             "mLabelSize": 14,
             "mTickSize": 10,
             "mLegendSize": 12,
             "mFigSize": (10, 6.8),
             "cmap": "magma"}

# Global consistent error message if matplotlib is missing
pltErrMsg = "\nSyncopy <core> WARNING: Could not import 'matplotlib'. \n" +\
          "{} requires a working matplotlib installation. \n" +\
          "Please consider installing 'matplotlib', e.g., via conda: \n" +\
          "\tconda install matplotlib\n" +\
          "or using pip:\n" +\
          "\tpip install matplotlib"
