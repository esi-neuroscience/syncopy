# -*- coding: utf-8 -*-
#
# Syncopy plotting setup
#

from syncopy import __plt__

foreground = "#2E3440"  # nord0
background = '#fcfcfc'  # hint of gray

# dark mode
# foreground = "#D8DEE9"  # nord4
# background = "#2E3440"  # nord0

if __plt__:
    import matplotlib as mpl
    mpl.style.use('seaborn-colorblind')
    # a hint of gray
    rc_props = {
        'patch.edgecolor': foreground,
        'text.color': foreground,
        'axes.facecolor': background,
        'axes.facecolor': background,
        'figure.facecolor': background,
        "axes.edgecolor": foreground,
        "axes.labelcolor": foreground,
        "xtick.color": foreground,
        "ytick.color": foreground,
        "legend.framealpha": 0,
        "figure.facecolor": background,
        "figure.edgecolor": background,
        "savefig.facecolor": background,
        "savefig.edgecolor": background,
        'ytick.color': foreground,
        'xtick.color': foreground,
        'text.color': foreground
    }


# Global style settings for single-/multi-plots
pltConfig = {"sTitleSize": 15,
             "sLabelSize": 16,
             "sTickSize": 12,
             "sLegendSize": 12,
             "sFigSize": (6.4, 4.2),
             "mTitleSize": 12.5,
             "mLabelSize": 12.5,
             "mTickSize": 11,
             "mLegendSize": 10,
             "mXSize": 3.2,
             "mYSize": 2.4,
             "mMaxAxes": 25,
             "cmap": "magma"}

# Global consistent error message if matplotlib is missing
pltErrMsg = "\nSyncopy <core> WARNING: Could not import 'matplotlib'. \n" +\
          "{} requires a working matplotlib installation. \n" +\
          "Please consider installing 'matplotlib', e.g., via conda: \n" +\
          "\tconda install matplotlib\n" +\
          "or using pip:\n" +\
          "\tpip install matplotlib"
