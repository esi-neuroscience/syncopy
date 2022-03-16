# -*- coding: utf-8 -*-
#
# Syncopy singlepanel plot backend
#

from syncopy.plotting.config import pltConfig, pltErrMsg
from syncopy import __plt__

if __plt__:
    import matplotlib.pyplot as ppl
else:
    print(pltErrMsg.format("singlepanelplot"))


def mk_line_figax(xlabel='time (s)', ylabel='signal (a.u.)'):

    """
    Create the figure and axes for a
    standard 2d-line plot
    """

    fig, ax = ppl.subplots(figsize=pltConfig['sFigSize'])
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, ax


def plot_lines(ax, data_x, data_y, **pkwargs):

    if 'alpha' not in pkwargs:
        ax.plot(data_x, data_y, alpha=0.9, **pkwargs)
    else:
        ax.plot(data_x, data_y, **pkwargs)
    if 'label' in pkwargs:
        ax.legend(ncol=2, loc='upper right')
