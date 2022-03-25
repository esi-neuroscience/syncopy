# -*- coding: utf-8 -*-
#
# Syncopy singlepanel plot backend
#

from syncopy.plotting.config import pltConfig, pltErrMsg
from syncopy import __plt__

if __plt__:
    import matplotlib.pyplot as ppl


# -- 2d-line plots --

def mk_line_figax(xlabel='time (s)', ylabel='signal (a.u.)'):

    """
    Create the figure and axes for a
    standard 2d-line plot
    """

    fig, ax = ppl.subplots(figsize=pltConfig['sFigSize'])
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major',
                   labelsize=pltConfig['sTickSize'])

    ax.set_xlabel(xlabel, fontsize=pltConfig['sLabelSize'])
    ax.set_ylabel(ylabel, fontsize=pltConfig['sLabelSize'])

    return fig, ax


def plot_lines(ax, data_x, data_y, **pkwargs):

    if 'alpha' not in pkwargs:
        ax.plot(data_x, data_y, alpha=0.9, **pkwargs)
    else:
        ax.plot(data_x, data_y, **pkwargs)
    if 'label' in pkwargs:
        ax.legend(ncol=2, loc='upper right',
                  fontsize=pltConfig['sLegendSize'])
        # make room for the legend
        mn, mx = ax.get_ylim()        
        ax.set_ylim((mn, 1.1 * mx))


# -- image plots --

def mk_img_figax(xlabel='time (s)', ylabel='frequency (Hz)', title=''):

    """
    Create the figure and axes for an
    image plot with `imshow`
    """

    fig, ax = ppl.subplots(figsize=pltConfig['sFigSize'])

    ax.tick_params(axis='both', which='major',
                   labelsize=pltConfig['sTickSize'])
    ax.set_xlabel(xlabel, fontsize=pltConfig['sLabelSize'])
    ax.set_ylabel(ylabel, fontsize=pltConfig['sLabelSize'])
    ax.set_title(title, fontsize=pltConfig['sTitleSize'])

    return fig, ax


def plot_tfreq(ax, data_yx, times, freqs, **pkwargs):

    """
    Plot time frequency data on a 2d grid, expects standard
    row-column (freq-time) axis ordering.

    Needs frequencies (`freqs`) and sampling rate (`fs`)
    for correct units.
    """

    # extent is defined in xy order
    df = freqs[1] - freqs[0]
    extent = [times[0], times[-1],
              freqs[0] - df / 2, freqs[-1] - df / 2]

    ax.imshow(data_yx, aspect='auto', cmap=pltConfig['cmap'],
              extent=extent)
