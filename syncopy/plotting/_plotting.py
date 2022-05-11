# -*- coding: utf-8 -*-
#
# Syncopy plotting backend
#

from syncopy.plotting.config import pltConfig
from syncopy import __plt__

if __plt__:
    import matplotlib.pyplot as ppl


# -- 2d-line plots --

def mk_line_figax(xlabel='time (s)', ylabel='signal (a.u.)'):

    """
    Create the figure and axis for a
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


def mk_multi_line_figax(nrows, ncols, xlabel='time (s)', ylabel='signal (a.u.)'):

    """
    Create the figure and axes for a
    multipanel 2d-line plot
    """

    # ncols and nrows get
    # restricted via the plotting frontend
    x_size = ncols * pltConfig['mXSize']
    y_size = nrows * pltConfig['mYSize']

    fig, axs = ppl.subplots(nrows, ncols, figsize=(x_size, y_size),
                            sharex=True, sharey=True, squeeze=False)

    # Hide the right and top spines
    # and remove all tick labels
    for ax in axs.flatten():
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(labelsize=0)

    # determine axis layout
    y_left = axs[:, 0]
    x_bottom = axs[-1, :]

    # write tick and axis labels only on outer axes to save space
    for ax in y_left:
        ax.tick_params(labelsize=pltConfig['mTickSize'])
        ax.set_ylabel(ylabel, fontsize=pltConfig['mLabelSize'])

    for ax in x_bottom:
        ax.tick_params(labelsize=pltConfig['mTickSize'])
        ax.set_xlabel(xlabel, fontsize=pltConfig['mLabelSize'])

    return fig, axs


def plot_lines(ax, data_x, data_y, leg_fontsize=pltConfig['sLegendSize'], **pkwargs):

    if 'alpha' not in pkwargs:
        ax.plot(data_x, data_y, alpha=0.9, **pkwargs)
    else:
        ax.plot(data_x, data_y, **pkwargs)
    if 'label' in pkwargs:
        ax.legend(ncol=2, loc='best', frameon=False,
                  fontsize=leg_fontsize)
        # make room for the legend
        mn, mx = ax.get_ylim()        
        ax.set_ylim((mn, 1.1 * mx))


# -- image plots --

def mk_img_figax(xlabel='time (s)', ylabel='frequency (Hz)'):

    """
    Create the figure and axes for an
    image plot with `imshow`
    """

    fig, ax = ppl.subplots(figsize=pltConfig['sFigSize'])

    ax.tick_params(axis='both', which='major',
                   labelsize=pltConfig['sTickSize'])
    ax.set_xlabel(xlabel, fontsize=pltConfig['sLabelSize'])
    ax.set_ylabel(ylabel, fontsize=pltConfig['sLabelSize'])

    return fig, ax


def mk_multi_img_figax(nrows, ncols, xlabel='time (s)', ylabel='frequency (Hz)'):

    """
    Create the figure and axes for an
    image plot with `imshow` for multiple
    sub plots
    """
    # ncols and nrows get
    # restricted via the plotting frontend
    x_size = ncols * pltConfig['mXSize']
    y_size = nrows * pltConfig['mYSize']

    fig, axs = ppl.subplots(nrows, ncols, figsize=(x_size, y_size),
                            sharex=True, sharey=True, squeeze=False)

    # determine axis layout
    y_left = axs[:, 0]
    x_bottom = axs[-1, :]

    # write tick and axis labels only on outer axes to save space
    for ax in y_left:
        ax.tick_params(labelsize=pltConfig['mTickSize'])
        ax.set_ylabel(ylabel, fontsize=pltConfig['mLabelSize'])

    for ax in x_bottom:
        ax.tick_params(labelsize=pltConfig['mTickSize'])
        ax.set_xlabel(xlabel, fontsize=pltConfig['mLabelSize'])

    return fig, axs


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

    ax.imshow(data_yx[::-1], aspect='auto', cmap=pltConfig['cmap'],
              extent=extent, **pkwargs)
