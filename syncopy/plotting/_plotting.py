# -*- coding: utf-8 -*-
#
# Syncopy plotting backend
#

# 3rd party imports
import numpy as np

from syncopy.plotting.config import pltConfig, rc_props, foreground
from syncopy import __plt__
from syncopy.plotting import _helpers

if __plt__:
    import matplotlib
    import matplotlib.pyplot as ppl


# for the legends
ncol_max = 3


# -- 2d-line plots --
@matplotlib.rc_context(rc_props)
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


@matplotlib.rc_context(rc_props)
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


@matplotlib.rc_context(rc_props)
def plot_lines(ax, data_x, data_y,
               leg_fontsize=pltConfig['sLegendSize'],
               shifted=False,
               **pkwargs):

    if shifted:
        offsets = _helpers.shift_multichan(data_y)
        data_y = data_y + offsets
        # no colors needed
        pkwargs['color'] = foreground

    if 'alpha' not in pkwargs:
        pkwargs['alpha'] = 0.9

    ax.plot(data_x, data_y, **pkwargs)

    # plot the legend
    if 'label' in pkwargs:
        # multi-chan stacking, use labels as ticks
        if shifted and data_y.ndim > 1:
            pos = np.array(data_y.mean(axis=0))
            ax.set_yticks(pos, pkwargs['label'])

        else:
            ax.legend(ncol=ncol_max, loc='best', frameon=False,
                      fontsize=leg_fontsize,
                      )
            # make room for the legend
            mn, mx = data_y.min(), data_y.max()

            stretch = lambda x, fac: np.abs((fac - 1) * x)

            ax.set_ylim((mn - stretch(mn, 1.1), mx + stretch(mx, 1.1)))


# -- image plots --
@matplotlib.rc_context(rc_props)
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


@matplotlib.rc_context(rc_props)
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


@matplotlib.rc_context(rc_props)
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

    cmap = pkwargs.pop('cmap', pltConfig['cmap'])
    ax.imshow(data_yx[::-1], aspect='auto', cmap=cmap,
              extent=extent, **pkwargs)
