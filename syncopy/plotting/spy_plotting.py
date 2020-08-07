# -*- coding: utf-8 -*-
# 
# Syncopy plotting routines
# 
# Created: 2020-03-17 17:33:35
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2020-08-07 14:41:09>

# Builtin/3rd party package imports
import numpy as np

# Local imports
from syncopy.shared.kwarg_decorators import unwrap_cfg
from syncopy.shared.errors import SPYError, SPYTypeError, SPYValueError, SPYWarning
from syncopy.shared.parsers import data_parser, scalar_parser
from syncopy.shared.tools import get_defaults
from syncopy import __plt__

# Conditional imports and mpl customizations (provided mpl defaults have not been 
# changed by user)
if __plt__:
    import matplotlib.pyplot as plt 
    import matplotlib.style as mplstyle
    import matplotlib as mpl
    from matplotlib import colors
    
    # Syncopy default plotting settings
    spyMplRc = {"figure.dpi": 100}
    
    # Check if we're running w/mpl's default settings: if user either changed 
    # existing setting or appended new ones (e.g., color definitions), abort
    changeMplConf = True
    rcDefaults = mpl.rc_params()
    rcKeys = rcDefaults.keys()
    rcParams = dict(mpl.rcParams)
    rcParams.pop("backend")
    rcParams.pop("interactive")
    for key, value in rcParams.items():
        if key not in rcKeys:
            changeMplConf = False
            break
        if rcDefaults[key] != value:
            changeMplConf = False
            break

    # If matplotlib's global config has not been changed, incorporate modifications            
    if changeMplConf:
        mplstyle.use("fast")
        for key, value in spyMplRc.items():
            mpl.rcParams[key] = value
    
# Global style settings for single-/multi-plots
pltConfig = {"singleTitleSize": 12,
             "singleLabelSize": 10,
             "singleTickSize": 8,
             "singleLegendSize": 10,
             "singleFigSize": (6.4, 4.8),
             "multiTitleSize": 10,
             "multiLabelSize": 8,
             "multiTickSize": 6,
             "multiLegendSize": 8,
             "multiFigSize": (10, 6.8)}

# Global consistent error message if matplotlib is missing
pltErrMsg = "Could not import 'matplotlib': {} requires a working matplotlib installation!"

__all__ = ["singlepanelplot", "multipanelplot"]


@unwrap_cfg
def singlepanelplot(*data, 
               trials="all", channels="all", toilim=None, 
               avg_channels=True, 
               title=None, grid=None, overlay=True, fig=None, **kwargs):
    """
    Plot contents of Syncopy data object(s) using single-panel figure(s)

    **Usage Summary**
    
    List of Syncopy data objects and respective valid plotting commands/selectors:
    
    :class:`~syncopy.AnalogData` : trials, channels, toi/toilim
        Examples
        
        >>> fig1, fig2 = spy.singlepanelplot(data1, data2, channels=["channel01", "channel02"], overlay=False)
        >>> cfg = spy.StructDict() 
        >>> cfg.trials = [5, 3, 0]; cfg.toilim = [0.25, 0.5]
        >>> fig = spy.singlepanelplot(cfg, data1, data2, overlay=True)
    
    Parameters
    ----------
    data : Syncopy data object(s)
        One or more non-empty Syncopy data object(s). **Note**: if multiple 
        datasets are provided, they must be all of the same type (e.g., 
        :class:`~syncopy.AnalogData`) and should contain the same or at 
        least comparable channels, trials etc. Consequently, some keywords are 
        only valid for certain types of Syncopy objects, e.g., "freqs" is not a 
        valid plotting-selector for an :class:`~syncopy.AnalogData` object. 
    trials : list (integers) or None or "all"
        Trials to average across. Either list of integers representing trial numbers 
        (can include repetitions and need not be sorted), "all" or `None`. If 
        `trials` is `None`, no trial information is used and the raw contents of 
        provided input dataset(s) is plotted (**Warning**: depending on the size 
        of the supplied dataset(s), this might be very memory-intensive). 
    channels : list (integers or strings), slice, range or "all"
        Channel-selection; can be a list of channel names (``['channel3', 'channel1']``), 
        a list of channel indices (``[3, 5]``), a slice (``slice(3, 10)``) or 
        range (``range(3, 10)``). Selections can be unsorted and may include 
        repetitions. If multiple input objects are provided, `channels` needs to be a
        valid selector for all supplied datasets. 
    toilim : list (floats [tmin, tmax]) or None
        Time-window ``[tmin, tmax]`` (in seconds) to be extracted from each trial. 
        Window specifications must be sorted and not NaN but may be unbounded. Edges 
        `tmin` and `tmax` are included in the selection. If `toilim` is `None`, 
        the entire time-span in each trial is selected. If multiple input objects 
        are provided, `toilim` needs to be a valid selector for all supplied datasets. 
        **Note** `toilim` is only a valid selector if `trials` is not `None`. 
    avg_channels : bool
        If `True`, plot input dataset(s) averaged across channels specified by
        `channels`. If `False`, no averaging is performed resulting in multiple
        time-course plots, each representing a single channel. 
    title : str or None
        If `str`, `title` specifies as axis panel-title, if `None`, an auto-generated
        title is used. 
    grid : bool or None
        If `True`, grid-lines are drawn, if `None` or `False` no grid-lines are 
        rendered. 
    overlay : bool
        If `True`, and multiple input objects were provided, supplied datasets are 
        plotted on top of each other (in the order of submission). If a single object 
        was provided, ``overlay = True`` and `fig` is a :class:`~matplotlib.figure.Figure`, 
        the supplied dataset is overlaid on top of any existing plot(s) in `fig`. 
        **Note**: using an existing figure to overlay dataset(s) is only 
        supported for figures created with this routine.
    fig : matplotlib.figure.Figure or None
        If `None`, new :class:`~matplotlib.figure.Figure` instance(s) are created
        for provided input dataset(s). If `fig` is a :class:`~matplotlib.figure.Figure`,
        the code attempts to overlay provided input dataset(s) on top of existing
        plots in `fig`. **Note**: overlay-plots are only supported for figures
        generated with this routine. Only a single figure can be provided. Thus,
        in case of multiple input datasets with ``overlay = False``, any supplied
        `fig` is ignored. 
        
    Returns
    -------
    fig : (list of) matplotlib.figure.Figure instance(s)
        Either single figure (single input dataset or multiple input datasets 
        with ``overlay = True``) or list of figures (multiple input datasets
        and ``overlay = False``). 
        
    Notes
    -----
    This function uses `matplotlib <https://matplotlib.org/>`_ to render data 
    visualizations. Thus, usage of Syncopy's plotting capabilities requires 
    a working matplotlib installation. 
    
    The actual rendering is performed by class methods specific to the provided 
    input object types (e.g., :class:`~syncopy.AnalogData`). Thus, 
    :func:`~syncopy.singlepanelplot` is mainly a convenience function and management routine
    that invokes the appropriate drawing code. 
    
    Data subset selection for plotting is performed using :func:`~syncopy.selectdata`, 
    thus additional in-place data-selection via a `select` keyword is **not** supported. 
        
    Examples
    --------
    Please refer to the respective `singlepanelplot` class methods for detailed usage
    examples specific to the respective Syncopy data object type. 
    
    See also
    --------
    :func:`~syncopy.multipanelplot` : visualize Syncopy objects using multi-panel figure(s)
    :meth:`syncopy.AnalogData.singlepanelplot` : `singlepanelplot` for :class:`~syncopy.AnalogData` objects
    """
    
    # Abort if matplotlib is not available
    if not __plt__:
        raise SPYError(pltErrMsg.format("singlepanelplot"))
    
    # Collect all keywords of corresponding class-method (w/possibly user-provided 
    # values) in dictionary 
    defaults = get_defaults(data[0].singlepanelplot)
    lcls = locals()
    kwords = {}
    for kword in defaults:
        kwords[kword] = lcls[kword]

    # Call plotting manager
    return _anyplot(*data, overlay=overlay, method="singlepanelplot", **kwords, **kwargs)


@unwrap_cfg
def multipanelplot(*data, 
              trials="all", channels="all", toilim=None, 
              avg_channels=False, avg_trials=True, 
              title=None, grid=None, overlay=True, fig=None, **kwargs):
    """
    Plot contents of Syncopy data object(s) using multi-panel figure(s)

    **Usage Summary**
    
    List of Syncopy data objects and respective valid plotting commands/selectors:
    
    :class:`~syncopy.AnalogData` : trials, channels, toi/toilim
        Examples
        
        >>> fig1, fig2 = spy.multipanelplot(data1, channels=["channel01", "channel02"])
        >>> cfg = spy.StructDict() 
        >>> cfg.trials = [5, 3, 0]; cfg.toilim = [0.25, 0.5]
        >>> fig = spy.multipanelplot(cfg, data1, data2, overlay=True)
    
    Parameters
    ----------
    data : Syncopy data object(s)
        One or more non-empty Syncopy data object(s). **Note**: if multiple 
        datasets are provided, they must be all of the same type (e.g., 
        :class:`~syncopy.AnalogData`) and should contain the same or at 
        least comparable channels, trials etc. Consequently, some keywords are 
        only valid for certain types of Syncopy objects, e.g., "freqs" is not a 
        valid plotting-selector for an :class:`~syncopy.AnalogData` object. 
    trials : list (integers) or None or "all"
        Either list of integers representing trial numbers 
        (can include repetitions and need not be sorted), "all" or `None`. If 
        `trials` is `None`, no trial information is used and the raw contents of 
        provided input dataset(s) is plotted (**Warning**: depending on the size 
        of the supplied dataset(s), this might be very memory-intensive). 
    channels : list (integers or strings), slice, range or "all"
        Channel-selection; can be a list of channel names (``['channel3', 'channel1']``), 
        a list of channel indices (``[3, 5]``), a slice (``slice(3, 10)``) or 
        range (``range(3, 10)``). Selections can be unsorted and may include 
        repetitions. If multiple input objects are provided, `channels` needs to be a
        valid selector for all supplied datasets. 
    toilim : list (floats [tmin, tmax]) or None
        Time-window ``[tmin, tmax]`` (in seconds) to be extracted from each trial. 
        Window specifications must be sorted and not NaN but may be unbounded. Edges 
        `tmin` and `tmax` are included in the selection. If `toilim` is `None`, 
        the entire time-span in each trial is selected. If multiple input objects 
        are provided, `toilim` needs to be a valid selector for all supplied datasets. 
        **Note** `toilim` is only a valid selector if `trials` is not `None`. 
    avg_channels : bool
        If `True`, plot input dataset(s) averaged across channels specified by
        `channels`. If `False`, and ``avg_trials = True`` no channel-averaging is 
        performed resulting in multiple panels, each representing the (trial-averaged) 
        time-course of a single channel. If ``avg_channels = avg_trials = False``,
        multiple panels containing multiple time-courses are rendered with each 
        panel representing a trial, and each time-course corresponding to a single 
        channel. For ``avg_channel = avg_trials = True`` no output is generated, 
        as this functionality is covered by :func:`~syncopy.singlepanelplot`. 
    avg_trials : bool
        If `True`, plot input dataset(s) averaged across trials specified by `trials`. 
        Specific panel allocation depends on value of `avg_channels` (see above). 
        If `avg_trials` is `True` but `trials` is `None`, 
        a :class:`~syncopy.shared.errors.SPYValueError` is raised. 
    title : str or None
        If `str`, `title` specifies figure title, if `None`, an auto-generated
        title is used. 
    grid : bool or None
        If `True`, grid-lines are drawn, if `None` or `False` no grid-lines are 
        rendered. 
    overlay : bool
        If `True`, and multiple input objects were provided, supplied datasets are 
        plotted on top of each other (in the order of submission). If a single object 
        was provided, ``overlay = True`` and `fig` is a :class:`~matplotlib.figure.Figure`, 
        the supplied dataset is overlaid on top of any existing plot(s) in `fig`. 
        **Note**: using an existing figure to overlay dataset(s) is only 
        supported for figures created with this routine.
    fig : matplotlib.figure.Figure or None
        If `None`, new :class:`~matplotlib.figure.Figure` instance(s) are created
        for provided input dataset(s). If `fig` is a :class:`~matplotlib.figure.Figure`,
        the code attempts to overlay provided input dataset(s) on top of existing
        plots in `fig`. **Note**: overlay-plots are only supported for figures
        generated with this routine. Only a single figure can be provided. Thus,
        in case of multiple input datasets with ``overlay = False``, any supplied
        `fig` is ignored. 
        
    Returns
    -------
    fig : (list of) matplotlib.figure.Figure instance(s)
        Either single figure (single input dataset or multiple input datasets 
        with ``overlay = True``) or list of figures (multiple input datasets
        and ``overlay = False``). 
        
    Notes
    -----
    This function uses `matplotlib <https://matplotlib.org/>`_ to render data 
    visualizations. Thus, usage of Syncopy's plotting capabilities requires 
    a working matplotlib installation. 
    
    The actual rendering is performed by class methods specific to the provided 
    input object types (e.g., :class:`~syncopy.AnalogData`). Thus, 
    :func:`~syncopy.multipanelplot` is mainly a convenience function and management routine
    that invokes the appropriate drawing code. 
    
    Data subset selection for plotting is performed using :func:`~syncopy.selectdata`, 
    thus additional in-place data-selection via a `select` keyword is **not** supported. 
        
    Examples
    --------
    Please refer to the respective `multipanelplot` class methods for detailed usage
    examples specific to the respective Syncopy data object type. 
    
    See also
    --------
    :func:`~syncopy.singlepanelplot` : visualize Syncopy objects using single-panel figure(s)
    :meth:`syncopy.AnalogData.multipanelplot` : `multipanelplot` for :class:`~syncopy.AnalogData` objects
    """

    # Abort if matplotlib is not available
    if not __plt__:
        raise SPYError(pltErrMsg.format("multipanelplot"))

    # Collect all keywords of corresponding class-method (w/possibly user-provided 
    # values) in dictionary 
    defaults = get_defaults(data[0].multipanelplot)
    lcls = locals()
    kwords = {}
    for kword in defaults:
        kwords[kword] = lcls[kword]

    # Call plotting manager
    return _anyplot(*data, overlay=overlay, method="multipanelplot", **kwords, **kwargs)


def _anyplot(*data, overlay=None, method=None, **kwargs):
    """
    Local management routine that invokes respective class methods based on 
    caller (`obj.singlepanelplot` or `obj.multipanelplot`)
    
    This is an auxiliary method that is intended purely for internal use. Please
    refer to the user-exposed methods :func:`~syncopy.singlepanelplot` and/or
    :func:`~syncopy.multipanelplot` to actually generate plots of Syncopy data objects. 
    """

    # The only error-checking done in here: ensure `overlay` is Boolean and assert 
    # `data` contains only non-empty Syncopy objects
    if not isinstance(overlay, bool):
        raise SPYTypeError(overlay, varname="overlay", expected="bool")
    for obj in data:
        try:
            data_parser(obj, varname="data", empty=False)
        except Exception as exc:
            raise exc
        # FIXME: while plotting is still WIP
        if obj.__class__.__name__ not in ["AnalogData", "SpectralData"]:
            errmsg = "Plotting currently only supported for `AnalogData` and `SpectralData` objects"
            raise NotImplementedError(errmsg)

    # See if figure was provided
    start = 0
    nData = len(data)
    fig = kwargs.pop("fig", None)
    if not overlay and fig is not None and nData > 1:
        msg = "User-provided figures not supported for non-overlay visualization " +\
            "of {} datasets. Supplied figure will not be used. "
        SPYWarning(msg.format(nData))
        fig = None

    # If we're overlaying, preserve initial figure object to plot over iteratively        
    if overlay:
        fig = getattr(data[0], method)(fig=fig, **kwargs)
        start = 1
    figList = []
    for n in range(start, nData):
        figList.append(getattr(data[n], method)(fig=fig, **kwargs))

    # Return single figure object (if `overlay` is `True`) or list of mulitple figs    
    if overlay:
        return fig
    return figList


def _compute_toilim_avg(self):
    """
    Coming soon..
    """
    
    tLengths = np.zeros((len(self._selection.trials),), dtype=np.intp)
    for k, tsel in enumerate(self._selection.time):
        start, stop = tsel.start, tsel.stop
        if start is None:
            start = 0
        if stop is None:
            stop = self._get_time([self._selection.trials[k]], 
                                    toilim=[-np.inf, np.inf])[0].stop
        tLengths[k] = stop - start
        
    if np.unique(tLengths).size > 1:
        lgl = "time-selections of equal length for averaging across trials"
        act = "time-selections of varying length"
        raise SPYValueError(legal=lgl, varname="toilim", actual=act)

    if tLengths[0] < 2:
        lgl = "time-selections containing at least two samples"
        act = "time-selections containing fewer than two samples"
        raise SPYValueError(legal=lgl, varname="toilim", actual=act)
        
    return tLengths


def _setup_figure(npanels, nrow=None, ncol=None, xLabel=None, yLabel=None,
                  include_colorbar=False, sharex=None, sharey=None, grid=None):
    """
    Coming soon...
    """
    
    # Note: if `xLabel` and/or `yLabel` is `None`, setting the corresponding axis
    # label simply uses an empty string '' and does not alter the axis - no need
    # for any ``if is None``` gymnastics below
    if npanels == 1:
        
        # Simplest case: single panel, no colorbar
        if not include_colorbar:
            fig, ax = plt.subplots(1, tight_layout=True, squeeze=True,
                                   figsize=pltConfig["singleFigSize"])

        # Single panel w/colorbar            
        else:
            fig, (ax, cax) = plt.subplots(1, 2, tight_layout=True, squeeze=True, 
                                          gridspec_kw={"wspace": 0.05, "width_ratios": [1, 0.025]},
                                          figsize=pltConfig["singleFigSize"])
            cax.tick_params(axis="both", labelsize=pltConfig["singleTickSize"])

        # Axes formatting gymnastics done for all single-panel plots            
        ax.set_xlabel(xLabel, size=pltConfig["singleLabelSize"])            
        ax.set_ylabel(yLabel, size=pltConfig["singleLabelSize"])            
        ax.tick_params(axis="both", labelsize=pltConfig["singleTickSize"])
        ax.autoscale(enable=True, axis="x", tight=True)
        ax.grid(grid)

        # Designate figure object as single-panel plotting target
        fig.singlepanelplot = True
        
    else:

        # Either use provided row/col settings or compute best fit
        nrow, ncol = _layout_subplot_panels(npanels, nrow, ncol)
        
        # If no explicit axis sharing settings were provided, make an executive decision
        if sharex is None:
            sharex = True
        if sharey is None:
            sharey = True
        
        # Multiple panels, no colorbar
        if not include_colorbar:
            (fig, ax_arr) = plt.subplots(nrow, ncol, constrained_layout=False, 
                                         gridspec_kw={"wspace": 0, "hspace": 0.35, 
                                                      "left": 0.05, "right": 0.97},
                                         figsize=pltConfig["multiFigSize"],
                                         sharex=sharex, sharey=sharey, squeeze=False)

        # Multiple panels, append colorbar via gridspec
        else:
            (fig, ax_arr) = plt.subplots(nrow, ncol, constrained_layout=False,
                                         gridspec_kw={"wspace": 0, "hspace": 0.35,
                                                      "left": 0.05, "right": 0.94},
                                         figsize=pltConfig["multiFigSize"],
                                         sharex=sharex, sharey=sharey, squeeze=False)
            gs = fig.add_gridspec(nrows=nrow, ncols=1, left=0.945, right=0.955)
            cax = fig.add_subplot(gs[:, 0])
            cax.tick_params(axis="both", labelsize=pltConfig["multiTickSize"])
            
        # Show xlabel only on bottom row of panels
        for col in range(ncol):
            ax_arr[-1, col].set_xlabel(xLabel, size=pltConfig["multiLabelSize"])
            
        # Omit first x-tick in all panels except first panel-row, show ylabel only 
        # on left border of first panel column
        for row in range(nrow):
            for col in range(1, ncol):
                ax_arr[row, col].xaxis.get_major_locator().set_params(prune="lower")
            ax_arr[row, 0].set_ylabel(yLabel, size=pltConfig["multiLabelSize"])
            
        # Flatten axis array (to ease panel counting) and remove surplus panels
        ax_arr = ax_arr.flatten(order="C")
        for ax in ax_arr:
            ax.tick_params(axis="both", labelsize=pltConfig["multiTickSize"])
            ax.autoscale(enable=True, axis="x", tight=True)
            ax.grid(True)
        for k in range(npanels, nrow * ncol):
            ax_arr[k].remove()
        ax = ax_arr
        
        # Designate figure object as multi-panel plotting target
        fig.multipanelplot = True
        
    # Attach custom Syncopy plotting attributes to newly created figure
    fig.objCount = 0
    fig.npanels = npanels
    
    # All done, return figure object, axis (array) and potentially color-bar axis
    if not include_colorbar:
        return fig, ax
    return fig, ax, cax


def _setup_colorbar(fig, ax, cax, label=None, outline=False, vmin=None, vmax=None):
    """
    Coming soon...
    """
    
    if fig.npanels == 1:
        axes = [ax]
    else:
        axes = ax
        
    if vmin is not None or vmax is not None:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for k in range(fig.npanels):
            axes[k].images[0].set_norm(norm)
    cbar = fig.colorbar(axes[0].images[0], cax=cax)
    cbar.set_label(label, size=pltConfig["singleLabelSize"])
    cbar.outline.set_visible(outline)
    return cbar


def _layout_subplot_panels(npanels, nrow=None, ncol=None, ndefault=5, maxpanels=50):
    """
    Create space-optimal subplot grid given required number of panels
    
    Parameters
    ----------
    npanels : int
        Number of required subplot panels in figure
    nrow : int or None
        Required number of panel rows. Note, if both `nrow` and `ncol` are not `None`,
        then ``nrow * ncol >= npanels`` has to be satisfied, otherwise a 
        :class:`~syncopy.shared.errors.SPYValueError` is raised. 
    ncol : int or None
        Required number of panel columns. Note, if both `nrow` and `ncol` are not `None`,
        then ``nrow * ncol >= npanels`` has to be satisfied, otherwise a 
        :class:`~syncopy.shared.errors.SPYValueError` is raised. 
    ndefault: int
        Default number of panel columns for grid construction (only relevant if 
        both `nrow` and `ncol` are `None`). 
    maxpanels : int
        Maximally allowed number of subplot panels for which a grid is constructed 
        
    Returns
    -------
    nrow : int
        Number of rows of constructed subplot panel grid
    nrow : int
        Number of columns of constructed subplot panel grid
        
    Notes
    -----
    If both `nrow` and `ncol` are `None`, the constructed grid will have the 
    dimension `N` x `ndefault`, where `N` is chosen "optimally", i.e., the smallest
    integer that satisfies ``ndefault * N >= npanels``. 
    Note further, that this is an auxiliary method that is intended purely for 
    internal use. Thus, error-checking is only performed on potentially user-provided 
    inputs (`nrow` and `ncol`). 
    
    Examples
    --------
    Create grid of default dimensions to hold eight panels
    
    >>> _layout_subplot_panels(8, ndefault=5)
    (2, 5)
    
    Create a grid that must have 4 rows
    
    >>> _layout_subplot_panels(8, nrow=4)
    (4, 2)
    
    Create a grid that must have 8 columns
    
    >>> _layout_subplot_panels(8, ncol=8)
    (1, 8)
    """

    # Abort if requested panel count is less than one or exceeds provided maximum    
    try:
        scalar_parser(npanels, varname="npanels", ntype="int_like", lims=[1, np.inf])
    except Exception as exc:
        raise exc
    if npanels > maxpanels:
        lgl = "a maximum of {} panels in total".format(maxpanels)
        raise SPYValueError(legal=lgl, actual=str(npanels), varname="npanels")

    # Row specifcation was provided, cols may or may not
    if nrow is not None:
        try:
            scalar_parser(nrow, varname="nrow", ntype="int_like", lims=[1, np.inf])
        except Exception as exc:
            raise exc
        if ncol is None:
            ncol = np.ceil(npanels / nrow).astype(np.intp)

    # Column specifcation was provided, rows may or may not
    if ncol is not None:
        try:
            scalar_parser(ncol, varname="ncol", ntype="int_like", lims=[1, np.inf])
        except Exception as exc:
            raise exc
        if nrow is None:
            nrow = np.ceil(npanels / ncol).astype(np.intp)

    # After the preparations above, this condition is *only* satisfied if both
    # `nrow` = `ncol` = `None` -> then use generic grid-layout
    if nrow is None:
        ncol = ndefault 
        nrow = np.ceil(npanels / ncol).astype(np.intp)
        ncol = min(ncol, npanels)

    # Complain appropriately if requested no. of panels does not fit inside grid
    if nrow * ncol < npanels:
        lgl = "row- and column-specification of grid to fit all panels"
        act = "grid with {0} rows and {1} columns but {2} panels"
        raise SPYValueError(legal=lgl, actual=act.format(nrow, ncol, npanels), 
                            varname="nrow/ncol")
        
    # In case a grid was provided too big for the requested no. of panels (e.g., 
    # 8 panels in an 4 x 3 grid -> would fit in 3 x 3), just warn, don't crash
    if nrow * ncol - npanels >= ncol:
        msg = "Grid dimension ({0} rows x {1} columns) larger than necessary " +\
            "for {2} panels. "
        SPYWarning(msg.format(nrow, ncol, npanels))
        
    return nrow, ncol
