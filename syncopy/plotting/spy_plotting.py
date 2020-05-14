# -*- coding: utf-8 -*-
# 
# Syncopy plotting routines
# 
# Created: 2020-03-17 17:33:35
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2020-05-14 14:29:21>

# # Builtin/3rd party package imports
# import tensorlfow

# Local imports
from syncopy.shared.kwarg_decorators import unwrap_cfg
from syncopy.shared.errors import SPYError, SPYTypeError, SPYWarning
from syncopy.shared.parsers import data_parser
from syncopy.shared.tools import get_defaults
from syncopy import __plt__
import syncopy as spy  # FIXME: for WIP type-error-checking

# Conditional imports and mpl customizations (provided mpl defaults have not been 
# changed by user)
if __plt__:
    import matplotlib.pyplot as plt 
    import matplotlib.style as mplstyle
    import matplotlib as mpl
    
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

__all__ = ["singleplot", "multiplot"]


@unwrap_cfg
def singleplot(*data, 
               trials="all", channels="all", toilim=None, 
               avg_channels=True, 
               title=None, grid=None, overlay=True, fig=None, **kwargs):
    """
    Plot contents of Syncopy data object(s) using single-panel figure(s)

    **Usage Summary**
    
    List of Syncopy data objects and respective valid plotting commands/selectors:
    
    :class:`~syncopy.AnalogData` : trials, channels, toi/toilim
        Examples
        
        >>> fig1, fig2 = spy.singleplot(data1, data2, channels=["channel01", "channel02"], overlay=False)
        >>> cfg = spy.StructDict() 
        >>> cfg.trials = [5, 3, 0]; cfg.toilim = [0.25, 0.5]
        >>> fig = spy.singleplot(cfg, data1, data2, overlay=True)
    
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
    :func:`~syncopy.singleplot` is mainly a convenience function and management routine
    that invokes the appropriate drawing code. 
    
    Data subset selection for plotting is performed using :func:`~syncopy.selectdata`, 
    thus additional in-place data-selection via a `select` keyword is **not** supported. 
        
    Examples
    --------
    Please refer to the respective `singleplot` class methods for detailed usage
    examples specific to the respective Syncopy data object type. 
    
    See also
    --------
    :func:`~syncopy.multiplot` : visualize Syncopy objects using multi-panel figure(s)
    :meth:`syncopy.AnalogData.singleplot` : `singleplot` for :class:`~syncopy.AnalogData` objects
    """
    
    # Abort if matplotlib is not available
    if not __plt__:
        raise SPYError(pltErrMsg.format("singleplot"))
    
    # Collect all keywords of corresponding class-method (w/possibly user-provided 
    # values) in dictionary 
    defaults = get_defaults(data[0].singleplot)
    lcls = locals()
    kwords = {}
    for kword in defaults:
        kwords[kword] = lcls[kword]

    # Call plotting manager
    return _anyplot(*data, overlay=overlay, method="singleplot", **kwords, **kwargs)


@unwrap_cfg
def multiplot(*data, 
              trials="all", channels="all", toilim=None, 
              avg_channels=False, avg_trials=True, 
              title=None, grid=None, overlay=True, fig=None, **kwargs):
    """
    Plot contents of Syncopy data object(s) using multi-panel figure(s)

    **Usage Summary**
    
    List of Syncopy data objects and respective valid plotting commands/selectors:
    
    :class:`~syncopy.AnalogData` : trials, channels, toi/toilim
        Examples
        
        >>> fig1, fig2 = spy.multiplot(data1, channels=["channel01", "channel02"])
        >>> cfg = spy.StructDict() 
        >>> cfg.trials = [5, 3, 0]; cfg.toilim = [0.25, 0.5]
        >>> fig = spy.multiplot(cfg, data1, data2, overlay=True)
    
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
        as this functionality is covered by :func:`~syncopy.singleplot`. 
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
    :func:`~syncopy.multiplot` is mainly a convenience function and management routine
    that invokes the appropriate drawing code. 
    
    Data subset selection for plotting is performed using :func:`~syncopy.selectdata`, 
    thus additional in-place data-selection via a `select` keyword is **not** supported. 
        
    Examples
    --------
    Please refer to the respective `multiplot` class methods for detailed usage
    examples specific to the respective Syncopy data object type. 
    
    See also
    --------
    :func:`~syncopy.single` : visualize Syncopy objects using single-panel figure(s)
    :meth:`syncopy.AnalogData.multiplot` : `multiplot` for :class:`~syncopy.AnalogData` objects
    """

    # Abort if matplotlib is not available
    if not __plt__:
        raise SPYError(pltErrMsg.format("multiplot"))

    # Collect all keywords of corresponding class-method (w/possibly user-provided 
    # values) in dictionary 
    defaults = get_defaults(data[0].multiplot)
    lcls = locals()
    kwords = {}
    for kword in defaults:
        kwords[kword] = lcls[kword]

    # Call plotting manager
    return _anyplot(*data, overlay=overlay, method="multiplot", **kwords, **kwargs)


def _anyplot(*data, overlay=None, method=None, **kwargs):
    """
    Coming soon...
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
        if not isinstance(obj, spy.AnalogData):
            errmsg = "Plotting currently only supported for `AnalogData` objects"
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
