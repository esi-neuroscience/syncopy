# -*- coding: utf-8 -*-
#
# Top-level interfaces for the plotting functionality
#

from syncopy import __plt__
from syncopy.plotting.config import pltErrMsg
from syncopy.shared.errors import SPYWarning

__all__ = ['singlepanelplot', 'multipanelplot']


def singlepanelplot(data, **show_kwargs):

    """
    Plot Syncopy data in a single panel

    Careful with selecting to many trials/channels
    as this can quickly lead to memory exhaustion for
    big datasets.

    Parameters
    ----------
    data : :class:`~syncopy.datatype.base_data`
        Any (derived) Syncopy data type
    show_kwargs : dict
        :func:`~syncopy.datatype.methods.show.show` arguments to select
        which parts of the data to plot

    Examples
    --------

    Plot the 1st trial of `data`:

    >>> spy.singlepanelplot(data, trials=0)

    Alternatively directly use the method attached to `data`:

    >>> data.singlepanelplot(trials=0)

    Select a time- and frequency window (for e.g. :func:`~syncopy.SpectralData`):

    >>> data.singlepanelplot(trials=0, foilim=[20, 50], toilim=[0, 0.25])
    """

    if not __plt__:
        SPYWarning(pltErrMsg)
        return

    data.singlepanelplot(**show_kwargs)


def multipanelplot(data, **show_kwargs):

    """
    Plot Syncopy data in multiple panels

    Careful with selecting to many trials/channels
    as this can quickly lead to memory exhaustion for
    big datasets.

    Parameters
    ----------
    data : :class:`~syncopy.datatype.base_data`
        Any (derived) Syncopy data type
    show_kwargs : dict
        :func:`~syncopy.datatype.methods.show.show` arguments to select
        which parts of the data to plot

    Examples
    --------

    Plot 4 channels of the 1st trial of `data`:

    >>> spy.singlepanelplot(data, trials=0, channel=[1, 2, 3, 4])

    Alternatively directly use the method attached to `data`:

    >>> data.singlepanelplot(trials=0, channel=[1, 2, 3, 4])

    Select a time- and frequency window (for e.g. :func:`~syncopy.SpectralData`):

    >>> data.singlepanelplot(trials=0, foilim=[20, 50], toilim=[0, 0.25], channel=['chanA', 'chanB'])
    """

    if not __plt__:
        SPYWarning(pltErrMsg)
        return

    data.multipanelplot(**show_kwargs)
