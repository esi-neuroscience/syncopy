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
    This is just an adapter to call the
    plotting methods of the respective datatype

    Parameters
    ----------
    data : an instance derived from :class:`~syncopy.datatype.base_data`
    show_kwargs : :func:`~syncopy.datatype.methods.show.show` arguments
    """

    if not __plt__:
        SPYWarning(pltErrMsg)
        return

    data.singlepanelplot(**show_kwargs)


def multipanelplot(data, **show_kwargs):

    """
    This is just an adapter to call the
    plotting methods of the respective datatype

    Parameters
    ----------
    data : an instance derived from :class:`~syncopy.datatype.base_data`
    show_kwargs : :func:`~syncopy.datatype.methods.show.show` arguments
    """

    if not __plt__:
        SPYWarning(pltErrMsg)
        return

    data.multipanelplot(**show_kwargs)
