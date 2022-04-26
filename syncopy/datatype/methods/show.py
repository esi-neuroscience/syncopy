# -*- coding: utf-8 -*-
#
# Syncopy data slicing methods
#

# Builtin/3rd party package imports
import numpy as np

# Local imports
from syncopy.shared.errors import SPYInfo, SPYTypeError
from syncopy.shared.kwarg_decorators import unwrap_cfg

__all__ = ["show"]


@unwrap_cfg
def show(data, squeeze=True, **kwargs):
    """
    Show (partial) contents of Syncopy object

    **Usage Notice**

    Syncopy uses HDF5 files as on-disk backing device for data storage. This
    allows working with larger-than-memory data-sets by streaming only relevant
    subsets of data from disk on demand without excessive RAM use. However, using
    :func:`~syncopy.show` this mechanism is bypassed and the requested data subset
    is loaded into memory at once. Thus, inadvertent usage of :func:`~syncopy.show`
    on a large data object can lead to memory overflow or even out-of-memory errors.

    **Usage Summary**

    Data selectors for showing subsets of Syncopy data objects follow the syntax
    of :func:`~syncopy.selectdata`. Please refer to :func:`~syncopy.selectdata`
    for a list of valid data selectors for respective Syncopy data objects.

    Parameters
    ----------
    data : Syncopy data object
        As for subset-selection via :func:`~syncopy.selectdata`, the type of `data`
        determines which keywords can be used.  Some keywords are only valid for
        certain types of Syncopy objects, e.g., "freqs" is not a valid selector
        for an :class:`~syncopy.AnalogData` object.
    squeeze : bool
        If `True` (default) any singleton dimensions are removed from the output
        array, i.e., the shape of the returned array does not contain ones (e.g.,
        ``arr.shape = (2,)`` not ``arr.shape = (1,2,1,1)``).
    **kwargs : keywords
        Valid data selectors (e.g., `trials`, `channels`, `toi` etc.). Please
        refer to :func:`~syncopy.selectdata` for a full list of available data
        selectors.

    Returns
    -------
    arr : NumPy nd-array
        A (selection) of data retrieved from the `data` input object.

    Notes
    -----
    This routine represents a convenience function for quickly inspecting the
    contents of Syncopy objects. It is always possible to manually access an object's
    numerical data by indexing the underlying HDF5 dataset: `data.data[idx]`.
    The dimension labels of the dataset are encoded in `data.dimord`, e.g., if
    `data` is a :class:`~syncopy.AnalogData` with `data.dimord` being `['time', 'channel']`
    and `data.data.shape` is `(15000, 16)`, then `data.data[:, 3]` returns the
    contents of the fourth channel across all time points.

    Examples
    --------
    Use :func:`~syncopy.tests.misc.generate_artificial_data` to create a synthetic
    :class:`syncopy.AnalogData` object.

    >>> from syncopy.tests.misc import generate_artificial_data
    >>> adata = generate_artificial_data(nTrials=10, nChannels=32)

    Show the contents of `'channel02'` across all trials:

    >>> spy.show(adata, channel='channel02')
    Syncopy <show> INFO: Showing all times 10 trials
    Out[2]: array([1.0871, 0.7267, 0.2816, ..., 1.0273, 0.893 , 0.7226], dtype=float32)

    Note that this is equivalent to

    >>> adata.show(channel='channel02')

    To preserve singleton dimensions use ``squeeze=False``:

    >>> adata.show(channel='channel02', squeeze=False)
    Out[3]:
    array([[1.0871],
           [0.7267],
           [0.2816],
           ...,
           [1.0273],
           [0.893 ],
           [0.7226]], dtype=float32)


    See also
    --------
    :func:`syncopy.selectdata` : Create a new Syncopy object from a selection
    """

    # Account for pathological cases
    if data.data is None:
        SPYInfo("Empty object, nothing to show")
        return

    # Parse single method-specific keyword
    if not isinstance(squeeze, bool):
        raise SPYTypeError(squeeze, varname="squeeze", expected="True or False")

    # Leverage `selectdata` to sanitize input and perform subset picking
    data.selectdata(inplace=True, **kwargs)

    # Truncate info message by removing any squeezed dimensions (if necessary)
    msg = data.selection.__str__().partition("with")[-1]
    if squeeze:
        removeKeys = ["one", "1 "]
        selectionTxt = np.array(msg.split(","))
        txtMask = [all(qualifier not in selTxt for qualifier in removeKeys) for selTxt in selectionTxt]
        msg = "".join(selectionTxt[txtMask])
        transform_out = np.squeeze
    else:
        transform_out = lambda x: x
    SPYInfo("Showing{}".format(msg))

    # Use an object's `_preview_trial` method fetch required indexing tuples
    idxList = []
    for trlno in data.selection.trials:
        idxList.append(data._preview_trial(trlno).idx)


    # Reset in-place subset selection
    data.selection = None

    # single trial selected
    if len(idxList) == 1:
        return transform_out(data.data[idxList[0]])
    # return multiple trials as list
    else:
        return [transform_out(data.data[idx]) for idx in idxList]
