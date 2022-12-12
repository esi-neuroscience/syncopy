# -*- coding: utf-8 -*-
#
# Syncopy data slicing methods
#

# Builtin/3rd party package imports
import numpy as np

# Local imports
from syncopy.shared.errors import SPYInfo, SPYTypeError, SPYValueError

__all__ = ["show"]


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

    # show (hdf5 indexing that is) only supports simple, ordered indexing
    # we have to painstakingly check for this
    invalid = False
    for sel_key in kwargs:
        sel = kwargs[sel_key]
        # sequence type
        if np.array(sel).size != 1:
            # some selections can be strings
            # with no clear way of sorting ('chanY', 'chanX')
            if isinstance(sel[0], str):
                # temporary selection to extract numerical indices
                sel_kw = {sel_key: sel}
                data.selectdata(inplace=True, **sel_kw)
                # extract only channel indexing (index of an index :/)
                ch_idx2 = data.dimord.index(sel_key)
                # this is now numeric!
                ch_idx = data._preview_trial(0).idx[ch_idx2]
                data.selection = None
                # consecutive, ordered selections are suddenly a slice :/
                # so all fine here actually
                if isinstance(ch_idx, slice):
                    continue
                if np.any(np.diff(ch_idx) < 0) or len(set(ch_idx)) != len(sel):
                    invalid = True
            # numeric selection, e.g. [0,4,2]
            else:
                if np.any(np.diff(sel) < 0) or len(set(sel)) != len(sel):
                    invalid = True
        elif isinstance(sel, slice):
            if sel.start > sel.stop:
                invalid = True
        if invalid:
            lgl = f"unique and sorted `{sel_key}` indices"
            act = sel
            raise SPYValueError(lgl, 'show kwargs', act)

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

    # catch totally out of range toi selection
    has_time = True if 'time' in data.dimord else False

    # Use an object's `_preview_trial` method fetch required indexing tuples
    idxList = []
    for trlno in data.selection.trial_ids:
        # each dim has an entry, list only needed for mutability
        idxs = list(data._preview_trial(trlno).idx)

        # time/toi is a special case, all other dims get checked
        # beforehand, e.g. foi, channel, ... but out of range toi's get mapped
        # repeatedly to the last index, causing invalid hdf5 indexing
        if has_time:
            idx = idxs[data.dimord.index('time')]
            if not isinstance(idx, slice) and (
                    len(idx) != len(set(idx))):
                lgl = "valid `toi` selection"
                act = sel
                raise SPYValueError(lgl, 'show kwargs', act)

        for i, prop_idx in enumerate(idxs):            
            if isinstance(prop_idx, list) and len(prop_idx) == 1:
                idxs[i] = prop_idx[0]
            
        idxList.append(tuple(idxs))

    # Reset in-place subset selection
    data.selection = None

    # single trial selected
    if len(idxList) == 1:
        return transform_out(data.data[idxList[0]])
    # return multiple trials as list
    else:
        return [transform_out(data.data[idx]) for idx in idxList]
