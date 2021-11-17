# -*- coding: utf-8 -*-
#
# Syncopy data slicing methods
#

# Builtin/3rd party package imports
import numpy as np

# Local imports
from syncopy.shared.errors import SPYInfo
from syncopy.shared.kwarg_decorators import unwrap_cfg

__all__ = ["show"]


@unwrap_cfg
def show(data, **kwargs):
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

    >>> spy.show(adata, channels=['channel02'])
    Syncopy <selectdata> INFO: In-place selection attached to data object: Syncopy AnalogData selector with 1 channels, all times, 10 trials
    Syncopy <show> INFO: Showing 1 channels, all times, 10 trials
    Out[11]:
    array([[1.627 ],
        [1.7906],
        [1.1757],
        ...,
        [1.1498],
        [0.7753],
        [1.0457]], dtype=float32)

    Note that this is equivalent to

    >>> adata.show(channels=['channel02'])

    See also
    --------
    :func:`syncopy.selectdata` : Create a new Syncopy object from a selection
    """

    # Account for pathological cases
    if data.data is None:
        SPYInfo("Empty object, nothing to show")
        return

    # Leverage `selectdata` to sanitize input and perform subset picking
    data.selectdata(inplace=True, **kwargs)

    # Use an object's `_preview_trial` method fetch required indexing tuples
    SPYInfo("Showing{}".format(data._selection.__str__().partition("with")[-1]))
    idxList = []
    for trlno in data._selection.trials:
        idxList.append(data._preview_trial(trlno).idx)

    # Perform some slicing/list-selection gymnastics: ensure that selections
    # that result in contiguous slices are actually returned as such (e.g.,
    # `idxList = [(slice(1,2), [2]), (slice(2,3), [2])` -> `returnIdx = [slice(1,3), [2]]`)
    singleIdx = [False] * len(idxList[0])
    returnIdx = list(idxList[0])
    for sk, selectors in enumerate(zip(*idxList)):
        if np.unique(selectors).size == 1:
            singleIdx[sk] = True
        else:
            if all(isinstance(sel, slice) for sel in selectors):
                gaps = [selectors[k + 1].start - selectors[k].stop for k in range(len(selectors) - 1)]
                if all(gap == 0 for gap in gaps):
                    singleIdx[sk] = True
                    returnIdx[sk] = slice(selectors[0].start, selectors[-1].stop)

    # Reset in-place subset selection
    data._selection = None

    # If possible slice underlying dataset only once, otherwise return a list
    # of arrays corresponding to selected trials
    if all(si == True for si in singleIdx):
        return data.data[tuple(returnIdx)]
    else:
        return [data.data[idx] for idx in idxList]
