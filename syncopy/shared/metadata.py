#  Function for handling additional return values from compute functions

import h5py
from hmac import compare_digest
from numbers import Number
import numpy as np

from syncopy.shared.errors import (SPYInfo, SPYTypeError, SPYValueError, SPYWarning)


def metadata_from_hdf5_file(h5py_filename, delete_afterwards=True):
    """
    Extract metadata from h5py file.

    This extracts metadata as a standard dictionary from the 'metadata' group of a (virtual or standard)
    hdf5 file. Note that it converts the attributes from the hdf5 attribute manager into a standard dictionary
    (that is independent of whether the hdf5 file is still open). This function is intended to be used on
    metadata temporarily attached to an hdf5 file by the cF.

    Parameters
    ----------
    h5py_filename str
        path to hdf5 file. The file will be opened for reading, and closed in the end.
        The file must contain a standard or virtual dataset named 'data'.
        If it does not contain 'metadata' group, the returned value will be `None`.
    delete_afterwards bool
        Whether to delete the metadata from the hdf5 file after extracting it.

    Returns
    -------
    metadata None or dict
        If a dict, that dict contains two more dictionaries at keys `'dsets'` and `'attrs'`. Both
        sub dicts are of type `(str, np.ndarray)`.
    """
    metadata = None
    open_mode = "a" if delete_afterwards else "r"
    with h5py.File(h5py_filename, mode=open_mode) as h5f:
        if 'data' in h5f:
            main_dset = h5f['data']
            if main_dset.is_virtual:
                metadata_list = list()  # A list of dicts.

                # Now open the virtual sources and check there for the metadata.
                for source_tpl in main_dset.virtual_sources():
                    with h5py.File(source_tpl.file_name, mode=open_mode) as h5f_virtual_part:
                        if 'metadata' in h5f_virtual_part:
                            virtual_metadata_grp = h5f_virtual_part['metadata']
                            metadata_list.append(extract_md_group(virtual_metadata_grp))
                            if delete_afterwards:
                                del h5f_virtual_part['metadata']
                metadata = _merge_md_list(metadata_list)
            else:
                # the main_dset is not virtual, so just grab the metadata group from the file root.
                if 'metadata' in h5f:
                    metadata = extract_md_group(h5f['metadata'])
                    if delete_afterwards:
                        del h5f['metadata']
        else:
            raise SPYValueError("'data' dataset in hd5f file {of}.".format(of=h5py_filename), actual="no such dataset")
    return metadata


def _merge_md_list(md_list):
    """
    Merge a list of dictionaries as returned by `extract_md_group()` into a single dictionary.

    For this to make any sense, the dicts in the `md_list` sub dicts must have unique keys. If that
    is not the case, later dicts will overwrite values of previous ones. This is not checked.

    Parameters
    ----------
    md_list: a list of dictionaries. Each entry dict has to be of type `(str, np.ndarray)`.

    Returns
    -------
    dict, where entries are of type `(str, np.ndarray)`.
    """
    if not md_list:
        return None
    metadata = dict()
    for md in md_list:
        # We just join all of them into a single dict, the unique keys allow this.
        metadata = {**metadata, **md}
    return metadata


def _parse_backend_metadata(metadata, check_attr_dsize=True):
    """
    Parse and validate extra cF return value.

    Parameters
    ----------
    metadata: dict
        The keys must be of type `str`.
        The values must be of type `np.ndarray`, but the size of the `ndarray`s is limited to 64kB, i.e., they must be small.
        The dtype of the arrays must not be `object`, as there is no hdf5 equivalent for that.
        These are limits of hdf5 attributes, see the h5py documentation on `attributes` for details.
    check_attr_dsize: boolean
        Wheter to compute size of arrays and print warnings if they are too large.

    Returns
    -------
    dict, where `(key, value)` are of type `(str, np.ndarray)`.
    """
    attribs = dict()

    if metadata is None:
        return attribs

    if not isinstance(metadata, dict):
        raise SPYTypeError(metadata, varname="metadata", expected="dict")

    for k, v in metadata.items():
        if not isinstance(v, np.ndarray):
            raise SPYTypeError(v, varname="value in metadata", expected="np.ndarray")
        if isinstance(k, str):
            attribs[k] = v
        else:
            raise SPYValueError("keys in metadata must be strings", varname="details")
    if check_attr_dsize:
        for k,v in attribs.items():
            dsize_kb = np.prod(v.shape) * v.dtype.itemsize / 1024.
            if dsize_kb > 64:
                SPYWarning("cF details: attribute '{attr}' has size {attr_size} kb, which is > the allowed 64 kb limit.".format(attr=k, attr_size=dsize_kb))
    return attribs


def parse_cF_returns(res):
    """
    Split the first and second return value of user-supplied cF, if a second one exists.

    Also checks the contract on the allowed return values.

    Returns
    -------
    res: np.ndarray
    details: dict or None
    """
    details = None  # This holds the 2nd return value from a cF, if any.
    if isinstance(res, tuple):  # The cF has a 2nd return value.
        if len(res) != 2:
            raise SPYValueError("user-supplied compute function must return a single ndarray or a tuple with length exactly 2", actual="tuple with length {tl}".format(tl=len(res)))
        else:
            res, details = res
        if details is not None: # Accept and silently ignore a 2nd return value of None.
            if isinstance(details, dict):
                for _, v in details.items():
                    if not isinstance(v, np.ndarray):
                        raise SPYValueError("the second return value of user-supplied compute functions must be a dict containing np.ndarrays")
                    if v.dtype == object:
                        raise SPYValueError("the second return value of user-supplied compute functions must be a dict containing np.ndarrays with datatype other than 'np.object'")
            else:
                raise SPYValueError("the second return value of user-supplied compute functions must be a dict")
    else:
        if not isinstance(res, np.ndarray):
            raise SPYValueError("user-supplied compute function must return a single ndarray or a tuple with length exactly 2", actual="neither tuple nor np.ndarray")
    return res, details


def h5_add_metadata(h5fout, metadata, unique_key_suffix=""):
    """
    Add details, the second return value of user-supplied cF, after parsing with `_parse_backend_metadata`,
    as a 'metadata' group to an existing hdf5 file.

    Parameters
    ----------
    hdf5_filename: h5py file instance | str
        Open h5py file, or path to existing hdf5 file. The file will be openend in write mode,
        written to, and then flushed and closed.
    metadata: dict
        The second return value of user-supplied cF, with the limitations described in `_parse_backend_metadata()`.
    unique_key_suffix: str or int
        A suffix to add to each attrib or dset name, to make it unique. Leave at the default for
        no suffix. Typically something like '__n_m', where `n` and `m` are integers and `n` is a
        trial index, and `m` is a chunk index.
        If an integer `n` is passed, it will be converted to the str '__n_0', where `n` is the
        integer (i.e., it will be treated as a trial index for which only a single chunk exists).
    """
    if metadata is None:
        return

    close_file = False
    if isinstance(h5fout, str):
        close_file = True # We openend it, we close it.
        h5fout = h5py.File(h5fout, mode="w")

    if isinstance(unique_key_suffix, Number):
        unique_key_suffix = "__" + str(unique_key_suffix) + "_0"

    grp = h5fout.require_group("metadata")
    attribs = _parse_backend_metadata(metadata)
    for k, v in attribs.items():
        k_unique = k + unique_key_suffix
        grp.attrs.create(k_unique, data=v)
    h5fout.flush()

    if close_file:
        h5fout.close()


def encode_unique_md_label(label, trial_idx, call_idx=0):
    """Assemble something like `test`, `2` and `0` into `test__2_0`."""
    return(label + "__" + str(trial_idx) + "_" + str(call_idx))


def decode_unique_md_label(unique_label):
    """
    Splits something like `test__2_0` into `test`, `2` and `0`.

    Parameters
    ----------
    unique_label: str, with format `<label>__<trial_idx>_<chunk_idx>'`.

    Returns
    -------
    tuple of `str`: the `'label'`, `'trial_idx'` and `'chunk_idx'`.
    """
    try:
        lab_ind = unique_label.rsplit("__")
        label = lab_ind[0]
        trialidx_callidx = lab_ind[1].rsplit("_")
        trialidx = trialidx_callidx[0]
        callidx = trialidx_callidx[1]
    except Exception as ex:
        raise SPYValueError(f"Could not decode metadata key '{unique_label}' into label, trial_index and chunk index. Expected input string in format `<label>__<trial_idx>_<chunk_idx>', e.g. 'pp__0_0': '{str(ex)}'")

    return label, trialidx ,callidx


def extract_md_group(md):
    """
    Extract metadata from h5py 'metadata' group and return a standard dict.

    Parameters
    ----------
    md: a h5py group, that contains metadata attributes as 'attrs'.

    Returns
    -------
    dict, containing entries of type `(str, np.ndarray)`.
    """
    metadata = dict()
    for k, v in md.attrs.items():
        metadata[k] = v.copy()  # copy the numpy array
    return metadata


def cast_0array(rule, arr):

    """
    Helper routine to "unpack" hdf5 0-dim attribute arrays,
    as even though they are effectively scalar,
    they can't be directly serialized to go into .info
    """

    rules = {'float': lambda x: float(x),
             'int': lambda x: int(x),
             'bool': lambda x: bool(x),
             'str': lambda x: str(x)
             }

    if rule not in rules:
        lgl = f"one of {rules.keys()}"
        raise SPYValueError(lgl, "rule", rule)

    if arr.ndim != 0:
        lgl = "0-dim numpy array"
        act = f"{arr.ndim}-dim array"
        raise SPYValueError(lgl, "arr", act)

    # return cast directly
    return rules[rule](arr)


def check_freq_hashes(metadata, out):

    # check individual freq. axis hashes
    ref_hash, ref_id = None, None
    trl_mismatches = []
    for md_label, fhash in metadata.items():
        _, trl_id, chk_id = decode_unique_md_label(md_label)
        if ref_hash is None:
            ref_hash = fhash
            ref_id = trl_id
        else:
            if not compare_digest(ref_hash, fhash):
                trl_mismatches.append(trl_id)
    # some freq axis were different
    if trl_mismatches:
        msg = (f"Frequency axes hashes mismatched for {len(trl_mismatches)} trials: "
               f"{trl_mismatches} against reference hash from first trial {ref_id}.")
        SPYWarning(msg)
        out.log = msg
        out.info['mismatched freq. axis trial ids'] = trl_mismatches


def metadata_nest(metadata):
    """
    Nest md dictionary keys with identical label prefixes into sub dictionaries.

    Put another way, this will add a layer of new dictionaries, which are the unique label
    names of the keys of the original dictionary. The unique label names of the keys are computed
    by running `decode_unique_md_label` on each key and considering the unique first return values.
    E.g., ```metadata = { 'ap__0_0': 1, 'ap__0_1': 2, 'pp__0_0': 3, 'pp__0_1': 4}``` becomes
    ```metadata_nested = { 'ap' : { 'ap__0_0': 1, 'ap__0_1': 2}, 'pp': {'pp__0_0': 3, 'pp__0_1': 4}}```.

    Parameters
    ----------
    metadata: dict
        Dictionary with metadata keys that can be handled by `decode_unique_md_label`.

    Returns
    -------
    metadata_nested: dict
        Nested version of the dict.

    See also
    --------
    metadata_unnest: performs the reverse operation of this function.
    """
    metadata_nested = dict()
    for unique_attr_label, v in metadata.items():
        label, trial_idx, call_idx = decode_unique_md_label(unique_attr_label)
        if not label in metadata_nested:
            metadata_nested[label] = dict()
        metadata_nested[label][unique_attr_label] = v
    return metadata_nested


def metadata_unnest(metadata):
    """
    Unnest md dictionary.

    E.g., ```metadata_nested = { 'ap' : { 'ap__0_0': 1, 'ap__0_1': 2}, 'pp': {'pp__0_0': 3, 'pp__0_1': 4}}``` becomes
    ```metadata = { 'ap__0_0': 1, 'ap__0_1': 2, 'pp__0_0': 3, 'pp__0_1': 4}```.

    Parameters
    ----------
    metadata: dict
        Dictionary with nested metadata keys.

    Returns
    -------
    metadata_unnested: dict
        Unnested version of the dict.

    See also
    --------
    metadata_nest: performs the reverse operation of this function.
    """
    metadata_unnested = dict()
    for nested_category_name, nested_dict in metadata.items():
        if not isinstance(nested_dict, dict):
            raise SPYValueError(legal="Dict containing only other dictionaries at first level.", varname="metadata", actual=f"Value at key '{nested_category_name}' is not a dict.")
        for unique_attr_label, nested_value in nested_dict.items():
            if unique_attr_label in metadata_unnested:  # It's already in there, from a previous dict!
                raise SPYValueError(legal="Dict containing no duplicated keys in nested sub dictionaries at first level.", varname="metadata", actual=f"Duplicate key '{unique_attr_label}': cannot unnest without losing data.")
            metadata_unnested[unique_attr_label] = nested_value
    return metadata_unnested


