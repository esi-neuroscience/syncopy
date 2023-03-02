# -*- coding: utf-8 -*-
#
# Load data from Field Trip .mat files
#

# Builtin/3rd party package imports
import re
import h5py
import numpy as np
from scipy import io as sio
from tqdm import tqdm

# Local imports
from syncopy.shared.errors import SPYValueError, SPYInfo, SPYWarning
from syncopy.shared.parsers import io_parser, sequence_parser, scalar_parser
from syncopy.datatype import AnalogData

__all__ = ["load_ft_raw"]

# Required fields for the ft_datatype_raw
req_fields_raw = ('time', 'trial', 'label')


def load_ft_raw(filename,
                list_only=False,
                select_structures=None,
                include_fields=None,
                mem_use=4000):

    """
    Imports raw time-series data from Field Trip
    into potentially multiple :class:`~syncopy.AnalogData` objects,
    one for each structure found within the MAT-file.

    The aim is to parse each FT data structure, which
    have the following fields (Syncopy analogon on the right):

    +--------------------+------------+
    | FT                 | Syncopy    |
    +====================+============+
    | label              | channel    |
    +--------------------+------------+
    | trial              | trial      |
    +--------------------+------------+
    | time               | time       |
    +--------------------+------------+
    | trialinfo          | trialinfo  |
    +--------------------+------------+
    | fsample (optional) | samplerate |
    +--------------------+------------+
    | cfg                | cfg        |
    +--------------------+------------+

    Limitations:

    The FT `cfg` contains a lot of meta data which at the
    moment we don't import into Syncopy. Syncopy however has
    it's own `cfg` mirroring FT's functionality (replay analyses)

    FT's `sampleinfo` is not generally compatible with Syncopy

    Parameters
    ----------
    filename: str
        Path to the MAT-file
    list_only: bool, optional
        Set to `True` to return only a list containing the names
        of the structures found
    select_structures: sequence or None, optional
        Sequence of strings, one for each structure,
        the default `None` will load all structures found
    include_fields: sequence, optional
        Additional MAT-File fields within each structure to
        be imported. They can be accessed via the `AnalogData.info`
        attribute.
    mem_use: int
        The amount of RAM requested for the import process in MB. Note that < v7.3
        MAT-File formats can only be loaded at once. For MAT-File v7.3 this should
        be at least twice the size of a single trial.

    Returns
    -------
    out_dict: dict
        Dictionary with the names of the structures as keys loaded from the MAT-File,
        and :class:`~syncopy.AnalogData` datasets as values

    Notes
    -----
    For MAT-File < v7.3 the MAT-file gets loaded completely
    into RAM using :func:`scipy.io.loadmat`, but its size should be capped by Matlab at 2GB.
    The >v7.3 MAT-files are in hdf5 format and will be read in trial-by-trial,
    this should be the Matlab default for MAT-files exceeding 2GB.

    See also
    --------
    `MAT-File formats <https://de.mathworks.com/help/matlab/import_export/mat-file-versions.html>`_
    `Field Trip datastructures <https://www.fieldtriptoolbox.org/development/datastructure/>`_

    Examples
    --------
    Load two structures `'Data_K'` and `'Data_KB'` from a MAT-File `example.mat`:

    >>> dct = load_ft_raw('example.mat', select_structures=('Data_K', 'Data_KB'))

    Access the individual :class:`~syncopy.AnalogData` datasets:

    >>> data_kb = dct['Data_KB']
    >>> data_k = dct['Data_K']

    Load all structures from `example.mat` plus additional field `'chV1'`:

    >>> dct = load_ft_raw('example.mat', include_fields=('chV1',))

    Access the additionally loaded field:

    >>> dct['Data_K'].info['chV1']

    Just peek into the MAT-File and get a list of the contained structures:

    >>> load_ft_raw('example.mat', list_only=True)
    >>> ['Data_K', 'Data_KB']
    """

    # -- Input validation --

    io_parser(filename, isfile=True)

    if select_structures is not None:
        sequence_parser(select_structures,
                        varname='select_structures',
                        content_type=str)
    if include_fields is not None:
        sequence_parser(include_fields,
                        varname='include_fields',
                        content_type=str)

    scalar_parser(mem_use, varname='mem_use', ntype="int_like", lims=[1, np.inf])

    # -- MAT-File Format --

    version = _get_Matlab_version(filename)
    msg = f"Reading MAT-File version {version} "
    SPYInfo(msg)

    # new hdf container format, use h5py
    if version >= 7.3:

        h5File = h5py.File(filename, 'r')
        struct_keys = [key for key in h5File.keys() if '#' not in key]

        struct_container = h5File
        struct_reader = lambda struct: _read_hdf_structure(struct,
                                                           h5File=h5File,
                                                           mem_use=mem_use,
                                                           include_fields=include_fields)

    # old format <2GB, use scipy's MAT reader
    else:

        if mem_use < 2000:
            msg = "MAT-File version < 7.3 does not support lazy loading"
            msg += f"\nReading {filename} might take up to 2GB of RAM, you requested only {mem_use / 1000}GB"
            SPYInfo(msg)
            lgl = '2000 or more MB'
            actual = f"{mem_use}"
            raise SPYValueError(lgl, varname='mem_use', actual=actual)

        raw_dict = sio.loadmat(filename,
                               mat_dtype=True,
                               simplify_cells=True)

        struct_keys = [skey for skey in raw_dict.keys() if '__' not in skey]

        struct_container = raw_dict
        struct_reader = lambda struct: _read_dict_structure(struct,
                                                            include_fields=include_fields)

    msg = f"Found {len(struct_keys)} structure(s): {struct_keys} in {filename}"
    SPYInfo(msg)

    if list_only:
        return struct_keys

    if len(struct_keys) == 0:
        SPYValueError(legal="At least one structure",
                      varname=filename,
                      actual="No structure found"
                      )

    # -- IO Operations --

    out_dict = {}

    # load only a subset
    if select_structures is not None:
        keys = select_structures
    # load all structures found
    else:
        keys = struct_keys

    for skey in keys:
        if skey not in struct_keys:
            msg = f"Could not find structure `{skey}` in {filename}"
            SPYWarning(msg)
            continue

        structure = struct_container[skey]
        _check_req_fields(req_fields_raw, structure)
        # the AnalogData objs
        adata = struct_reader(structure)

        # Write log-entry
        msg = f"loaded struct `{skey}` from Matlab file version {version}\n"
        msg += f"\tsource file: {filename}"
        adata.log = msg

        out_dict[skey] = adata

    return out_dict


def _read_hdf_structure(h5Group,
                        h5File,
                        mem_use,
                        include_fields=None):

    """
    Each Matlab structure contained in
    a hdf5 MAT-File is a h5py Group object.

    Each key of this Group corresponds to
    a field in the Matlab structure.

    This is the translation from FT to Syncopy:

    +--------------------+------------+
    | FT                 | Syncopy    |
    +--------------------+------------+
    | label              | channel    |
    | trial              | trial      |
    | time               | time       |
    | fsample (optional) | samplerate |
    | cfg                | X          |
    +--------------------+------------+

    """
    # for user info
    struct_name = h5Group.name[1:]

    # this should be fixed upstream such that
    # the `defaultDimord` is indeed the default :)
    AData = AnalogData(dimord=AnalogData._defaultDimord)

    # probably better to define an abstract mapping
    # if we want to support more FT formats in the future

    # these are numpy arrays holding hdf5 object references
    # i.e. one per trial, channel, time (per trial)
    trl_refs = h5Group['trial'][:, 0]
    time_refs = h5Group['time'][:, 0]
    chan_refs = h5Group['label'][0, :]

    if 'fsample' in h5Group:
        AData.samplerate = h5Group['fsample'][0, 0]
    else:
        AData.samplerate = _infer_fsample(h5File[time_refs[0]])

    # -- retrieve shape information --
    nTrials = trl_refs.size

    # peek in 1st trial to determine the number of channels
    # and one trial size for
    nSamples1, nChannels = h5File[trl_refs[0]].shape
    # compute total hdf5 shape
    # we stack along 1st axis
    trlSamples = [h5File[ref].shape[0] for ref in trl_refs]
    # in samples
    mean_trl_size = np.mean(trlSamples)
    nTotalSamples = np.sum(trlSamples)
    # get sample indices
    si = np.r_[0, np.cumsum(trlSamples)]
    sampleinfo = np.column_stack([si[:-1], si[1:]])

    itemsize = h5File[trl_refs[0]].dtype.itemsize
    # in Mbyte
    trl_size = itemsize * mean_trl_size * nChannels / 1e6

    # assumption: single trial fits into RAM
    if trl_size >= 0.4 * mem_use:
        lgl = f'{2.5 * trl_size} or more MB'
        actual = f"{mem_use}"
        raise SPYValueError(lgl, varname='mem_use', actual=actual)

    # -- IO process --

    # create new hdf5 dataset for our AnalogData
    # with the default dimord ['time', 'channel']
    # and our default data type np.float32 -> implicit casting!
    with h5py.File(AData.filename, mode="w") as h5FileOut:
        ADset = h5FileOut.create_dataset("data",
                                         dtype=np.float32,
                                         shape=[nTotalSamples, nChannels])

        pbar = tqdm(trl_refs, desc=f"{struct_name} - loading {nTrials} trials", disable=None)

        SampleCounter = 0   # trial stacking
        # one swipe per trial
        for tr in pbar:
            trl_array = h5File[tr]
            # in samples
            trl_samples = trl_array.shape[0]
            ADset[SampleCounter:SampleCounter + trl_samples, :] = trl_array
            SampleCounter += trl_samples
        pbar.close()

        AData.data = ADset

    AData._reopen()

    # -- trialdefinition --

    offsets = []
    # we need to look into the time vectors for each trial
    for time_r in time_refs:
        offsets.append(h5File[time_r][0, 0])
    offsets = np.rint(np.array(offsets) * AData.samplerate)
    trl_def = np.hstack([sampleinfo, offsets[:, None]])

    # check if there is a 'trialinfo'
    try:
        trl_def = np.hstack([trl_def, h5Group['trialinfo']])
    except KeyError:
        pass

    AData.trialdefinition = trl_def

    # each channel label is an integer array with shape (X, 1),
    # where `X` is the number of ascii encoded characters
    channels = [''.join(map(chr, h5File[cr][:, 0])) for cr in chan_refs]
    AData.channel = channels

    # -- Additional Fields --
    if include_fields is not None:
        AData.info = {}
        # additional fields in MAT-File
        afields = [k for k in h5Group.keys() if k not in req_fields_raw]
        msg = f"Found following additional fields: {afields}"
        SPYInfo(msg, caller='load_ft_raw')
        for field in include_fields:
            if field not in h5Group:
                msg = f"Could not find additional field {field} in {struct_name}"
                SPYWarning(msg, caller='load_ft_raw')
                continue

            dset = h5Group[field]
            # we only support fields pointing
            # directly to a dataset containing actual data
            # and not references to larger objects
            if isinstance(dset[0], h5py.Reference):
                msg = f"Could not read additional field '{field}'\n"
                msg += "Only simple fields holding str labels or 1D arrays are supported atm"
                SPYWarning(msg)
                continue

            # ASCII encoding via uint16
            if dset.dtype == np.uint16 and len(dset.shape) == 2:
                AData.info[field] = _parse_MAT_hdf_strings(dset).tolist()

            # numerical data can be written
            # directly as into info dict
            elif dset.dtype == np.float64:
                AData.info[field] = dset[...].tolist()

            else:
                msg = f"Could not read additional field '{field}'\n"
                msg += "Unknown data type, only 1D numerical or string arrays/fields supported"
                SPYWarning(msg)
                continue

    return AData


def _read_dict_structure(structure, include_fields=None):
    """
    Local helper to parse a single FT structure
    and return an :class:`~syncopy.AnalogData` object

    Only for for Matlab data format version < 7.3
    which was opened via scipy.io.loadmat!

    This is the translation from FT to Syncopy:

    +--------------------+------------+
    | FT                 | Syncopy    |
    +--------------------+------------+
    | label              | channel    |
    | trial              | trial      |
    | time               | time       |
    | fsample (optional) | samplerate |
    | cfg                | X          |
    +--------------------+------------+

    Each trial in FT has nChannels x nSamples ordering,
    Syncopy has nSamples x nChannels
    """


    # initialize AnalogData
    if 'fsample' in structure:
        samplerate = structure['fsample']
    else:
        samplerate = _infer_fsample(structure['time'][0])

    AData = AnalogData(samplerate=samplerate)

    # compute total hdf5 shape
    # we use fixed stacking along 1st axis
    # but channel x sample ordering in FT
    nTotalSamples = np.sum([trl.shape[1] for trl in structure['trial']])
    nChannels = structure['trial'][0].shape[0]
    sampleinfo = []

    with h5py.File(AData._filename, 'w') as h5file:

        dset = h5file.create_dataset("data",
                                     dtype=np.float32,
                                     shape=[nTotalSamples, nChannels])

        stack_count = 0
        for trl in structure['trial']:
            trl_size = trl.shape[1]
            # default data type np.float32 -> implicit casting!
            dset[stack_count:stack_count + trl_size] = trl.T.astype(np.float32)

            # construct on the fly to cover all the trials
            sampleinfo.append(np.array([stack_count, stack_count + trl_size]))

            stack_count += trl_size

        AData.data = dset

    AData._reopen()

    sampleinfo = np.array(sampleinfo)

    # get the channel ids
    channels = structure['label']
    # set the channel ids
    AData.channel = list(channels.astype(str))

    # get the offets
    offsets = np.array([tvec[0] for tvec in structure['time']])
    offsets *= AData.samplerate

    # build trialdefinition
    trl_def = np.column_stack([sampleinfo, offsets])

    # check if there is a 'trialinfo'
    try:
        trl_def = np.hstack([trl_def, structure['trialinfo']])
    except KeyError:
        pass

    AData.trialdefinition = trl_def

    # -- Additional Fields --

    if include_fields is not None:
        AData.info = {}
        # additional fields in MAT-File
        afields = [k for k in structure.keys() if k not in req_fields_raw]
        msg = f"Found following additional fields: {afields}"
        SPYInfo(msg, caller='load_ft_raw')

        for field in include_fields:
            if field not in structure:
                msg = f"Could not find additional field {field}"
                SPYWarning(msg, caller='load_ft_raw')
                continue
            # we only support fields pointing directly to some data
            # no nested structures!
            if np.ndim(structure[field]) != 0:
                msg = f"Could not read additional nested field '{field}'\n"
                msg += "Only simple fields holding str labels or 1D arrays are supported"
                SPYWarning(msg)
                continue

            AData.info[field] = structure[field].tolist()

    return AData


def _get_Matlab_version(filename):

    """
    Peeks into the 1st line of a .mat file
    and extracts the version information.
    Works for both < 7.3 and newer MAT-files.
    """

    with open(filename, 'rb') as matfile:
        line1 = next(matfile)
        # relevant information
        header = line1[:76].decode()

    # matches for example 'MATLAB 5.01'
    # with the version as only capture group
    pattern = re.compile(r"^MATLAB\s(\d*\.\d*)")
    match = pattern.match(header)

    if not match:
        lgl = 'recognizable .mat file'
        actual = 'can not recognize .mat file'
        raise SPYValueError(lgl, filename, actual)

    version = float(match.group(1))

    return version


def _check_req_fields(req_fields, structure):

    """
    Just check the the minimal required fields
    (aka keys in Python) are present in a
    Matlab structure

    Works for both old-style (dict) and
    new-style (hdf5 Group) MAT-file structures.
    """

    for key in req_fields:
        if key not in structure:
            lgl = f"{key} present in MAT structure"
            actual = f"{key} missing"
            raise SPYValueError(lgl, 'MAT structure', actual)


def _infer_fsample(time_vector):

    """
    Akin to `ft_datatype_raw` determine
    the sampling frequency from the sampling
    times
    """

    return np.mean(np.diff(time_vector))


def _parse_MAT_hdf_strings(dataset):

    """
    Expects a hdf5 dataset of shape (X, N),
    where X is the number of characters in
    a single string, and N is the number of strings.

    The entries themselves are are of integer type,
    the ASCII encoding of strings in Matlab v7.3.

    Intended for small(!!) string datasets containing
    for example some labels
    """

    # FIXME: a simple `for in ascii_arr in dataset` might do the trick as well
    # (no need to enumerate)?
    str_seq = []
    for i, ascii_arr in enumerate(dataset[...].T):
        string = ''.join(map(chr, ascii_arr))
        str_seq.append(string)

    return np.array(str_seq)
