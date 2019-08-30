# -*- coding: utf-8 -*-
# 
# Module for all kinds of parsing gymnastics
# 
# Created: 2019-01-08 09:58:11
# Last modified by: Joscha Schmiedt [joscha.schmiedt@esi-frankfurt.de]
# Last modification time: <2019-08-30 11:24:48>

# Builtin/3rd party package imports
import os
import numpy as np
import numbers
import functools
from inspect import signature

# Local imports
from syncopy.shared.errors import SPYIOError, SPYTypeError, SPYValueError, SPYError
import syncopy as spy

__all__ = ["get_defaults"]


def io_parser(fs_loc, varname="", isfile=True, ext="", exists=True):
    """
    Parse file-system location strings for reading/writing files/directories

    Parameters
    ----------
    fs_loc : str
        String pointing to (hopefully valid) file-system location
        (absolute/relative path of file or directory ).
    varname : str
        Local variable name used in caller, see Examples for details.
    isfile : bool
        Indicates whether `fs_loc` points to a file (`isfile = True`) or
        directory (`isfile = False`)
    ext : str or 1darray-like
        Valid filename extension(s). Can be a single string (e.g., `ext = "lfp"`)
        or a list/1darray of valid extensions (e.g., `ext = ["lfp", "mua"]`).
    exists : bool
        If `exists = True` ensure that file-system location specified by `fs_loc` exists
        (typically used when reading from `fs_loc`), otherwise (`exists = False`)
        check for already present conflicting files/directories (typically used when
        creating/writing to `fs_loc`).

    Returns
    -------
    fs_path : str
        Absolute path of `fs_loc`.
    fs_name : str (only if `isfile = True`)
        Name (including extension) of input file (without path).

    Examples
    --------
    To test whether `"/path/to/dataset.lfp"` points to an existing file, one
    might use

    >>> io_parser("/path/to/dataset.lfp")
    '/path/to', 'dataset.lfp'

    The following call ensures that a folder called "mydata" can be safely
    created in the current working directory

    >>> io_parser("mydata", isfile=False, exists=False)
    '/path/to/cwd/mydata'

    Suppose a routine wants to save data to a file with potential
    extensions `".lfp"` or `".mua"`. The following call may be used to ensure
    the user input `dsetname = "relative/dir/dataset.mua"` is a valid choice:

    >>> abs_path, filename = io_parser(dsetname, varname="dsetname", ext=["lfp", "mua"], exists=False)
    >>> abs_path
    '/full/path/to/relative/dir/'
    >>> filename
    'dataset.mua'
    """

    # Start by resovling potential conflicts
    if not isfile and len(ext) > 0:
        print("<io_parser> WARNING: filename extension(s) specified but " +\
              "`isfile = False`. Exiting...")
        return

    # Make sure `fs_loc` is actually a string
    if not isinstance(fs_loc, str):
        raise SPYTypeError(fs_loc, varname=varname, expected=str)

    # Avoid headaches, use absolute paths...
    fs_loc = os.path.abspath(os.path.expanduser(fs_loc))

    # Ensure that filesystem object does/does not exist
    if exists and not os.path.exists(fs_loc):
        raise SPYIOError(fs_loc, exists=False)
    if not exists and os.path.exists(fs_loc):
        raise SPYIOError(fs_loc, exists=True)

    # First, take care of directories...
    if not isfile:
        isdir = os.path.isdir(fs_loc)
        if (isdir and not exists):
            raise SPYIOError (fs_loc, exists=isdir)
        elif (not isdir and exists):
            raise SPYValueError(legal="directory", actual="file")
        else:
            return fs_loc

    # ...now files
    else:

        # Separate filename from its path
        file_name = os.path.basename(fs_loc)

        # If wanted, parse filename extension(s)
        if len(ext):

            # Extract filename extension and get rid of its dot
            file_ext = os.path.splitext(file_name)[1]
            file_ext = file_ext.replace(".", "")

            # In here, having no extension counts as an error
            error = False
            if len(file_ext) == 0:
                error = True
            if file_ext not in str(ext) or error:
                if isinstance(ext, (list, np.ndarray)):
                    ext = "'" + "or '".join(ex + "' " for ex in ext)
                raise SPYValueError(ext, varname="filename-extension", actual=file_ext)

        # Now make sure file does or does not exist
        isfile = os.path.isfile(fs_loc)
        if (isfile and not exists):
            raise SPYIOError(fs_loc, exists=isfile)
        elif (not isfile and exists):
            raise SPYValueError(legal="file", actual="directory")
        else:
            return fs_loc.split(file_name)[0], file_name

        
def scalar_parser(var, varname="", ntype=None, lims=None):
    """
    Parse scalars

    Parameters
    ----------
    var : scalar
        Scalar quantity to verify
    varname : str
        Local variable name used in caller, see Examples for details.
    ntype : None or str
        Expected numerical type of `var`. Possible options include any valid
        builtin type as well as `"int_like"` (`var` is expected to have
        no significant digits after its decimal point, e.g., 3.0, -12.0 etc.).
        If `ntype` is `None` the numerical type of `var` is not checked.
    lims : None or two-element list_like
        Lower (`lims[0]`) and upper (`lims[1]`) bounds for legal values of `var`.
        Note that the code checks for non-strict inequality, i.e., `var = lims[0]` or
        `var = lims[1]` are both considered to be valid values of `var`.
        Using `lims = [-np.inf, np.inf]` may be employed to ensure that `var` is
        finite and non-NaN. For complex scalars bounds-checking is performed
        element-wise, that is both real and imaginary part of `var` have to be
        inside the  bounds provided by `lims` (see Examples for details).
        If `lims` is `None` bounds-checking is not performed.

    Returns
    -------
    Nothing : None

    Examples
    --------
    Assume `freq` is supposed to be a scalar with integer-like values between
    10 and 1000. The following calls confirm the validity of `freq`

    >>> freq = 440
    >>> scalar_parser(freq, varname="freq", ntype="int_like", lims=[10, 1000])
    >>> freq = 440.0
    >>> scalar_parser(freq, varname="freq", ntype="int_like", lims=[10, 1000])
        
    Conversely, these values of `freq` yield errors

    >>> freq = 440.5    # not integer-like
    >>> scalar_parser(freq, varname="freq", ntype="int_like", lims=[10, 1000])
    >>> freq = 2        # outside bounds
    >>> scalar_parser(freq, varname="freq", ntype="int_like", lims=[10, 1000])
    >>> freq = '440'    # not a scalar
    >>> scalar_parser(freq, varname="freq", ntype="int_like", lims=[10, 1000])

    For complex scalars bounds-checking is performed element-wise on both
    real and imaginary part:

    >>> scalar_parser(complex(2,-1), lims=[-3, 5])  # valid
    >>> scalar_parser(complex(2,-1), lims=[-3, 1])  # invalid since real part is greater than 1

    See also
    --------
    array_parser : similar functionality for parsing array-like objects
    """

    # Make sure `var` is a scalar-like number
    if not isinstance(var, numbers.Number):
        raise SPYTypeError(var, varname=varname, expected="scalar")

    # If required, parse type ("int_like" is a bit of a special case here...)
    if ntype is not None:
        if ntype == "int_like":
            if np.round(var) != var:
                raise SPYValueError(ntype, varname=varname, actual=str(var))
        else:
            if type(var) != getattr(__builtins__, ntype):
                raise SPYTypeError(var, varname=varname, expected=ntype)

    # If required perform bounds-check: transform scalar to NumPy array
    # to be able to handle complex scalars too
    if lims is not None:
        if isinstance(var, complex):
            val = np.array([var.real, var.imag])
            legal = "both real and imaginary part to be "
        else:
            val = np.array([var])
            legal = "value to be "
        if np.any(val < lims[0]) or np.any(val > lims[1]) or not np.isfinite(var):
            legal += "greater or equals {lb:s} and less or equals {ub:s}"
            raise SPYValueError(legal.format(lb=str(lims[0]), ub=str(lims[1])),
                                varname=varname, actual=str(var))

    return


def array_parser(var, varname="", ntype=None, hasinf=None, hasnan=None,
                 lims=None, dims=None):
    """
    Parse array-like objects

    Parameters
    ----------
    var : array_like
        Array object to verify
    varname : str
        Local variable name used in caller, see Examples for details.
    ntype : None or str
        Expected data type of `var`. Possible options are any valid
        builtin type, all NumPy dtypes as as well as `"numeric"` (a catch-all
        to ensure `var` only contains numeric elements) and "int_like"` 
        (all elements of `var` are expected to have no significant digits 
        after the decimal point, e.g., 3.0, -12.0 etc.). 
        If `ntype` is `None` the data type of `var` is not checked. 
    hasinf : None or bool
        If `hasinf` is `False` the input array `var` is considered invalid 
        if it contains non-finite elements (`np.inf`), vice-versa if `hasinf`
        is `True`. If `hasinf` is `None` elements of `var` are not probed 
        for finiteness. 
    hasnan : None or bool
        If `hasnan` is `False` the input array `var` is considered invalid 
        if it contains undefined elements (`np.nan`), vice-versa if `hasnan`
        is `True`. If `hasnan` is `None` elements of `var` are not probed 
        for well-posedness. 
    lims : None or two-element list_like
        Lower (`lims[0]`) and upper (`lims[1]`) bounds for legal values of `var`'s 
        elements. Note that the code checks for non-strict inequality, 
        i.e., `var[i] = lims[0]` or `var[i] = lims[1]` are both considered 
        to be valid elements of `var`. 
        For complex arrays bounds-checking is performed on both real and 
        imaginary parts of each component of `var`. That is, all elements of 
        `var` have to satisfy `lims[0] <= var[i].real <= lims[1]` as well as 
        `lims[0] <= var[i].imag <= lims[1]` (see Examples for details). 
        Note that `np.inf` and `np.nan` entries are ignored during bounds-
        checking. Use the keywords `hasinf` and `hasnan` to probe an array 
        for infinite and non-numeric entries, respectively. 
        If `lims` is `None` bounds-checking is not performed. 
    dims : None or int or tuple
        Expected number of dimensions (if `dims` is an integer) or shape 
        (if `dims` is a tuple) of `var`. By default, singleton dimensions 
        of `var` are ignored if `dims` is a tuple, i.e., for `dims = (10, )` 
        an array `var` with `var.shape = (10, 1)` is considered valid. However, 
        if singleton dimensions are explicitly queried by setting `dims = (10, 1)`
        any array `var` with `var.shape = (10, )` or `var.shape = (1, 10)` is 
        considered invalid. 
        Unknown dimensions can be represented as `None`, i.e., for 
        `dims = (10, None)` arrays with shape `(10, 1)`, `(10, 100)` or 
        `(10, 0)` are all considered valid, however, any 1d-array (e.g., 
        `var.shape = (10,)`) is invalid. 
        If `dims` is an integer, `var.ndim` has to match `dims` exactly, i.e.,
        any array `var` with `var.shape = (10, )` is considered invalid if 
        `dims = 2` and conversely, `dims = 1` and `var.shape = (10,  1)` 
        triggers an exception. 
    
    Returns
    -------
    Nothing : None

    Examples
    --------
    Assume `time` is supposed to be a 1d-array with floating point components
    bounded by 0 and 10. The following calls confirm the validity of `time`

    >>> time = np.linspace(0, 10, 100)
    >>> array_parser(time, varname="time", lims=[0, 10], dims=1)
    >>> array_parser(time, varname="time", lims=[0, 10], dims=(100,))

    Artificially appending a singleton dimension to `time` does not affect
    parsing:

    >>> time = time[:,np.newaxis]
    >>> time.shape
    (100, 1)
    >>> array_parser(time, varname="time", lims=[0, 10], dims=(100,))

    However, explicitly querying for a row-vector fails

    >>> array_parser(time, varname="time", lims=[0, 10], dims=(1,100))

    Complex arrays are parsed analogously:

    >>> spec = np.array([np.complex(2,3), np.complex(2,-2)])
    >>> array_parser(spec, varname="spec", dims=1)
    >>> array_parser(spec, varname="spec", dims=(2,))

    Note that bounds-checking is performed component-wise on both real and
    imaginary parts:

    >>> array_parser(spec, varname="spec", lims=[-3, 5])    # valid
    >>> array_parser(spec, varname="spec", lims=[-1, 5])    # invalid since spec[1].imag < lims[0]

    Character lists can be parsed as well:

    >>> channels = ["channel1", "channel2", "channel3"]
    >>> array_parser(channels, varname="channels", dims=1)
    >>> array_parser(channels, varname="channels", dims=(3,))
    
    See also
    --------
    scalar_parser : similar functionality for parsing numeric scalars
    """

    # Make sure `var` is array-like and convert it to ndarray to simplify parsing
    if not isinstance(var, (np.ndarray, list)):
        raise SPYTypeError(var, varname=varname, expected="array_like")
    arr = np.array(var)

    # If bounds-checking is requested but `ntype` is not set, use the
    # generic "numeric" option to ensure array is actually numeric
    if lims is not None and ntype is None:
        ntype = "numeric"

    # If required, parse type (handle "int_like" and "numeric" separately)
    if ntype is not None:
        msg = "dtype = {dt:s}"
        if ntype in ["numeric", "int_like"]:
            if not np.issubdtype(arr.dtype, np.number):
                raise SPYValueError(msg.format(dt="numeric"), varname=varname,
                                    actual=msg.format(dt=str(arr.dtype)))
            if ntype == "int_like":
                if not np.all([np.round(a) == a for a in arr]):
                    raise SPYValueError(msg.format(dt=ntype), varname=varname)
        else:
            if not np.issubdtype(arr.dtype, np.dtype(ntype).type):
                raise SPYValueError(msg.format(dt=ntype), varname=varname,
                                    actual=msg.format(dt=str(arr.dtype)))

    # If required, parse finiteness of array-elements
    if hasinf is not None:
        if not hasinf and np.isinf(arr).any():
            lgl = "finite numerical array"
            act = "array with {} `inf` entries".format(str(np.isinf(arr).sum()))
            raise SPYValueError(legal=lgl, varname=varname, actual=act)
        if hasinf and not np.isinf(arr).any():
            lgl = "numerical array with infinite (`np.inf`) entries"
            act = "finite numerical array"
            raise SPYValueError(legal=lgl, varname=varname, actual=act)

    # If required, parse well-posedness of array-elements
    if hasnan is not None:
        if not hasnan and np.isnan(arr).any():
            lgl = "well-defined numerical array"
            act = "array with {} `NaN` entries".format(str(np.isnan(arr).sum()))
            raise SPYValueError(legal=lgl, varname=varname, actual=act)
        if hasnan and not np.isnan(arr).any():
            lgl = "numerical array with undefined (`np.nan`) entries"
            act = "well-defined numerical array"
            raise SPYValueError(legal=lgl, varname=varname, actual=act)

    # If required perform component-wise bounds-check (remove NaN's and Inf's first)
    if lims is not None:
        fi_arr = arr[np.isfinite(arr)]
        if np.issubdtype(fi_arr.dtype, np.dtype("complex").type):
            amin = min(fi_arr.real.min(), fi_arr.imag.min())
            amax = max(fi_arr.real.max(), fi_arr.imag.max())
        else:
            amin = fi_arr.min()
            amax = fi_arr.max()
        if amin < lims[0] or amax > lims[1]:
            legal = "all array elements to be bounded by {lb:s} and {ub:s}"
            raise SPYValueError(legal.format(lb=str(lims[0]), ub=str(lims[1])),
                                varname=varname)

    # If required parse dimensional layout of array
    if dims is not None:

        # Account for the special case of 1d character arrays (that
        # collapse to 0d-arrays when squeezed)
        ischar = int(np.issubdtype(arr.dtype, np.dtype("str").type))

        # Compare shape or dimension number
        if isinstance(dims, tuple):
            if len(dims) > 1:
                ashape = arr.shape
            else:
                ashape = max((ischar,), arr.squeeze().shape)
            if len(dims) != len(ashape):
                msg = "{}-dimensional array"
                raise SPYValueError(legal=msg.format(len(dims)), varname=varname,
                                    actual=msg.format(len(ashape)))
            for dk, dim in enumerate(dims):
                if dim is not None and ashape[dk] != dim:
                    raise SPYValueError("array of shape " + str(dims),
                                        varname=varname, actual="shape = " + str(arr.shape))
        else:
            ndim = max(ischar, arr.ndim)
            if ndim != dims:
                raise SPYValueError(str(dims) + "d-array", varname=varname,
                                    actual=str(ndim) + "d-array")

    return


def data_parser(data, varname="", dataclass=None, writable=None, empty=None, dimord=None):
    """
    Docstring

    writable = True/False/None
    empty=True/False (False: ensure we're working with some contents)
    """

    # Make sure `data` is (derived from) `BaseData`
    if not any(["BaseData" in str(base) for base in data.__class__.__mro__]):
        raise SPYTypeError(data, varname=varname, expected="Syncopy data object")

    # If requested, check specific data-class of object
    if dataclass is not None:
        if data.__class__.__name__ not in str(dataclass):
            msg = "Syncopy {} object".format(dataclass)
            raise SPYTypeError(data, varname=varname, expected=msg)

    # If requested, ensure object contains data (or not)
    if empty is not None:
        legal = "{status:s} Syncopy data object"
        if empty and (data.data is not None or data.samplerate is not None):
            raise SPYValueError(legal=legal.format(status="empty"),
                                varname=varname,
                                actual="non-empty")
        elif not empty and (data.data is None or data.samplerate is None):
            raise SPYValueError(legal=legal.format(status="non-empty"),
                                varname=varname,
                                actual="empty")

    # If requested, ensure proper access to object
    if writable is not None:
        legal = "{access:s} to Syncopy data object"
        actual = "mode = {mode:s}"
        if writable and data.mode == "r":
            raise SPYValueError(legal=legal.format(access="write-access"),
                                varname=varname,
                                actual=actual.format(mode=data.mode))
        elif not writable and data.mode != "r":
            raise SPYValueError(legal=legal.format(access="read-only-access"),
                                varname=varname,
                                actual=actual.format(mode=data.mode))

    # If requested, check integrity of dimensional information (if non-empty)
    if dimord is not None:
        base = "Syncopy {diminfo:s} data object"
        if data.dimord != dimord:
            legal = base.format(diminfo="'" + "' x '".join(str(dim) for dim in dimord) + "'")
            actual = base.format(diminfo="'" + "' x '".join(str(dim) for dim in data.dimord)
                                 + "' " if data.dimord else "empty")
            raise SPYValueError(legal=legal, varname=varname, actual=actual)

    return


def json_parser(json_dct, wanted_dct):
    """
    Docstring coming soon(ish)
    """

    if not set(wanted_dct.keys()).issubset(json_dct.keys()):
        legal = "mandatory fields " + "".join(key + ", " for key in wanted_dct.keys())[:-2]
        raise SPYValueError(legal=legal, varname="JSON")
    
    for key, tp in wanted_dct.items():
        if not isinstance(json_dct[key], tp):
            raise SPYTypeError(json_dct[key], varname="JSON: {}".format(key),
                               expected=tp)
    return


def get_defaults(obj):
    """
    Parse input arguments of `obj` and return dictionary

    Parameters
    ----------
    obj : function or class
        Object whose input arguments to parse. Can be either a class or
        function.

    Returns
    -------
    argdict : dictionary
        Dictionary of `argument : default value` pairs constructed from
        `obj`'s call-signature/instantiation.

    Examples
    --------
    To see the default input arguments of :meth:`syncopy.specest.mtmfft` use
    
    >>> spy.get_defaults(spy.mtmfft)
    """

    if not callable(obj):
        raise SPYTypeError(obj, varname="obj", expected="SyNCoPy function or class")
    dct = {k: v.default for k, v in signature(obj).parameters.items()\
           if v.default != v.empty and v.name != "cfg"}
    return spy.StructDict(dct)


def filename_parser(filename, is_in_valid_container=None):
    """Extract information from Syncopy file and folder names

    Parameters
    ----------
        filename: str
            Syncopy data file (\*.<dataclass>.info), Syncopy info 
            file (\*.<dataclass>) or Syncopy container folder (\*.spy)
        is_in_valid_container: bool            
            If `True`, the `filename` must be inside a folder with a .spy 
            extension. 
            If `False`, `filename` must not be inside a .spy folder.
            If `None`, the extension of the parent folder is not checked.

    Returns
    -------
        fileinfo : dict 
            Information extracted from filename and foldername with keys
            ['filename', 'container', 'folder', 'tag', 'basename', 'extension'].           

    Examples
    --------
    >>> filename_parser('/home/user/monkeyB_20190709_rfmapping_1_amua-stimon.analog')  
    {'filename': 'monkeyB_20190709_rfmapping_1_amua-stimon.analog', 
     'container': None, 
     'folder': '/home/schmiedtj_it/Projects/SyNCoPy', 
     'tag': None,
     'basename': 'monkeyB_20190709_rfmapping_1_amua-stimon', 
     'extension': '.analog'}

    >>> filename_parser('/home/user/monkeyB_20190709_rfmapping_1_amua-stimon.analog.info')
    {'filename': 'monkeyB_20190709_rfmapping_1_amua-stimon.analog',
    'container': None,
    'folder': '/home/user',
    'tag': None,
    'basename': 'monkeyB_20190709_rfmapping_1_amua-stimon',
    'extension': '.analog'}  

    >>> filename_parser('session_1.spy/session_1_amua-stimon.analog')  
    {'filename': 'session_1_amua-stimon.analog',
     'container': 'session_1.spy',
     'folder': '/home/user/session_1.spy',
     'tag': 'amua-stimon',
     'basename': 'session_1',
     'extension': '.analog'}

    >>> filename_parser('session_1.spy')
    {'filename': None,
     'container': 'session_1.spy',
     'folder': '/home/user',
     'tag': None,
     'basename': 'session_1',
     'extension': '.spy'}


    See also
    --------
    io_parser : check file and folder names for existence

    """      
    if filename is None:
        return {
        "filename": None,
        "container": None,
        "folder": None,
        "tag": None,
        "basename": None,
        "extension": None
        }

    filename = os.path.abspath(os.path.expanduser(filename))

    folder, filename = os.path.split(filename)
    container = folder.split(os.path.sep)[-1]
    basename, ext = os.path.splitext(filename)    
    
    if filename.count(".") > 2:
        raise SPYValueError(legal="single extension, found {}".format(filename.count(".")), 
                            actual=filename, varname="filename")
    if ext == spy.FILE_EXT["dir"] and basename.count(".") > 0:
        raise SPYValueError(legal="no extension, found {}".format(basename.count(".")), 
                            actual=basename, varname="container")
        
    if ext == spy.FILE_EXT["info"]:
        filename = basename
        basename, ext = os.path.splitext(filename)
    elif ext == spy.FILE_EXT["dir"]:
        return {
        "filename": None,
        "container": filename,
        "folder": folder,
        "tag": None,
        "basename": basename,
        "extension": ext
        }
    
    if ext not in spy.FILE_EXT["data"] + (spy.FILE_EXT["dir"],):
        raise SPYValueError(legal=spy.FILE_EXT["data"], 
                            actual=ext, varname="filename extension")

    folderExtIsSpy = os.path.splitext(container)[1] == spy.FILE_EXT["dir"]
    if is_in_valid_container is not None:
        if not folderExtIsSpy and is_in_valid_container:
            raise SPYValueError(legal=spy.FILE_EXT["dir"], 
                                actual=os.path.splitext(container)[1], 
                                varname="folder extension")
        elif folderExtIsSpy and not is_in_valid_container:
            raise SPYValueError(legal='not ' + spy.FILE_EXT["dir"], 
                                actual=os.path.splitext(container)[1], 
                                varname="folder extension")

    
    if folderExtIsSpy:
        containerBasename = os.path.splitext(container)[0]
        if not basename.startswith(containerBasename):
            raise SPYValueError(legal=containerBasename, 
                                actual=filename, 
                                varname='start of filename')
        tag = basename.partition(containerBasename)[-1]
        if tag == "":
            tag = None
        else:    
            if tag[0] == '_': tag = tag[1:]
        basename = containerBasename       
    else:
        container = None
        tag = None

    return {
        "filename": filename,
        "container": container,
        "folder": folder,
        "tag": tag,
        "basename": basename,
        "extension": ext
        }
    

def unwrap_cfg(func):
    """
    Decorator that unwraps cfg object in function call
    """

    @functools.wraps(func)
    def wrapper_cfg(*args, **kwargs):

        # First, parse positional arguments for dict-type inputs
        cfg = None
        k = 0
        for argidx, arg in enumerate(args):
            if isinstance(arg, dict):
                cfgidx = argidx
                k += 1

        # If a dict was found, assume it's a `cfg` dict and extract it from
        # the positional argument list; if more than one dict was found, abort
        # IMPORTANT: create a copy of `cfg` using `StructDict` constructor to
        # not manipulate `cfg` in user's namespace!
        if k == 1:
            args = list(args)
            cfg = spy.StructDict(args.pop(cfgidx))
        elif k > 1:
            raise SPYValueError(legal="single `cfg` input",
                                varname="cfg",
                                actual="{0:d} `cfg` objects in input arguments".format(k))

        # Now parse provided keywords for `cfg` entry - if `cfg` was already
        # provided as positional argument, abort
        # IMPORTANT: create a copy of `cfg` using `StructDict` constructor to
        # not manipulate `cfg` in user's namespace!
        if kwargs.get("cfg") is not None:
            if cfg:
                lgl = "`cfg` either as positional or keyword argument, not both"
                raise SPYValueError(legal=lgl, varname="cfg")
            cfg = spy.StructDict(kwargs.pop("cfg"))
            if not isinstance(cfg, dict):
                raise SPYTypeError(kwargs["cfg"], varname="cfg",
                                   expected="dictionary-like")

        # If `cfg` was detected either in positional or keyword arguments, process it
        if cfg:

            # If a method is called using `cfg`, non-default values for
            # keyword arguments *have* to be provided within the `cfg`
            defaults = get_defaults(func)
            for key, value in kwargs.items():
                if defaults[key] != value:
                    raise SPYValueError(legal="no keyword arguments",
                                        varname=key,
                                        actual="non-default value for {}".format(key))

            # Translate any existing "yes" and "no" fields to `True` and `False`,
            # respectively, and subsequently call the function with `cfg` unwrapped
            for key in cfg.keys():
                if cfg[key] == "yes":
                    cfg[key] = True
                elif cfg[key] == "no":
                    cfg[key] = False

            # If `cfg` contains keys 'data' or 'dataset' extract corresponding
            # entry and make it a positional argument (abort if both 'data'
            # and 'dataset' are present)
            data = cfg.pop("data", None)
            if cfg.get("dataset"):
                if data:
                    lgl = "either 'data' or 'dataset' field in `cfg`, not both"
                    raise SPYValueError(legal=lgl, varname="cfg")
                data = cfg.pop("dataset")
            if data:
                args = [data] + args

            # Call function with modified positional/keyword arguments
            return func(*args, **cfg)

        # No meaningful `cfg` keyword found, proceed with regular function call
        return func(*args, **kwargs)

    return wrapper_cfg
