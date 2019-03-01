# load_spw_container.py - Fill object with data from disk
# 
# Created: February  6 2019
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-03-01 16:34:24>

# Builtin/3rd party package imports
import os
import json
import numpy as np
from collections import OrderedDict
from numpy.lib.format import open_memmap
from glob import iglob

# Local imports
from spykewave.utils import spy_io_parser, spy_json_parser, spy_data_parser, SPWTypeError, SPWValueError
from spykewave.io import hash_file, FILE_EXT
import spykewave.datatype as swd

__all__ = ["load_spw"]

##########################################################################################
def load_spw(in_name, fname=None, checksum=False, out=None, **kwargs):
    """
    Docstring coming soon...

    in case 'dir' and 'dir.spw' exist, preference will be given to 'dir.spw'

    fname can be search pattern 'session1*' or base file-name ('asdf' will
    load 'asdf.<hash>.json/.dat/.trl') or hash-id ('d4c1' will load 
    'asdf.d4c1.json/.dat/.trl')
    """

    # Make sure `in_name` is a valid filesystem-location: in case 'dir' and
    # 'dir.spw' exists, preference will be given to 'dir.spw'
    if not isinstance(in_name, str):
        raise SPWTypeError(in_name, varname="in_name", expected="str")
    _, in_ext = os.path.splitext(in_name)
    if in_ext != FILE_EXT["dir"]:
        in_spw = in_name + FILE_EXT["dir"]
    try:
        in_name = spy_io_parser(in_spw, varname="in_name", isfile=False, exists=True)
    except:
        try:
            in_name = spy_io_parser(in_name, varname="in_name", isfile=False, exists=True)
        except Exception as exc:
            raise exc

    # Prepare dictionary of relevant filename-extensions
    f_ext = dict(FILE_EXT)
    f_ext.pop("dir")
    
    # Either (try to) load newest fileset or look for a specific one
    if fname is None:

        # Get most recent json file in `in_name`, default to "*.json" if not found
        in_file = max(iglob(os.path.join(in_name, "*" + FILE_EXT["json"])),
                      key=os.path.getctime, default="*.json")
        
    else:

        # Remove (if any) path as well as extension from provided file-name(-pattern)
        # and convert `fname` to search pattern if it does not already conatin wildcards
        fname = os.path.basename(os.path.expanduser(fname))
        if "*" not in fname:
            fname = "*" + fname + "*"
        in_base, in_ext = os.path.splitext(fname)

        # If `fname` contains a dat/trl/json extension, we expect to find
        # exactly one match, otherwise we want to see exactly three files 
        if in_ext in f_ext.values():
            expected_count = 1
        elif in_ext == "":
            expected_count = 3
        else:
            legal = "no extension or " + "".join(ex + ", " for ex in f_ext.values())[:-2]
            raise SPWValueError(legal=legal, varname="fname", actual=fname)

        # Specifically use `iglob` to not accidentally construct a gigantic
        # list in pathological situations (`fname = "*"`)
        in_count = 0
        for fk, fle in enumerate(iglob(os.path.join(in_name, fname))):
            in_count = fk + 1
            in_file = fle
        if in_count != expected_count:
            legal = "{exp:d} file(s), found {cnt:d}"
            raise SPWValueError(legal=legal.format(exp=expected_count, cnt=in_count),
                                varname="fname", actual=fname)

    # Construct dictionary of files to read from 
    in_base  = os.path.splitext(in_file)[0]
    in_files = {}
    for kind, ext in f_ext.items():
        in_files[kind] = in_base + ext

    # Load contents of json file and make sure nothing was lost in translation
    expected = {"dimord" : list,
                "version" : str,
                "log" : str,
                "channel" : list}
    with open(in_files["json"], "r") as fle:
        json_dict = json.load(fle)
    mandatory = set(["type", "cfg"] + list(expected.keys()))
    if not mandatory.issubset(json_dict.keys()):
        legal = "mandatory fields " + "".join(attr + ", " for attr in mandatory)[:-2]
        actual = "keys " + "".join(attr + ", " for attr in json_dict.keys())[:-2]
        raise SPWValueError(legal=legal, varname=in_files["json"], actual=actual)

    # Make sure the implied data-genre makes sense
    legal_types = [attr for attr in dir(swd) \
                   if not (attr.startswith("_") \
                           or attr in ["data_classes", "Indexer", "VirtualData"])]
    if json_dict["type"] not in legal_types:
        legal = "one of " + "".join(ltype + ", " for ltype in legal_types)[:-2]
        raise SPWValueError(legal=legal, varname="JSON: type", actual=json_dict["type"])
    
    # Parse remaining meta-info fields
    try:
        spy_json_parser(json_dict, expected)
    except Exception as exc:
        raise exc

    # Depending on data genre specified in file, check respective fields
    if json_dict["type"] == "AnalogData":
        expected = {"samplerate" : float,
                    "channel" : list}
        try:
            spy_json_parser(json_dict, expected)
        except Exception as exc:
            raise exc
        if set(json_dict["dimord"]) != set(["channel", "time"]):
            raise SPWValueError(legal="dimord = ['channel', 'time']",
                                varname="JSON: dimord",
                                actual=str(json_dict["dimord"]))
        
    elif json_dict["type"] == "SpectralData":
        expected = {"samplerate" : float,
                    "channel" : list,
                    "time" : list,
                    "taper" : list}
        try:
            spy_json_parser(json_dict, expected)
        except Exception as exc:
            raise exc
        if set(json_dict["dimord"]) != set(["taper", "channel", "freq", "time"]):
            raise SPWValueError(legal="dimord = ['taper', 'channel', 'freq', 'time']",
                                varname="JSON: dimord",
                                actual=str(json_dict["dimord"]))

    # If wanted, perform checksum matching
    if checksum:
        hsh_msg = "hash = {hsh:s}"
        for fle in ["trl", "data"]:
            hsh = hash_file(in_files[fle])
            if hsh != json_dict[fle + "_checksum"]:
                raise SPWValueError(legal=hsh_msg.format(hsh=json_dict[fle + "_checksum"]),
                                    varname=os.path.basename(in_files[fle]),
                                    actual=hsh_msg.format(hsh=hsh))
    
    # Parsing is done, create new or check provided container 
    if out is not None:
        try:
            spy_data_parser(out, varname="out", writable=True,
                            dimord=json_dict["dimord"], dataclass=json_dict["type"])
        except Exception as exc:
            raise exc
        new_out = False
    else:
        out = getattr(swd, json_dict["type"])()
        new_out = True

    # Assign meta-info common to all sub-classes
    out._log = json_dict["log"]
    out._dimlabels = OrderedDict(zip(json_dict["dimord"], [None]*len(json_dict["dimord"])))
    for key in ["cfg", "notes"]:
        if json_dict.get(key):
            setattr(out, "_{}".format(key), json_dict[key])

    # Access data on disk
    out._data = open_memmap(in_files["data"], mode="r+")
    out._filename = in_files["data"]

    # Sub-class-specific things follow
    trialdef = np.load(in_files["trl"])
    if json_dict["type"] == "AnalogData":
        out._samplerate = json_dict["samplerate"]
        out._dimlabels["channel"] = json_dict["channel"]
        out._dimlabels["time"] = range(out.data.shape[out.dimord.index("time")])
        out._sampleinfo = trialdef[:,:2]
        out._trialinfo = trialdef[:,2:]
        out._hdr = None
    elif json_dict["type"] == "SpectralData":
        out._samplerate = json_dict["samplerate"]
        out._dimlabels["channel"] = json_dict["channel"]
        out._dimlabels["taper"] = json_dict["taper"]
        out._dimlabels["freq"] = np.array(json_dict["freq"])
        out._dimlabels["time"] = np.array(json_dict["time"])
        out._trialinfo = trialdef

    # Write log-entry
    msg = "Read files v. {ver:s} {fname:s}"
    out.log = msg.format(ver=json_dict["version"], fname=in_base + "[dat/info/trl]")

    # Happy breakdown
    return out if new_out else None
    
