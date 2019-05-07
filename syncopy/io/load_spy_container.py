# -*- coding: utf-8 -*-
#
# Load data from SynCoPy containers
#
# Created: 2019-02-06 11:40:56
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-04-29 11:58:29>

# Builtin/3rd party package imports
import os
import json
import inspect
import h5py
import sys
import numpy as np
from collections import OrderedDict
from glob import iglob

# Local imports
from syncopy.utils import io_parser, json_parser, data_parser, SPYTypeError, SPYValueError
from syncopy.io import hash_file, FILE_EXT
import syncopy.datatype as spd

__all__ = ["load_spy"]


def load_spy(in_name, fname=None, checksum=False, out=None, **kwargs):
    """
    Docstring coming soon...

    in case 'dir' and 'dir.spy' exist, preference will be given to 'dir.spy'

    fname can be search pattern 'session1*' or base file-name ('asdf' will
    load 'asdf.<hash>.json/.dat') or hash-id ('d4c1' will load
    'asdf.d4c1.json/.dat')
    """

    # Make sure `in_name` is a valid filesystem-location: in case 'dir' and
    # 'dir.spy' exists, preference will be given to 'dir.spy'
    # >>>>>>>>>>>>>>>>> FIXME: this doesn't work with arbitrary extensions, e.g., test.pxw!!!
    if not isinstance(in_name, str):
        raise SPYTypeError(in_name, varname="in_name", expected="str")
    _, in_ext = os.path.splitext(in_name)
    if in_ext != FILE_EXT["dir"]:
        in_spy = in_name + FILE_EXT["dir"]
    try:
        in_name = io_parser(in_spy, varname="in_name", isfile=False, exists=True)
    except:
        try:
            in_name = io_parser(in_name, varname="in_name", isfile=False, exists=True)
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
        fname = os.path.basename(os.path.abspath(os.path.expanduser(fname)))
        in_base, in_ext = os.path.splitext(fname)
        if "*" not in fname:
            fname = "*" + fname + "*"

        # If `fname` contains a dat/json extension, we expect to find
        # exactly one match, otherwise we want to see exactly two files
        in_ext = in_ext.replace("*", "")
        if in_ext in f_ext.values():
            expected_count = 1
        elif in_ext == "":
            expected_count = 2
        else:
            legal = "no extension or " + "".join(ex + ", " for ex in f_ext.values())[:-2]
            raise SPYValueError(legal=legal, varname="fname", actual=fname)

        # Specifically use `iglob` to not accidentally construct a gigantic
        # list in pathological situations (`fname = "*"`)
        in_count = 0
        for fk, fle in enumerate(iglob(os.path.join(in_name, fname))):
            in_count = fk + 1
            in_file = fle
        if in_count != expected_count:
            legal = "{exp:d} file(s), found {cnt:d}"
            raise SPYValueError(legal=legal.format(exp=expected_count, cnt=in_count),
                                varname="fname", actual=fname)

    # Construct dictionary of files to read from
    in_base = os.path.splitext(in_file)[0]
    in_files = {}
    for kind, ext in f_ext.items():
        in_files[kind] = in_base + ext

    # Load contents of json file and make sure nothing was lost in translation
    expected = {"dimord": list,
                "version": str,
                "log": str,
                "cfg": dict,
                "data": str,
                "data_dtype": str,
                "data_shape": list,
                "data_offset": int,
                "trl_dtype": str,
                "trl_shape": list,
                "trl_offset": int}
    with open(in_files["json"], "r") as fle:
        json_dict = json.load(fle)
    mandatory = set(["type"] + list(expected.keys()))
    if not mandatory.issubset(json_dict.keys()):
        legal = "mandatory fields " + "".join(attr + ", " for attr in mandatory)[:-2]
        actual = "keys " + "".join(attr + ", " for attr in json_dict.keys())[:-2]
        raise SPYValueError(legal=legal, varname=in_files["json"], actual=actual)

    # Make sure the implied data-genre makes sense
    legal_types = [dclass for dclass in spd.__all__
                   if not (inspect.isfunction(getattr(spd, dclass)))]
    if json_dict["type"] not in legal_types:
        legal = "one of " + "".join(ltype + ", " for ltype in legal_types)[:-2]
        raise SPYValueError(legal=legal, varname="JSON: type", actual=json_dict["type"])

    # Parse remaining meta-info fields
    try:
        json_parser(json_dict, expected)
    except Exception as exc:
        raise exc

    # Depending on data genre specified in file, check respective add'l fields
    # Note: `EventData` currently does not have any mandatory add'l fields
    if json_dict["type"] == "AnalogData":
        expected = {"samplerate": float,
                    "channel": list}
        try:
            json_parser(json_dict, expected)
        except Exception as exc:
            raise exc
        if set(json_dict["dimord"]) != set(["channel", "time"]):
            raise SPYValueError(legal="dimord = ['channel', 'time']",
                                varname="JSON: dimord",
                                actual=str(json_dict["dimord"]))

    elif json_dict["type"] == "SpectralData":
        expected = {"samplerate": float,
                    "channel": list,
                    "freq": list,
                    "taper": list}
        try:
            json_parser(json_dict, expected)
        except Exception as exc:
            raise exc
        if set(json_dict["dimord"]) != set(["taper", "channel", "freq", "time"]):
            raise SPYValueError(legal="dimord = ['taper', 'channel', 'freq', 'time']",
                                varname="JSON: dimord",
                                actual=str(json_dict["dimord"]))

    elif json_dict["type"] == "SpikeData":
        expected = {"channel": list,
                    "unit": list}
        try:
            json_parser(json_dict, expected)
        except Exception as exc:
            raise exc
        if set(json_dict["dimord"]) != set(["sample", "unit", "channel"]):
            raise SPYValueError(legal="dimord = ['sample', 'unit', 'channel']",
                                varname="JSON: dimord",
                                actual=str(json_dict["dimord"]))

    # If wanted, perform checksum matching
    if checksum:
        hsh_msg = "hash = {hsh:s}"
        hsh = hash_file(in_files["data"])
        if hsh != json_dict["data_checksum"]:
            raise SPYValueError(legal=hsh_msg.format(hsh=json_dict["data_checksum"]),
                                varname=os.path.basename(in_files["data"]),
                                actual=hsh_msg.format(hsh=hsh))

    # Parsing is done, create new or check provided container
    if out is not None:
        try:
            data_parser(out, varname="out", writable=True,
                        dimord=json_dict["dimord"], dataclass=json_dict["type"])
        except Exception as exc:
            raise exc
        new_out = False
    else:
        out = getattr(spd, json_dict["type"])()
        new_out = True

    # Assign meta-info common to all sub-classes
    out._log = json_dict["log"]
    out._dimlabels = OrderedDict(zip(json_dict["dimord"], [None] * len(json_dict["dimord"])))
    for key in ["cfg", "notes"]:
        if json_dict.get(key):
            setattr(out, "_{}".format(key), json_dict[key])

    # Access data on disk
    out.data = in_files["data"]

    # Abuse ``definetrial`` to set trial-related props
    trialdef = h5py.File(in_files["data"], mode="r")["trialdefinition"][()]
    out.definetrial(trialdef)

    # Sub-class-specific things follow
    if json_dict["type"] == "AnalogData":
        out.samplerate = json_dict["samplerate"]
        out.channel = np.array(json_dict["channel"])
    elif json_dict["type"] == "SpectralData":
        out.samplerate = json_dict["samplerate"]
        out.channel = np.array(json_dict["channel"])
        out.taper = np.array(json_dict["taper"])
        out.freq = np.array(json_dict["freq"])
    elif json_dict["type"] == "SpikeData":
        if json_dict.get("samplerate") is not None:
            out.samplerate = json_dict["samplerate"]
        out.channel = np.array(json_dict["channel"])
        out.unit = np.array(json_dict["unit"])
    elif json_dict["type"] == "EventData":
        if json_dict.get("samplerate") is not None:
            out.samplerate = json_dict["samplerate"]

    # Write `cfg` entries
    out.cfg = {"method": sys._getframe().f_code.co_name,
               "files": in_base + "[dat/info]"}

    # Write log-entry
    msg = "Read files v. {ver:s} {fname:s}"
    out.log = msg.format(ver=json_dict["version"], fname=in_base + "[dat/info/trl]")

    # Happy breakdown
    return out if new_out else None
