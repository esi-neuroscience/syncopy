# -*- coding: utf-8 -*-
# @Author: Diljit Singh Kajal
# @Date:   2022-04-08 15:00:00
#
# load_tdt.py Merge separate TDT SEV files into one HDF5 file

import os
from datetime import datetime
import re
import numpy as np
from getpass import getuser
from tqdm.auto import tqdm
import h5py

# Local imports
from syncopy.shared.parsers import io_parser, scalar_parser
from syncopy.shared.errors import SPYWarning, SPYValueError
from syncopy.shared.tools import StructDict
import syncopy as spy


# --- The user exposed function ---


def load_tdt(data_path, start_code=None, end_code=None,
             subtract_median=False):
    """
    Imports TDT time series data and meta-information
    into a single :class:`~syncopy.AnalogData` object.

    An ad-hoc trialdefinition will be attached if
    both `start_code` and `end_code` are given.
    Otherwise a single all-to-all trialdefinition
    is used. Custom trialdefinitions can be done
    afterwards with :func:`~syncopy.definetrial`.

    All meta-information is stored within the `.info`
    dict of the  :class:`~syncopy.AnalogData` object.

    PDio related keys:
        ("PDio_onset", "PDio_offset", "PDio_data")

    Trigger related keys:
        ("Trigger_code", "Trigger_timestamp", "Trigger_sample")

    Parameters
    ----------
    data_path : str
        Path to the directory containing the `.sev` files
    start_code : int or None, optional
        Trigger code defining the beginning of a trial
    end_code : int or None, optional
        Trigger code defining the end of a trial
    subtract_median : bool
        Set  to `True` to subtract the median from all
        individual time series

    Returns
    -------
    adata : :class:`~syncopy.AnalogData`
        The tdt data in syncopy format

    Examples
    --------
    Load all channels from a source directory `/data/session3/`:

    >>> adata = load_tdt('/data/session3')

    Access the trigger codes and samples:

    >>> trg_code = adata.info['Trigger_code']
    >>> trg_sample = adata.info['Trigger_sample']

    Load the same data and construct a trialdefinition on the fly:

    >>> adata = load_tdt('/data/session3', start_code=23000, end_code=30020)

    Access the 3rd trial:

    >>> trl_dat = adata.trials[2]
    """

    io_parser(data_path, isfile=False)

    if start_code is not None and end_code is None:
        lgl = "trigger codes for both trial start and end"
        raise SPYValueError(lgl, "end_code", end_code)
    if end_code is not None and start_code is None:
        lgl = "trigger codes for both trial start and end"
        raise SPYValueError(lgl, "start_code", start_code)
    elif end_code is None and start_code is None:
        pass
    # both are given
    else:
        scalar_parser(start_code, "start_code", ntype='int_like')
        scalar_parser(end_code, "end_code", ntype='int_like')

    # initialize tdt info loader class
    TDT_Load_Info = ESI_TDTinfo(data_path)
    # this is a StructDict
    tdt_info = TDT_Load_Info.load_tdt_info()

    # nicely sorted by channel names
    file_paths = _get_source_paths(data_path, ".sev")

    tdt_data_handler = ESI_TDTdata(
        data_path, subtract_median=subtract_median, channels=None
    )

    adata = tdt_data_handler.data_aranging(file_paths, tdt_info)
    # we have to open for reading again
    adata.data = h5py.File(adata.filename, "r")['data']

    # Write log-entry
    msg = f"loaded TDT data from {len(file_paths)} files\n"
    msg += f"\tsource folder: {data_path}\n"
    msg += f"\tsubtract median: {subtract_median}"
    adata.log = msg

    if start_code is not None:
        adata.trialdefinition = _mk_trialdef(adata, start_code, end_code)
        msg = f"created trialdefinition from start code `{start_code}` and end code `{end_code}`"
        adata.log = msg

    return adata


class ESI_TDTinfo:
    def __init__(self, block_path):
        self.block_path = block_path
        self.nodata = False
        self.t1 = 0
        self.t2 = 0
        self.UNKNOWN = int("00000000", 16)
        self.STRON = int("00000101", 16)
        self.STROFF = int("00000102", 16)
        self.SCALAR = int("00000201", 16)
        self.STREAM = int("00008101", 16)
        self.SNIP = int("00008201", 16)
        self.MARK = int("00008801", 16)
        self.HASDATA = int("00008000", 16)
        self.UCF = int("00000010", 16)
        self.PHANTOM = int("00000020", 16)
        self.MASK = int("0000FF0F", 16)
        self.INVALID_MASK = int("FFFF0000", 16)
        self.STARTBLOCK = int("0001", 16)
        self.STOPBLOCK = int("0002", 16)
        self.ALLOWED_FORMATS = [np.float32, np.int32, np.int16, np.int8, np.float64, np.int64]
        self.ALLOWED_EVTYPES = ["all", "epocs", "snips", "streams", "scalars"]

    def code_to_type(self, code):
        # given event code, return string 'epocs', 'snips', 'streams', or 'scalars'
        strobe_types = [self.STRON, self.STROFF, self.MARK]
        scalar_types = [self.SCALAR]
        snip_types = [self.SNIP]

        if code in strobe_types:
            s = "epocs"
        elif code in snip_types:
            s = "snips"
        elif code & self.MASK == self.STREAM:
            s = "streams"
        elif code in scalar_types:
            s = "scalars"
        else:
            s = "unknown"
        return s

    def time2sample(self, ts, fs=195312.5, t1=False, t2=False, to_time=False):
        sample = ts * fs
        if t2:
            # drop precision beyond 1e-9
            exact = np.round(sample * 1e9) / 1e9
            sample = np.floor(sample)
            if exact == sample:
                sample -= 1
        else:
            sample = np.ceil(sample) if t1 else np.round(sample)
        sample = np.uint64(sample)
        if to_time:
            return np.float64(sample) / fs
        return sample

    def check_ucf(self, code):
        # given event code, check if it has unique channel files
        return code & self.UCF == self.UCF

    def epoc_to_type(self, code):
        # given epoc event code, return if it is 'onset' or 'offset' event
        strobe_on_types = [self.STRON, self.MARK]
        strobe_off_types = [self.STROFF]
        if code in strobe_on_types:
            return "onset"
        elif code in strobe_off_types:
            return "offset"
        return "unknown"

    def code_to_name(self, code):
        return int(code).to_bytes(4, byteorder="little").decode("cp437")

    def load_tdt_info(self):
        header = StructDict()
        data = StructDict()
        data.epocs = StructDict()
        data.streams = StructDict()
        data.scalars = StructDict()
        data.info = StructDict()

        epocs = StructDict()
        epocs.name = []
        epocs.buddies = []
        epocs.ts = []
        epocs.code = []
        epocs.type = []
        epocs.type_str = []
        epocs.data = []
        epocs.dform = []

        tsq_list = _get_source_paths(self.block_path, ".tsq")
        if len(tsq_list) > 1:
            raise Exception("multiple TSQ files found\n{0}".format(", ".join(tsq_list)))
        tsq = open(tsq_list[0], "rb")
        tsq.seek(0, os.SEEK_SET)
        tsq.seek(48, os.SEEK_SET)
        code1 = np.fromfile(tsq, dtype=np.int32, count=1)
        assert code1 == self.STARTBLOCK, "Block start marker not found"
        tsq.seek(56, os.SEEK_SET)
        header.start_time = np.fromfile(tsq, dtype=np.float64, count=1)

        # read stop time
        tsq.seek(-32, os.SEEK_END)
        code2 = np.fromfile(tsq, dtype=np.int32, count=1)
        if code2 != self.STOPBLOCK:
            SPYWarning(
                "Block end marker not found, block did not end cleanly. Try setting T2 smaller if errors occur"
            )
            header.stop_time = np.nan
        else:
            tsq.seek(-24, os.SEEK_END)
            header.stop_time = np.fromfile(tsq, dtype=np.float64, count=1)

        [data.info.tankpath, data.info.blockname] = os.path.split(os.path.normpath(self.block_path))
        data.info.start_date = datetime.fromtimestamp(header.start_time[0])
        if not np.isnan(header.start_time):
            data.info.utc_start_time = data.info.start_date.strftime("%H:%M:%S")
        else:
            data.info.utc_start_time = np.nan

        if not np.isnan(header.stop_time):
            data.info.stop_date = datetime.fromtimestamp(header.stop_time[0])
            data.info.utc_stop_time = data.info.stop_date.strftime("%H:%M:%S")
        else:
            data.info.stop_date = np.nan
            data.info.utc_stop_time = np.nan

        if header.stop_time > 0:
            data.info.duration = data.info.stop_date - data.info.start_date  # datestr(s2-s1, 'HH:MM:SS')

        tsq.seek(40, os.SEEK_SET)

        read_size = 10000000 if self.t2 > 0 else 50000000
        header.stores = StructDict()
        while True:
            heads = np.frombuffer(tsq.read(read_size * 4), dtype=np.uint32)
            rem = len(heads) % 10
            if rem != 0:
                SPYWarning("Block did not end cleanly, removing last {0} headers".format(rem))
                heads = heads[:-rem]

            # reshape so each column is one header
            heads = heads.reshape((-1, 10)).T

            # check the codes first and build store maps and note arrays
            codes = heads[2, :]

            good_codes = codes > 0
            bad_codes = np.logical_not(good_codes)

            if np.sum(bad_codes) > 0:
                SPYWarning(
                    "Bad TSQ headers were written, removing {0}, keeping {1} headers".format(
                        sum(bad_codes), sum(good_codes)
                    )
                )
                heads = heads[:, good_codes]
                codes = heads[2, :]
            # get set of codes but preserve order in the block
            store_codes = []
            unique_codes, unique_ind = np.unique(codes, return_index=True)
            for counter, x in enumerate(unique_codes):
                store_codes.append(
                    {
                        "code": x,
                        "type": heads[1, unique_ind[counter]],
                        "type_str": self.code_to_type(heads[1, unique_ind[counter]]),
                        "ucf": self.check_ucf(heads[1, unique_ind[counter]]),
                        "epoc_type": self.epoc_to_type(heads[1, unique_ind[counter]]),
                        "dform": heads[8, unique_ind[counter]],
                        "size": heads[0, unique_ind[counter]],
                        "buddy": heads[3, unique_ind[counter]],
                        "temp": heads[:, unique_ind[counter]],
                    }
                )

            # Looking for only Mark, PDi\ and PDio
            looking_for = ["Mark", "PDio", "LFPs", "LFP1", "PDi\\"]  #
            targets = StructDict()
            for chk, content in enumerate(store_codes):
                if self.code_to_name(content["code"]) in looking_for:
                    targets[self.code_to_name(content["code"])] = chk

            for tar in targets.items():
                store_code = store_codes[tar[1]]
                store_code["name"] = self.code_to_name(store_code["code"])
                # print(code_to_name(store_code['code']), store_code['code'])

                store_code["var_name"] = store_code["name"]
                var_name = store_code["var_name"]

                if store_code["type_str"] == "epocs":
                    if not store_code["name"] in epocs.name:
                        buddy = "".join([str(chr(c)) for c in np.array([store_code["buddy"]]).view(np.uint8)])
                        buddy = buddy.replace("\x00", " ")

                        # if skip_by_name:
                        #     if store:
                        #         if isinstance(store, str):
                        #             if buddy == store:
                        #                 skip_by_name = False
                        #         elif isinstance(store, list):
                        #             if buddy in store:
                        #                 skip_by_name = False
                        # if skip_by_name:
                        #     continue
                        epocs.name.append(store_code["name"])
                        epocs.buddies.append(buddy)
                        epocs.code.append(store_code["code"])
                        epocs.ts.append([])
                        epocs.type.append(store_code["epoc_type"])
                        epocs.type_str.append(store_code["type_str"])
                        epocs.data.append([])
                        epocs.dform.append(store_code["dform"])

                if var_name not in header.stores.keys():
                    if store_code["type_str"] != "epocs":
                        header.stores[var_name] = StructDict(
                            name=store_code["name"],
                            code=store_code["code"],
                            size=store_code["size"],
                            type=store_code["type"],
                            type_str=store_code["type_str"],
                        )
                        if header.stores[var_name].type_str == "streams":
                            header.stores[var_name].ucf = store_code["ucf"]
                        if header.stores[var_name].type_str != "scalars":
                            # Finding the sampling rate
                            header.stores[var_name].fs = np.double(
                                np.array([store_code["temp"][9]]).view(np.float32)
                            )
                        header.stores[var_name].dform = store_code["dform"]
                valid_ind = np.where(codes == store_code["code"])[0]
                temp = heads[3, valid_ind].view(np.uint16)

                if store_code["type_str"] != "epocs":
                    if not hasattr(header.stores[var_name], "ts"):
                        header.stores[var_name].ts = []
                    vvv = (
                        np.reshape(
                            heads[[[4], [5]], valid_ind].T,
                            (-1, 1)).T.view(np.float64) - header.start_time
                    )
                    # round timestamps to the nearest sample
                    vvv = self.time2sample(vvv, to_time=True)
                    header.stores[var_name].ts.append(vvv)
                    if (not self.nodata) or (store_code["type_str"] == "streams"):
                        if not hasattr(header.stores[var_name], "data"):
                            header.stores[var_name].data = []
                        header.stores[var_name].data.append(
                            np.reshape(heads[[[6], [7]], valid_ind].T, (-1, 1)).T.view(np.float64)
                        )
                    if not hasattr(header.stores[var_name], "chan"):
                        header.stores[var_name].chan = []
                    header.stores[var_name].chan.append(temp[::2])
                else:
                    loc = epocs.name.index(store_code["name"])
                    # round timestamps to the nearest sample
                    vvv = (
                        np.reshape(
                            heads[[[4], [5]], valid_ind].T,
                            (-1, 1)).T.view(np.float64) - header.start_time
                    )
                    # round timestamps to the nearest sample
                    vvv = self.time2sample(vvv, to_time=True)
                    epocs.ts[loc] = np.append(epocs.ts[loc], vvv)
                    epocs.data[loc] = np.append(
                        epocs.data[loc], np.reshape(heads[[[6], [7]], valid_ind].T, (-1, 1)).T.view(np.float64)
                    )
            last_ts = heads[[4, 5], -1].view(np.float64) - header.start_time
            last_ts = last_ts[0]
            if self.t2 > 0 and last_ts > self.t2:
                break
            # eof reached
            if heads.size < read_size:
                break
        print(
            "Reading data from t = {0}s to t = {1}s".format(
                np.round(self.t1, 2), np.round(np.maximum(last_ts, self.t2), 2)
            )
        )

        for ii in range(len(epocs.name)):
            # find all non-buddies first
            if epocs.type[ii] == "onset":
                var_name = epocs.name[ii]
                header.stores[var_name] = StructDict()
                header.stores[var_name].name = epocs.name[ii]
                ts = epocs.ts[ii]
                header.stores[var_name].onset = ts
                header.stores[var_name].offset = np.append(ts[1:], np.inf)
                header.stores[var_name].type = epocs.type[ii]
                header.stores[var_name].type_str = epocs.type_str[ii]
                header.stores[var_name].data = epocs.data[ii]
                header.stores[var_name].dform = epocs.dform[ii]
                header.stores[var_name].size = 10

        for ii in range(len(epocs.name)):
            if epocs.type[ii] == "offset":
                var_name = epocs.buddies[ii]
                if var_name not in header.stores.keys():
                    SPYWarning(epocs.buddies[ii] + " buddy epoc not found, skipping")
                    continue
                header.stores[var_name].offset = epocs.ts[ii]
                # handle odd case where there is a single offset event and no onset events
                if "onset" not in header.stores[var_name].keys():
                    header.stores[var_name].name = epocs.buddies[ii]
                    header.stores[var_name].onset = 0
                    header.stores[var_name].type_str = "epocs"
                    header.stores[var_name].type = "onset"
                    header.stores[var_name].data = 0
                    header.stores[var_name].dform = 4
                    header.stores[var_name].size = 10
                # fix time ranges
                if header.stores[var_name].offset[0] < header.stores[var_name].onset[0]:
                    header.stores[var_name].onset = np.append(0, header.stores[var_name].onset)
                    header.stores[var_name].data = np.append(
                        header.stores[var_name].data[0], header.stores[var_name].data
                    )
                if header.stores[var_name].onset[-1] > header.stores[var_name].offset[-1]:
                    header.stores[var_name].offset = np.append(header.stores[var_name].offset, np.inf)

        for var_name in header.stores.keys():
            # convert cell arrays to regular arrays
            if "ts" in header.stores[var_name].keys():
                header.stores[var_name].ts = np.concatenate(header.stores[var_name].ts, axis=1)[0]
            if "chan" in header.stores[var_name].keys():
                header.stores[var_name].chan = np.concatenate(header.stores[var_name].chan)
            if "sortcode" in header.stores[var_name].keys():
                header.stores[var_name].sortcode = np.concatenate(header.stores[var_name].sortcode)
            if "data" in header.stores[var_name].keys():
                if header.stores[var_name].type_str != "epocs":
                    header.stores[var_name].data = np.concatenate(header.stores[var_name].data, axis=1)[0]

            # if it's a data type, cast as a file offset pointer instead of data
            if header.stores[var_name].type_str in ["streams", "snips"]:
                if "data" in header.stores[var_name].keys():
                    header.stores[var_name].data = header.stores[var_name].data.view(np.uint64)
            if "chan" in header.stores[var_name].keys():
                if np.max(header.stores[var_name].chan) == 1:
                    header.stores[var_name].chan = [1]

        valid_time_range = np.array([[self.t1], [self.t2]]) if self.t2 > 0 else np.array([[self.t1], [np.inf]])
        ranges = None
        if hasattr(ranges, "__len__"):
            valid_time_range = ranges

        num_ranges = valid_time_range.shape[1]
        if num_ranges > 0:
            data.time_ranges = valid_time_range

        for var_name in header.stores.keys():
            current_type_str = header.stores[var_name].type_str
            data[current_type_str][var_name] = header.stores[var_name]
            firstStart = valid_time_range[0, 0]
            last_stop = valid_time_range[1, -1]
            if "ts" in header.stores[var_name].keys():
                if current_type_str == "streams":
                    data[current_type_str][var_name].start_time = [0 for jj in range(num_ranges)]
                else:
                    this_dtype = data[current_type_str][var_name].ts.dtype
                    data[current_type_str][var_name].filtered_ts = [
                        np.array([], dtype=this_dtype) for jj in range(num_ranges)
                    ]
                if hasattr(data[current_type_str][var_name], "chan"):
                    data[current_type_str][var_name].filtered_chan = [[] for jj in range(num_ranges)]
                if hasattr(data[current_type_str][var_name], "sortcode"):
                    this_dtype = data[current_type_str][var_name].sortcode.dtype
                    data[current_type_str][var_name].filtered_sort_code = [
                        np.array([], dtype=this_dtype) for jj in range(num_ranges)
                    ]
                if hasattr(data[current_type_str][var_name], "data"):
                    this_dtype = data[current_type_str][var_name].data.dtype
                    data[current_type_str][var_name].filtered_data = [
                        np.array([], dtype=this_dtype) for jj in range(num_ranges)
                    ]

                filter_ind = [[] for i in range(num_ranges)]
                for jj in range(num_ranges):
                    start = valid_time_range[0, jj]
                    stop = valid_time_range[1, jj]
                    ind1 = data[current_type_str][var_name].ts >= start
                    ind2 = data[current_type_str][var_name].ts < stop
                    filter_ind[jj] = np.where(ind1 & ind2)[0]
                    bSkip = 0
                    if len(filter_ind[jj]) == 0:
                        # if it's a stream and a short window, we might have missed it
                        if current_type_str == "streams":
                            ind2 = np.where(ind2)[0]
                            if len(ind2) > 0:
                                ind2 = ind2[-1]
                                # keep one prior for streams (for all channels)
                                nchan = max(data[current_type_str][var_name].chan)
                                if ind2 - nchan >= -1:
                                    filter_ind[jj] = ind2 - np.arange(nchan - 1, -1, -1)
                                    temp = data[current_type_str][var_name].ts[filter_ind[jj]]
                                    data[current_type_str][var_name].start_time[jj] = temp[0]
                                    bSkip = 1

                    if len(filter_ind[jj]) > 0:
                        # parse out the information we need
                        if current_type_str == "streams":
                            # keep one prior for streams (for all channels)
                            if not bSkip:
                                nchan = max(data[current_type_str][var_name].chan)
                                temp = filter_ind[jj]
                                if temp[0] - nchan > -1:
                                    filter_ind[jj] = np.concatenate(
                                        [-np.arange(nchan, 0, -1) + temp[0], filter_ind[jj]]
                                    )
                                temp = data[current_type_str][var_name].ts[filter_ind[jj]]
                                data[current_type_str][var_name].start_time[jj] = temp[0]
                        else:
                            data[current_type_str][var_name].filtered_ts[jj] = data[current_type_str][
                                var_name
                            ].ts[filter_ind[jj]]
                        if hasattr(data[current_type_str][var_name], "chan"):
                            if len(data[current_type_str][var_name].chan) > 1:
                                data[current_type_str][var_name].filtered_chan[jj] = data[current_type_str][
                                    var_name
                                ].chan[filter_ind[jj]]
                            else:
                                data[current_type_str][var_name].filtered_chan[jj] = data[current_type_str][
                                    var_name
                                ].chan
                        if hasattr(data[current_type_str][var_name], "sortcode"):
                            data[current_type_str][var_name].filtered_sort_code[jj] = data[current_type_str][
                                var_name
                            ].sortcode[filter_ind[jj]]
                        if hasattr(data[current_type_str][var_name], "data"):
                            data[current_type_str][var_name].filtered_data[jj] = data[current_type_str][
                                var_name
                            ].data[filter_ind[jj]]
                if current_type_str == "streams":
                    delattr(data[current_type_str][var_name], "ts")
                    delattr(data[current_type_str][var_name], "data")
                    delattr(data[current_type_str][var_name], "chan")
                    if not hasattr(data[current_type_str][var_name], "filtered_chan"):
                        data[current_type_str][var_name].filtered_chan = [[] for i in range(num_ranges)]
                    if not hasattr(data[current_type_str][var_name], "filtered_data"):
                        data[current_type_str][var_name].filtered_data = [[] for i in range(num_ranges)]
                    if not hasattr(data[current_type_str][var_name], "start_time"):
                        data[current_type_str][var_name].start_time = -1
                else:
                    # consolidate other fields
                    if hasattr(data[current_type_str][var_name], "filtered_ts"):
                        data[current_type_str][var_name].ts = np.concatenate(
                            data[current_type_str][var_name].filtered_ts
                        )
                        delattr(data[current_type_str][var_name], "filtered_ts")
                    else:
                        data[current_type_str][var_name].ts = []
                    if hasattr(data[current_type_str][var_name], "chan"):
                        if hasattr(data[current_type_str][var_name], "filtered_chan"):
                            data[current_type_str][var_name].chan = np.concatenate(
                                data[current_type_str][var_name].filtered_chan
                            )
                            delattr(data[current_type_str][var_name], "filtered_chan")
                            if current_type_str == "snips":
                                if len(set(data[current_type_str][var_name].chan)) == 1:
                                    data[current_type_str][var_name].chan = [
                                        data[current_type_str][var_name].chan[0]
                                    ]

                        else:
                            data[current_type_str][var_name].chan = []
                    if hasattr(data[current_type_str][var_name], "sortcode"):
                        if hasattr(data[current_type_str][var_name], "filtered_sort_code"):
                            data[current_type_str][var_name].sortcode = np.concatenate(
                                data[current_type_str][var_name].filtered_sort_code
                            )
                            delattr(data[current_type_str][var_name], "filtered_sort_code")
                        else:
                            data[current_type_str][var_name].sortcode = []
                    if hasattr(data[current_type_str][var_name], "data"):
                        if hasattr(data[current_type_str][var_name], "filtered_data"):
                            data[current_type_str][var_name].data = np.concatenate(
                                data[current_type_str][var_name].filtered_data
                            )
                            delattr(data[current_type_str][var_name], "filtered_data")
                        else:
                            data[current_type_str][var_name].data = []
            else:
                # handle epoc events
                filter_ind = []
                for jj in range(num_ranges):
                    start = valid_time_range[0, jj]
                    stop = valid_time_range[1, jj]
                    ind1 = data[current_type_str][var_name].onset >= start
                    ind2 = data[current_type_str][var_name].onset < stop
                    filter_ind.append(np.where(ind1 & ind2)[0])
                filter_ind = np.concatenate(filter_ind)
                if len(filter_ind) > 0:
                    data[current_type_str][var_name].onset = data[current_type_str][var_name].onset[filter_ind]
                    data[current_type_str][var_name].data = data[current_type_str][var_name].data[filter_ind]
                    data[current_type_str][var_name].offset = data[current_type_str][var_name].offset[
                        filter_ind
                    ]

                    if data[current_type_str][var_name].offset[0] < data[current_type_str][var_name].onset[0]:
                        if data[current_type_str][var_name].onset[0] > firstStart:
                            data[current_type_str][var_name].onset = np.concatenate(
                                [[firstStart], data[current_type_str][var_name].onset]
                            )
                    if data[current_type_str][var_name].offset[-1] > last_stop:
                        data[current_type_str][var_name].offset[-1] = last_stop
                else:
                    # default case is no valid events for this store
                    data[current_type_str][var_name].onset = []
                    data[current_type_str][var_name].data = []
                    data[current_type_str][var_name].offset = []
                    if var_name == "Note":
                        data[current_type_str][var_name].notes = []

        for current_name in header.stores.keys():
            current_type_str = header.stores[current_name].type_str
            # if current_type_str not in evtype:
            #     continue

            current_type_str = data[current_type_str][current_name].type_str
            current_data_format = self.ALLOWED_FORMATS[data[current_type_str][current_name].dform]
            if current_type_str == "scalars":
                if len(data[current_type_str][current_name].chan) > 0:
                    nchan = int(np.max(data[current_type_str][current_name].chan))
                else:
                    nchan = 0
                if nchan > 1:
                    # organize data by sample
                    # find channels with most and least amount of data
                    ind = []
                    min_length = np.inf
                    max_length = 0
                    for xx in range(nchan):
                        ind.append(np.where(data[current_type_str][current_name].chan == xx + 1)[0])
                        min_length = min(len(ind[-1]), min_length)
                        max_length = max(len(ind[-1]), max_length)
                    if min_length != max_length:
                        SPYWarning(
                            "Truncating store {0} to {1} values (from {2})".format(
                                current_name, min_length, max_length
                            ))
                        ind = [ind[xx][:min_length] for xx in range(nchan)]
                    if not self.nodata:
                        data[current_type_str][current_name].data = (
                            data[current_type_str][current_name].data[np.concatenate(ind)].reshape(nchan, -1)
                        )
                    # only use timestamps from first channel
                    data[current_type_str][current_name].ts = data[current_type_str][current_name].ts[ind[0]]
                    # remove channels field
                    delattr(data[current_type_str][current_name], "chan")
        tsq.close()
        del epocs
        del header
        Data = StructDict()
        Data.PDio = data.epocs.PDio
        try:
            Data.LFPs = data.streams.LFPs
        except Exception:
            Data.LFPs = data.streams.LFP1
        Data.Mark = data.scalars.Mark
        Data.info = data.info
        return Data


class ESI_TDTdata:
    def __init__(
        self,
        inputdir,
        subtract_median=False,
        channels=None,
    ):

        self.inputdir = inputdir
        self.chan_in_chunks = 16
        self.subtract_median = subtract_median
        self.channels = "all" if channels is None else channels

    def arrange_header(self, DataInfo_loaded, Files):
        header = StructDict()
        header["fs"] = DataInfo_loaded.LFPs.fs
        header["total_num_channel"] = len(Files)
        return header

    def read_data(self, filename):
        HEADERSIZE = 40
        """Read data from a TDT SEV file created by the RS4 streamer"""
        with open(filename, "rb") as f:
            f.seek(HEADERSIZE)
            data = np.fromfile(f, dtype="single")
        return data

    def md5sum(self, filename):
        from hashlib import md5

        hash = md5()
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(128 * hash.block_size), b""):
                hash.update(chunk)
        return hash.hexdigest()

    def data_aranging(self, Files, DataInfo_loaded):
        AData = spy.AnalogData(dimord=['time', 'channel'])
        hdf_out_path = AData.filename
        LenOfData = self.read_data(Files[0]).shape[0]  # Lenght of the data is always set to the length of the first channel
        with h5py.File(hdf_out_path, "w") as combined_data_file:
            idxStartStop = [
                np.clip(np.array((jj, jj + self.chan_in_chunks)), a_min=None, a_max=len(Files))
                for jj in range(0, len(Files), self.chan_in_chunks)
            ]
            print(
                "Merging {0} files in {1} chunks each with {2} channels into \n   {3}".format(
                    len(Files), len(idxStartStop), self.chan_in_chunks, hdf_out_path
                )
            )
            for (start, stop) in tqdm(iterable=idxStartStop, desc="chunk", unit="chunk", disable=None):
                data = [self.read_data(Files[jj])[:LenOfData] for jj in range(start, stop)]
                data = np.vstack(data).T
                if start == 0:
                    # this is the actual dataset for the AnalogData
                    target = combined_data_file.create_dataset(
                        "data", shape=(data.shape[0], len(Files)), dtype="single"
                    )
                if self.subtract_median:
                    data -= np.median(data, keepdims=True).astype(data.dtype)
                target[:, start:stop] = data

            # link dataset to AnalogData instance
            AData.data = target
            # temporary fix to get at least all-to-all
            AData.trialdefinition = None

        chanlist = (
            None if self.channels == "all" else ["channel" + str(trch + 1).zfill(3) for trch in self.channels]
        )
        AData.samplerate = DataInfo_loaded.LFPs.fs
        AData.channel = chanlist

        # helper to make serializable
        def serial(arr):
            return arr.tolist()

        # write info file
        AData.info["originalFiles"] = (Files,)
        # AData.info["md5sum"] = self.md5sum(hdf_out_path)
        AData.info["blockname"] = DataInfo_loaded.info.blockname
        AData.info["start_date"] = str(DataInfo_loaded.info.start_date)
        AData.info["utc_start_time"] = DataInfo_loaded.info.utc_start_time
        AData.info["stop_date"] = str(DataInfo_loaded.info.stop_date)
        AData.info["utc_stop_time"] = DataInfo_loaded.info.utc_stop_time
        AData.info["duration"] = str(DataInfo_loaded.info.duration)
        AData.info["PDio_onset"] = serial(DataInfo_loaded.PDio.onset)
        AData.info["PDio_offset"] = serial(DataInfo_loaded.PDio.offset)
        AData.info["PDio_data"] = serial(DataInfo_loaded.PDio.data)
        AData.info["Trigger_timestamp"] = serial(DataInfo_loaded.Mark.ts)
        AData.info["Trigger_sample"] = serial(np.round(DataInfo_loaded.Mark.ts * DataInfo_loaded.LFPs.fs))
        AData.info["Trigger_code"] = serial(DataInfo_loaded.Mark.data[0])

        return AData


# --- Helpers ---

def _mk_trialdef(adata, start_code, end_code):
    """
    Create a basic trialdefinition from the trial
    start and end trigger codes
    """

    # trigger codes and samples
    trg_codes = np.array(adata.info['Trigger_code'], dtype=int)
    trg_sample = np.array(adata.info['Trigger_sample'], dtype=int)

    # boolean indexing
    trl_starts = trg_sample[trg_codes == start_code]
    trl_ends = trg_sample[trg_codes == end_code]

    if trl_starts.size == 0:
        lgl = "at least one occurence of trial start code"
        raise SPYValueError(lgl, "start_code", start_code)

    if trl_ends.size == 0:
        lgl = "at least one occurence of trial end code"
        raise SPYValueError(lgl, "end_code", end_code)

    if trl_starts.size > trl_ends.size:
        msg = f"Found {trl_starts.size} trial starts and {trl_ends.size} trial end codes!\n"
        msg += "truncating to number of trial ends.."
        SPYWarning(msg)
        N = trl_ends.size
    elif trl_ends.size > trl_starts.size:
        msg = f"Found {trl_starts.size} trial starts and {trl_ends.size} trial end codes!\n"
        msg += "truncating to number of trial starts.."
        SPYWarning(msg)
        N = trl_starts.size
    # both are equal
    else:
        N = trl_starts.size

    trldef = np.zeros((N, 3))
    trldef[:, 0] = trl_starts[:N]
    trldef[:, 1] = trl_ends[:N]

    return trldef


def _get_source_paths(directory, ext=".sev"):
    """
    Returns all abs. paths in `directory`
    for files which end with `ext`
    """

    # get all data source file names
    f_names = [f for f in os.listdir(directory) if f.endswith(ext)]
    # parse absolute paths to source files
    tdtPaths = []
    for fname in f_names:
        fname = os.path.join(directory, fname)
        f_path, f_name = io_parser(fname, varname="tdt source file name", isfile=True, exists=True)
        tdtPaths.append(os.path.join(f_path, f_name))
    tdtPaths = _natural_sort(tdtPaths)
    return tdtPaths


def _natural_sort(file_names):
    """Sort a list of strings using numbers
        Ch1 will be followed by Ch2 and not Ch11.
    """

    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(file_names, key=alphanum_key)
