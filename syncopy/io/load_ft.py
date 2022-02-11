# -*- coding: utf-8 -*-
#
# Load data Field Trip .mat files
#

# Builtin/3rd party package imports
import numpy as np
from scipy import io as sio
import re

# Local imports
from syncopy.shared.errors import (SPYTypeError, SPYValueError, SPYIOError, SPYInfo, 
                                   SPYError, SPYWarning)

from syncopy.datatype import AnalogData


__all__ = ["load_ft_signals"]


def load_ft_signals(filename,
                    select_structures=None,
                    add_fields=None,
                    **lm_kwargs):

    '''
    Imports time-series data from Field Trip
    into potentially multiple `~syncopy.AnalogData` objects,
    one for each structure. 

    Intended for Matlab versions < 7.3, since 7.3 Matlab
    also uses hdf5. Here another local helper should 
    be implemented

    The aim is to parse each FT data structure, which
    have the following fields (Syncopy analogon on the right):

    FT     Syncopy

    label - channel
    trial - trial
    time  - time
    fsample - samplerate
    cfg - ?
    
    The FT `cfg` contains a lot of meta data which at the
    moment we don't import into Syncopy.
    
    This is still experimental code, use with caution!!
    '''

    raw_dict = sio.loadmat(filename,
                           mat_dtype=True,
                           simplify_cells=True,
                           **lm_kwargs)
                                
    bytes_ = raw_dict['__header__']
    header = bytes_.decode()
    version = _get_Matlab_version(header)

    if version >= 7.3:
        raise NotImplementedError("Only Matlab < 7.3 is supported")
    
    struct_keys = [key for key in raw_dict.keys() if '__' not in key]

    if len(struct_keys) == 0:
        SPYValueError(legal="At least one structure",
                      varname=filename,
                      actual="No structure found"
                      )
    
    msg = f"Found {len(struct_keys)} structure(s): {struct_keys}"
    SPYInfo(msg)

    out_dict = {}
    
    # load all structures
    if select_structures is None:
        for key in struct_keys:
            
            structure = raw_dict[key]
            data = _read_mat_structure(structure, add_fields=add_fields)
            out_dict[key] = data
                    
    # load only a subset
    else:
        for key in select_structures:
            if key not in struct_keys:
                msg = f"Could not find structure `{key}` in {filename}"
                SPYWarning(msg)
                continue

            structure = raw_dict[key]

            data = _read_mat_structure(structure)
            out_dict[key] = data
            
    return out_dict


def _read_mat_structure(structure, add_fields=None):

    '''
    Local helper to parse a single FT structure
    and return an `~syncopy.AnalogData` object

    Intended for Matlab Version < 7.3

    This is the translation from FT to Syncopy:

    FT     Syncopy

    label - channel
    trial - trial
    time  - time
    fsample - samplerate
    cfg - X 

    Each trial in FT has nChannels x nSamples ordering,
    Syncopy has nSamples x nChannels
    '''
    
    # nTrials = structure["trial"].shape[0]    
    trials = []
        
    # 1st trial as reference
    nChannels, nSamples = structure["trial"][0].shape

    # check equal trial lengths
    for trl in structure["trial"]:

        if trl.shape[-1] != nSamples:
            lgl = 'Trials of equal lengths'
            actual = 'Trials of unequal lengths'
            raise SPYValueError(lgl, varname="load .mat", actual=actual)
                
        # channel x sample ordering in FT
        trials.append(trl.T.astype(np.float32))

    # initialize AnalogData    
    adata = AnalogData(trials, samplerate=structure['fsample'])
    adata.add_info = {}
    # get the channel ids
    channels = structure["label"]
    # set the channel ids 
    adata.channel =list(channels.astype(str))

    # update trialdefinition
    times_array = np.vstack(structure["time"])
    
    # nTrials x nSamples
    offsets = times_array[:, 0] * adata.samplerate

    trl_def = np.hstack([adata.sampleinfo, offsets[:, None]])
    adata.trialdefinition = trl_def

    # write additional fields(non standard FT-format)
    # into Syncopy config
    afields =  add_fields if add_fields is not None else range(0)
    for field in afields:
        adata.add_info[field] = structure[field]
    return adata
        

def _get_Matlab_version(header):

    # matches for example 'MATLAB 5.01'
    # with the version as only capture group
    
    pattern = re.compile("^MATLAB\s(\d*\.\d*)")

    match = pattern.match(header)

    if not match:
        lgl = 'Recognized .mat file'
        actual = 'not recognized .mat file'
        raise SPYValueError(lgl, 'load .mat', actual)

    version = float(match.group(1))

    return version
