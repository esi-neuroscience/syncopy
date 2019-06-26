# -*- coding: utf-8 -*-
#
# Helper methods for testing routines
# 
# Created: 2019-04-18 14:41:32
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-06-26 11:34:06>

import subprocess
import sys
import h5py
import numpy as np

# Local imports
import syncopy as spy


def is_win_vm():
    """
    Returns `True` if code is running on virtual Windows machine, `False`
    otherwise
    """

    # If we're not running on Windows abort
    if sys.platform != "win32":
        return False

    # Use the windows management instrumentation command-line to extract machine manufacturer
    out, err = subprocess.Popen("wmic computersystem get manufacturer",
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, shell=True).communicate()

    # If the vendor name contains any "virtual"-flavor, we're probably running
    # in a VM - if the above command triggered an error, abort
    if len(err) == 0:
        vendor = out.split()[1].lower()
        vmlist = ["vmware", "virtual", "virtualbox", "vbox", "qemu"]
        return any([virtual in vendor for virtual in vmlist])
    else:
        return False


def is_slurm_node():
    """
    Returns `True` if code is running on a SLURM-managed cluster node, `False`
    otherwise
    """

    # Simply test if the srun command is available
    out, err = subprocess.Popen("srun --version",
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, shell=True).communicate()
    if len(out) > 0:
        return True
    else:
        return False

    
def generate_artifical_data(nTrials=2, nChannels=2, equidistant=True,
                            overlapping=False, inmemory=True, dimord="default"):
    """
    Populate `AnalogData` object w/ artificial signal
    """

    # Create dummy 1d signal that will be blown up to fill channels later
    dt = 0.001
    t = np.arange(0, 3, dt, dtype="float32") - 1.0
    sig = np.cos(2 * np.pi * (7 * (np.heaviside(t, 1) * t - 1) + 10) * t)

    # Depending on chosen `dimord` either get default position of time-axis
    # in `AnalogData` objects or use provided `dimord` and reshape signal accordingly
    if dimord == "default":
        dimord = spy.AnalogData().dimord
    timeAxis = dimord.index("time")
    idx = [1, 1]
    idx[timeAxis] = -1
    sig = np.repeat(sig.reshape(*idx), axis=idx.index(1), repeats=nChannels)

    # Either create the full data array in memory using array tiling or
    # create an HDF5 container in `__storage__` and fill it trial-by-trial
    out = spy.AnalogData(samplerate=1/dt, dimord=dimord)
    if inmemory:
        idx[timeAxis] = nTrials 
        sig = np.tile(sig, idx)
        sig += np.random.standard_normal(sig.shape).astype(sig.dtype) * 0.5
        out.data = sig
    else:
        with h5py.File(out._filename, "w") as h5f:
            shp = list(sig.shape)
            shp[timeAxis] *= nTrials
            dset = h5f.create_dataset("AnalogData", shape=tuple(shp), dtype=sig.dtype)
            shp = [slice(None), slice(None)]
            for iTrial in range(nTrials):
                shp[timeAxis] = slice(iTrial*t.size, (iTrial + 1)*t.size)
                dset[tuple(shp)] = sig + np.random.standard_normal(sig.shape).astype(sig.dtype) * 0.5
                dset.flush()
        out.data = h5py.File(out._filename, "r+")["AnalogData"]

    # Define by-trial offsets to generate (non-)equidistant/(non-)overlapping trials
    trialdefinition = np.zeros((nTrials, 3), dtype='int')
    if equidistant:
        if not overlapping:
            offsets = np.zeros((nTrials,), dtype=sig.dtype)
        else:
            offsets = np.full((nTrials,), 100, dtype=sig.dtype)
    else:
        offsets = np.random.randint(low=int(0.1*t.size),
                                    high=int(0.2*t.size), size=(nTrials,))

    # Using generated offsets, construct trialdef array and make sure initial
    # and end-samples are within data bounds (only relevant if overlapping
    # trials are built)
    shift = (-1)**(not overlapping)
    for iTrial in range(nTrials):
        trialdefinition[iTrial, :] = np.array([iTrial*t.size - shift*offsets[iTrial],
                                               (iTrial + 1)*t.size + shift*offsets[iTrial],
                                               1000])
    trialdefinition[0, 0] = 0
    trialdefinition[-1, 1] = nTrials*t.size

    out.definetrial(trialdefinition)

    return out
