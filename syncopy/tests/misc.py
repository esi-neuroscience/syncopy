# -*- coding: utf-8 -*-
#
# Helper methods for testing routines
# 
# Created: 2019-04-18 14:41:32
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-05-17 11:06:57>

import subprocess
import sys
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
                                universal_newlines=True).communicate()

    # If the vendor name contains any "virtual"-flavor, we're probably running
    # in a VM - if the above command triggered an error, abort
    if len(err) == 0:
        vendor = out.split()[1].lower()
        vmlist = ["vmware", "virtual", "virtualbox", "vbox", "qemu"]
        return any([virtual in vendor for virtual in vmlist])
    else:
        return False


def generate_artifical_data(nTrials=2, nChannels=2):
    """
    Populate `AnalogData` object w/ artificial signal
    """
    dt = 0.001
    t = np.arange(0, 3, dt) - 1.0
    sig = np.cos(2 * np.pi * (7 * (np.heaviside(t, 1) * t - 1) + 10) * t)
    sig = np.repeat(sig[np.newaxis, :], axis=0, repeats=nChannels)
    sig = np.tile(sig, [1, nTrials])
    sig += np.random.randn(*sig.shape) * 0.5
    sig = np.float32(sig)

    trialdefinition = np.zeros((nTrials, 3), dtype='int')
    for iTrial in range(nTrials):
        trialdefinition[iTrial, :] = np.array([iTrial * t.size, (iTrial + 1) * t.size, 1000])

    return spy.AnalogData(data=sig, dimord=["channel", "time"],
                          channel='channel', samplerate=1 / dt,
                          trialdefinition=trialdefinition)
