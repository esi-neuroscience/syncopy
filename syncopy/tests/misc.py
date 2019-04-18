# -*- coding: utf-8 -*-
#
# Helper methods for testing routines
# 
# Created: 2019-04-18 14:41:32
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-04-18 15:08:26>

import subprocess
import sys

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
