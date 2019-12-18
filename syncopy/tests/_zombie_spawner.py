# -*- coding: utf-8 -*-
# 
# Script launching a Syncopy zombie called by test_packagesetup.py
# 
# Created: 2019-11-11 10:56:48
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-11-11 15:37:14>

import os
import time
from syncopy.tests.misc import generate_artificial_data
from syncopy import __storage__


if __name__ == "__main__":
    dummy = generate_artificial_data(nTrials=2, nChannels=16, 
                                     equidistant=True, inmemory=False)
    dummy.save(os.path.join(__storage__, "spy_dummy"))
    while True:
        time.sleep(1.)
