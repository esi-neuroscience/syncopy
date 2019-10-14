# -*- coding: utf-8 -*-
#
# Wavelet analysis example
#
# Created: 2019-02-25 13:08:56
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-05-13 10:07:30>

# Builtin/3rd party package imports
import numpy as np
import matplotlib.pyplot as plt

# Add SyNCoPy to Python search path
import os
import sys
spy_path = os.path.abspath(".." + os.sep + "..")
if spy_path not in sys.path:
    sys.path.insert(0, spy_path)

# Import SyNCoPy
import syncopy as spy


if __name__ == "__main__":
    data = spy.tests.misc.generate_artificial_data(nChannels=5, nTrials=37)
    spec, lastres = spy.wavelet(data)
    import matplotlib.pyplot as plt
    plt.ion()
    plt.close('all')
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(data.time[0], data.trials[0].T)
    ax[1].pcolormesh(spec.time[0], spec.freq,
                     np.absolute(np.squeeze(spec.trials[3][:, 0, 0, :])).T)
    ax[1].set_yscale('log')
    ax[1].set_ylim([1, 100])
