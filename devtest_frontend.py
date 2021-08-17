''' This is a temporary development file '''

import numpy as np
import matplotlib.pyplot as ppl
from scipy import signal
from syncopy.specest import freqanalysis
from syncopy.tests.misc import generate_artificial_data
from syncopy.shared.tools import get_defaults


tdat = generate_artificial_data()

toi_ival = np.linspace(-0.5, 1, 100)

toi_ival = [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
toi_ival = np.linspace(-0.2, 1, 50)
# toi_ival = 'all'
# foi = np.logspace(-1, 2.6, 50)
foi = np.linspace(0.1, 45, 50)

pad = "relative"
padlength = 5

# r_mtm = freqanalysis(tdat, method="mtmconvol",
#                      toi=toi_ival,
#                      output='abs',
#                      taper='dpss',
#                      tapsmofrq=None,
#                      t_ftimwin=1,
#                      wav=None,
#                      pad=pad,
#                      foi=foi
# )
#, foilim=[5, 500])

# test classical wavelet analysis
r_wav = freqanalysis(tdat, method="wavelet",
                     toi=toi_ival,
                     output='abs',
                     foi=foi,
                     pad=pad,
                     padlength=padlength) #, foilim=[5, 500])

# # test superlet analysis
# r_sup = freqanalysis(tdat, method="superlet", toi=toi_ival,
#                      order_max=30, output='abs',
#                      order_min=1,
#                      c_1 = 3,
#                      foi=foi,
#                      adaptive=True)

