''' This is a temporary development file '''

import numpy as np
import matplotlib.pyplot as ppl
from scipy import signal

from syncopy.datatype import padding
from syncopy.shared.tools import get_defaults
from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYWarning

from syncopy.specest import freqanalysis
from syncopy.tests.misc import generate_artificial_data

from syncopy.specest.compRoutines import (
    SuperletTransform,
    WaveletTransform,
    MultiTaperFFTConvol
)


tdat = generate_artificial_data()

toi_ival = np.linspace(-0.5, 1, 100)
toi_ival = np.arange(0, 0.5, step=0.2)
toi_ival = [0,0.1,0.3]
# toi_ival = [0,1]
# toi_ival = 'all'
# toi_ival = None
# foi = np.logspace(-1, 2.6, 50)
foi = np.linspace(0.1, 45, 50)

pad = 'relative'
# pad = 'absolute'
# pad = 'maxlen'
padlength = 3100
prepadlength = 150
postpadlength = 150

r_mtm = freqanalysis(tdat, method="mtmfft",
                     toi=toi_ival,
                     t_ftimwin=10.5,
                     output='abs',
                     taper='dpss',
                     tapsmofrq=None,
                     pad=pad,
                     padlength=100,
                     foi=foi)

# r_mtmc = freqanalysis(tdat, method="mtmconvol",
#                       toi=toi_ival,
#                       t_ftimwin=.5,
#                       output='abs',
#                       taper='hann',
#                       tapsmofrq=2,
#                       pad=True,
#                       padlength=100,
#                       order_max=10,
#                       foi=foi)

# , foilim=[5, 500])

# test classical wavelet analysis
# r_wav = freqanalysis(
#     tdat, method="wavelet",
#     toi=toi_ival,
#     output='abs',
#     foi=foi,
#     pad=True,
#     padtype='sth',
#     adaptive=True
# ) 

# # test superlet analysis
# r_sup = freqanalysis(tdat, method="superlet", toi=toi_ival,
#                      order_max=30, output='abs',
#                      order_min=1,
#                      c_1 = 3,
#                      foi=foi,
#                      adaptive=True,
#                      wav="Paul")

