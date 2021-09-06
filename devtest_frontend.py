''' This is a temporary development file '''

import numpy as np
import matplotlib.pyplot as ppl
from scipy import signal

from syncopy.shared.parsers import data_parser, scalar_parser, array_parser 
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


tdat = generate_artificial_data(inmemory=True)

toi_ival = [-1, 2]
toi_eqd = np.arange(0, 0.5, step=0.01)
toi_neqd = [0,0.1,0.3,0.30001]
# toi_ival = 'all'
# toi_ival = None
# foi = np.logspace(-1, 2.6, 50)
foi = np.linspace(0, 45, 46)
foi = None

# pad = 'relative'
# pad = 'absolute'
# pad = 'absolute'
pad = 'maxlen'
padlength = 4000
# prepadlength = 150
# postpadlength = 150


r_mtmc = freqanalysis(tdat, method="mtmconvol",
                      toi=toi_eqd,
                      t_ftimwin=1.5,
                      output='pow',
                      taper='dpss',
                      nTaper=10,
                      tapsmofrq=None,
                      keeptapers=False,
                      pad=pad,
                      padlength=padlength,
                      foi=foi)


# r_mtm = freqanalysis(tdat, method="mtmfft",
#                      toi=toi_ival,
#                      t_ftimwin=1.5,
#                      output='pow',
#                      taper='dpss',
#                      nTaper=10,
#                      tapsmofrq=None,
#                      keeptapers=False,
#                      pad=pad,
#                      padlength=padlength,
#                      foi=foi)

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
r_wav = freqanalysis(
    tdat, method="wavelet",
    toi=toi_ival,
    # toi='all',
    wav='Paul',
    order=4,
    output='abs',
    #foilim=[2,10],
    foi=foi,
    pad=None,
    padtype='sth',
    adaptive=True
) 

# # test superlet analysis
# r_sup = freqanalysis(tdat, method="superlet", toi=toi_ival,
#                      order_max=30, output='abs',
#                      order_min=1,
#                      c_1 = 3,
#                      foi=foi,
#                      adaptive=True,
#                      wav="Paul")

