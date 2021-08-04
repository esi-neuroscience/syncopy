''' This is a temporary development file '''

import numpy as np
import matplotlib.pyplot as ppl
from syncopy.specest import freqanalysis
from syncopy.tests.misc import generate_artificial_data

tdat = generate_artificial_data()

# test mtmfft analysis
r_mtm = freqanalysis(tdat)

toi_ival = np.linspace(-0.5, 1, 100)

#toi_ival = [0,0.2,0.5,1]
toi_ival = 'all'
foi = np.logspace(-1, 2.6, 25)
# test classical wavelet analysis
r_wav = freqanalysis(tdat, method="wavelet",
                     toi=toi_ival,
                     output='abs',
                     foi=None) #, foilim=[5, 500])

# test superlet analysis
r_sup = freqanalysis(tdat, method="superlet", toi=toi_ival,
                     order_max=20, output='abs',
                     order_min=1,
                     c_1 = 5,
                     adaptive=True)

r_sup = freqanalysis(tdat, method="superlet", toi='all', order_max=30, foi=foi, output='abs',order_min=5, adaptive=True)
#res_strials = [t for t in r_sup.trials]
