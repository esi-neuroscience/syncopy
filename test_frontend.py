''' This is a temporary development file '''

import numpy as np
import matplotlib.pyplot as ppl
from syncopy.specest import freqanalysis
from syncopy.tests.misc import generate_artificial_data

tdat = generate_artificial_data()
trials = [t for t in tdat.trials]

# test mtmfft analysis
r_mtm = freqanalysis(tdat)


toi_ival = np.linspace(-0.5, 1, 100)
#toi_ival = [0,0.2,0.5,1]
# test classical wavelet analysis
r_wav = freqanalysis(tdat, method="wavelet",
                     toi=toi_ival,
                     output='abs') # , foilim=[5, 500])
res_wtrials = [t for t in r_wav.trials]

# test superlet analysis
foi = np.logspace(-1, 2.6, 25)

r_sup = freqanalysis(tdat, method="superlet", toi='all', order_max=30, foi=foi, output='abs',order_min=5, adaptive=True)
#res_strials = [t for t in r_sup.trials]
