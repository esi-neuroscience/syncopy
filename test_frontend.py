''' This is a temporary development file '''

import numpy as np
import matplotlib.pyplot as ppl
from syncopy.specest import freqanalysis
from syncopy.tests.misc import generate_artificial_data

tdat = generate_artificial_data()
trials = [t for t in tdat.trials]

# test mtmfft analysis
r_mtm = freqanalysis(tdat)


# test classical wavelet analysis
r_wav = freqanalysis(tdat, method="wavelet", toi='all', output='abs')
res_trials = [t for t in r_wav.trials]

# test superlet analysis
# r_sup = freqanalysis(tdat, method="superlet", toi='all')
