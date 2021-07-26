''' This is a temporary development file '''

import numpy as np
from syncopy.specest import freqanalysis
from syncopy.tests.misc import generate_artificial_data

tdat = generate_artificial_data()

# test mtmfft analysis
r_mtm = freqanalysis(tdat)

# test classical wavelet analysis
r_wav = freqanalysis(tdat, method="wavelet", toi='all')

# test superlet analysis
r_sup = freqanalysis(tdat, method="superlet", toi='all')
