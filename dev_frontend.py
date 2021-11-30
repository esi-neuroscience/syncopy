import numpy as np
import scipy.signal as sci

from syncopy.datatype import CrossSpectralData, padding, SpectralData
from syncopy.connectivity.ST_compRoutines import cross_spectra_cF, ST_CrossSpectra
from syncopy.connectivity.ST_compRoutines import cross_covariance_cF
from syncopy.connectivity import connectivityanalysis
from syncopy.specest import freqanalysis
import matplotlib.pyplot as ppl

from syncopy.shared.parsers import data_parser, scalar_parser, array_parser
from syncopy.shared.tools import get_defaults
from syncopy.datatype import SpectralData, padding

from syncopy.tests.misc import generate_artificial_data
tdat = generate_artificial_data(inmemory=True, seed=1230, nTrials=5)

foilim = [1, 30]
# this still gives type(tsel) = slice :)
sdict1 = {"trials": [0], 'channels' : ['channel1'], 'toi': np.arange(-1, 1, 0.001)}

# # no problems here..
# coherence = connectivityanalysis(data=tdat, method=
#                                 foilim=None,
#                                 output='pow')

# a lot of problems here..
correlation = connectivityanalysis(data=tdat, method='corr', keeptrials=False)


# the hard wired dimord of the cF
dimord = ['None', 'freq', 'channel_i', 'channel_j']

# res = freqanalysis(data=tdat,
#                    method='mtmfft',
#                    samplerate=tdat.samplerate,
# #                   order_max=20,
# #                   foilim=foilim,
# #                   foi=np.arange(502),
#                    output='pow',
# #                   polyremoval=1,
#                    t_ftimwin=0.5,
#                    keeptrials=False,
#                    taper='dpss',
#                    nTaper = 11,
#                    tapsmofrq=5,
#                    keeptapers=True,
#                    parallel=False, # try this!!!!!!
#                    select={"trials" : [0,1]})
