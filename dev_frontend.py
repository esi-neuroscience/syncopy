import numpy as np
import scipy.signal as sci

from syncopy.datatype import CrossSpectralData, padding, SpectralData, AnalogData
from syncopy.connectivity.ST_compRoutines import cross_spectra_cF, ST_CrossSpectra
from syncopy.connectivity.ST_compRoutines import cross_covariance_cF
from syncopy.connectivity import connectivity
from syncopy.specest import freqanalysis
import matplotlib.pyplot as ppl

from syncopy.shared.parsers import data_parser, scalar_parser, array_parser
from syncopy.shared.tools import get_defaults
from syncopy.datatype import SpectralData, padding

from syncopy.tests.misc import generate_artificial_data
tdat = generate_artificial_data(inmemory=True, seed=1230, nTrials=50, nChannels=5)

foilim = [1, 30]
# this still gives type(tsel) = slice :)
sdict1 = {"trials": [0], 'channels' : ['channel1'], 'toi': np.arange(-1, 1, 0.001)}

# no problems here..
coherence = connectivity(data=tdat,
                         foilim=None,
                         output='pow',
                         taper='dpss',
                         tapsmofrq=5,
                         foo = 3, # non-sensical 
                         keeptrials=False)

granger = connectivity(data=tdat,
                       method='granger',
                       foilim=[0, 50],
                       output='pow',
                       taper='dpss',
                       tapsmofrq=5,                                 
                       keeptrials=False)

# D = SpectralData(dimord=['freq','test1','test2','taper'])
# D2 = AnalogData(dimord=['freq','test1'])

# a lot of problems here..
# correlation = connectivity(data=tdat, method='corr', keeptrials=False, taper='df')

# the hard wired dimord of the cF

res = freqanalysis(data=tdat,
                   method='mtmfft',
                   samplerate=tdat.samplerate,
#                   order_max=20,
#                   foilim=foilim,
#                   foi=np.arange(502),
                   output='pow',
#                   polyremoval=1,
                   t_ftimwin=0.5,
                   keeptrials=True,
                   taper='dpss',
                   nTaper = 19,
                   tapsmofrq=5,
                   keeptapers=True,
                   parallel=False, # try this!!!!!!
                   select={"trials" : [0,1]})
