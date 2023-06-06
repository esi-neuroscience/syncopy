import numpy as np
import syncopy as spy

# cfg dictionary
cfg = spy.StructDict()
cfg.nTrials = 50
cfg.nSamples = 1000
cfg.nChannels = 2
cfg.samplerate = 500   # in Hz



# set the log level
spy.set_loglevel("INFO")
