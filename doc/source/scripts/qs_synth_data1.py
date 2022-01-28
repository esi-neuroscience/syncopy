import numpy as np
import syncopy as spy

nTrials = 50
nSamples = 1000
nChannels = 3
samplerate = 500 # in Hz
# the sampling times vector
tvec = np.arange(nSamples) * 1 / samplerate 
f1, f2 = 30, 42 # the harmonic frequencies in Hz

# define the two harmonics
harm1 = np.cos(2 * np.pi * f1 * tvec)
harm2 = np.cos(2 * np.pi * f2 * tvec)
   
trials = []
for _ in range(nTrials):
    # the white noise
    trl = 0.5 * np.random.randn(nSamples, nChannels)
    # add 1st harmonic to 1st channel
    trl[:, 0] += harm1 
    # add 2nd harmonic to 2nd channel
    trl[:, 1] += harm2 
	
    trials.append(trl)
	
data = spy.AnalogData(trials, samplerate=samplerate)
