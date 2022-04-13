import numpy as np
import syncopy as spy


def generate_noisy_harmonics(nSamples, nChannels, samplerate):

    f1, f2 = 20, 50 # the harmonic frequencies in Hz
    
    # the sampling times vector
    tvec = np.arange(nSamples) * 1 / samplerate 

    # define the two harmonics
    harm1 = np.cos(2 * np.pi * f1 * tvec)
    harm2 = np.cos(2 * np.pi * f2 * tvec)
   
    # add some white noise
    trial = 0.5 * np.random.randn(nSamples, nChannels)
    # add 1st harmonic to 1st channel
    trial[:, 0] += harm1 
    # add 2nd harmonic to 2nd channel
    trial[:, 1] += 0.5 * harm2 
	
    return trial


nTrials = 50
nSamples = 1000
nChannels = 3
samplerate = 500   # in Hz

trials = []
for _ in range(nTrials):
    trial = generate_noisy_harmonics(nSamples, nChannels, samplerate)
    trials.append(trial)
    
synth_data = spy.AnalogData(trials, samplerate=samplerate)
