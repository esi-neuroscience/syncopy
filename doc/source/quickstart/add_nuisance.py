# built-in synthetic data generators
import syncopy.tests.synth_data as spy_synth

# a linear trend
lin_trend = spy_synth.linear_trend(y_max=3,
                                   nTrials=50,
				   samplerate=500,
				   nSamples=1000,
				   nChannels=2)
# a 2nd 'nuisance' harmonic
harm50 = spy_synth.harmonic(freq=50,
                            nTrials=50,
			    samplerate=500,
			    nSamples=1000,
			    nChannels=2)
