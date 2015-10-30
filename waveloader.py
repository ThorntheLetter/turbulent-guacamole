import wave
import struct
import numpy as np
import sys
import model
import data-util

DEFAULT_SAMPLE_LENGTH = 150






def predict_samples(outfile, infile, length = 250000, sample_length = DEFAULT_SAMPLE_LENGTH):
	outfile.setparams(infile.getparams())
	x = np.zeros((1,sample_length,1), dtype = 'float32')
	for i in range(length):
		y = model.themodel.predict(x[:,-1*sample_length,:])[0][0]
		x = np.append(x, [[[y]]], axis = 1)
		print(i,"/",length)
	outfile.writeframes(data-util.vunsquash(x[sample_length:]).astype('int16').tobytes())


#currently will only support mono 16-bit signed int PCM wave files, but setting up to add more support later.
if __name__ == "__main__":
	infile = wave.open(sys.argv[1], 'rb')
	outfile = wave.open(sys.argv[2], 'wb')
	x, y = arrange_samples(infile)
	model.themodel.fit(x, y, batch_size = 150, verbose = 1, nb_epoch = 2)
	predict_samples(outfile, infile)