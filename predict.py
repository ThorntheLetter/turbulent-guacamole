import wave
from model import themodel
import argparse
from datautil import *
import numpy as np


parser = argparse.ArgumentParser(description = "train the model")
parser.add_argument('model', help='The name of the HDF5 file with the model\'s weights.')
parser.add_argument('output', help='The name of the wav file to write the predictions of.')
parser.add_argument('-s', '--sample', type = int, dest='sample_size', default = 44100, help='The number of frames to include in each sample.')
parser.add_argument('-l', '--length', type = int, dest='length', default = 441000, help='The number of frames to predict.')

def predict_samples(outfile, length, sample_length):
	outfile.setparams((nchannels = 1, sampwidth = 2, framerate = 44100, nframes = 0, comptype = 'NONE', compname = 'not compressed')
	x = np.zeros((1, sample_length, 1), dtype = 'float32')
	for i in range(length):
		y = model.themodel.predict(x[:,-1*sample_length,:])[0,0]
		x = np.append(x, [[[y]]], axis = 1)
		print(i,"/",length, end = '\r')
	outfile.writeframes(vunsquash(x[sample_length:]).astype('int16').tobytes())

def main():
	args = parser.parse_args()
	themodel.load_weights(args.model)
	predict_samples(args.output, args.length, args.sample_size)



if __name__ == "__main__":
	main()