import wave
import struct
import numpy as np
import sys
import model

DEFAULT_SAMPLE_LENGTH = 1000

#Gets next frame in the file
def get_next_frame(file):
	return (struct.unpack("<h",file.readframes(1))[0])

#Arranges the file into numpy matrix for input.
def arrange_samples(file, sample_length = DEFAULT_SAMPLE_LENGTH):
	file.rewind()
	position = file.tell()
	number_of_frames = file.getnframes()
	number_of_samples = number_of_frames - sample_length
	x = np.zeros((number_of_samples, sample_length, 1), dtype = 'int16')
	y = np.zeros((number_of_samples, 1), dtype = 'int16')
	for i in range(number_of_samples):
		print(i, "/", number_of_samples)
		file.setpos(position)
		current_frame = get_next_frame(file)
		position = file.tell()
		for j in range(sample_length):
			x[i,j,0] = current_frame
			current_frame = get_next_frame(file)
		y[i,0] = current_frame
	return (x, y)


def predict_samples(outfile, infile, length = 500000, sample_length = DEFAULT_SAMPLE_LENGTH):
	outfile.setparams(infile.getparams())

	x = np.zeros((1,sample_length,1), dtype = int16)
	for i in range(length):
		y = model.themodel.predict(x)[0][0]
		x = np.delete(x,0,1)
		x = np.append(x, [[[y]]], axis = 1)
		outfile.writeframes(struct.pack('h', int(np.round(y))))


#currently will only support mono 16-bit signed int PCM wave files, but setting up to add more support later.
if __name__ == "__main__":
	infile = wave.open(sys.argv[1], 'rb')
	outfile = wave.open('output.wav', 'wb')
	x, y = arrange_samples(infile)
	model.themodel.fit(x, y, batch_size = 1000, verbose = 1, nb_epochs = 1)
	predict_samples(outfile, infile)
