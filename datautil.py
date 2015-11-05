import numpy as np
import wave

#takes input frame and width of sample in bytes and transforms it into a number between -1 and 1
def squash(input, width = 2):
	return(input/(2 ** ((8 * width)-1)))

#reverses squash()
def unsquash(input, width = 2):
	return(input * (2 ** ((8 * width) - 1)))

vunsquash = np.vectorize(unsquash)

vsquash = np.vectorize(squash)

#Gets next frame in the file
def get_next_frames(file, nframes):
	return np.fromstring(file.readframes(nframes), dtype = 'int16')

#Arranges the file into numpy matrix for input using every possible sequence.
def arrange_samples_full(file, sample_length):
	file = wave.open(filename, 'rb')
	file.rewind()
	position = file.tell()
	number_of_frames = file.getnframes()
	number_of_samples = number_of_frames - sample_length - 1
	x = np.zeros((number_of_samples, sample_length, 1), dtype = 'float32')
	y = np.zeros((number_of_samples, 1), dtype = 'float32')
	for i in range(number_of_samples):
		print(i, "/", number_of_samples, end = '\r')
		file.setpos(position)
		x[i,:,0] = get_next_frames(file, sample_length)
		y[i,:] = get_next_frames(file, 1)
		position = file.tell()
		file.readframes(1)
	print()
	return (vsquash(x), vsquash(y))


#Arranges the file into numpy matrix for input sequentially.
def arrange_samples_sequential(filename, sample_length):
	file = wave.open(filename, 'rb')
	file.rewind()
	number_of_frames = file.getnframes()
	number_of_samples = number_of_frames // (sample_length + 1)
	x = np.zeros((number_of_samples, sample_length, 1), dtype = 'float32')
	y = np.zeros((number_of_samples, 1), dtype = 'float32')
	for i in range(number_of_samples):
		print(i, "/", number_of_samples, end ='\r')
		x[i,:,0] = get_next_frames(file, sample_length)
		y[i,:] = get_next_frames(file, 1)
	print()	
	return (vsquash(x), vsquash(y))
