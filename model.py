import os
os.environ["THEANO_FLAGS"] = "FAST_RUN,device=gpu,floatX=float32"


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import numpy as np


layer1_nodes = 256
layer2_nodes = 256
#I think I will do a sine wave instead
model = Sequential()
model.add(LSTM(1, return_sequences = True, output_dim = layer1_nodes))
#Dropout goes here, but i wanted it to somewhat work before making it more complex.
model.add(LSTM(layer1_nodes, return_sequences = False, output_dim = layer2_nodes))
model.add(Dense(layer1_nodes, 1)) #The Output: layer2_nodes connections in, one connection out.
model.add(Activation("tanh"))

model.compile(loss='mse', optimizer='rmsprop') #mse because I somewhat know how that one works, rmsprop because I haven't looked for good desctiptions of those yet.\
print("done") #so that I know it is done. test 1 done in 174.8s. does adding gpu help for this? test 2 in 93.4s.

#simple data set code that will get rewritten to show progress later.
x = np.zeros((100,10,1))
y = np.zeros((100,1))
for i in range(100):
	for j in range(10):
		x[i,j,0] = (i + j) / 10
	y[i,0] = (i + 11) / 11

x = np.sin(x)
y = np.sin(y) 

for i in range(5):
	print("iteration: ", i)
	model.fit(x, y, batch_size = 20, verbose = 2)

xseed = np.zeros((1,10,1))
for i in range(10):
	for j in range(10):
		xseed[0,j,0] = i + (j / 10)
	xseed = np.sin(xseed)
	print("prediction: ", model.predict(xseed))
	print("actual: ", np.sin(i + 1.1))
	print()
