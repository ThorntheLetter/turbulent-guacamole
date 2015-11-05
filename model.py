import os
os.environ["THEANO_FLAGS"] = "device=gpu,floatX=float32"
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM


LAYER1_SIZE = 256
LAYER2_SIZE = 256
DROPOUT_RATE = .3

themodel = Sequential()
themodel.add(LSTM(LAYER1_SIZE, return_sequences = True, input_dim = 1))
themodel.add(Dropout(DROPOUT_RATE))
themodel.add(LSTM(LAYER1_SIZE, return_sequences = False))
themodel.add(Dropout(DROPOUT_RATE))
themodel.add(Dense(1))
themodel.compile(loss='mse', optimizer='rmsprop') #mse because I somewhat know how that one works, rmsprop because I haven't looked for good desctiptions of those yet.
