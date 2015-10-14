import os
os.environ["THEANO_FLAGS"] = "device=gpu,floatX=float32"
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import numpy as np


LAYER1_SIZE = 256
LAYER2_SIZE = 256
DROPOUT_RATE = .3

model = Sequential()
model.add(LSTM(1, return_sequences = True, output_dim = LAYER1_SIZE))
model.add(Dropout(DROPOUT_RATE))
model.add(LSTM(LAYER1_SIZE, return_sequences = False, output_dim = LAYER2_SIZE))
model.add(Dropout(DROPOUT_RATE))
model.add(Dense(LAYER2_SIZE, 1))
model.compile(loss='mse', optimizer='rmsprop') #mse because I somewhat know how that one works, rmsprop because I haven't looked for good desctiptions of those yet.