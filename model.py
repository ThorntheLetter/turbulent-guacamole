import numpy as np
import theano, theano.tensor as T
import random
import theano_lstm
from theano_lstm import LSTM, StackedCells, Layer

class Model:
	"""Model to predict idk a square wave or something, mostly based on the theano_lstm tutorial"""
	def __init__(self, hiddenSize, inputSize, stackSize = 1, cellType = LSTM):
		self.model = StackedCells(inputSize, cellType = cellType, layers =[hidden_size] * stack_size) #Most of model
		self.model.layers.append(Layer(hiddenSize, inputSize, activation = np.tanh)) #Output layer, same size as input layer.
		self.inputMat = T.imatrix() #The input matrix is integers, square wave can work with that, change for other data types
		self.primingWord = T.iscalar() #not sure what this does yet so i am keeping it even though it looks like word stuff
		self.srng = T.sharedRandomstreams.RandomStreams(np.random.randint(0, 1024)) #rng
		self.predictions = self.createPrediction() #not sure why this is ana attribute and method but okay
		self.greedyPredictions = self.createPrediction(greedy=True) #same, supposedly these are for symbolic variables.
		self.create_cost_fun() #these 3 are for training
		self.create_training_function()
		self.create_predict_function()

	@property
	def params(self): #accessor for params
		return self.model.params

	