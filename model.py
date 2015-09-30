import numpy as np
import theano, theano.tensor as T
import random
import theano_lstm

class Model:
	"""Model to predict idk a square wave or something"""
	def __init__(self, hiddenSize, inputSize):
		