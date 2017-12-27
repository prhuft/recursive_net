"""
Incentivized neural network for autonomous learning. 

Preston Huft, December 2017

Teach a point to walk down a hill --> very simple, I hope.

Qualitative description of how it should work: 

Input: point position (x,y), in domain [x1, x2] and range [y1,y2]
Output: delta x, within y's domain, and delta y, within y's range. 
Cost: Cost will be evaluated based on whether x and y position roughly obey 
	some predefined function, unbeknownst to the network. That is, there is
	not an expected output, but an implicit standard or goal. 
Training: The network will not be trained by showing it a bunch of x y pairs
	flagged correct or incorrect. Instead, the the output, which is the next
	input is compared to the previous input to determine a measure of the 
	extent to which the network is obeying the rule. This cost is used to
	update the weights and biases. 
"""

import random
import numpy as np
import matplotlib

#### Compute the cost and its gradient
class Cost(object):

	@staticmethod
	def ce(a, y):
		"""Returns the cross entropy cost function for an ouptut ``a`` and a
		desired output ``y``. np.nan_to_num forces infinite values to be large 
		finite values of the appropriate sign. """
	
		return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
		
	@staticmethod
	def delta(a, y):
		"""Returns the error delta for the output layer L."""
	
		return (a-y)

#### The neural network 
class Network(object):

	def __init__(self, sizes):
		"""Sizes is a list of the numbers of neurons in each layer"""
		
		self.num_layers = len(sizes)
		self.sizes = sizes
		
	def weight_initializer(self):
		"""Initialize each weight with a Gaussian distribution with mean 0 and
		standard deviation 1/sqrt(number weights in current layer). Initialize 
		each bias with a Gaussian distribution with mean 0 and standard 
		deviation 1. 
		
		Note: layer one, i.e. the input layer, has no biases because biases
		are only used for outputs of previous layers. """
		
		self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
		
		# no. of rows = no. of neurons in layer l+1, 
		# no. of cols = no. of neurons in layer l
		self.weights = [np.random(y,x) for x,y in zip(self.sizes[:-1],
						self.sizes[1:])] 
	
	def feedforward(self, a):
		"""Return the output of the neural network for an input a."""
		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a)+b) # TODO def sigmoid
		return a
		
	def SGD(self, eta, epochs, epoch_size):
		"""Train the neural network using stochastic gradient descent. There is
		no training data, so the cost depends on rule that the network should 
		learn to obey, which I call the criterion. The cost will initially be 
		random, such that the network will have to flail about until it heads 
		in the right direction. When it does, the cost will begin to depend 
		more on the recent output. ``Eta`` is the learning parameter, ``epochs``
		is the number of training cycles, and ``epoch_size`` is the number of
		times to cycle the network during a single epoch."""
		
		

#### Miscellaneous functions 
	
def b2_to_b10(number):
	"""Converts binary number to base 10. The binary number should be a list 
	with the largest bit place on the left, as one would write it."""
	
	result = 0
	bits = len(number)
	
	# The weird indexing here allows intuitive binary entry
	for n in range(0,bits):
		result += number[bits-1-n]*np.power(2,n)
	
	return result
	
def b10_to_b2(number, bits):
	"""Converts a base 10 number to binary. The number of bits specified here
	should be greater or equal than the number of bits required."""
	
	result = np.zeros(bits)
	for n in range(0,bits):
		bit = np.power(2,bits-1-n)
		if number >= bit:
			result[n] = 1
			number -= bit
			
	return result
	
def input_to_num(x): # Why this not working????
	"""Converts input value array to base 10 float with one decimal point. 
	For example, [.8, .3, .1, .6] would become 831.6. The last value in the
	array is assumed to be the tenths place."""
	
	result = 0
	max = len(x)-1
	for n in range(max,0):
		result += x[n]*np.power(10,max-n)
	
	return result
	
	
@staticmethod
def criterion(input_i, input_j):
	"""Evaluates the extent to which change from input i to input j met the 
	specified criterion defined here. A coefficient in [0,1] is returned.
	
	Recall that the rule, or the criterion is 
	what is used to tune the output. """
	
	# Decompose the inputs
	x1, y1 = input_i
	x2, y2 = input_j
	
	# The criterion is that the points x,y should trace a parabola, so we 
	# can subtract the actual slope from the desired slope and scale the result
	
	scale = 1/509
	if x2-x1 != 0: delta_x = x2-x1
	else: delta_x = 1
	kappa = scale*abs((2*x2 - (y2-y1)/(delta_x)))
	return kappa

def a_r(output_size):
	"""Returns a 1D array of length output_size with values between random 
	values between 0 and 1."""
	
	return np.random.sample(output_size)
