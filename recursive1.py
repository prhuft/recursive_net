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
from matplotlib import pyplot as plt

#### The cost function and its 
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

#### The output criterion
class Criterion(object):

	@staticmethod
	def kappa(input_i, input_j):
		"""Evaluates the extent to which change from input i to input j met the 
		specified criterion defined here. A coefficient in [0,1] is returned.
		
		Recall that the rule, or the criterion is 
		what is used to tune the output. """
		
		# Decompose the inputs
		x1, y1 = input_to_xy(input_i)
		x2, y2 = input_to_xy(input_j)
		
		# The criterion is that the points x,y should trace a parabola, so we 
		# can subtract the actual slope from the desired slope and scale the result
		
		scale = 1/np.power(2,len(input_i)) # guarantees kappa <= 1
		if x2-x1 != 0: delta_x = x2-x1
		else: delta_x = 1
		kappa = scale*abs((2*x2 - (y2-y1)/(delta_x)))
		return kappa
	
	@staticmethod
	def a_L(kappa, a_L):
		"""Returns the artificially-constructed standard used to measure the
		output. When the network performs well, i.e. criterion ``kappa`` is 
		near 1, a_L depends highly on the previous output. When the network
		is far off, i.e. ``kappa`` near 0, a_L will be mostly random."""
		
		# Note that a_L is in [0,1]
		a_L = .5*((1-kappa)*a_r(len(a_L)) + kappa*a_L)
		return a_L
	
	@staticmethod
	def a_r(output_size):
		"""Returns a 1D array of length output_size with values between random 
		values between 0 and 1."""
		
		return np.random.sample(output_size)
		
	def new_input(output):
		"""For this simple net, the raw output is the new input."""
	
		return output

#### The neural network 
class Network(object):

	def __init__(self, sizes, input, bits=None):
		"""``sizes`` is a list of the numbers of neurons in each layer, ``a_L``
		is the randomly initialized 'previous output', with elements in [0,1],
		and input is a list [x,y]. 
		"""
		
		if bits is not None:
			self.bits = bits
		else: 
			self.bits = 8
		
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.a_L = a_r(sizes[-1])
		
		x = b10_to_b2(input[0],bits)
		y = b10_to_b2(input[0],bits)
		
		self.last_input = np.append(x,y)
		self.new_input = zeros(2*bits)
		self.kappa = 0 # The network standard will start out being random
		self.kappas = []
		
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
		
	def SGD(self, eta, epochs, epoch_size, lmbda = 0.0):
		"""Train the neural network using stochastic gradient descent. There is
		no training data, so the cost depends on rule that the network should 
		learn to obey, which I call the criterion. The cost will initially be 
		random, such that the network will have to flail about until it heads 
		in the right direction. When it does, the cost will begin to depend 
		more on the recent output. ``Eta`` is the learning parameter, ``epochs``
		is the number of training cycles, and ``epoch_size`` is the number of
		times to cycle the network during a single epoch."""
		
		# Loop through the training epochs
		for epoch in epochs:
			for n in range(0, epoch_size):
			
				self.kappas = []

				# Update the weights and biases after running a single input
				self.update(epoch_size, eta, lmbda, epoch)
				
				# How'd we do? 
				plot_kappa(self.kappas)
			
			
			
	def update(self, epoch_size, eta, lmbda, epochs):
		    """Update the network's weights and biases by applying gradient
        descent using backpropagation over a single epoch, i.e. ``epoch_size``
		tries to succeed. Recall there is no training data; subsequent inputs
		are born of previous outputs. ``eta`` is the learning rate, ``lmbda`` is
		the regularization parameter, and ``epochs`` is the total number of 
		epochs. """
		
		# n = epochs*epoch_size
		# nabla_b = [np.zeros(b.shape) for b in self.biases]
		# nabla_w = [np.zeros(w.shape) for w in self.weights]
		
		for i in range(0, epoch_size):
		
			# The current input and output standard. Note that self.a_L 
			# and self.kappa are updated in self.backprop
			x = Criterion.new_input(self.a_L)
			y = Criterion.a_L(self.kappa, self.a_L)
			
			delta_nabla_b, delta_nabla_w = self.backprop(x, y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
			
        self.weights = [(1-eta*(lmbda/n))*w-(eta/epoch_size)*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/epoch_size)*nb
                       for b, nb in zip(self.biases, nabla_b)]		
		
	def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the input vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
       
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
			
		# Store the last activation layer and compute kappa
		self.a_L = activations[-1]
		self.new_input = Criterion.new_input(self.a_L)
		self.kappa = Criterion.criterion(self.last_input, self.new_input)
		self.kappas.append[self.kappa]
		
		# Why is last_input == new_input? Self.new input is updated before 
		# we compare the two, so this is fine.
		self.last_input = self.new_input
		
        return (nabla_b, nabla_w)

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
	
def input_to_xy(input)
	"""Convert the neural input/output list to an (x,y) pair."""
	
	# Round to 0 or 1
	for i in range(0,len(input))
		input[i] = int(input[i]+0.5)
		
	cols = len(input)
	x = b2_to_b10(input[:int(cols/2)])
	y = b2_to_b10(input[int(cols/2):])
	
	return x,y
	
	
def plot_kappa(kappas):
	"""Plot the list of kappas versus iterations in one epoch."""
	x = plt.plot(kappas)
	plt.show(x)
	
def plot_inputs(inputs):
	"""Convert the input arrays back to (x,y) pairs and plot x vs y."""
	
	# Convert list inputs to a numpy array
	input_arr = np.array(inputs)
	rows, cols = x.shape
	
	x = np.zeros(rows)
	y = np.zeros(rows)
	for i in range(0, rows)
		x[i], y[i] = input_to_xy(input_arr[i])
		
	p = plt.plot(x,y)
	plt.show(p)
		
	
		
		
	
		
	

	
