#import os
#os.chdir('C:/Users/Bryan/Desktop')
#exec(open("./neuralNetwork.py").read())
#Filename paths are on lines '115' and '131'.

import numpy
import matplotlib.pyplot
import scipy.special

#Neural Network Class Definition
class neuralNetwork:
	# Initialize the neural network
	def __init__(self, inputnodes, hiddenlayers, hiddennodesperlayer, outputnodes, learningrate):
		#Set number of nodes in each input, hidden and output layer
		self.inodes = inputnodes
		self.onodes = outputnodes
		self.numhlayers = hiddenlayers
		self.numperhlay = hiddennodesperlayer
		self.dict = {}
		
		# Link weight matrices, wih, whh and who
		# Weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
		layernumlist = self.numperhlay.split(",")
		self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (int(layernumlist[0]), self.inodes))
		
		for x in range(1, self.numhlayers):
			self.dict["%s" % x] = numpy.random.normal(0.0, pow(int(layernumlist[x-1]), -0.5), (int(layernumlist[x]), int(layernumlist[x-1])))
		self.who = numpy.random.normal(0.0, pow(int(layernumlist[self.numhlayers - 1]), -0.5), (self.onodes, int(layernumlist[self.numhlayers - 1])))
		# learning rate
		self.lr = learningrate
		
#		print(self.wih)
#		for x in range(0, self.numhlayers - 1):
#			print(self.x)
#		print(self.who)
		
		# Activation function is the sigmoid function
		self.activation_function = lambda x: scipy.special.expit(x)
		
		pass
		
	#train the neural network
	def train(self, inputs_list, targets_list):
		#convert inputs list to 2d arrays
		inputs = numpy.array(inputs_list, ndmin=2).T
		targets = numpy.array(targets_list, ndmin=2).T
		
		d = {}
		#Calculate signals into hidden layer
		hidden_inputs0 = numpy.dot(self.wih, inputs)
		d["hidden_outputs0"] = self.activation_function(hidden_inputs0)
		
		for x in range(1, self.numhlayers):
			y = x - 1
			d["hidden_inputs%s" % x] = numpy.dot(self.dict["%s" % x], d["hidden_outputs%s" % y])
			d["hidden_outputs%s" % x] = self.activation_function(d["hidden_inputs%s" % x])
			
		final_inputs = numpy.dot(self.who, d["hidden_outputs%s" % (self.numhlayers - 1)])
		final_outputs = self.activation_function(final_inputs)
		
#		print(final_outputs)
		
		#output layer error is (target - actual) ^ 2
		d["output_errors%s" % self.numhlayers] = (targets - final_outputs)
		d["output_errors%s" % (self.numhlayers - 1)] = numpy.dot(self.who.T, d["output_errors%s" % self.numhlayers])
		
		#hidden layer errors are the output_errors, split by weights, recombined a previous hidden nodes.
		for x in reversed(range(1, self.numhlayers)):
			d["output_errors%s" % (x - 1)] = numpy.dot(self.dict["%s" % x].T, d["output_errors%s" % x])
		
		#Update the weights for the links between layers
		self.who += self.lr * numpy.dot((d["output_errors%s" % self.numhlayers] * final_outputs * (1.0 - final_outputs)), numpy.transpose(d["hidden_outputs%s" % (self.numhlayers - 1)]))
		for x in reversed(range(1, self.numhlayers)):
			self.dict["%s" % x] += self.lr * numpy.dot((d["output_errors%s" % x] * d["hidden_outputs%s" % x] * (1.0 - d["hidden_outputs%s" % x])), numpy.transpose(d["hidden_outputs%s" % (x - 1)]))
		self.wih += self.lr * numpy.dot((d["output_errors0"] * d["hidden_outputs0"] * (1.0 - d["hidden_outputs0"])), numpy.transpose(inputs))
		
#		print(self.wih)
#		for x in range(0, self.numhlayers - 1):
#			print(self.x)
#		print(self.who)
		
		pass
	
	def query(self, inputs_list):
		#convert inputs list to 2d array
		inputs = numpy.array(inputs_list, ndmin=2).T
		
		d = {}
		#Calculate signals into hidden layer
		hidden_inputs0 = numpy.dot(self.wih, inputs)
		d["hidden_outputs0"] = self.activation_function(hidden_inputs0)
		
		for x in range(1, self.numhlayers):
			y = x - 1
			d["hidden_inputs%s" % x] = numpy.dot(self.dict["%s" % x], d["hidden_outputs%s" % y])
			d["hidden_outputs%s" % x] = self.activation_function(d["hidden_inputs%s" % x])
			
		final_inputs = numpy.dot(self.who, d["hidden_outputs%s" % (self.numhlayers - 1)])
		final_outputs = self.activation_function(final_inputs)
		
		print(final_outputs)
		
		return(final_outputs)
		
		
input_nodes = 784
hidden_layers = 1
hidden_nodes = "200"
output_nodes = 10
learning_rate = 0.1

n = neuralNetwork(input_nodes,hidden_layers,hidden_nodes,output_nodes,learning_rate)

print('Hello')
data_file = open("mnistTestData/mnist_train.csv", 'r')
data_list = data_file.readlines()
data_file.close()

for record in data_list:
	all_values = record.split(',')
	inputs = (numpy.asfarray(all_values[1:])/255.0*0.99) + 0.01
	targets = numpy.zeros(output_nodes) + 0.01
	targets[int(all_values[0])] = 0.99
	n.train(inputs, targets)
	n.train(inputs, targets)
	n.train(inputs, targets)
	n.train(inputs, targets)
	n.train(inputs, targets)
	pass

test_data_file = open("mnistTestData/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

#all_values = test_data_list[0].split(',')
#image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
#matplotlib.pyplot.imshow(image_array, cmap='Greys',interpolation='None')
#matplotlib.pyplot.savefig('myfig.png')

#n.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)

#Test the Neural Network
scorecard = []

#Go through all the records in the test data set
for record in test_data_list:
	all_values = record.split(",")
	correct_label = int(all_values[0])
	
	#scale and shift inputs
	inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
	#Query the network
	outputs = n.query(inputs)
	label = numpy.argmax(outputs)
	print(label, "network's answer")
	print(correct_label, "correct label")
	#Append correct or incorrect to list
	if (label == correct_label):
		scorecard.append(1)
	else:
		scorecard.append(0)
	pass
	
print(scorecard)
scorecard_array = numpy.asarray(scorecard)
print("Performance = ", scorecard_array.sum() / scorecard_array.size)
