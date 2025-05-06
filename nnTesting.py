import numpy as np
import matplotlib
import nnfs
import math
from matplotlib import pyplot as plt
from nnfs.datasets import spiral_data # use this for testing data

#-------------------------------------------------

inputs =  [1, 2, 3, 2.5]
weights =  [0.2, 0.8, -0.5, 1.0]
bias = 2

#output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + inputs[3]*weights[3] + bias
#print(output) 


x = 0
y = 0
for num in inputs:
    x += inputs[y]*weights[y]
    y += 1
x += bias

#print(x)

#-------------------------------------------------
# 3 neurons instead of 1 means 3 lists of weights, 3 biases      

inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
bias = [2, 3, 0.5]

#myTuple = tuple(zip(weights, bias))
#print(myTuple)

layerOutputs = []
# for each neuron
for neuron_weights, neuron_bias in zip(weights, bias):
    #reset output each time we visit a new neuron
    neuron_output = 0
    # for each input and weight to specified neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    # dont forget to add the bias before appending
    neuron_output += neuron_bias
    layerOutputs.append(neuron_output)

#print(layerOutputs)

#--------------------------------------------------
#

a = [1, 2, 3]
b = [2, 3, 4]
dotProduct = 0
for item in zip(a,b):
    dotProduct += item[0]*item[1]

#print(dotProduct)


#---------------------------------------------------
#

inputs =  [1, 2, 3, 2.5]
weights =  [0.2, 0.8, -0.5, 1.0]
bias = 2

dotProduct = 0
for item in zip(inputs, weights):
    dotProduct += item[0]*item[1]
# dont forget to add the bias
dotProduct += bias
#print(dotProduct)


#---------------------------------------------------
#

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

# setup empty list for appending
dotProducts = []
# for each neuron
for w, b in zip(weights, biases):
    # reset starting point cuz were visiting new neuron
    output = 0
    # for each input and weight
    for i, weight in zip(inputs, w):
        output += i*weight
    # dont forget to add the bias    
    output += b
    dotProducts.append(output)


#print(dotProducts)


#---------------------------------------------------
#

#a = np.array([[1, 2, 3]])
#b = np.array([[2, 3, 4]]).T #.T for transpose so we can calculate dot product
#print(np.dot(a,b))

# neural network that takes in group of samples (input) and outputs a group of predictions
inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

layerOutputs = np.dot(inputs, np.array(weights).T) + biases
#print(layerOutputs)

#---------------------------------------------------
#

inputs = [[1, 2, 3, 2.5],
          [2., 5., -1., 2],
          [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
# take previous layer outputs, these are our new inputs
# calculate layer 2 with second set of weights, dont forget to add second set of biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
#print(layer2_outputs)

#---------------------------------------------------
#

#nnfs.init()

X, y = spiral_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()

#--------------------------------------------------


import numpy as np
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = np.maximum(0, inputs)
#print(output)

#--------------------------------------------------

layer_outputs = [4.8, 1.21, 2.385]
E = math.e

exp_val = []
for output in layer_outputs:
    # ** is power operator
    exp_val.append(E ** output) 
#print(exp_val)

# normalize values
norm_base = sum(exp_val)
norm_val = []
for val in exp_val:
    norm_val.append(val / norm_base)
#print(norm_val)
#print(sum(norm_val))

#exp_val = np.exp("inputs")
#probabilities = exp_val / np.sum(exp_val, axis=1, keepdims=True)

layer_outputs = np.array([[4.8, 1.21, 2.385],
                          [8.9, -1.81, 0.2],
                          [1.41, 1.051, 0.026]])
print('Sum without axis')
print(np.sum(layer_outputs))
print('This will be identical to the above since default is None:')
print(np.sum(layer_outputs, axis=None))
print('Another way to think of it w/ a matrix == axis 0: columns:')
print(np.sum(layer_outputs, axis=0))

for i in layer_outputs:
    print(sum(i))

print('So we can sum axis 1, but note the current shape:')
print(np.sum(layer_outputs, axis=1))

print('Sum axis 1, but keep the same dimensions as input:')
print(np.sum(layer_outputs, axis=1, keepdims=True))


#---------------------------------------------------------------



import numpy as np
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])
class_targets = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 1, 0]])


# Probabilities for target values -
# only if categorical labels
if len(class_targets.shape) == 1:
    correct_confidences = softmax_outputs[
        range(len(softmax_outputs)),
        class_targets
    ]

# Mask values - only for one-hot encoded labels
elif len(class_targets.shape) == 2:
    correct_confidences = np.sum(
        softmax_outputs*class_targets,
        axis=1, keepdims=True)
    print(correct_confidences)
    
# Losses
neg_log = -np.log(correct_confidences)
average_loss = np.mean(neg_log)
print("average loss: {:}".format(average_loss))

print(np.e**(-np.inf))











