"""
William Schlough
March 17th, 2025

Neural Network based on Harrison kinsely + Daniel Kukiela "Neural Networks from Scratch in Python, Building Neural Networks in Raw Python"

Goal is to use this neural network in conjunction with a written number databse to accurately predict written numbers. 
Expansion goal is TBD. 
"""

import numpy as np
import nnfs
import matplotlib
from matplotlib import pyplot as plt
from nnfs import datasets 
from nnfs.datasets import spiral_data

nnfs.init()


class denseLayer:
    # initialize with random values 
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) 
        self.biases = np.zeros((1, n_neurons))
    # forward pass    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class activationReLU:
    # works on a per neuron basis
    # forward pass of ReLU
    def forward(self, inputs):
        # np.maximum cuts off any values lower than 0 (y = x : x>=0 ;;; y = 0 : x<0)
        self.output = np.maximum(0, inputs)

class activationSoftmax:
    # accepts non-normalized values as inputs and outputs a probability distribution
    # to be used as confidence scores
    # example --> [0.7, 0.1, 0.2] --> 0th index is 70%, 1st index is 10%, 2nd index is 20%, 
    #   with all percents relating to likelyhood of belonging to a specific class
    # forward pass of softmax function
    def forward(self, inputs):
        # unnormalized values
        # include subraction of the largest of the inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
        self.output = probabilities

class Loss:
    # cross-entropy is used to compare a "ground truth" probability with a predictied distribution
    # commonly used loss function with softmax activation on output layer
    def calculate(self, output, y):
        # calc sample losses
        sample_losses = self.forward(output, y)
        # calc mean loss
        data_loss = np.mean(sample_losses)

        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped*y_true,
                axis=1
            )
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

def main():
    # spiral_data gives us tuple
    X, y = spiral_data(samples=100, classes=3)

    # call denseLayer and populate with 2 input and 3 outputs
    dense1 = denseLayer(2, 3)

    # create ReLU activation to be used with dense layer
    activation1 = activationReLU()
    
    # second dense layer
    # 3 input, 3 output (we are taking 3 output from previous dense layer)
    dense2 = denseLayer(3, 3)

    # second activation function
    activation2 = activationSoftmax()

    # call forward() on dense layer with data created from spiral_data() 
    dense1.forward(X)

    # forward pass through activation function (which is ReLU)
    activation1.forward(dense1.output)

    # forward pass through second dense layer 
    # takes outputs from first dense layer run through activation function
    dense2.forward(activation1.output)

    # forward pass through second activation function
    # takes output from second dense layer
    activation2.forward(dense2.output)

    # print the first 5 rows
    print(activation2.output[:5])

    loss_function = Loss_CategoricalCrossentropy()
    loss = loss_function.calculate(activation2.output, y)
    print("loss:", loss)

if __name__ == "__main__":
    main()