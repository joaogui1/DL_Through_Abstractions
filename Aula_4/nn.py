import numpy as np

# Activation functions
def tanh(x):
    return np.tanh(x)

# Derivative of tanh from its output
def dtanh(y):
    return 1 - y ** 2

# The neural network framework
class Layer:
    def __init__(self, num_inputs, num_outputs):
    # Init all weights between [-1 .. 1].
    # Each input is connected to all outputs.
    # One line per input and one column per output.
        self.weights = np.random.uniform(-1, 1, (num_inputs, num_outputs))

    def forward(self, input):
        self.output = tanh(input.dot(self.weights))
        return self.output

    def computeGradient(self, error):
        self.delta = error * dtanh(self.output)
        # Returns the gradient
        return self.delta.dot(self.weights.T)

    def updateWeights(self, input, learning_rate):
        self.weights += input.T.dot(self.delta) * learning_rate

class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backprop(self, input, error, learning_rate):
        # Compute deltas at each layer starting from the last one
        for layer in reversed(self.layers):
            error = layer.computeGradient(error)

    # Update the weights
        for layer in self.layers:
            layer.updateWeights(input, learning_rate)
            input = layer.output