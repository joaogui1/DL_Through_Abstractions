import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# The neural network framework
class Perceptron:
    def __init__(self, dimension):
    # Init all weights between [-1 .. 1].
    # Each input is connected to all outputs.
    # One line per input and one column per output.
        self.weights = np.random.uniform(-1, 1, dimension)
        self.bias = np.random.uniform(-1, 1)

    def forward(self, inp):
        self.output = np.sign(np.dot(inp, self.weights) + self.bias)
        return self.output

    def updateWeights(self, inp, error, learning_rate):
        self.weights += np.dot(inp, error) * learning_rate
        self.bias += error * learning_rate

    def train(self, inputs, targets, learning_rate=0.01, epochs=50):
      # Compute deltas at each layer starting from the last one
        for _ in range(epochs):
            mse = 0.0
            for inp, target in zip(inputs, targets):
                output = self.forward(inp)
                error = target - output
                mse += error**2
                self.updateWeights(inp, error, learning_rate)
            print(mse/len(inputs))
            # print(NN.weights[0], NN.bias)

df = pd.read_csv("bug.csv", names=['x1', 'x2', 'y'])

inputs = np.asarray([[x1, x2] for x1, x2 in zip(df['x1'], df['x2'])], dtype=np.float32)
outputs = df['y'].values

# plt.plot(inputs, outputs)
# plt.show()

print(outputs)

NN = Perceptron(2)
NN.train(inputs, outputs)
