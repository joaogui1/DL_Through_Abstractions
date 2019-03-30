import os
import random
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as mpl

def get_data(datafile):
    data = pd.read_csv(os.getcwd() + '/datasets/' + datafile)
    y = data['0']
    X = data.ix[:,'1':]

    targets = np.zeros((y.shape[0], 3))
    for i in range(y.shape[0]):
        targets[i][y[i] - 1] = 1

    return X.values, targets

def get_data_reg(datafile):
    data = pd.read_csv(os.getcwd() + '/datasets/' + datafile)
    y = data.loc[:, '68':'69']
    X = data.loc[:, :'67']

    # targets = np.zeros((y.shape[0], 3))
    # for i in range(y.shape[0]):
    #     targets[i][y[i] - 1] = 1

    return X.values, y.values

def normalize(X):
    x = X
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return x_scaled
    #(X - X.min())/(X.max() - X.min())

class Mlp():

    def __init__(self, size_layers, bias_flag=True):

        self.size_layers = size_layers
        self.n_layers    = len(size_layers)
        self.bias_flag   = bias_flag

        # Ramdomly initialize theta (MLP weights)
        self.initialize_theta_weights()

    def train(self, X, Y, iterations=300, reset=False, eta = 0.5, momentum = 0.0):

        n_examples = Y.shape[0]
        old_delta = 0*self.unroll_weights(self.theta_weights)

        if reset:
            self.initialize_theta_weights()
        for iteration in range(iterations):
            self.gradients = self.backpropagation(X, Y)
            self.gradients_vector = self.unroll_weights(self.gradients)
            self.theta_vector = self.unroll_weights(self.theta_weights)
            self.theta_vector += momentum*old_delta - eta*self.gradients_vector
            self.theta_weights = self.roll_weights(self.theta_vector)

    def predict(self, X):

        A , Z = self.feedforward(X)
        Y_hat = A[-1]
        return Y_hat

    def classify(self, X):

        X_hat = self.predict(X)
        y = [np.argmax(x) for x in X_hat]
        return y

    def initialize_theta_weights(self):
        self.theta_weights = []
        size_next_layers = self.size_layers.copy()
        size_next_layers.pop(0)
        for size_layer, size_next_layer in zip(self.size_layers, size_next_layers):
            if self.bias_flag:
                theta_tmp = ((np.random.rand(size_next_layer, size_layer + 1) * 2.0 ) - 1)
            else:
                theta_tmp = ((np.random.rand(size_next_layer, size_layer) * 2.0 ) - 1)

            self.theta_weights.append(theta_tmp)
        return self.theta_weights

    def backpropagation(self, X, Y):

        g_dz = lambda x: self.sigmoid_derivative(x)

        n_examples = X.shape[0]
        # Feedforward
        A, Z = self.feedforward(X)

        # Backpropagation
        deltas = [None] * self.n_layers
        deltas[-1] = (A[-1] - Y)
        # For the second last layer to the second one
        for ix_layer in np.arange(self.n_layers - 1 - 1 , 0 , -1):
            theta_tmp = self.theta_weights[ix_layer]
            if self.bias_flag:
                # Removing weights for bias
                theta_tmp = np.delete(theta_tmp, np.s_[0], 1)
            deltas[ix_layer] = (np.matmul(theta_tmp.transpose(), deltas[ix_layer + 1].transpose() ) ).transpose() * g_dz(Z[ix_layer])

        # Compute gradients
        gradients = [None] * (self.n_layers - 1)
        for ix_layer in range(self.n_layers - 1):
            grads_tmp = np.matmul(deltas[ix_layer + 1].transpose() , A[ix_layer])
            grads_tmp = grads_tmp / n_examples

            gradients[ix_layer] = grads_tmp;
        return gradients

    def feedforward(self, X):
        '''
        Implementation of the Feedforward
        '''
        g = lambda x: self.sigmoid(x)

        A = [None] * self.n_layers
        Z = [None] * self.n_layers
        input_layer = X

        for ix_layer in range(self.n_layers - 1):
            n_examples = input_layer.shape[0]
            if self.bias_flag:
                # Add bias element to every example in input_layer
                input_layer = np.concatenate((np.ones([n_examples ,1]) ,input_layer), axis=1)
            A[ix_layer] = input_layer
            # Multiplying input_layer by theta_weights for this layer
            Z[ix_layer + 1] = np.matmul(input_layer,  self.theta_weights[ix_layer].transpose() )
            # Activation Function
            output_layer = g(Z[ix_layer + 1])
            # Current output_layer will be next input_layer
            input_layer = output_layer

        A[self.n_layers - 1] = output_layer
        return A, Z


    def unroll_weights(self, rolled_data):
        '''
        Unroll a list of matrices to a single vector
        Each matrix represents the Weights (or Gradients) from one layer to the next
        '''
        unrolled_array = np.array([])
        for one_layer in rolled_data:
            unrolled_array = np.concatenate((unrolled_array, one_layer.flatten(1)) )
        return unrolled_array

    def roll_weights(self, unrolled_data):
        '''
        Unrolls a single vector to a list of matrices
        Each matrix represents the Weights (or Gradients) from one layer to the next
        '''
        size_next_layers = self.size_layers.copy()
        size_next_layers.pop(0)
        rolled_list = []
        if self.bias_flag:
            extra_item = 1
        else:
            extra_item = 0
        for size_layer, size_next_layer in zip(self.size_layers, size_next_layers):
            n_weights = (size_next_layer * (size_layer + extra_item))
            data_tmp = unrolled_data[0 : n_weights]
            data_tmp = data_tmp.reshape(size_next_layer, (size_layer + extra_item), order = 'F')
            rolled_list.append(data_tmp)
            unrolled_data = np.delete(unrolled_data, np.s_[0:n_weights])
        return rolled_list

    def sigmoid(self, z):

        result = 1.0 / (1.0 + np.exp(-z))
        return result

    def sigmoid_derivative(self, z):

        result = self.sigmoid(z) * (1 - self.sigmoid(z))
        return result

    def classification(self, X, y):
        output = self.classify(X)
        error = 0
        for i in range(len(output)):
            error += 1 - y[i][output[i]]

        return 1.0 - (error/len(output))

    def regression(self, X, y):
        output = self.predict(X)
        error = 0
        for i in range(len(output)):
            error += np.dot(y[i] - output[i], y[i] - output[i])
        error /= len(output)
        error = error**0.5

        return error

accuracy = [[], []]

# x_ax = range(1, 40)
# for et in x_ax:
#     #print(list(zip(X, y)))
#     mean = [0.0, 0.0]
#     for i in range(30):
#         X, y = get_data_reg("wine.csv")
#         X = normalize(X)
#         ANN = Mlp([len(X[0]), 10, len(y[0])], et)
#         ANN.train(X, y)
#         mean[0] += ANN.classification(X, y)
#         X, y = get_data_reg("wine_test.csv")
#         X = normalize(X)
#         mean[1] += ANN.classification(X, y)
#         print(et)

#     accuracy[0].append(mean[0]/30.0)
#     accuracy[1].append(mean[1]/30.0)
#     #
#     # print(mean[0]/50.0)
#     # print(mean[1]/50.0)

# mpl.plot(x_ax, accuracy[0], 'b', x_ax, accuracy[1], 'r')
# mpl.plot(x_ax, accuracy[0], label='Training dataset')
# mpl.plot(x_ax, accuracy[1], label='Test dataset')
# mpl.legend()
# mpl.title("Mean square error with different number of epochs")
# mpl.ylabel("Accuracy")
# mpl.xlabel("Number of epochs")
# mpl.ylim(0.0, 1.0)
# mpl.show()
