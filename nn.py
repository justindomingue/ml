#!/usr/bin/python

# Neural Networks
# source: http://iamtrask.github.io/2015/07/12/basic-python-network/

import math
import numpy as np

def sigmoid(t, deriv=False):
    if deriv:
        return t*(1-t)
    return 1/(1+np.exp(-t))

def nn_2layers(X, y, iterations=100):
    """ Trains a simple 2-layer neural network on feature matrix `X` and outcomes `y`
    :param X: input feature matrix where rows are observation and columns are features
    :param y: outcome associated to each observation in `X`
    :return: 1D vector of weights
    """
    num_features = X.shape[1]

    # initialize synapse 0 randomly with mean 0
    syn0 = 2*np.random.random((num_features,1))-1

    for _ in xrange(iterations):

        # forward propagation
        l1 = sigmoid(np.dot(X, syn0))

        # prediction error for layer 1
        l1_error = l1 - y

        # adjust the weights by multiplying error by derivative of sigmoid at l1
        l1_delta = l1_error * sigmoid(l1, True)
        syn0 += np.dot(X.T, l1_delta)

    return [syn0]

def neural_network_with_shapes(X, y, neurons_per_layer, iterations=100000, verbose=False):
    """ Trains an `n`-layer neural network on feature matrix `X` and outcomes `y`
    :param X: input feature matrix where rows are observation and columns are features
    :param y: outcome associated to each observation in `X`
    :param neurons_per_layer: array of integers representing the number of neurons for each layer
    :return: matrix of weights


    Usage:
        To train a neural network with one input layer, one hidden layer and one output layer,

        # For reproducibility, set the random seed
        >>> np.random.seed(1)

        First, generate training data
        >>> X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])     # observations
        >>> y = np.array([0,0,1,1]).reshape(4,1)                # output

        Then, set the network shape:
            3: input layer with 3 neurons    (features)
            5: hidden layer with 4 neurons
            1: output layer with 1 neurons   (predicted value)
        >>> neurons = [ 3, 5, 1, ]

        From that neuron configuration, the network will look like this:
            (ascii is failing me here... every neuron in the input layer is connected
            to every neuron in the hdiden layer and the same for the hidden and output
            layers).

          input   hidden   output
            O-------O-----\
                \---O------\
            O-------O-------O
                /---O------/
            O-------O-----/

        Now the fun part. Build and train a neural network with the structures you just defined.
        `neural_network_with_shapes()` with return the matrix of weights (all the synapses)
        >>> weights = neural_network_with_shapes(X, y, neurons_per_layer=neurons, iterations=100)

        To use the network, call `forward_propagation` with the weights you just trained.
        >>> print predict(X, weights)
        [[ 0.09494851]
         [ 0.06620708]
         [ 0.922808  ]
         [ 0.90488084]]
    """

    num_synapses = len(neurons_per_layer)-1  # number of synapses in the nn (consequently, there are `num_synapses+1` layers.
    synapses = []                            # list of matrix of weights connecting layer i and i+1

    # initialize weights randomly with mean 0
    for i in xrange(num_synapses):
        shape = (neurons_per_layer[i], neurons_per_layer[i+1])
        synapses.append(2*np.random.random(shape)-1)

    # train the network
    for j in xrange(iterations):

        # FORWARD PROPAGATION
        layers = forward_propagation(X, synapses)

        # BACK PROPAGATION

        # first process last layer using `y`
        ln = layers[-1]
        ln_error = y - ln

        if verbose and j%10000 == 0:
            print "Error: {}".format(np.mean(np.abs(ln_error)))

        delta = ln_error * sigmoid(ln, deriv=True)

        # then update every preceding layers
        for i in reversed(xrange(num_synapses)):
            l_i = layers[i]
            syn_i = synapses[i]

            # update the synapse based on succeeding layers
            syn_i += l_i.T.dot(delta)

            # if we've reached the input layer
            if i == 0:
                break

            # how much did `l_i` contribute to the `l_i+1` error
            error = delta.dot(syn_i.T)

            # error weighted derivative
            delta = error * sigmoid(l_i, deriv=True)

    return synapses

def forward_propagation(X, synapses):
    """Performs feed forward propagation through layers 0, 1, ..., len(synapses)-1"""
    layers = [X]
    for i in xrange(len(synapses)):
        layers.append(sigmoid(np.dot(layers[i], synapses[i])))
    return layers

def predict(X, weights):
    return forward_propagation(X, weights)[-1]
