#!/usr/bin/python

# Linear regression library
# Implements gradient descent and extensions of it
#
# The excellent Coursera online course on Linear Regression by Emily Fox and Carlos Guestrin
# from the University of Washington (https://www.coursera.org/learn/ml-regression) has
# inspired most of the implementations

from math import sqrt
import numpy as np

# ===============================
#       LINEAR REGRESSION
# ===============================

def feature_derivative(errors, feature):
    """ Computes the derivative of the linear regression cost function
    :param errors:  numpy array
    :param feature: numpy array of the same length as errors
    :return: return the product of the two vectors as derivative
    """
    derivative = 2 * np.dot(errors, feature)
    return(derivative)

def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)

    while not converged:
        predictions = predict_output(feature_matrix, weights)
        errors = predictions - output

        gradient_sum_squares = 0
        for i in range(len(weights)):   # update each feature's weight
            derivative = feature_derivative(errors, feature_matrix[:,i])
            gradient_sum_squares += derivative**2

            weights[i] -= derivative * step_size

        gradient_magnitude = sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True

    return weights

# ===============================
#    RIDGE LINEAR REGRESSION
# ===============================

def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    if feature_is_constant:
        derivative = 2 * np.dot(errors, feature)
    else:
        derivative = 2 * np.dot(errors, feature) + 2 * l2_penalty*weight

    return derivative

def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations=100, verbose=True):
    weights = np.array(initial_weights)
    batch_size = 256

    j = []
    for _ in range(max_iterations):

        # Shuffle data
        feature_matrix, output = shuffle_in_unison_inplace(feature_matrix, output)

        # Split data into batches of size `batch_size`
        batch_features, batch_outputs = np.array_split(feature_matrix, batch_size), np.array_split(output, batch_size)

        # Update weights for each batch
        for batch_id in range(len(batch_features)):
            batch_feature_matrix = batch_features[batch_id]
            batch_output = batch_outputs[batch_id]

            # compute predictions and errors
            predictions = predict_output(batch_feature_matrix, weights)
            errors = predictions - batch_output

            j.append(rss(predictions, batch_output))

            if verbose:
                print('Error: {}'.format(rss(predictions, batch_output)))

            for i in xrange(len(weights)):   # update each feature's weight
                feature_is_constant = i==0  # do not regularize the intercept
                derivative = feature_derivative_ridge(errors, batch_feature_matrix[:,i], weights[i], l2_penalty, feature_is_constant)

                weights[i] = weights[i] - derivative * step_size

    # import matplotlib.pyplot as plot
    # x = range(max_iterations)
    # y = j
    # plot.plot(x,y)
    # plot.show()
    return(weights)

# ----- CROSS VALIDATION ------

def ridge_k_fold_cross_validation(k, l2_penalty, feature_matrix, output, step_size=1e-12, max_iterations=100):
    ''' Performs k-fold cross validation
    After dividing the training set into k segments of roughly equal size, measures the
    validation error with each of the segments designated as the validation set and returns
    the average as an estimate of the generalization error.

    :param k: number of folds
    :param l2_penalty: parameter to test
    :param feature_matrix: shuffled feature matrix
    :param output: output vector
    :return: average validation error using k segments as validation sets
    '''

    # set some paramaters
    n = len(feature_matrix)

    # #columns initial weights all set to 0
    initial_weights = np.zeros(feature_matrix.shape[1])

    running_val_error = .0
    for i in range(0,k):
        # compute starting and ending indices of segment i
        start = n*i/k
        end   = n*(i+1)/k-1

        # form validation set by taking a slice (start:end+1)
        validation_feature_matrix = feature_matrix[start:end+1]
        validation_output = output[start:end+1]


        # form training set by taking everything but the val_set
        first = feature_matrix[0:start]
        second   = feature_matrix[end+1:n]
        train_set = np.concatenate((first, second), axis=0)
        train_output = np.concatenate([output[0:start], output[end+1:n]], axis=0)

        # train a ridge regression linear model
        # model is just a list weights
        model = ridge_regression_gradient_descent(
                train_set,
                train_output,
                initial_weights,
                l2_penalty=l2_penalty,
                step_size=step_size,
                max_iterations=max_iterations,
                verbose=False)

        # compute the validation error
        error = rss(predict_output(validation_feature_matrix, model), validation_output)
        running_val_error += error

    return running_val_error/k

def run_ridge_cross_validate(l2_penalty_space, feature_matrix, output, step_size=1e-12, max_iterations=1000, plot=False):
    """ Find the model that minimizes the average validation error
    :param l2_penalty_space: Array. l2_penalty values to try, e.g. np.logspace(1,7,num=13)
    :param feature_matrix:
    :param output:
    :param plot: if True, plots the l2_penalty space in the x axis and the cross-validation error in the y axis
    :return optimal l2_penalty in `l2_penalty_space`
    """
    errors=[]
    lowest_error = float("inf")
    optimal_l2_penalty = None
    for l2_penalty in l2_penalty_space:
        error = ridge_k_fold_cross_validation(10, l2_penalty, feature_matrix, output, step_size=step_size, max_iterations=max_iterations)
        errors.append(error)

        if error < lowest_error:
            lowest_error = error
            optimal_l2_penalty = l2_penalty

    if plot:
        import matplotlib.pyplot as plt
        plt.xscale('log')
        plt.plot(l2_penalty_space, errors)

    return optimal_l2_penalty


# ===============================
#            UTILITIES
# ===============================

def predict_output(feature_matrix, weights):
    """ Predicts the output vector given `feature_matrix` and `weights`
    :param feature_matrix: numpy matrix containing the features as columns
    :param weights: weights is a corresponding numpy array
    :return:  predictions vector
    """
    return np.dot(feature_matrix, weights)

def rss(predictions, actual):
    return ((predictions-actual)**2).sum()

def rsquared(predictions, actual):
    """ Quotient of the variances of the fitted values and observed values of the dependent variable """
    ybar = 1/len(actual)*actual.sum()

    var_yhat = predictions.var()
    var_y    = actual.var()
    return var_yhat/var_y

def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]