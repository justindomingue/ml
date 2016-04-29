#!/usr/bin/python

# Learns the weights of a linear evaluation function using both neural networks and linear regression

import numpy as np
import pandas as pd
from regression import regression_gradient_descent, ridge_regression_gradient_descent, predict_output, rss
import regression
import nn

import sys

def get_numpy_data(dataframe, features, output, with_constant=False, output_factor=1):
    """ Converts a Pandas dataframe to two numpy arrays for `features` and `output`
    :param dataframe: Pandas dataframe to convert
    :param features: [String] features to extrat
    :param output: (String) name of the column containing the output
    :return: Feature columns as np matrix and output column as np array
    """
    if with_constant:
      dataframe['constant'] = 1
      features = ['constant'] + features  # add 'constant' to list of features

    # dataframe->np_array selecting only columns in `features`
    feature_matrix = dataframe.as_matrix(columns=features)
    output_array = dataframe[output].values*output_factor

    return (feature_matrix, output_array)

def sample(df, first=0.010):
    msk = np.random.rand(len(df)) < first
    train = df[msk]
    test  = df[~msk]
    return (train, test)

def shuffle(df):
    return df.iloc[np.random.permutation(len(df))]

if __name__ == "__main__":
    file = sys.argv[1]
    random_states = pd.read_csv(file)

    min_threshold, max_threshold = 0.2, 0.8
    min_turn, max_turn = 0, 15

    # random_states = random_states[random_states['turn_number']>min_turn | random_states['turn_number']<max_turn]

    # random_states = random_states[(random_states['outcome']<0.1) | (random_states['outcome']>0.9)]
    train_data, test_data = sample(random_states)
    # train_data = train_data[(train_data['outcome']<min_threshold) | (train_data['outcome']>max_threshold)]

    print 'Training on file {} containing {} data points'.format(file, len(train_data))

    # define parameters for the model
    model_features = [
        # 'p0_pit0'
        'p0_pit0', 'p0_pit1', 'p0_pit2', 'p0_pit3', 'p0_pit4', 'p0_pit5', 'p0_pit6', 'p0_pit7', 'p0_pit8', 'p0_pit9', 'p0_pit10', 'p0_pit11', 'p0_pit12', 'p0_pit13', 'p0_pit14', 'p0_pit15', 'p0_pit16', 'p0_pit17', 'p0_pit18', 'p0_pit19', 'p0_pit20', 'p0_pit21', 'p0_pit22', 'p0_pit23', 'p0_pit24', 'p0_pit25', 'p0_pit26', 'p0_pit27', 'p0_pit28', 'p0_pit29', 'p0_pit30', 'p0_pit31'
        # 'p1_pit0', 'p1_pit1', 'p1_pit2', 'p1_pit3', 'p1_pit4', 'p1_pit5', 'p1_pit6', 'p1_pit7', 'p1_pit8', 'p1_pit9', 'p1_pit10', 'p1_pit11', 'p1_pit12', 'p1_pit13', 'p1_pit14', 'p1_pit15', 'p1_pit16', 'p1_pit17', 'p1_pit18', 'p1_pit19', 'p1_pit20', 'p1_pit21', 'p1_pit22', 'p1_pit23', 'p1_pit24', 'p1_pit25', 'p1_pit26', 'p1_pit27', 'p1_pit28', 'p1_pit29', 'p1_pit30', 'p1_pit31'
        ]
    my_output = 'outcome'

    model = 'lr'
    model = 'nn'

# Neural Network
    if model == 'nn':

      (X, y) = get_numpy_data(train_data, model_features, my_output, with_constant=False, output_factor=1)
      y =  y.reshape(y.shape[0], 1) # make sure y has 2 dimensions

      iterations=10
      neurons = [ X.shape[1], 1 ]

      weights = nn.neural_network_with_shapes(X, y, neurons, iterations=iterations, verbose=True, plot=False)

      (test_X, test_y) = get_numpy_data(test_data, model_features, my_output)
      test_predictions = nn.predict(test_X, weights)

      train_r2 = regression.rsquared(nn.predict(X, weights), y)
      test_r2 = regression.rsquared(test_predictions, test_y)

      train_rss = regression.avg_prediction_error(nn.predict(X, weights), y)
      test_rss  = regression.avg_prediction_error(test_predictions, test_y)

      print weights

      print weights
      print 'Train r2: {}, test r2: {}'.format(train_r2, test_r2)
      print 'train rss: {}, test rss: {}'.format(train_rss, test_rss)

# Linear Regression
    elif model == 'lr':
      (feature_matrix, output) = get_numpy_data(train_data, model_features, my_output, with_constant=True, output_factor=100)
      # initial_weights = np.ones(feature_matrix.shape[1])
      initial_weights = 2 * np.random.random(feature_matrix.shape[1]) - 1
      # initial_weights = [120.86201553,-1.16925753,-1.17057485,-1.2383677,-1.09748451 ,-1.31677655,-1.39924277,-1.54396165,-1.6524178,-1.74319548 ,-1.90869109,-1.89703014,-1.96203515,-1.8102061,-1.91907187 ,-1.77835281,-1.76678584,-1.79334199,-1.83527532,-1.77566051 ,-1.83925756,-1.68633652,-1.8537131,-1.92154856,-1.79306029 ,-1.63872948,-1.52182232,-1.50174726,-1.34444211,-1.33287026 ,-1.32522779,-1.41841499,-1.3169736,]
      step_size = 4e-7  # determined experimentally 4e-6 or 4e-7 works great
      tolerance = 1e8 # for simple linear regression (not regularization)
      max_iterations=1000
      l2_penalty=0.05 # determined using cross validation

      # cross validation
      # plot = False
      # l2_penalty_space = np.linspace(1e-5,1e-1,num=5)
      # optimal = regression.run_ridge_cross_validate(l2_penalty_space, feature_matrix, output, k=3, plot=plot, step_size=step_size, max_iterations=max_iterations)
      # print 'From {}, optimal is {}'.format(l2_penalty_space, optimal)

      # learn the weights
      # weights = regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)
      weights = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size=step_size, l2_penalty=l2_penalty, max_iterations=max_iterations, verbose=True, plot=False)

      (test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output, with_constant=True, output_factor=100)
      test_predictions = predict_output(test_feature_matrix, weights)

      train_r2 = regression.rsquared(predict_output(feature_matrix, weights), output)
      test_r2 = regression.rsquared(test_predictions, test_output)

      train_rss = regression.avg_prediction_error(predict_output(feature_matrix, weights), output)
      test_rss  = regression.avg_prediction_error(test_predictions, test_output)

      print weights
      print 'Train r2: {}, test r2: {}'.format(train_r2, test_r2)
      print 'train rss: {}, test rss: {}'.format(train_rss, test_rss)
