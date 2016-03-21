#!/usr/bin/python

import numpy as np
import pandas as pd
from regression import regression_gradient_descent, ridge_regression_gradient_descent, predict_output, rss
import regression

def get_numpy_data(dataframe, features, output):
    """ Converts a Pandas dataframe to two numpy arrays for `features` and `output`
    :param dataframe: Pandas dataframe to convert
    :param features: [String] features to extrat
    :param output: (String) name of the column containing the output
    :return: Feature columns as np matrix and output column as np array
    """
    dataframe['constant'] = 1
    features = ['constant'] + features  # add 'constant' to list of features

    # dataframe->np_array selecting only columns in `features`
    feature_matrix = dataframe.as_matrix(columns=features)
    output_array = dataframe[output].values*100

    return (feature_matrix, output_array)

def sample(df, first=0.8):
    msk = np.random.rand(len(df)) < first
    train = df[msk]
    test  = df[~msk]
    return (train, test)

def shuffle(df):
    return df.iloc[np.random.permutation(len(df))]

if __name__ == "__main__":
    random_states = pd.read_csv('data/random_states_complete.csv')
    random_states = random_states[(random_states['outcome']<0.3) | (random_states['outcome']>0.7)]
    train_data, test_data = sample((random_states))

    print 'Training on {} data points'.format(len(train_data))

    # define parameters for the model
    model_features = [
        'p0_pit0', 'p0_pit1', 'p0_pit2', 'p0_pit3', 'p0_pit4', 'p0_pit5', 'p0_pit6', 'p0_pit7', 'p0_pit8', 'p0_pit9', 'p0_pit10', 'p0_pit11', 'p0_pit12', 'p0_pit13', 'p0_pit14', 'p0_pit15', 'p0_pit16', 'p0_pit17', 'p0_pit18', 'p0_pit19', 'p0_pit20', 'p0_pit21', 'p0_pit22', 'p0_pit23', 'p0_pit24', 'p0_pit25', 'p0_pit26', 'p0_pit27', 'p0_pit28', 'p0_pit29', 'p0_pit30', 'p0_pit31'
        # ,'p1_pit0', 'p1_pit1', 'p1_pit2', 'p1_pit3', 'p1_pit4', 'p1_pit5', 'p1_pit6', 'p1_pit7', 'p1_pit8', 'p1_pit9', 'p1_pit10', 'p1_pit11', 'p1_pit12', 'p1_pit13', 'p1_pit14', 'p1_pit15', 'p1_pit16', 'p1_pit17', 'p1_pit18', 'p1_pit19', 'p1_pit20', 'p1_pit21', 'p1_pit22', 'p1_pit23', 'p1_pit24', 'p1_pit25', 'p1_pit26', 'p1_pit27', 'p1_pit28', 'p1_pit29', 'p1_pit30', 'p1_pit31'
        ]
    my_output = 'outcome'
    (feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
    initial_weights = np.append([1], np.ones(len(model_features)))

    step_size = 4e-7
    tolerance = 1e8 # for simple linear regression (not regularization)
    max_iterations=10000
    l2_penalty=0.01 # determined using cross validation

    # cross validation
    # plot = False
    # l2_penalty_space = np.logspace(np.log10(1e-2),np.log(1e3),num=10)
    # optimal = regression.run_ridge_cross_validate(l2_penalty_space, feature_matrix, output, plot=plot, step_size=step_size, max_iterations=max_iterations)
    # print 'From {}, optimal is {}'.format(l2_penalty_space, optimal)

    # learn the weights
    # weights = regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)
    weights = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size=step_size, l2_penalty=l2_penalty, max_iterations=max_iterations, verbose=False)

    (test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)
    test_predictions = predict_output(test_feature_matrix, weights)

    train_r2 = regression.rsquared(predict_output(feature_matrix, weights), output)
    test_r2 = regression.rsquared(test_predictions, test_output)

    print weights
    print 'Train r2: {}, test r2: {}'.format(train_r2, test_r2)
