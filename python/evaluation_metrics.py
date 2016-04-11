#/usr/bin/env python3.5
# -*- coding: utf-8 -*-

from numpy import dot, squeeze

__all__ = ['mean_squared_error']


def main():
    pass


def mean_squared_error(y, yhat):
    ''' Measures the mean sqaured error thus yielding the bias-variance 
        tradeoff of a model incorporated in the predictions {yhat}. The mean
        squared error, as the name already states, computes the expected 
        squared error E[ (y - yhat)² ].
        
        Input:
            y    - data of the form (N x 1) used as true labels
            yhat - data of the form (N x 1) used as approximated labels

        Output:
            mse  - error of the form (1 x 1) with mse = bias + var + noise
    '''
    err = y - yhat
    return squeeze(dot(err.T, err))/y.shape[0]



if __name__ == '__main__': main()
