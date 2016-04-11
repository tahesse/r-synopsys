#/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import dot, ones, newaxis, argsort
from evaluation_metrics import mean_squared_error as mse

__all__ = ['ordinary_least_squares']


def main():
    # generate some artifical normal data
    X = np.random.rand(150, 2)
    y = np.random.rand(150, 1)
    
    # compute slope and y intercept
    w0, w = ordinary_least_squares(X, y)
    
    # some reshaping for plotting
    idc0 = argsort(X[:,0], axis=0)
    idc1 = argsort(X[:,1], axis=0)
    X0 = X[idc0,0][:,newaxis]
    X1 = X[idc1,1][:,newaxis]
    Xw = dot(np.concatenate((X0, X1), axis=1), w.T) + w0
    
    # plotting
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.hold(True)
    ax.scatter(X[:,0], X[:,1], y, c='b', depthshade=False)
    ax.plot(X0[[0,-1],0], X1[[0,-1],0], '-r', zs=Xw[[0,-1],0])
    ax.set_xlabel(r'$X_1$')
    ax.set_ylabel(r'$X_2$')
    ax.set_zlabel(r'$y$')
    print(mse(y, Xw))
    ax.set_title('Mean Squared Error: %.6f' % mse(y, Xw))


def ordinary_least_squares(X, y):
    ''' Ordinary least squares (or Linear Regression) is a method for pointwise
        parameter estimation for linear models. In this function we are dealing
        with the Frequentist approach that assumes Normal distributed data for
        which we try to minimize the squared error (see mean squared error).
        
        Input:
            X - labeled data of the form (N x M)
            y - data labels of the form (N x 1) matching the rows of X

        Output:
            w0 - the y intecept term with dimension (1 x 1)
            w  - slope vector of the form (1 x M)
    '''
    from numpy.linalg import inv
    Xtilda = np.concatenate((ones((X.shape[0], 1)), X), axis=1)
    theta = dot(dot(inv(dot(Xtilda.T, Xtilda)), Xtilda.T), y).T
    return theta[0, 0:1][newaxis, :], theta[0, 1:][newaxis, :]


if __name__ == '__main__': main()
