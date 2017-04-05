#/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import dot, ones, newaxis, argsort, eye, squeeze
from evaluation_metrics import mean_squared_error as mse

__all__ = ['ordinary_least_squares']


def main():
    # generate some artifical normal data
    X = np.random.rand(150, 2)
    X[:,0] = (X[:,0]**2)
    X[:,1] = (X[:,1]**3)
    y = np.random.rand(150, 1)
    
    # compute slope and y intercept
    w0, w = ordinary_least_squares(X, y)
    
    v0, v = ridge_regression(X, y, ridge=1e-6)

    
    #print(X.shape, X[:,0].shape)
    i = np.argsort(X[:,0])
    j = np.argsort(X[:,1])
    Xs = np.concatenate((X[i,0][:,newaxis], X[j,1][:,newaxis]), axis=1)
    
    # some reshaping for plottin
    Xw = dot(Xs, w.T) + w0
    Xv = dot(Xs, v.T) + v0

    # compute surface plot quantities
    x1, x2 = np.meshgrid(np.arange(0, X[:,0].max(axis=0), X[:,0].max(axis=0)/150),\
                         np.arange(0, X[:,1].max(axis=0), X[:,1].max(axis=0)/150))
    Zols = w[0,0]*x1+w[0,1]*x2+w0
    Zrr  = v[0,0]*x1+v[0,1]*x2+v0

    # plotting
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=16)
    fig = plt.figure()
    
    # Ordinary Least Squares plot
    ax = fig.add_subplot(211, projection='3d')
    ax.scatter(X[:,0], X[:,1], y, c='r', marker='D', depthshade=False,\
               facecolor=(0,0,0,0), edgecolor='None')
    ax.plot_surface(x1, x2, Zols, color='b', label='Ordinary Least Squares',\
                   facecolor=(0,0,0,0), edgecolor='None', antialiased=True)
    ax.set_xlabel(r'$X_1$')
    ax.set_ylabel(r'$X_2$')
    ax.set_zlabel(r'$y$')
    ax.set_title('Ordinary Least Squares - Mean Squared Error: %.6f' % mse(y, Xw))
    
    # Ridge Regression plot
    ax = fig.add_subplot(212, projection='3d')
    ax.scatter(X[:,0], X[:,1], y, c='r', marker='D', depthshade=False,\
               facecolor=(0,0,0,0), edgecolor='None')
    ax.plot_surface(x1, x2, Zrr, color='g', label='Ridge Regression',\
                   facecolor=(0,0,0,0), edgecolor='None', antialiased=True)
    ax.set_xlabel(r'$X_1$')
    ax.set_ylabel(r'$X_2$')
    ax.set_zlabel(r'$y$')
    ax.set_title('Ridge Regression - Mean Squared Error: %.6f' % mse(y, Xv))
    #ax.legend()
    plt.show(block=True)


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


def ridge_regression(X, y, ridge=1e-16):
    ''' Ridge Regression
        
        Input:
            X - labeled data of the form (N x M)
            y - data labels of the form (N x 1) matching the rows of X

        Output:
            w0 - the y intecept term with dimension (1 x 1)
            w  - slope vector of the form (1 x M)
    '''
    from numpy.linalg import inv
    Xtilda = np.concatenate((ones((X.shape[0], 1)), X), axis=1)
    theta = dot(dot(inv(dot(Xtilda.T, Xtilda)+ridge*eye(Xtilda.shape[1])), Xtilda.T), y).T
    return theta[0, 0:1][newaxis, :], theta[0, 1:][newaxis, :]


if __name__ == '__main__':
    main()
