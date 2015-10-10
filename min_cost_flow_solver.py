'''
Minimum-cost-flow solver using cvxopt
We refer to the following tutorial: http://cvxopt.org/examples/tutorial/lp.html
'''



from cvxopt import matrix, spmatrix, sparse, solvers, spdiag
import numpy as np

__author__ = 'jeromethai'


def solver(coeff, sources, adjacency):
    # apply CVXOPT to solve the min-cost flow problem
    # coeff    : numpy array with the coefficients of the objective
    # sources  : numpy array with source terms s.t. \sum_j (x_{ji} - x_{ij}) = s_i
    # adjacency: numpy array that contains the adjacency matrix

    # dimension
    N = len(sources)
    # objective
    c = matrix(coeff.flatten())
    # build constraints
    b, A = constraints(sources, adjacency, N)
    # lp solver here
    x = np.squeeze(solvers.lp(c,A,b)['x'])
    return x.reshape((N, N))


def constraints(sources, adjacency, N):
    # build the constraint matrix for the problem
    b = matrix(np.concatenate((sources, -sources, np.zeros((N*N,)))))
    # build the constraint matrix
    I, J = np.where(adjacency)
    adj = spmatrix(1., J, J + N*I, (N, N*N)) - spmatrix(1., I, J + N*I, (N, N*N))
    A = sparse([[adj, -adj, -spdiag([1.]*(N*N))]])
    return b, A