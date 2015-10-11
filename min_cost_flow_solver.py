'''
Minimum-cost-flow solver using cvxopt
We refer to the following tutorial: http://cvxopt.org/examples/tutorial/lp.html
'''



from cvxopt import matrix, spmatrix, sparse, solvers, spdiag
import numpy as np

__author__ = 'jeromethai'


def solver(coeff, sources, adjacency):
    print 'start the solver for the min-cost flow problem ...'
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
    print b.size
    print A.size
    print 'start iterations of the LP'
    x = np.squeeze(solvers.lp(c,A,b)['x'])
    return x.reshape((N, N))


def constraints(sources, adjacency, N):
    print 'build constraints for the min-cost flow problem ...'
    # build the constraint matrix for the problem
    b = matrix(np.concatenate((sources, -sources, np.zeros((N*N,)))))
    # build the constraint matrix
    print 'get adjacency matrix ...'
    I, J = np.where(adjacency)
    print 'construct adjacency constraints ...'
    adj = spmatrix(1., J, J + N*I, (N, N*N)) - spmatrix(1., I, J + N*I, (N, N*N))
    print 'build constraint matrix ...'
    tmp = sparse([[adj, -adj]])
    print 'still building the constraint matrix ...'
    D = spmatrix(-np.ones((N*N,)), range(N*N), range(N*N))
    print 'finish building the constraint matrix ...'
    A = sparse([[tmp, D]])
    return b, A