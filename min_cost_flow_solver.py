'''
Minimum-cost-flow solver using cvxopt
We refer to the following tutorial: http://cvxopt.org/examples/tutorial/lp.html
'''


import cplex_interface
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
    print 'start iterations of the LP'
    x = np.squeeze(solvers.lp(c,A,b)['x'])
    return x.reshape((N, N))


def constraints(sources, adjacency, N):
    print 'build constraints for the min-cost flow problem ...'
    # build the constraint matrix for the problem
    b = matrix(np.concatenate((sources, -sources, np.zeros((N*N,)))))
    # build the constraint matrix
    I, J = np.where(adjacency)
    adj = spmatrix(1., J, J + N*I, (N, N*N)) - spmatrix(1., I, J + N*I, (N, N*N))
    A = sparse([[adj, -adj, spmatrix(-np.ones((N*N,)), range(N*N), range(N*N))]])
    return b, A


def cplex_solver(coeff, sources, adjacency):
    N = len(sources)
    open('tmp.lp', 'w').write(to_cplex_lp_file(coeff, sources))
    variables, sols = cplex_interface.solve_from_file('tmp.lp', 'o')
    # reconstruct flow matrix from 'sols' returned by cplex solver 
    non_zeros = np.where(sols)
    flow = np.zeros((N,N))
    for i in non_zeros[0]:
        a,b = [int(j) for j in variables[i][2:].split('_')]
        flow[a,b] = sols[i]
    return flow


def to_cplex_lp_file(coeff, sources):
    # generate input file for CPLEX solver
    # http://lpsolve.sourceforge.net/5.5/CPLEX-format.htm
    N = len(sources)
    out = 'Minimize\n  obj: '
    for i in range(N):
        for j in range(N):
            out = out + '{} x_{}_{} + '.format(coeff[i,j], i, j)
    out = out[:-3] + '\nSubject To\n  '
    # equality constraints
    for i in range(N):
        for j in range(i) + range(i+1, N):
            out = out + 'x_{}_{} - x_{}_{} + '.format(j, i, i, j)
        out = out[:-2] + '= {}\n  '.format(sources[i])
    # bounds
    out = out[:-2] + 'Bounds\n  '
    for i in range(N):
        for j in range(N):
            out = out + '0 <= x_{}_{}\n  '.format(i,j)
    out = out[:-2] + 'End'
    return out
