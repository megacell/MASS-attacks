'''Attack routing solver
'''


import cplex_interface
from cvxopt import matrix, spmatrix, sparse, spdiag, solvers
import numpy as np

__author__ = 'jeromethai'


def attack_routing_solver(network, attack_rates, k, eps=10e-8, cplex=False):
    # solver for the attack routing
    # attack_rates : fixed attack rates, solve for the availabilities and routing
    # weights      : weights for the availabilities in the objective
    solver = cplex_solver if cplex else cvxopt_solver
    flow = solver(network, attack_rates, k)
    return flow_to_availabilities_routing(network.size, flow, attack_rates, eps)


def cvxopt_solver(network, attack_rates, k):
    N = network.size
    c = matrix(np.repeat(network.weights, network.size))
    b, A = constraints(network, attack_rates, k)
    # lp solver here
    x = np.squeeze(solvers.lp(c,A,b)['x'])
    return x.reshape((N, N))


def cplex_solver(network, attack_rates, k):
    N = network.size
    open('tmp.lp', 'w').write(to_cplex_lp_file(network, attack_rates, k))
    variables, sols = cplex_interface.solve_from_file('tmp.lp', 'o')
    # reconstruct flow matrix from 'sols' returned by cplex solver 
    non_zeros = np.where(sols)
    flow = np.zeros((N,N))
    for i in non_zeros[0]:
        if variables[i][0] == 'a': continue
        a,b = [int(j) for j in variables[i][2:].split('_')]
        flow[a,b] = sols[i]
    return flow


def constraints(network, attack_rates, k):
    # construct the constraints for the attack routing problem
    N = network.size
    u = np.tile(range(N), N)
    v = np.repeat(range(N),N)
    w = np.array(range(N*N))
    # build constraint matrix
    A1 = spmatrix(np.repeat(attack_rates, N), u, w, (N, N*N))
    A2 = -spmatrix(np.repeat(attack_rates + network.rates, N), v, w, (N, N*N))

    I = np.array(range(N))
    J = I + np.array(range(N)) * N
    A3 = spmatrix(network.rates, I, J, (N, N*N))

    tmp = np.dot(np.diag(network.rates), network.routing).transpose()
    A4 = matrix(np.repeat(tmp, N, axis=1))
    A5 = -spmatrix(tmp.flatten(), v, np.tile(J, N), (N, N*N))

    A6 = A1 + A2 + A3 + A4 + A5 

    I = np.array([0]*(N-1))
    J = np.array(range(k)+range((k+1),N)) + N * k
    A7 = spmatrix(1., I, J, (1, N*N))
    A = sparse([[A6, -A6, A7, -A7, -spdiag([1.]*(N*N))]])

    tmp = np.zeros(2*N + 2 + N*N)
    tmp[2*N] = 1.
    tmp[2*N + 1] = -1.
    b = matrix(tmp)
    return b, A


def flow_to_availabilities_routing(size, flow, attack_rates, eps=10e-8):
    # convert the flow solution of the min cost flow problem back to 
    # availabilities and routing probabilities
    flow[range(size), range(size)] = 0.0
    # availabilities
    avail = np.sum(flow, 1)
    assert np.sum(avail > eps) == size
    # routing probabilities for the attacks
    tmp = np.divide(np.ones((size,)), avail)
    opt_routing = np.dot(np.diag(tmp), flow)
    return avail, opt_routing


def to_cplex_lp_file(network, attack_rates, k):
    # generate input file for CPLEX solver
    # http://lpsolve.sourceforge.net/5.5/CPLEX-format.htm
    N = network.size
    w = network.weights
    lam = network.rates + attack_rates
    tmp = np.dot(np.diag(network.rates), network.routing).transpose()
    out = 'Minimize\n  obj: '
    for i in range(k) + range(k+1, N):
        out = out + '{} a_{} + '.format(w[i], i)
    out = out[:-3] + '\nSubject To\n  '
    # equality constraints
    for i in range(k) + range(k+1, N):
        for j in range(i) + range(i+1, N):
            out = out + '{} y_{}_{} - '.format(attack_rates[j], j, i)
            out = out + '{} y_{}_{} + '.format(lam[i], i, j)
            if j != k: out = out + '{} a_{} + '.format(tmp[i,j], j)
        out = out[:-2] + '= - {}\n  '.format(tmp[i,k])
    # constraints on the a_i
    for i in range(N):
        for j in range(i) + range(i+1, N):
            out = out + 'y_{}_{} + '.format(i,j)
        if i == k:
            out = out[:-2] + '= 1.0\n  '.format(i)
        else:
            out = out[:-2] + '- a_{} = 0.0\n  '.format(i)
    # bounds
    out = out[:-2] + 'Bounds\n  '
    for i in range(N):
        for j in range(i) + range(i+1, N):
            out = out + '0 <= y_{}_{}\n  '.format(i,j)
    out = out[:-2] + 'End'
    return out


