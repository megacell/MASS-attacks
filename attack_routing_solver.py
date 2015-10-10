'''Attack routing solver
'''

from cvxopt import matrix, spmatrix, sparse, spdiag, solvers
import numpy as np

__author__ = 'jeromethai'


def attack_routing_solver(network, attack_rates, k, eps = 10e-8):
    # solver for the attack routing
    # attack_rates : fixed attack rates, solve for the availabilities and routing
    # weights      : weights for the availabilities in the objective
    N = network.size
    c = matrix(np.repeat(network.weights, network.size))
    b, A = constraints(network, attack_rates, k)
    # lp solver here
    x = np.squeeze(solvers.lp(c,A,b)['x'])
    flow = x.reshape((N, N))
    return flow_to_availabilities_routing(N, flow, attack_rates, eps)


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


def flow_to_availabilities_routing(size, flow, attack_rates, eps = 10e-8):
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


