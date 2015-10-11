'''Min attack solver
'''


import min_cost_flow_solver as mcf
import numpy as np

__author__ = 'jeromethai'


def check_target_cost(target, cost, size, eps = 10e-8):
    # check that the target is the right size and positive
    assert target.shape[0] == size
    assert np.max(target) == 1.0
    assert np.min(target) >= eps
    # check that costs are the right size
    assert cost.shape[0] == size
    assert cost.shape[1] == size
    assert np.min(cost) >= eps
    


def min_cost_flow_init(network, target, cost, eps = 10e-8):
    print 'initialize paramaters of the LP ...'
    # produce the casting into a min-cost flow problem
    # compute source terms
    check_target_cost(target, cost, network.size)
    pickup_rates = np.multiply(network.rates, target)
    sources = pickup_rates - np.dot(network.routing.transpose(), pickup_rates)
    assert abs(np.sum(sources)) < eps
    # compute coefficients in the objective
    inverse_target = np.divide(np.ones((network.size,)), target)
    coeff = np.dot(np.diag(inverse_target), cost)
    return coeff, sources


def flow_to_rates_routing(size, flow, target, eps = 10e-8):
    # convert the flow solution of the min cost flow problem
    # back into rates
    # makes sure that the diagonal is equal to zero
    flow[range(size), range(size)] = 0.0
    opt_rates = np.divide(np.sum(flow, 1), target)
    # convert the flow into routing
    tmp = np.multiply(opt_rates, target)
    zeroes = np.where(tmp < eps)
    tmp[zeroes] = size - 1.
    flow[zeroes, :] = 1.
    flow[range(size), range(size)] = 0.0
    inverse_tmp = np.divide(np.ones((size,)), tmp)
    opt_routing = np.dot(np.diag(inverse_tmp), flow)
    opt_rates[opt_rates < 0.0] = 0.0
    opt_routing[opt_routing < 0.0] = 0.0
    return opt_rates, opt_routing


def min_attack_solver(network, target, cost, eps = 10e-8):
    print 'start min_attack_solver ...'
    # initialize the parameters for the min-cost-flow problem
    # it returns that optimal rates and routing probabilities
    coeff, sources = min_cost_flow_init(network, target, cost, eps)
    # runs the min-cost-flow problem
    flow = mcf.solver(coeff, sources, network.adjacency)
    opt_rates, opt_routing = flow_to_rates_routing(network.size, flow, target, eps)
    return opt_rates, opt_routing


def to_cplex_lp_file(network, target, cost, eps = 10e-8): 
    # generate input file for CPLEX solver
    # http://lpsolve.sourceforge.net/5.5/CPLEX-format.htm
    coeff, sources = min_cost_flow_init(network, target, cost, eps = 10e-8)
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
        out = out[:-2] + '= {}\n'.format(sources[i])
    out = out + 'Bounds\n  '
    # bounds
    for i in range(N):
        for j in range(N):
            out = out + '0 <= x_{}_{}\n'.format(i,j)
    out = out + 'End'
    return out
