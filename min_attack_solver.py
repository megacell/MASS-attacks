'''
Min attack solver: 
Optimizing for the Optimal Attack problem with the availabilities fixed
'''

import cplex_interface
import min_cost_flow_solver as mcf
import numpy as np

__author__ = 'jeromethai'

class MinAttackSolver:
    # class for computing the min attacks with the availabilities fixed
    def __init__(self, network, target_availabilities, cost, eps=10e-8, cplex=False):
        self.network = network
        self.a = target_availabilities # availabilities are fixed
        self.cost = cost # cost on the rates of attacks
        self.phi = network.rates # rates before the attacks
        self.delta = network.routing # routing prob. before attacks
        self.eps = eps
        self.cplex = cplex
        self.N = network.size
        self.check()


    def check(self):
        # check that the target is the right size and positive
        assert self.a.shape[0] == self.N, 'availabilities do not match network size'
        assert np.max(self.a) == 1.0, 'availabilities not <= 1.0'
        assert np.min(self.a) >= self.eps, 'availabilities not positive'
        # check that costs are the right size
        assert self.cost.shape[0] == self.N, 'costs do not match network size on 0 axis'
        assert self.cost.shape[1] == self.N, 'costs do not match network size on 1 axis'
        assert np.min(self.cost) >= self.eps, 'costs not positive'



    def min_cost_flow_init(self):
        print 'initialize paramaters of the LP ...'
        # produce the casting into a min-cost flow problem
        # compute source terms
        pickup_rates = np.multiply(self.phi, self.a)
        sources = pickup_rates - np.dot(self.delta.transpose(), pickup_rates)
        assert abs(np.sum(sources)) < self.eps
        # compute coefficients in the objective
        inverse_target = np.divide(np.ones((self.N,)), self.a)
        coeff = np.dot(np.diag(inverse_target), self.cost)
        return coeff, sources


    def flow_to_rates_routing(self, flow):
        # convert the flow solution of the min cost flow problem
        # back into rates
        # makes sure that the diagonal is equal to zero
        N = self.N
        flow[range(N), range(N)] = 0.0
        opt_rates = np.divide(np.sum(flow, 1), self.a)
        # convert the flow into routing
        tmp = np.multiply(opt_rates, self.a)
        zeroes = np.where(tmp < self.eps)
        tmp[zeroes] = N - 1.
        flow[zeroes, :] = 1.
        flow[range(N), range(N)] = 0.0
        inverse_tmp = np.divide(np.ones((N,)), tmp)
        opt_routing = np.dot(np.diag(inverse_tmp), flow)
        opt_rates[opt_rates < 0.0] = 0.0
        opt_routing[opt_routing < 0.0] = 0.0
        return opt_rates, opt_routing


    def solve(self):
        print 'start min_attack_solver ...'
        # initialize the parameters for the min-cost-flow problem
        # it returns that optimal rates and routing probabilities
        coeff, sources = self.min_cost_flow_init()
        # runs the min-cost-flow problem
        solver = mcf.cplex_solver if self.cplex else mcf.solver
        flow = solver(coeff, sources, self.network.adjacency)
        return self.flow_to_rates_routing(flow)
