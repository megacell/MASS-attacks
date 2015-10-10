'''Tests for the min)attack_solver
'''

import unittest
import numpy as np
from utils import generate_uniform, generate_asymmetric
from min_attack_solver import min_cost_flow_init, flow_to_rates_routing, \
    min_attack_solver
import MAS_network as MAS


__author__ = 'jeromethai'


class TestMinAttackSolver(unittest.TestCase):
    

    def test_min_cost_flow_init(self):
        eps = 10e-8
        # generate asymmetric network
        rates, routing, travel_times = generate_asymmetric()
        network = MAS.Network(rates, routing, travel_times)
        # target for the availabilities
        target = np.array([.5, 1., 1.])
        # cost is uniform
        cost = np.ones((3,3))
        coeff, sources = min_cost_flow_init(network, target, cost)
        # check coefficients
        tmp = np.ones((3,3))
        tmp[0,:] = np.array([2., 2., 2.])
        delta = abs(coeff - tmp)
        self.assertTrue(np.sum(delta) < eps)
        # check sources
        delta = abs(sources - np.array([.0, .5, -.5]))
        self.assertTrue(np.sum(delta) < eps)


    def test_flow_to_rates_routing(self):
        eps = 10e-8
        size = 3
        flow = np.zeros((3, 3))
        flow[0,1] = .25
        flow[0,2] = .25
        flow[1,0] = .5
        flow[1,2] = .5
        target = np.array([.5, 1., 1.])
        rates, routing = flow_to_rates_routing(size, flow, target)
        # check rates
        delta = abs(rates - np.array([1., 1., 0.]))
        self.assertTrue(np.sum(delta) < eps)
        # check routing
        tmp = .5 * np.ones((size, size))
        tmp[range(size), range(size)] = 0.0
        self.assertTrue(np.sum(abs(routing - tmp)) < eps)


    def test_min_attack_solver(self):
        eps = 10e-8
        # generate asymmetric network
        rates, routing, travel_times = generate_asymmetric()
        network = MAS.Network(rates, routing, travel_times)
        target = np.ones((3,))
        cost = np.ones((3,3))
        opt_rates, opt_routing = min_attack_solver(network, target, cost)
        self.assertTrue(np.sum(abs(opt_rates-np.array([0.,0.,1.]))) < eps)
        tmp = .5 * np.ones((3, 3))
        tmp[range(3), range(3)] = 0.0
        self.assertTrue(np.sum(abs(opt_routing-tmp)) < eps)


if __name__ == '__main__':
    unittest.main()