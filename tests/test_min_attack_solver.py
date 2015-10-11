'''Tests for the min_attack_solver
'''

import unittest
import numpy as np
from utils import generate_uniform, generate_asymmetric, is_equal
from min_attack_solver import min_cost_flow_init, flow_to_rates_routing, \
    min_attack_solver, to_cplex_lp_file
import MAS_network as MAS


__author__ = 'jeromethai'


class TestMinAttackSolver(unittest.TestCase):
    

    def test_min_cost_flow_init(self):
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
        self.assertTrue(is_equal(coeff, tmp))
        # check sources
        self.assertTrue(is_equal(sources, np.array([.0, .5, -.5])))


    def test_flow_to_rates_routing(self):
        size = 3
        flow = np.zeros((3, 3))
        flow[0,1] = .25
        flow[0,2] = .25
        flow[1,0] = .5
        flow[1,2] = .5
        target = np.array([.5, 1., 1.])
        rates, routing = flow_to_rates_routing(size, flow, target)
        # check rates
        self.assertTrue(is_equal(rates, np.array([1., 1., 0.])))
        # check routing
        tmp = .5 * np.ones((size, size))
        tmp[range(size), range(size)] = 0.0
        self.assertTrue(is_equal(routing, tmp))


    def test_min_attack_solver(self):
        # generate asymmetric network
        rates, routing, travel_times = generate_asymmetric()
        network = MAS.Network(rates, routing, travel_times)
        target = np.ones((3,))
        cost = np.ones((3,3))
        opt_rates, opt_routing = min_attack_solver(network, target, cost)
        self.assertTrue(is_equal(opt_rates, np.array([0.,0.,1.])))
        tmp = .5 * np.ones((3, 3))
        tmp[range(3), range(3)] = 0.0
        self.assertTrue(is_equal(opt_routing, tmp))


    def test_to_cplex_lp_file(self):
        # test if it generates the right string
        rates, routing, travel_times = generate_asymmetric()
        network = MAS.Network(rates, routing, travel_times)
        target = np.array([ 0.25, 0.5, 1.])
        cost = np.ones((3,3))
        opt_rates, opt_routing = min_attack_solver(network, target, cost)
        print to_cplex_lp_file(network, target, cost)


if __name__ == '__main__':
    unittest.main()