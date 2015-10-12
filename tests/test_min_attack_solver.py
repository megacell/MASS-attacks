'''
Tests for the min_attack_solver
'''

import unittest
import numpy as np
from utils import generate_uniform, generate_asymmetric, is_equal
from min_attack_solver import MinAttackSolver
from min_cost_flow_solver import to_cplex_lp_file
import MAS_network as MAS


__author__ = 'jeromethai'


class TestMinAttackSolver(unittest.TestCase):


    def test_min_cost_flow_init(self):
        # generate asymmetric network
        network = MAS.Network(*generate_asymmetric())
        # target for the availabilities
        target = np.array([.5, 1., 1.])
        # cost is uniform
        cost = np.ones((3,3))
        coeff, sources = MinAttackSolver(network, target, cost).min_cost_flow_init()
        # check coefficients
        tmp = np.ones((3,3))
        tmp[0,:] = np.array([2., 2., 2.])
        self.assertTrue(is_equal(coeff, tmp))
        # check sources
        self.assertTrue(is_equal(sources, np.array([.0, .5, -.5])))


    def test_flow_to_rates_routing(self):
        network = MAS.Network(*generate_asymmetric())
        flow = np.zeros((3, 3))
        flow[0,1] = .25
        flow[0,2] = .25
        flow[1,0] = .5
        flow[1,2] = .5
        cost = np.ones((3,3))
        target = np.array([.5, 1., 1.])
        rates, routing = MinAttackSolver(network, target, cost).flow_to_rates_routing(flow)
        # check rates
        self.assertTrue(is_equal(rates, np.array([1., 1., 0.])))
        # check routing
        tmp = .5 * np.ones((3, 3))
        tmp[range(3), range(3)] = 0.0
        self.assertTrue(is_equal(routing, tmp))


    def test_min_attack_solver(self):
        # generate asymmetric network
        network = MAS.Network(*generate_asymmetric())
        target = np.ones((3,))
        cost = np.ones((3,3))
        opt_rates, opt_routing = MinAttackSolver(network, target, cost).solve()
        self.assertTrue(is_equal(opt_rates, np.array([0.,0.,1.])))
        tmp = .5 * np.ones((3, 3))
        tmp[range(3), range(3)] = 0.0
        self.assertTrue(is_equal(opt_routing, tmp))


    def test_to_cplex_lp_file(self):
        # test if it generates the right string
        network = MAS.Network(*generate_asymmetric())
        target = np.array([ 0.25, 0.5, 1.])
        cost = np.ones((3,3))
        coeff, sources = MinAttackSolver(network, target, cost).min_cost_flow_init()
        string = to_cplex_lp_file(coeff, sources, network.adjacency)
        # print string


if __name__ == '__main__':
    unittest.main()
