'''Tests for the min_attack_solver
'''


import unittest
from utils import generate_uniform, generate_asymmetric, is_equal
import MAS_network as MAS
import numpy as np
from attack_routing_solver import constraints, attack_routing_solver


__author__ = 'jeromethai'

class TestAttackRoutingSolver(unittest.TestCase):
    

    def test_constraints(self):
        # generate asymmetric network with availabilities = [ 0.5  0.5  1. ]
        rates, routing, travel_times = generate_asymmetric()
        network = MAS.Network(rates, routing, travel_times)
        # an attack strategy that results in availabilities = [ 0.25  0.5   1. ]
        attack_rates = np.array([1., 0., 0.])
        attack_routing = np.array([[0., 0., 1.],[.5, 0., .5],[.5, .5, 0.]])
        network.update(attack_rates, attack_routing)
        # fix the availability at station 2 to be equal to 1
        k = 2
        b, A = constraints(network, attack_rates, k)
   

    def test_attack_routing_solver(self):
        # generate symmetric network with availabilities = [1., 1., 1.]
        # weight on the availabilities are [1., 1., 1.]
        rates, routing, travel_times = generate_uniform()
        network = MAS.Network(rates, routing, travel_times)
        # attack rates are fixed
        attack_rates = np.array([1., 1., 1.])
        # find routing minimizing the weighted weighted sum of availabilities
        # fix the availability at station 2 to be equal to 1
        k = 2 
        # get the availabilities 'a' and routing that led to 'a'
        a, routing = attack_routing_solver(network, attack_rates, k)
        self.assertTrue(is_equal(a, np.array([2./3, 2./3, 1.])))
        tmp = np.array([[0., 0., 1.], [0., 0., 1.], [.5, .5, 0.]])
        self.assertTrue(is_equal(routing, tmp))


if __name__ == '__main__':
    unittest.main()
        