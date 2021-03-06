'''Tests for the min_attack_solver
'''


import unittest
from utils import generate_uniform, generate_asymmetric, is_equal
import MAS_network as MAS
import numpy as np
from attack_routing_solver import AttackRoutingSolver


__author__ = 'jeromethai'

class TestAttackRoutingSolver(unittest.TestCase):
    

    def test_constraints(self):
        # generate asymmetric network with availabilities = [ 0.5  0.5  1. ]
        network = MAS.Network(*generate_asymmetric())
        # an attack strategy that results in availabilities = [ 0.25  0.5   1. ]
        attack_rates = np.array([1., 0., 0.])
        attack_routing = np.array([[0., 0., 1.],[.5, 0., .5],[.5, .5, 0.]])
        network.update(attack_rates, attack_routing)
        # fix the availability at station 2 to be equal to 1
        k = 2
        b, A = AttackRoutingSolver(network, attack_rates, k).constraints()
   

    def test_attack_routing_solver(self):
        # generate symmetric network with availabilities = [1., 1., 1.]
        # weight on the availabilities are [1., 1., 1.]
        network = MAS.Network(*generate_uniform())
        # attack rates are fixed
        attack_rates = np.array([1., 1., 1.])
        # find routing minimizing the weighted weighted sum of availabilities
        # fix the availability at station 2 to be equal to 1
        k = 2 
        # get the availabilities 'a' and routing that led to 'a'
        a, routing = AttackRoutingSolver(network, attack_rates, k).solve()
        network.update(attack_rates, routing)
        self.assertTrue(abs(np.sum(network.new_availabilities()) - 7./3))


    def test_to_cplex_lp_file(self):
        # same example as above
        network = MAS.Network(*generate_uniform())
        attack_rates = np.array([1., 1., 1.])
        k = 2
        string = AttackRoutingSolver(network, attack_rates, k).to_cplex_lp_file()
        print string


    def test_cplex_attack_routing_small_network(self):
        # see test_attack_routing_solver() above for details on the example
        network = MAS.Network(*generate_uniform())
        attack_rates = np.array([1., 1., 1.])
        k = 2
        a, routing = AttackRoutingSolver(network, attack_rates, k, cplex=True).solve()
        network.update(attack_rates, routing)
        self.assertTrue(abs(np.sum(network.new_availabilities()) - 7./3))


if __name__ == '__main__':
    unittest.main()
        