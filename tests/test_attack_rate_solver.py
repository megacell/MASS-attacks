'''
Tests for the attack_rate_solver
'''

import unittest
import numpy as np
from utils import generate_uniform, generate_asymmetric, is_equal
from attack_rate_solver import AttackRateSolver
import MAS_network as MAS


__author__ = 'jeromethai'

class TestAttackRateSolver(unittest.TestCase):


    def generate_solver(self):
        # generate asymmetric network
        network = MAS.Network(*generate_asymmetric())
        attack_routing = np.array([[0., 0., 1.],[.5, 0., .5],[.5, .5, 0.]])
        nu = np.array([1., 0., 0.])
        k = 2
        return AttackRateSolver(network, attack_routing, k, nu)


    def test_objective(self):
        obj, a = self.generate_solver().objective(nu = np.array([1., 0., 0.]))
        self.assertTrue(is_equal(obj, 1.75))
        self.assertTrue(is_equal(a, np.array([.25, .5, 1.])))


    def test_gradient_computation(self): 
        g = self.generate_solver().gradient()
        self.assertTrue(is_equal(g, np.array([-.125, -.375, .75])))


    def test_init_solver(self):
        ars_solver = self.generate_solver()
        ars_solver.init_solver(None, None)
        self.assertTrue(is_equal(ars_solver.obj_values[0], 1.75))
        self.assertTrue(is_equal(ars_solver.a, np.array([.25, .5, 1.])))
        self.assertTrue(ars_solver.iter == 0)




if __name__ == '__main__':
    unittest.main()