'''
Tests for optimal attack solver
'''

import unittest
import numpy as np
import MAS_network as MAS
from optimal_attack_solver import OptimalAttackSolver


__author__ = 'jeromethai'


class TestOptimalAttackSolver(unittest.TestCase):

    def test_optimal_attack_solver_full_network(self):
        network = MAS.load_network('data/queueing_params.mat')
        network.budget = 200.
        k = np.where(network.new_availabilities() - 1. == 0.0)[0][0]
        network.balance(cplex=True)
        print 'total customer rate', np.sum(network.rates)
        print 'total rebalancing rate', np.sum(network.attack_rates)
        network.combine()
        print 'min availability', np.min(network.availabilities())
        print 'combined customer and rebalancing rates', np.sum(network.rates)
        # oas = OptimalAttackSolver(network, cplex=True, k=k)
        oas = OptimalAttackSolver(network, cplex=True)
        oas.solve(alpha=10., beta=1., max_iters_attack_rate=5)