'''
Implement the solver for the optimal attack problem
The problem is solved via block-coordinated descent
using a combination of three algorithms in this order:
    - attack_routing_solver in which the rate of attacks 'nu' are fixed
    - min_attack_solver in which the availabilities 'a' are fixed
    - attack_rate_solver in whith the attack routing 'kappa' is fixed
'''

import numpy as np

__author__ = 'jeromethai'


class OptimalAttackSolver:
    # class for the optimal attack solver
    def __init__(self, network, k, eps=1e-8, cplex=False):
        self.network = network
        self.phi = network.rates
        self.delta = network.routing
        self.N = network.size
        self.b = network.budget
        self.k = k
        self.eps = eps
        self.cplex = cplex
        self.w = network.weights
        # objects specific to the block-coordinate descent
        self.iter = -1 # iteration number
        self.max_iters = 10


    def solve(self):
        # solves using block-coordinate descent
        # uses the single_destination_attack policy as a starting point
        print 'initial objective value'
        print np.sum(np.multiply(self.w, self.network.availabilities))
        self.network.single_destination_attack(self.k)
        print 'apply single_destination_attack'
        print np.sum(np.multiply(self.w, self.network.new_availabilities))
        for i in range(self.max_iters):
            # apply attack_routing_solver
            pass
            # apply min_attack_solver

            # apply attack_rate_solver