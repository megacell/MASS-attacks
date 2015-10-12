'''
Implement the solver for the optimal attack problem
The problem is solved via block-coordinated descent
using a combination of three algorithms:
    - min_attack_solver in which the availabilities 'a' are fixed
    - attack_routing_solver in which the 
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


    def init_solver(self):
        # used the single_destination_attack policy as a starting point
        


    def solve(self):
        # solves using block-coordinate descent
        for i in range(self.max_iters):
            pass