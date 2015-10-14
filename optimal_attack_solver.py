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
    def __init__(self, network, max_iters=10, full_adj=True, eps=1e-8, cplex=True, \
                    k=None, omega=0):
        self.network = network
        self.k = k
        self.omega = omega
        self.N = network.size
        self.eps = eps
        self.cplex = cplex
        self.w = network.weights
        self.full_adj = full_adj
        # objects specific to the block-coordinate descent
        self.max_iters = max_iters


    def objective(self, availabilities):
        return np.sum(np.multiply(self.w, availabilities))


    def solve(self, alpha=10., beta=1., max_iters_attack_rate=5, split_budget=False):
        # solves using block-coordinate descent
        network = self.network
        full_adj, eps, cplex =  self.full_adj, self.eps, self.cplex
        omega = self.omega
        # uses the single_destination_attack policy as a starting point
        print '============= initial objective value ============='
        print self.objective(network.new_availabilities())
        k = network.best_single_destination_attack() if self.k is None else self.k
        print 'station {} is fixed to be equal to 1'.format(k)

        if full_adj:
            network.single_destination_attack(k)
        else:
            network.split_budget_attack()
        print '============= after initialization ============='
        # import pdb; pdb.set_trace()
        assert network.verify_adjacency() == True
        print self.objective(network.new_availabilities())
        for i in range(self.max_iters):
            print ' ============= iter ============='
            print i
            network.opt_attack_routing(network.attack_rates, k, full_adj, eps, cplex)
            network.re_normalize_attack_routing()
            print '============= after opt_attack_routing ============='
            assert network.verify_adjacency() == True
            print self.objective(network.new_availabilities())
            network.min_attack(network.new_availabilities(), full_adj, eps, cplex)
            network.re_normalize_attack_routing()
            print '============= after min_attack ============='
            #import pdb; pdb.set_trace()
            assert network.verify_adjacency() == True
            print self.objective(network.new_availabilities())
            network.opt_attack_rate(network.attack_routing, k, network.attack_rates, \
                    alpha, beta, max_iters_attack_rate, omega, eps)
            print '============= after opt_attack_rate ============='
            assert network.verify_adjacency() == True
            print self.objective(network.new_availabilities())
            print '============= max budget ============= '
            print network.budget
            print '============= final budget ============= '
            print np.sum(network.attack_rates)

        print network.attack_routing
