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
    def __init__(self, network, eps=1e-8, cplex=False, k=None):
        self.network = network
        self.k = k
        self.N = network.size
        self.eps = eps
        self.cplex = cplex
        self.w = network.weights
        # objects specific to the block-coordinate descent
        self.max_iters = 10


    def objective(self, availabilities):
        return np.sum(np.multiply(self.w, availabilities))


    def solve(self, alpha=10., beta=1., max_iters_attack_rate=5, split_budget=False):
        # solves using block-coordinate descent
        network = self.network
        eps, cplex =  self.eps, self.cplex
        # uses the single_destination_attack policy as a starting point
        print '============= initial objective value ============='
        print self.objective(network.new_availabilities())

        k = network.best_single_destination_attack() if self.k is None else self.k
        if spilt_budget:
            network.split_budget_attack()
        else:
            network.single_destination_attack(k)

        print '============= after single_destination_attack ============='
        print self.objective(network.new_availabilities())
        for i in range(self.max_iters):
            print ' ============= iter ============='
            print i
            network.opt_attack_routing(network.attack_rates, k, eps, cplex)
            print '============= after opt_attack_routing ============='
            print self.objective(network.new_availabilities())
            network.min_attack(network.new_availabilities(), eps, cplex)
            print '============= after min_attack ============='
            print self.objective(network.new_availabilities())
            network.opt_attack_rate(network.attack_routing, k, network.attack_rates, \
                    alpha, beta, max_iters_attack_rate, eps)
            print '============= apply opt_attack_rate ============='
            print self.objective(network.new_availabilities())
            print '============= max budget ============= '
            print network.budget
            print '============= final budget ============= '
            print np.sum(network.attack_rates)

        print network.attack_routing
        import pdb; pdb.set_trace()
