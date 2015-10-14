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
    def __init__(self, network, max_iters=10, full_adj=True, omega=0.0, eps=1e-8, cplex=True, \
                    k=None):
        self.network = network
        self.k = k
        self.omega = omega
        self.N = network.size
        self.omega = omega
        self.eps = eps
        self.cplex = cplex
        self.w = network.weights
        self.full_adj = full_adj
        # objects specific to the block-coordinate descent
        self.max_iters = max_iters


    def objective(self, a, nu):
        obj = np.sum(np.multiply(self.w, a))
        thru = 0.0 if nu is None else np.sum(np.multiply(nu, a))
        return (obj, thru)


    def solve(self, alpha=10., beta=1., max_iters_attack_rate=5, split_budget=False):
        # solves using block-coordinate descent
        network = self.network
        full_adj, eps, cplex =  self.full_adj, self.eps, self.cplex
        omega = self.omega
        # uses the single_destination_attack policy as a starting point
        print '============= initial objective value ============='
        print self.objective(network.new_availabilities(), network.attack_rates)
        k = network.best_single_destination_attack() if self.k is None else self.k
        print 'station {} is fixed to be equal to 1'.format(k)

        # if full_adj:
        #     network.single_destination_attack(k)
        # else:
        #     network.split_budget_attack()
        network.split_budget_attack()
        print '============= after initialization ============='
        # import pdb; pdb.set_trace()
        if not full_adj: assert network.verify_adjacency() == True
        print self.objective(network.new_availabilities(), network.attack_rates)
        for i in range(self.max_iters):
            print ' ============= iter ============='
            print i
            network.opt_attack_routing(network.attack_rates, k, full_adj, omega, \
                                                                        eps, cplex)
            network.re_normalize_attack_routing()
            print '============= after opt_attack_routing ============='
            if not full_adj: assert network.verify_adjacency() == True
            print self.objective(network.new_availabilities(), network.attack_rates)
            network.max_attack(network.new_availabilities(), full_adj, eps)
            network.re_normalize_attack_routing()
            print '============= after min_attack ============='
            #import pdb; pdb.set_trace()
            if not full_adj: assert network.verify_adjacency() == True
            print self.objective(network.new_availabilities(), network.attack_rates)
            network.opt_attack_rate(network.attack_routing, k, network.attack_rates, \
                    alpha, beta, max_iters_attack_rate, omega, eps)
            print '============= after opt_attack_rate ============='
            if not full_adj: assert network.verify_adjacency() == True
            print self.objective(network.new_availabilities(), network.attack_rates)
            print '============= max budget ============= '
            print network.budget
            print '============= final budget ============= '
            print np.sum(network.attack_rates)

        import pdb; pdb.set_trace()
        #print network.attack_routing
