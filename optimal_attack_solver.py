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
    def __init__(self, network, max_iters=10, full_adj=True, omega=0.0, ridge=0.0, \
                    eps=1e-8, cplex=True, k=None):
        self.network = network
        self.k = k
        self.omega = omega
        self.N = network.size
        self.omega = omega # term to maximize throughput
        if isinstance(ridge, int) or isinstance(ridge, float): 
            ridge = ridge * np.ones((network.size,))
        self.ridge = ridge # l2-regularization on the attack_rates
        self.eps = eps
        self.cplex = cplex
        self.w = network.weights
        self.full_adj = full_adj
        self.max_iters = max_iters
        self.b = network.budget
        self.initial = None
        self.last = None
        self.messages = []

    def status(self, iter, last_step):
        # check that eveything is all right
        if iter>0: self.network.re_normalize_attack_routing()
        if not self.full_adj: assert self.network.verify_adjacency() == True
        # print status
        nu = self.network.attack_rates
        a = self.network.new_availabilities()
        obj = np.sum(np.multiply(self.w, a))
        if nu is None: nu = 0.0 
        thru = np.sum(np.multiply(nu, a))
        reg = 0.5 * np.sum(np.multiply(self.ridge, np.square(nu)))
        template = \
        '''
        ==================================================================
        iteration      : {}
        last step      : {}
        objective      : {} / {}
        relative obj   : {} %
        progress       : {} %
        throughput     : {}
        regularization : {}
        budget/total   : {} / {}
        ==================================================================
        '''
        msg = template.format(iter, \
                        last_step, \
                        int(obj), int(self.initial), \
                        int(100.0 * obj / self.initial), \
                        100.0 * (self.last - obj) / obj, \
                        int(thru), \
                        reg, \
                        int(np.sum(nu)), int(self.b))
        print msg
        self.last = obj
        self.messages.append(msg)


    def solve(self, alpha=10., beta=1., max_iters_attack_rate=5, split_budget=False):
        # solves using block-coordinate descent
        network = self.network
        full_adj, eps, cplex =  self.full_adj, self.eps, self.cplex
        omega = self.omega
        ridge = self.ridge
        # uses the single_destination_attack policy as a starting point
        self.initial = np.sum(np.multiply(self.w, network.new_availabilities()))
        self.last = self.initial
        self.status(0, 'before initialization')
        k = network.best_single_destination_attack() if self.k is None else self.k
        # if full_adj:
        #     network.single_destination_attack(k)
        # else:
        #     network.split_budget_attack()
        network.split_budget_attack()
        self.status(0, 'after initialization')
        for i in range(1, self.max_iters+1):
            # apply optimal attack routing
            network.opt_attack_routing(network.attack_rates, k, full_adj, omega, \
                                                                        eps, cplex)

            self.status(i, 'optimal_attack_routing')
            # apply maximum throughput attack
            network.max_attack(network.new_availabilities(), ridge, full_adj, eps)
            self.status(i, 'maximum_throughput_attack')
            # apply optimal attack rate
            network.opt_attack_rate(network.attack_routing, k, network.attack_rates, \
                    alpha, beta, max_iters_attack_rate, \
                    omega, ridge, eps)
            self.status(i, 'optimal_attack_rate')
        print '\n'.join(self.messages)
        print sorted(network.attack_rates)
        #import pdb; pdb.set_trace()
        #print network.attack_routing
        print 'generating visualization ...'
