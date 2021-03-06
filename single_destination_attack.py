'''
Implement the single-destination attack policy
where all the availabilities are decreased by the same factor alpha
and a_k is set to 1
'''

import numpy as np

__author__ = 'jeromethai'


class SingleDestinationAttack:
    # class for implementing the single-destination attack
    # as described in the paper
    def __init__(self, network, k):
        self.network = network
        self.delta = network.routing # routing prob before attacks
        self.k = k
        self.phi = network.rates # rates before the attacks
        self.N = network.size
        self.b = network.budget
        # we fix the attack to be directed to station k
        kappa = np.zeros((self.N, self.N))
        kappa[range(self.N), k] = 1.0
        kappa[k, range(self.N)] = 1. / (self.N - 1.)
        kappa[k, k] = 0.0
        self.kappa = kappa


    def apply(self):
        # apply the policy and returns attack_rates and attack_routing
        self.a = self.network.availabilities()
        tmp = np.sum(np.divide(self.delta[self.k,:], self.a))
        if self.b < (1. - self.a[self.k]) * self.phi[self.k] * tmp:
            # inefficient attack, hence attack_rates = 0
            attack_rates, alpha = np.zeros((self.N,)), -1.0
        else:
            # efficient attack
            attack_rates = (self.b/tmp) * np.divide(self.delta[self.k,:], self.a)
            alpha = self.a[self.k] + self.b / (self.phi[self.k] * tmp)
        return {'attack_rates': attack_rates, 'attack_routing': self.kappa, \
                'alpha': alpha}



