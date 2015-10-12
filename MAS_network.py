''' Simple class for Mobility-As-a-Service (MAS) networks
'''

import numpy as np
import scipy.io
from utils import is_equal, pi_2_a, r_2_pi
# attack solvers
from attack_rate_solver import AttackRateSolver
from min_attack_solver import MinAttackSolver
from attack_routing_solver import AttackRoutingSolver
from single_destination_attack import SingleDestinationAttack

__author__ = 'jeromethai'


class Network:
    def __init__(self, rates, routing, travel_times):
        # Class for a Jackson network
        # rates        : numpy array with rates of arrival of passengers
        # routing      : numpy matrix for routing probabilitites
        # travel_times : numpy matrix for travel times
        self.rates = rates
        self.routing = routing
        self.travel_times = travel_times
        self.size = len(rates)
        self.mean_travel_time = np.sum(self.travel_times) / (self.size * self.size)
        # compute adjacency matrix
        adjacency = np.ones((self.size, self.size))
        adjacency[range(self.size), range(self.size)] = 0.0
        self.adjacency = adjacency
        # attack rates and routing
        self.attack_rates = None
        self.attack_routing = None
        # rates and routing after the attacks
        self.new_rates = rates
        self.new_routing = routing
        # weights wuch that attacks minimize weighted sum of availabilities
        self.weights=np.ones((self.size,))
        # budget for the attacks
        self.budget = 1.0


    def check(self, eps=10e-8):
        assert eps > 0., "eps too small"

        # check that dimensions match
        assert self.routing.shape == (self.size, self.size), \
            "sizes of rates and routing do not match"
        assert self.travel_times.shape == (self.size, self.size), \
            "sizes of rates and travel_times don't match"

        # check that we have probabilities for routing
        assert np.min(self.routing) >= 0.0, "negative probabilities"
        assert np.sum(self.routing.diagonal()) == 0.0, \
            "diagonal of routing matrix not null"
        assert is_equal(np.sum(self.routing, axis=1), 1.0, eps), \
            "routing matrix are not probabilities"

        # make sure that the Jackson network is not ill-defined
        assert np.min(self.rates) > eps, "rates too small"
        assert float(np.min(self.rates)) / np.max(self.rates) > eps, \
            "ratio min(rates) / max(rates) too small"

        # check travel times
        tmp = self.travel_times
        tmp[range(self.size), range(self.size)] = np.max(self.travel_times)
        assert np.sum(tmp > eps) == self.size * self.size, \
            "travel times too small"
        assert float(np.min(tmp)) / np.max(tmp) > eps, \
            "ratio min(travel_times) / max(travel_times) too small"

        # check the weights
        assert len(self.weights) == self.size, 'weights wrong size'
        assert np.sum(self.weights > eps) == self.size, 'weights not positive'


    def set_weights_to_min_time_usage(self):
        # set weights to minimize the time usage of the network
        tmp = np.multiply(self.routing, self.travel_times)
        self.weights = np.multiply(self.rates, np.sum(tmp, axis=1))
        self.check()


    def throughputs(self, eps=10e-8):
        # get throughputs by solving the balanced equations before attacks
        return r_2_pi(self.routing, eps)


    def new_throughputs(self, eps=10e-8):
        # get throughputs by solving the balanced equations after attacks
        return r_2_pi(self.new_routing, eps)


    def availabilities(self, eps=10e-8):
        # get asymptotic availabilities at each station before the attacks
        return pi_2_a(self.throughputs(eps), self.rates)


    def new_availabilities(self, eps=10e-8):
        # get asymptotic availabilities at each station after the attacks
        return pi_2_a(self.new_throughputs(eps), self.new_rates)


    def balance(self, eps=10e-8, cplex=False):
        # balance the network as posed in Zhang2015
        target = np.ones((self.size,))
        # cost are travel times
        cost = self.travel_times
        # modify cost so that the problem is bounded
        cost[range(self.size), range(self.size)] = self.mean_travel_time
        opt_rates, opt_routing = MinAttackSolver(self, target, cost, eps, cplex).solve()
        # update the network
        self.update(opt_rates, opt_routing)
        return opt_rates, opt_routing


    def min_attack(self, target, eps=10e-8, cplex=False):
        # target is the vector of target availabilities
        assert np.max(target) == 1.0, 'max(target) > 1.0'
        assert np.min(target) >= eps, 'target not positive'
        cost = np.ones((self.size, self.size))
        opt_rates, opt_routing = MinAttackSolver(self, target, cost, eps, cplex).solve()
        # update the network
        self.update(opt_rates, opt_routing)
        return opt_rates, opt_routing


    def update(self, attack_rates, attack_routing):
        # update new_rates and new_routing given attack rates and routing
        self.attack_rates = attack_rates
        self.attack_routing = attack_routing
        self.new_rates = self.rates + attack_rates
        tmp = np.dot(np.diag(attack_rates), attack_routing) + \
            np.dot(np.diag(self.rates), self.routing)
        inverse_new_rates = np.divide(np.ones((self.size,)), self.new_rates)
        self.new_routing = np.dot(np.diag(inverse_new_rates), tmp)


    def opt_attack_routing(self, attack_rates, k, eps=10e-8, cplex=False):
        # given fixed attack_rates
        # find the best routing of attacks
        # to minimize the weighted sum of the availabilities
        assert len(attack_rates) == self.size, 'attack_rates wrong size'
        assert (k >= 0 and  k < self.size), 'index k is out of range'
        assert np.sum(attack_rates >= 0.0) == self.size, 'negative attack_rate'
        a, attack_routing = AttackRoutingSolver(self, attack_rates, k, eps, cplex).solve()
        # update the network
        self.update(attack_rates, attack_routing)
        return a, attack_routing


    def opt_attack_rate(self, attack_routing, k, nu_init, alpha=5., beta=1., max_iters=10):
        # given fixed attack routing, a_k set to 1 and initial 'nu_init'
        ars_solver = AttackRateSolver(self, attack_routing, k, nu_init)
        sol = ars_solver.solve(ars_solver.make_sqrt_step(alpha,beta),
                               ars_solver.make_stop(max_iters))
        self.update(sol['attack_rates'], attack_routing)
        # print sol['obj_values']
        return sol['attack_rates']


    def single_destination_attack(self, k):
        # best attack that scales down all the availabilities by the same factor
        attack_rates, attack_routing = SingleDestinationAttack(self, k).apply()
        self.update(attack_rates, attack_routing)
        return attack_rates, attack_routing



def load_network(file_path):
    # generate MAS network from file
    data = scipy.io.loadmat(file_path)
    network = Network(np.squeeze(data['lam']), data['p'], data['T'])
    network.check()
    return network
