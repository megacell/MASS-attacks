''' Simple class for Mobility-As-a-Service (MAS) networks
'''

import numpy as np
from min_attack_solver import min_attack_solver 
from attack_routing_solver import attack_routing_solver
import scipy.io
from utils import is_equal

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



    def check(self, eps = 10e-8):
        assert eps > 0., "eps too small"

        # check that dimensions match
        assert self.routing.shape == (self.size, self.size), \
            "sizes of rates and routing do not match"
        assert self.travel_times.shape == (self.size, self.size), \
            "sizes of rates and travel_times don't match"

        # check that we have probabilities for routing
        assert np.sum(self.routing >= 0.0) == self.size * self.size, \
            "negative probabilities"
        assert np.sum(self.routing.diagonal()) == 0.0, \
            "diagonal of routing matrix not null"
        assert is_equal(np.sum(self.routing, axis=1), 1.0, eps), \
            "routing matrix are not probabilities"

        # make sure that the Jackson network is not ill-defined
        assert np.sum(self.rates > eps) == self.size, "rates too small"
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


    def throughputs(self, eps = 10e-8):
        # get throughputs by solving the balanced equations
        eigenvalues, eigenvectors = np.linalg.eig(self.routing.transpose())
        index = np.argwhere(abs(eigenvalues - 1.0) < eps)[0][0]
        pi = np.real(eigenvectors[:, index])
        return pi / np.sum(pi)


    def new_throughputs(self, eps = 10e-8):
        # get throughputs by solving the balanced equations
        eigenvalues, eigenvectors = np.linalg.eig(self.new_routing.transpose())
        index = np.argwhere(abs(eigenvalues - 1.0) < eps)[0][0]
        pi = np.real(eigenvectors[:, index])
        return pi / np.sum(pi)


    def availabilities(self, eps = 10e-8):
        # get asymptotic availabilities at each station
        a = np.divide(self.throughputs(eps), self.rates)
        return a / np.max(a)


    def new_availabilities(self, eps = 10e-8):
        # get asymptotic availabilities at each station
        a = np.divide(self.new_throughputs(eps), self.new_rates)
        return a / np.max(a)


    def balance(self, eps = 10e-8):
        # balance the network as posed in Zhang2015
        target = np.ones((self.size,))
        # cost are travel times
        cost = self.travel_times
        # modify cost so that the problem is bounded
        cost[range(self.size), range(self.size)] = self.mean_travel_time
        opt_rates, opt_routing = min_attack_solver(self, target, cost, eps)
        # update the network
        self.update(opt_rates, opt_routing)
        return opt_rates, opt_routing


    def min_attack(self, target, eps = 10e-8):
        # target is the vector of target availabilities
        assert np.max(target) == 1.0, 'max(target) > 1.0'
        assert np.min(target) >= eps, 'target not positive' 
        cost = np.ones((self.size, self.size))
        opt_rates, opt_routing = min_attack_solver(self, target, cost, eps)
        # update the network
        self.update(opt_rates, opt_routing)
        return opt_rates, opt_routing


    def update(self, rates, routing):
        # update new_rates and new_routing given attack rates and routing
        self.attack_rates = rates
        self.attack_routing = routing
        self.new_rates = self.rates + rates
        tmp = np.dot(np.diag(rates), routing) + np.dot(np.diag(self.rates), self.routing)
        inverse_new_rates = np.divide(np.ones(self.size,), self.new_rates)
        self.new_routing = np.dot(np.diag(inverse_new_rates), tmp)


    def opt_attack_routing(self, attack_rates, k, eps = 10e-8):
        # given fixed attack_rates
        # find the best routing of attacks 
        # to minimize the weighted sum of the availabilities
        assert len(attack_rates) == self.size, 'attack_rates wrong size'
        assert (k >= 0 and  k < self.size), 'index k is out of range'
        assert np.sum(attack_rates >= 0.0) == self.size, 'negative attack_rate'
        a, routing = attack_routing_solver(self, attack_rates, k, eps)
        # update the network
        self.update(attack_rates, routing)
        return a, routing


def load_network(file_path):
    # generate MAS network from file
    data = scipy.io.loadmat(file_path)
    network = Network(np.squeeze(data['lam']), data['p'], data['T'])
    network.check()
    return network






