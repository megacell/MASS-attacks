''' Simple class for Mobility-As-a-Service (MAS) networks
'''

import numpy as np
from min_attack_solver import min_attack_solver

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
        # rates and routing after the attacks
        self.new_rates = rates
        self.new_routing = routing


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
        assert np.sum(np.sum(self.routing, axis=1) == 1.0) == self.size, \
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


    def throughputs(self, eps = 10e-8):
        # get throughputs by solving the balanced equations
        eigenvalues, eigenvectors = np.linalg.eig(self.routing.transpose())
        index = np.argwhere(abs(eigenvalues - 1.0) < eps)[0][0]
        pi = eigenvectors[:, index]
        return pi / np.sum(pi)


    def new_throughputs(self, eps = 10e-8):
        # get throughputs by solving the balanced equations
        eigenvalues, eigenvectors = np.linalg.eig(self.new_routing.transpose())
        index = np.argwhere(abs(eigenvalues - 1.0) < eps)[0][0]
        pi = eigenvectors[:, index]
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
        return opt_rates, opt_routing


    def min_attack(self, target, eps = 10e-8):
        # target is the vector of target availabilities
        assert np.max(target) == 1.0
        assert np.min(target) >= eps
        cost = np.ones((self.size, self.size))
        opt_rates, opt_routing = min_attack_solver(self, target, cost, eps)
        return opt_rates, opt_routing


    def update(self, rates, routing):
        # update new_rates and new_routing given attack rates and routing
        self.new_rates = self.rates + rates
        tmp = np.dot(np.diag(rates), routing) + np.dot(np.diag(self.rates), self.routing)
        inverse_new_rates = np.divide(np.ones(self.size,), self.new_rates)
        self.new_routing = np.dot(np.diag(inverse_new_rates), tmp)






