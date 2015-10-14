''' Simple class for Mobility-As-a-Service (MAS) networks
'''

import numpy as np
import scipy.io
from utils import is_equal, pi_2_a, r_2_pi, norm_adjacencies
# attack solvers
from attack_rate_solver import AttackRateSolver
from min_attack_solver import MinAttackSolver
from attack_routing_solver import AttackRoutingSolver
from single_destination_attack import SingleDestinationAttack
from optimal_attack_solver import OptimalAttackSolver
from max_attack_solver import MaxAttackSolver

__author__ = 'jeromethai'


class Network:
    def __init__(self, rates, routing, travel_times, adjacency_1=None):
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
        self.full_adjacency = np.ones((self.size, self.size))
        self.full_adjacency[range(self.size), range(self.size)] = 0.0
        self.adjacency_1 = adjacency_1
        self.adjacency = adjacency_1
        # attack rates and routing
        self.attack_rates = None
        self.attack_routing = None
        # combined rates and routing after the attacks
        self.new_rates = rates
        self.new_routing = routing
        # weights wuch that attacks minimize weighted sum of availabilities
        self.weights=np.ones((self.size,))
        # budget for the attacks
        self.budget = 1.0


    def check(self, eps=1e-8):
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

        # check the adjacency matrix
        if self.adjacency is not None:
            assert self.adjacency.shape == (self.size, self.size)
            tmp = (self.adjacency == 0.) + (self.adjacency == 1.)
            assert np.sum(tmp) == self.size * self.size, "entries in adj not in 0, 1"
            assert np.sum(self.adjacency.diagonal()) == 0.0, \
                "diagonal of routing matrix not null"


    def re_normalize_attack_routing(self):
            tmp = np.divide(np.ones((self.size,)), np.sum(self.attack_routing, axis=1))
            self.attack_routing = np.dot(np.diag(tmp), self.attack_routing)


    def set_weights_to_min_time_usage(self):
        # set weights to minimize the time usage of the network
        tmp = np.multiply(self.routing, self.travel_times)
        self.weights = np.multiply(self.rates, np.sum(tmp, axis=1))
        self.check()


    def throughputs(self, eps=1e-8):
        # get throughputs by solving the balanced equations before attacks
        return r_2_pi(self.routing, eps)


    def new_throughputs(self, eps=1e-8):
        # get throughputs by solving the balanced equations after attacks
        return r_2_pi(self.new_routing, eps)


    def availabilities(self, eps=1e-8):
        # get asymptotic availabilities at each station before the attacks
        return pi_2_a(self.throughputs(eps), self.rates)


    def new_availabilities(self, eps=1e-8):
        # get asymptotic availabilities at each station after the attacks
        return pi_2_a(self.new_throughputs(eps), self.new_rates)


    def balance(self, full_adj=True, eps=1e-8, cplex=True):
        # balance the network as posed in Zhang2015
        target = np.ones((self.size,))
        # cost are travel times
        cost = self.travel_times
        # modify cost so that the problem is bounded
        cost[range(self.size), range(self.size)] = self.mean_travel_time
        opt_rates, opt_routing = MinAttackSolver(self, target, cost, full_adj, eps, cplex).solve()
        # update the network
        self.update(opt_rates, opt_routing)
        return opt_rates, opt_routing


    def min_attack(self, target, full_adj=True, eps=1e-8, cplex=True):
        # target is the vector of target availabilities
        assert np.max(target) == 1.0, 'max(target) > 1.0'
        assert np.min(target) >= eps, 'target not positive'
        cost = np.ones((self.size, self.size))
        opt_rates, opt_routing = MinAttackSolver(self, target, cost, full_adj=full_adj, eps=eps, cplex=cplex).solve()
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


    def combine(self):
        # combine the new_rates (rates + attack_rates) into rates
        # combine the new_routing (routing + attack_routing) into routing
        self.rates = self.new_rates
        self.routing = self.new_routing
        # erase attack_rates and attack_routing since they are combined
        # into rates and routing
        self.attack_rates = None
        self.attack_routing = None
        self.new_rates = self.rates
        self.new_routing = self.routing


    def get_adjacencies(self, r):
        # Returns an adjacency matrix if we allow raduis r number of steps
        assert self.adjacency_1 is not None
        assert type(r) == int and r > 0, 'Incorrect r'
        if r == 1:
            return np.copy(self.adjacency_1)
        else:
            # Since the graph is undirected, if we can reach node i in at most r
            # steps, we can reach it in either r-1 steps (odd length path from
            # origin to i) or r steps (even length path from origin to i)
            r_steps = np.linalg.matrix_power(self.adjacency_1, r)
            r_minus_1_steps = np.linalg.matrix_power(self.adjacency_1, r - 1)
            res = r_steps + r_minus_1_steps
            norm_adjacencies(res)
            return res


    def update_adjacency(self, r):
        # update the adjacency such that it allows r number of steps
        self.adjacency = self.get_adjacencies(r)


    def verify_adjacency(self):
        # return True if it respects the adjacency matrix
        return np.sum(np.multiply(self.attack_routing, 1.0-self.adjacency)) == 0.0


    def opt_attack_routing(self, attack_rates, k, full_adj=True, omega=0.0, \
                                                    eps=1e-8, cplex=True):
        # given fixed attack_rates
        # find the best routing of attacks
        # to minimize the weighted sum of the availabilities
        assert len(attack_rates) == self.size, 'attack_rates wrong size'
        assert (k >= 0 and  k < self.size), 'index k is out of range'
        assert np.sum(attack_rates >= 0.0) == self.size, 'negative attack_rate'
        a, attack_routing = AttackRoutingSolver(self, attack_rates, k, full_adj, 
                                        omega, eps, cplex).solve()
        # update the network
        self.update(attack_rates, attack_routing)
        return a, attack_routing


    def opt_attack_rate(self, attack_routing, k, nu_init, \
                    alpha=5., beta=1., max_iters=10, omega=0.0, eps=1e-8):
        # given fixed attack routing, a_k set to 1 and initial 'nu_init'
        ars_solver = AttackRateSolver(self, attack_routing, k, nu_init, omega=omega)
        sol = ars_solver.solve(ars_solver.make_sqrt_step(alpha,beta),
                               ars_solver.make_stop(max_iters))
        self.update(sol['attack_rates'], attack_routing)
        # print sol['obj_values']
        return sol['attack_rates']


    def single_destination_attack(self, k):
        # best attack that scales down all the availabilities
        # by the same factor except for k
        sol = SingleDestinationAttack(self, k).apply()
        self.update(sol['attack_rates'], sol['attack_routing'])
        return sol


    def split_budget_attack(self):
        # Splits budget amongst all stations and attack
        rates = np.ones(self.size) / float(self.budget)
        tmp = np.divide(np.ones((self.size,)), np.sum(self.adjacency, axis=1))
        routing = np.dot(np.diag(tmp), self.adjacency)
        # routing = np.array([[1 / (self.size - 1.) if i != j else 0
        #                     for i in range(self.size)]
        #                     for j in range(self.size)])
        self.update(rates, routing)


    def best_single_destination_attack(self):
        # find the best index for the single_destination_attack
        min_obj = np.sum(self.weights)
        a = self.availabilities()
        for i in range(self.size):
            print 'search index', i
            sol = self.single_destination_attack(i)
            if sol['alpha'] < 0: continue
            new_a = (1. / sol['alpha']) * a
            new_a[i] = 1.0
            obj = np.sum(np.multiply(self.weights, new_a))
            #obj = np.sum(np.multiply(self.weights, self.new_availabilities()))
            if obj < min_obj:
                best, min_obj = i, obj
        return best


    def optimal_attack(self, max_iters=10, full_adj=True, omega=0.0, \
                        eps=1e-8, cplex=True, \
                        k=None, alpha=10., beta=1., max_iters_attack_rate=5):
        oas = OptimalAttackSolver(self, max_iters, full_adj, omega, eps, cplex, k)
        oas.solve(alpha, beta, max_iters_attack_rate)


    def max_attack(self, target, full_adj=True, eps=1e-8):
        # maximizes the throughput of attacks
        assert np.max(target) == 1.0, 'max(target) > 1.0'
        assert np.min(target) >= eps, 'target not positive'
        opt_rates, opt_routing = MaxAttackSolver(self, target, full_adj=full_adj, eps=eps).solve()
        # update the network
        self.update(opt_rates, opt_routing)
        return opt_rates, opt_routing


def load_network(file_path):
    # generate MAS network from file
    data = scipy.io.loadmat(file_path)
    # num1 = (20 if someBoolValue else num1)
    adjacency = data['adj'] if 'adj' in data else None
    network = Network(np.squeeze(data['lam']), data['p'], data['T'], adjacency)
    network.station_names = data['stations']
    network.check()
    return network
