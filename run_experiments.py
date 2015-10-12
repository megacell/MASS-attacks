import numpy as np
from MAS_network import load_network
from logo_to_availabilities import get_availabilities
from optimal_attack_solver import OptimalAttackSolver

__author__ = 'yuanchenyang', 'jeromethai'

from pdb import set_trace as T


def cal_logo_experiment(adj):
    nw = load_network('data/queueing_params.mat')
    target = get_availabilities(nw.station_names)

    bal_rates, bal_routing = nw.balance()
    nw.combine()

    res = []
    for i in adj:
        nw.full_adjacency = nw.get_adjacencies(i)
        att_rates, att_routing = nw.min_attack(target, full_adj=True)
        res.append(int(np.sum(att_rates)))

    print 'Passenger Arrival Rate:', np.sum(nw.rates)
    print 'Balance Cost: ', np.sum(bal_rates)
    print 'Attack After Balance Cost (adjacency {}): {}'.format(adj, res)
    return res


def optimal_attack_full_network():
    # when there is no limit on the type of attacks,
    # a very good type of attack is obtained
    # when they are all routes to the same station
    # we proceed so by choosing the best station to route the attacks to
    # which is the initialization, then there is not much room for progress
    # with the block-coordinate descent algorithm
    network = load_network('data/queueing_params.mat')
    network.budget = 200.
    k = np.where(network.new_availabilities() - 1. == 0.0)[0][0]
    network.balance(cplex=True)
    print 'total customer rate', np.sum(network.rates)
    print 'total rebalancing rate', np.sum(network.attack_rates)
    network.combine()
    print 'min availability', np.min(network.availabilities())
    print 'combined customer and rebalancing rates', np.sum(network.rates)
    oas = OptimalAttackSolver(network, max_iters=3)
    oas.solve(alpha=10., beta=1., max_iters_attack_rate=5)


def optimal_attack_full_network_2():
    network = load_network('data/queueing_params_with_adjacency.mat')
    network.budget = 200.

if __name__ == '__main__':
    cal_logo_experiment(range(1, 15))
    # optimal_attack_full_network()
