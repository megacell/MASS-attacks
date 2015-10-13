import numpy as np
from MAS_network import load_network
from logo_to_availabilities import get_availabilities
from optimal_attack_solver import OptimalAttackSolver
from simulation import Network, simulate

__author__ = 'yuanchenyang', 'jeromethai'

from pdb import set_trace as T


def cal_logo_experiment(adj):
    nw = load_network('data/queueing_params.mat')
    target = get_availabilities(nw.station_names)

    bal_rates, bal_routing = nw.balance()
    nw.combine()

    res = []
    for i in adj:
        nw.update_adjacency(i)
        att_rates, att_routing = nw.min_attack(target, full_adj=False)
        T()
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
    nw = load_network('data/queueing_params.mat')
    nw.balance()
    nw.combine()
    nw.budget = 200.
    nw.optimal_attack(max_iters=3).solve(alpha=10., beta=1., max_iters_attack_rate=5)


def optimal_attack_with_different_adjacencies():
    # try to compute the optimal attacks with different radii of adjacencies
    network = load_network('data/queueing_params.mat')
    nw.balance()
    nw.budget = 200.

def network_simulation():
    nw = load_network('data/queueing_params.mat')
    target = get_availabilities(nw.station_names)

    bal_rates, bal_routing = nw.balance()
    nw.combine()

    n = Network(nw.size, nw.rates, nw.travel_times, nw.routing, [20]* nw.size)
    for i in range(100):
        if i % 10 == 0:
            print i
        n.jump()
    T()

if __name__ == '__main__':
    # cal_logo_experiment(range(1, 15))
    # optimal_attack_full_network()
    network_simulation()
