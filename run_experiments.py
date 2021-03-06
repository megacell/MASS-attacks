import numpy as np
import pickle
import scipy.io as sio
from MAS_network import load_network
from logo_to_availabilities import get_availabilities
from optimal_attack_solver import OptimalAttackSolver
from param_inference.utils import FeatureCollection
from param_inference.generate_matrices import rbs, get_xy
from simulation import Network, simulate

__author__ = 'yuanchenyang', 'jeromethai'

from pdb import set_trace as T

MAT_FILE = 'data/queueing_params.mat'

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
    nw.budget = 20.
    nw.optimal_attack(max_iters=3, alpha=10., beta=1., max_iters_attack_rate=5)


def optimal_attack_with_radius(r, save_to=None):
    # try to compute the optimal attacks with different radii of adjacencies
    nw = load_network('data/queueing_params.mat')
    nw.set_weights_to_min_time_usage()
    #nw.rates += np.ones(nw.size) * 100
    nw.balance()
    nw.combine()
    nw.budget = 1000
    nw.update_adjacency(r)
    # k has been pre-processed and is given by best_single_destination_attack()
    k = 302
    nw.optimal_attack(max_iters=1, full_adj=False, alpha=10., beta=1., \
                            max_iters_attack_rate=3, k=k)

    rates = nw.attack_rates / (nw.attack_rates + nw.rates)
    T()
    if save_to:
        obj = {'rates': rates, 'routing': nw.attack_routing}
        pickle.dump(obj, open(save_to, 'wb'))

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

def draw_rates(filename):
    fc = FeatureCollection()
    rates = pickle.load(open(filename))['rates']
    for weight, station in zip(rates, sio.loadmat(MAT_FILE)['stations']):
        fc.add_polygon(rbs.get_poly(*get_xy(station)), {'weight': weight})
    fc.dump('rates.geojson')

def draw_routing(filename, dir):
    fc = FeatureCollection()
    routing = pickle.load(open(filename))['routing']
    stations = map(get_xy, sio.loadmat(MAT_FILE)['stations'])
    for row, (sx, sy) in zip(routing, stations):
        total = 0
        for rate, (ex, ey) in zip(row, stations):
            dx, dy = ex - sx, ey - sy
            if dx > 0:
                total += rate * dx / np.sqrt(dx**2 + dy**2)
        fc.add_polygon(rbs.get_poly(sx, sy), {'weight': total})
    fc.dump('routing.geojson')

if __name__ == '__main__':
    # cal_logo_experiment(range(1, 15))
    # optimal_attack_full_network()
    optimal_attack_with_radius(5)
    # network_simulation()
    # optimal_attack_with_radius(2, save_to='tmp1.pkl')
    # draw_rates('tmp1.pkl')
    # draw_routing('tmp1.pkl', 1)
    #network_simulation()
