import numpy as np
import pickle
from MAS_network import load_network
from logo_to_availabilities import get_availabilities
from optimal_attack_solver import OptimalAttackSolver
from param_inference.utils import FeatureCollection
from param_inference.generate_matrices import rbs, get_xy
from simulation import Network, simulate

__author__ = 'yuanchenyang', 'jeromethai'

from pdb import set_trace as T

MAT_FILE = 'data/queueing_params.pkl'

def cal_logo_experiment(adj):
    nw = load_network(MAT_FILE)
    target = get_availabilities(nw.station_names)

    bal_rates, bal_routing = nw.balance()
    nw.combine()

    res = []
    for i in adj:
        nw.update_adjacency(i)
        att_rates, att_routing = nw.min_attack(target, full_adj=False)
        res.append(int(np.sum(att_rates)))
    print 'Passenger Arrival Rate:', np.sum(nw.rates)
    print 'Balance Cost: ', np.sum(bal_rates)
    print 'Attack After Balance Cost (adjacency {}): {}'.format(adj, res)
    return res

def cal_logo_draw(adj):
    mat = pickle.load(open(MAT_FILE))
    nw = load_network(MAT_FILE)
    target = get_availabilities(nw.station_names)

    bal_rates, bal_routing = nw.balance()
    nw.combine()

    nw.update_adjacency(adj)
    att_rates, att_routing = nw.min_attack(target, full_adj=False)

    print 'Passenger Arrival Rate:', np.sum(nw.rates)
    print 'Balance Cost: ', np.sum(bal_rates)
    print 'Attack After Balance Cost (adjacency {}): {}'.format(adj, np.sum(att_rates))

    draw_rates('logo_rates.geojson', mat, att_rates)
    draw_routing('logo_routing.geojson', mat, att_rates, att_routing)
    draw_availabilities('logo_avail.geojson', mat, nw.new_availabilities())

def optimal_attack_full_network():
    # when there is no limit on the type of attacks,
    # a very good type of attack is obtained
    # when they are all routes to the same station
    # we proceed so by choosing the best station to route the attacks to
    # which is the initialization, then there is not much room for progress
    # with the block-coordinate descent algorithm
    nw = load_network(MAT_FILE)
    nw.balance()
    nw.combine()
    nw.budget = 20.
    nw.optimal_attack(max_iters=3, alpha=10., beta=1., max_iters_attack_rate=5)


def optimal_attack_with_radius(r, save_to=None):
    # try to compute the optimal attacks with different radii of adjacencies
    nw = load_network(MAT_FILE)
    nw.set_weights_to_min_time_usage()
    T()
    #nw.rates += np.ones(nw.size) * 100
    nw.balance()
    nw.combine()
    nw.budget = 1000
    if r > 0:
        nw.update_adjacency(r)
    # k has been pre-processed and is given by best_single_destination_attack()
    k = 86 #442 #386 #129
    nw.optimal_attack(max_iters=3, full_adj=(r == 0), alpha=10., beta=1., \
                      max_iters_attack_rate=5, k=k)

    save_results(nw, save_to)


def save_results(nw, save_to):
    rates = nw.attack_rates / (nw.attack_rates + nw.rates)
    obj = {'rates': rates,
           'routing': nw.attack_routing,
           'avails': nw.new_availabilities()}
    pickle.dump(obj, open(save_to, 'wb'))


def optimal_attack_with_max_throughput():
    nw = load_network(MAT_FILE)
    nw.rates = nw.rates + 50.*np.ones((nw.size,))
    nw.balance()
    nw.combine()
    nw.budget = 1000.0
    k = 86
    nw.optimal_attack(omega=0.0, max_iters=3, alpha=10., beta=1., \
                max_iters_attack_rate=5, k=k)


def optimal_attack_with_regularization(omega, ridge, save_to):
    nw = load_network(MAT_FILE)
    nw.rates = nw.rates + 50.*np.ones((nw.size,))
    # nw.balance()
    # nw.combine()
    nw.budget = 1000.0
    k = 86
    nw.optimal_attack(omega=omega, ridge=ridge, max_iters=3, alpha=10., beta=1., \
                max_iters_attack_rate=5, k=k)
    save_results(nw, save_to)



def network_simulation():
    nw = load_network(MAT_FILE)
    target = get_availabilities(nw.station_names)

    bal_rates, bal_routing = nw.balance()
    nw.combine()

    n = Network(nw.size, nw.rates, nw.travel_times, nw.routing, [20]* nw.size)
    for i in range(100):
        if i % 10 == 0:
            print i
        n.jump()


def draw_rates(outfile, mat, rates):
    fc = FeatureCollection()
    stations = mat['stations']
    clusters = mat['clusters']
    for weight, station in zip(rates, stations):
        for s in mat['clusters'][station]:
            fc.add_polygon(rbs.get_poly(*get_xy(s)), {'weight': weight})
    fc.dump(outfile)

def draw_availabilities(outfile, mat, avails):
    fc = FeatureCollection()
    stations = mat['stations']
    clusters = mat['clusters']
    for weight, station in zip(avails, stations):
        for s in mat['clusters'][station]:
            fc.add_polygon(rbs.get_poly(*get_xy(s)), {'weight': weight})
    # So that scale is from 0 to 1
    fc.add_polygon(rbs.get_poly(100, 100), dict(weight=0))
    fc.dump(outfile)


def draw_routing(outfile, mat, rates, routing):
    fc = FeatureCollection()
    stations = map(get_xy, mat['stations'])
    counter = 0
    for rate, row, (sx, sy) in zip(rates, routing, stations):
        total = np.array([0, 0])
        for prob, (ex, ey) in zip(row, stations):
            if prob > 0:
                delta = np.array(rbs.get_center(ex, ey)) - np.array(rbs.get_center(sx, sy))
                total = total + rate * prob * delta / np.linalg.norm(delta)
        if (total[0] + total[1]) > 0:
            counter += 1
        fc.add_point(rbs.get_center(sx, sy), {'u': total[0], 'v': total[1]})

    #fc.add_point(rbs.get_center(0, 0), {'u': 1, 'v': 1})
    fc.dump(outfile)


def draw_network(filename):
    mat = pickle.load(open(MAT_FILE))
    saved = pickle.load(open(filename))
    rates, routing, avails = saved['rates'], saved['routing'], saved['avails']
    draw_rates('rates.geojson', mat, rates)
    draw_routing('routing.geojson', mat, rates, routing)
    draw_availabilities('avails.geojson', mat, avails)


def run_jerome():
    optimal_attack_with_regularization(omega=0.01, ridge=0.01, save_to='tmp1.pkl')
    draw_network('tmp1.pkl')


def run_chenyang():
        # k = 86 for grand central terminal, and k = 302 for a section with small lam
    # cal_logo_experiment(range(1, 15))
    # optimal_attack_full_network()
    # optimal_attack_with_radius(5)
    # network_simulation()

    #cal_logo_draw(1)

    #optimal_attack_with_max_throughput()
    optimal_attack_with_radius(10, save_to='tmp1.pkl')
    #optimal_attack_with_regularization(omega=0.1, ridge=0.01)

    mat = pickle.load(open(MAT_FILE))
    saved = pickle.load(open('tmp1.pkl'))
    rates, routing, avails = saved['rates'], saved['routing'], saved['avails']
    draw_rates('rates.geojson', mat, rates)
    draw_routing('routing.geojson', mat, rates, routing)
    draw_availabilities('avails.geojson', mat, avails)


if __name__ == '__main__':
    run_jerome()
    # run_chenyang()
