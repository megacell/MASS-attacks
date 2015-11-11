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

#MAT_FILE = 'data/queueing_params.pkl'
MAT_FILE = 'data/queueing_params_no_cluster.pkl'


def draw_customer_demand():
    nw = load_network(MAT_FILE)
    # import pdb; pdb.set_trace()
    save_results(nw, save_to='tmp1.pkl')
    saved = pickle.load(open('tmp1.pkl'))
    rates = saved['rates']
    mat = pickle.load(open(MAT_FILE))
    draw_rates('demand_rates.geojson', mat, rates)


def cal_logo_experiment(adj):
    # load MAT_FILE containing...
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

    if adj:
        nw.update_adjacency(adj)
    att_rates, att_routing = nw.min_attack(target, full_adj=(adj == 0))

    print 'Passenger Arrival Rate:', np.sum(nw.rates)
    print 'Balance Cost: ', np.sum(bal_rates)
    print 'Attack After Balance Cost (adjacency {}): {}'.format(adj, np.sum(att_rates))

    draw_rates('logo_rates.geojson', mat, att_rates)
    draw_routing('logo_routing.geojson', mat, att_rates, att_routing, normalize=True)
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


def save_results(nw, save_to='tmp1.pkl', just_rates=False):
    rates = nw.rates
    obj = {'rates': rates,
           'routing': nw.attack_routing,
           'times': nw.travel_times,
           'avails': nw.new_availabilities()}
    pickle.dump(obj, open(save_to, 'wb'))

def save_total_results(nw, real_rates, save_to):
    obj = {'rates': nw.rates.tolist(),
           'real_rates': real_rates,
           'routing': nw.routing.tolist(),
           'times': nw.travel_times.tolist(),
           'avails': nw.new_availabilities().tolist()}
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



def draw_rates(outfile, mat, rates):
    fc = FeatureCollection()
    stations = mat['stations']
    clusters = mat['clusters']
    for weight, station in zip(rates, stations):
        clusters = mat['clusters'][station]
        for s in clusters:
            fc.add_polygon(rbs.get_poly(*get_xy(s)),
                           # Divide by 2 to get rates per hour
                           {'weight': weight/float(len(clusters) * 2)})
    fc.dump(outfile)

def draw_availabilities(outfile, mat, avails):
    fc = FeatureCollection()
    stations = mat['stations']
    clusters = mat['clusters']
    for weight, station in zip(avails, stations):
        clusters = mat['clusters'][station]
        for s in clusters:
            fc.add_polygon(rbs.get_poly(*get_xy(s)),
                           {'weight': weight/float(len(clusters))})
    # So that scale is from 0 to 1
    fc.add_polygon(rbs.get_poly(100, 100), dict(weight=0))
    fc.dump(outfile)


def draw_routing(outfile, mat, rates, routing, normalize=False):
    fc = FeatureCollection()
    stations = mat['stations']
    clusters = mat['clusters']
    for rate, row, s in zip(rates, routing, stations):
        total = np.array([0, 0])
        for c in clusters[s]:
            sx, sy = get_xy(c)
            for prob, e in zip(row, stations):
                ex, ey = get_xy(e)
                if prob > 0:
                    delta = np.array(rbs.get_center(ex, ey)) \
                          - np.array(rbs.get_center(sx, sy))
                    total = total + prob * delta
            norm = np.linalg.norm(total)
            if norm > 0 and normalize:
                total = total / np.linalg.norm(total)

            fc.add_point(rbs.get_center(sx, sy), {'u': total[0], 'v': total[1]})

    # Calibration
    # dy  = np.array(rbs.get_center(0, 1)) - np.array(rbs.get_center(0, 0))
    # dx  = np.array(rbs.get_center(1, 0)) - np.array(rbs.get_center(0, 0))
    # dxy = np.array(rbs.get_center(1, 1)) - np.array(rbs.get_center(0, 0))
    # fc.add_polygon(rbs.get_poly(0, 0))
    # fc.add_polygon(rbs.get_poly(1, 0))
    # fc.add_polygon(rbs.get_poly(0, 1))
    # fc.add_polygon(rbs.get_poly(1, 1))
    # fc.add_point(rbs.get_center(0, 0), {'u': dx[0], 'v': dx[1]})
    # fc.add_point(rbs.get_center(0, 0), {'u': dy[0], 'v': dy[1]})
    # fc.add_point(rbs.get_center(0, 0), {'u': dxy[0], 'v': dxy[1]})
    fc.dump(outfile)


def draw_network(filename='tmp1.pkl', normalize=False):
    mat = pickle.load(open(MAT_FILE))
    saved = pickle.load(open(filename))
    #T()
    rates, routing, avails = saved['rates'], saved['routing'], saved['avails']
    draw_rates('rates.geojson', mat, rates)
    draw_routing('routing.geojson', mat, rates, routing, normalize=normalize)
    draw_availabilities('avails.geojson', mat, avails)


def optimal_attack_with_regularization(omega, ridge, budget,\
                      save_to='tmp1.pkl', iters=3, r=None):
    nw = load_network(MAT_FILE)
    real_rates = nw.rates.tolist()
    #nw.rates = nw.rates + 50.*np.ones((nw.size,))
    nw.set_weights_to_min_time_usage()
    nw.balance()
    nw.combine()
    nw.budget = budget
    k=86
    if r is not None: nw.update_adjacency(r)
    nw.optimal_attack(omega=omega, ridge=ridge, max_iters=iters, \
                        alpha=10., beta=1., max_iters_attack_rate=5, \
                        k=k, full_adj=(r is None))
    save_results(nw, save_to + '.real')
    nw.combine()
    save_total_results(nw, real_rates, save_to)

def attack():
    max_iters=3
    omega=1000.
    ridge=0.1
    r = None
    nw = load_network(MAT_FILE)
    real_rates = nw.rates.tolist()
    nw.set_weights_to_min_time_usage()
    nw.balance()
    nw.combine()
    nw.budget = 1000.0
    k=86
    if r is not None: nw.update_adjacency(r)
    nw.optimal_attack(omega=omega, ridge=ridge, max_iters=max_iters, \
                        alpha=10., beta=1., max_iters_attack_rate=5, \
                        k=k, full_adj=(r is None))
    nw.combine()
    save_total_results(nw, real_rates, 'attack_1000.pkl')

def balance():
    nw = load_network(MAT_FILE)
    real_rates = nw.rates.tolist()
    nw.set_weights_to_min_time_usage()
    nw.balance()
    nw.combine()
    save_total_results(nw, real_rates, 'balanced.pkl')

def run_jerome():
    # omega=100., ridge=0.1, bdg=1000 -> obj: 188/4670, bdg: 1000/1000, thru: 43
    # omega=100., ridge=0.1, bdg=100 -> obj: 1093/4670, bdg: 100/100, thru: 23
    # omega=100., ridge=10., bdg=1000 -> obj: 212/4670, bdg: 1000/1000, thru: 42
    # omega=1000., ridge=1000, bdg=1000 -> obj: 222/4670, bdg: 980/1000, thru: 43
    
    # this works with max attack
    # optimal_attack_with_regularization(omega=100., ridge=.1, budget=1000.0)
    
    # optimal_attack_with_regularization(omega=0., ridge=.1, budget=100.0, save_to='output/attack_100.pkl')
    # optimal_attack_with_regularization(omega=0., ridge=.1, budget=200.0, save_to='output/attack_200.pkl')
    # optimal_attack_with_regularization(omega=0., ridge=.1, budget=500.0, save_to='output/attack_500.pkl')
    # optimal_attack_with_regularization(omega=0., ridge=.1, budget=1000.0, save_to='output/attack_1000.pkl')
    # optimal_attack_with_regularization(omega=0., ridge=.01, budget=2000.0, save_to='output/attack_2000.pkl')
    # optimal_attack_with_regularization(omega=0., ridge=.01, budget=5000.0, save_to='output/attack_5000.pkl')
    # optimal_attack_with_regularization(omega=0., ridge=.01, budget=10000.0, save_to='output/attack_10000.pkl')
    # optimal_attack_with_regularization(omega=0., ridge=.01, budget=1500.0, save_to='output/attack_1500.pkl')
    # optimal_attack_with_regularization(omega=0., ridge=.01, budget=2500.0, save_to='output/attack_2500.pkl')
    # optimal_attack_with_regularization(omega=0., ridge=.01, budget=3000.0, save_to='output/attack_3000.pkl')
    # optimal_attack_with_regularization(omega=0., ridge=.01, budget=4000.0, save_to='output/attack_4000.pkl')
    # optimal_attack_with_regularization(omega=0., ridge=.01, budget=6000.0, save_to='output/attack_6000.pkl')
    # optimal_attack_with_regularization(omega=0., ridge=.01, budget=7000.0, save_to='output/attack_7000.pkl')
    # optimal_attack_with_regularization(omega=0., ridge=.01, budget=8000.0, save_to='output/attack_8000.pkl')
    # optimal_attack_with_regularization(omega=0., ridge=.01, budget=9000.0, save_to='output/attack_9000.pkl')


    #optimal_attack_with_regularization(max_iters=5, omega=10., ridge=0.1, \
    #    save_to='tmp1.pkl', r=3)
    #draw_network(filename='output/attack_100.pkl.real', normalize=True)

    # draw_network(filename='output/attack_1000.pkl.real', normalize=True)
    draw_customer_demand()

def run_chenyang():
    # k = 86 for grand central terminal, and k = 302 for a section with small lam
    #
    # optimal_attack_full_network()
    # optimal_attack_with_radius(5)
    # network_simulation()

    #balance()
    attack()

    # optimal_attack_with_regularization(max_iters=5, omega=1000., ridge=0.01,
    #                                    save_to='tmp1.pkl', r=None)
    # draw_network('tmp1.pkl', normalize=True)

    #cal_logo_draw(7)
    #cal_logo_experiment(range(1, 15))

    #optimal_attack_with_max_throughput()
    #optimal_attack_with_radius(10, save_to='tmp1.pkl')
    #optimal_attack_with_regularization(omega=0.1, ridge=0.01)

    #mat = pickle.load(open(MAT_FILE))
    #saved = pickle.load(open('tmp1.pkl'))
    #rates, routing, avails = saved['rates'], saved['routing'], saved['avails']
    #draw_rates('rates.geojson', mat, rates)
    #draw_routing('routing.geojson', mat, rates, routing)
    #draw_availabilities('avails.geojson', mat, avails)


if __name__ == '__main__':
    run_jerome()
    #run_chenyang()
