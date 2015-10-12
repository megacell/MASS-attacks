import numpy as np
from MAS_network import load_network
from logo_to_availabilities import get_availabilities
from pdb import set_trace as T

def cal_logo_experiment():
    nw = load_network('data/queueing_params.mat')
    target = get_availabilities(nw.station_names)

    bal_rates, bal_routing = nw.balance()
    nw.combine()
    nw.adjacencies = nw.full_adjacency #nw.get_adjacencies(100)
    att_rates, att_routing = nw.min_attack(target, full_adj=False)

    nw = load_network('data/queueing_params.mat')
    bal_rates, bal_routing = nw.balance()
    nw.combine()
    att_rates_full, att_routing_full = nw.min_attack(target, full_adj=True)

    print 'Passenger Arrival Rate:', np.sum(nw.rates)
    print 'Balance Cost: ', np.sum(bal_rates)
    print 'Attack After Balance Cost (full adjacency 1): ', np.sum(att_rates)
    print 'Attack After Balance Cost (full adjacency 2): ', np.sum(att_rates_full)
    print 'Last 2 should be the same'

if __name__ == '__main__':
    cal_logo_experiment()
