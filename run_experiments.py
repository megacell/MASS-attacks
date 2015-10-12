import numpy as np
from MAS_network import load_network
from logo_to_availabilities import get_availabilities
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

if __name__ == '__main__':
    cal_logo_experiment(range(1, 10))
