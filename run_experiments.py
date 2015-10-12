import numpy as np
from MAS_network import load_network
from logo_to_availabilities import get_availabilities


def cal_logo_experiment():
    nw = load_network('data/queueing_params.mat')
    target = get_availabilities(nw.station_names)

    bal_rates, bal_routing = nw.balance(cplex=True)
    nw.combine()
    att_rates, att_routing = nw.min_attack(target, cplex=True)

    nw = load_network('data/queueing_params.mat')
    att_before_rates, att_before_routing = nw.min_attack(target, cplex=True)

    print 'Passenger Arrival Rate:', np.sum(nw.rates)
    print 'Balance Cost: ', np.sum(bal_rates)
    print 'Attack After Balance Cost: ', np.sum(att_rates)
    print 'Attack Before Balance Cost: ', np.sum(att_before_rates)

if __name__ == '__main__':
    cal_logo_experiment()
