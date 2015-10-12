import numpy as np
from MAS_network import load_network
from big_qu


def cal_logo_experiment():
    nw = load_network('data/queueing_params.mat')
    opt_rates, opt_routing = nw.balance(cplex=True)
    print 'Balance Cost: ', np.sum(opt_rates)


    print

if __name__ == '__main__':
    cal_logo_experiment()
