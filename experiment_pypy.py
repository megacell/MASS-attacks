import pickle
import sys
from pdb import set_trace as T
from simulation import Network, simulate

def get_balanced():
    nw = pickle.load(open('balanced.pkl'))
    rates = nw['rates']
    routing = nw['routing']
    times = nw['times']

    n = Network(len(rates), rates, times, routing, [8] * len(rates),
                real_rates=nw['real_rates'])

    i = 0
    while n.t < 30.0/60:
        if i % 10 == 0:
            print i
        i += 1
        n.jump()

    pickle.dump({nid: node.n for nid, node in n.graph.items()},
                open('balanced_rates.pkl', 'wb'))

    for _ in range(1000):
        T()

def main():
    nw = pickle.load(open(sys.argv[1]))
    rates = nw['rates']
    routing = nw['routing']
    times = nw['times']

    n = Network(len(rates), rates, times, routing, [0] * len(rates),
                real_rates=nw['real_rates'])

    balanced = pickle.load(open('balanced_rates.pkl'))
    for nid, number in balanced.items():
        n.graph[nid].n = number

    i = 0
    st_lost = []
    re_lost = []
    while n.t < 30.0/60:
        if i % 100 == 0:
            print i
        i += 1
        st_lost.append(n.station_lost)
        re_lost.append(n.real_lost)
        n.jump()

    print n.station_lost
    print n.real_lost
    pickle.dump([st_lost, re_lost], open('loss_' + sys.argv[1], 'wb'))


main()
#get_balanced()
