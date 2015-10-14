''' This file is pypy-compatibl e'''

import random
from collections import Counter, defaultdict, namedtuple

# Here we define the a Jackson Network Node.
def make_callable(x):
    return x if callable(x) else lambda _: x

# Checks if r is a probability distribution
def check_dist(r):
    for x, px in r.items():
        assert px >= 0, 'Not a distribution!'
    assert abs(sum(r.values()) - 1.0) < 1e-8, 'Not a distribution!'

def from_to(i, j):
    return '{}->{}'.format(i, j)

def sample(dist):
    p, s = random.random(), 0
    for id, prob in dist.items():
        s += prob
        if p < s: return id
    raise ValueError(p, s, id, dist)

NodeState = namedtuple('NodeState', ['id', 'n', 'lost', 't'])

class Node:
    '''A Node object in a Jackson network simulates a queue with an exponential
    service time.'''
    def __init__(self, id, n, mu):
        '''
        id     : Hashable id given to the node
        mu     : Service rate (average vehicles per time), a function of the
                 number of vehicles in node
        n      : Initial number of vehicles in the node
        '''
        self.id = id
        self.n = n
        self.mu = mu
        self.lost = 0 # Number of passengers lost

    def add(self, n):
        self.n += n

    def service_rate(self):
        '''Returns the current service rate of this node'''
        raise NotImplementedError

    def get_state(self, t):
        '''Returns the time-varying state of this node as a tuple'''
        return NodeState(self.id, self.n, self.lost, t)

    def route_to(self):
        '''Randomly samples a destination (node id) from the routing probability
        distribution'''
        raise NotImplementedError

class RoadNode(Node):
    def __init__(self, id, n, mu, rid):
        '''
        rid    : Single ID to route to
        mu     : rate of service for each car
        '''
        Node.__init__(self, id, n, mu)
        self.rid = rid

    def service_rate(self):
        return self.mu * self.n

    def route_to(self):
        return self.rid

class StationNode(Node):
    def __init__(self, id, n, mu, r):
        '''
        r     : Routing probability, dict from node id to probability
        '''
        Node.__init__(self, id, n, mu)
        check_dist(r)
        self.r = r

    def service_rate(self):
        return self.mu

    def route_to(self):
        return sample(self.r)

class Network:
    def __init__(self, n, lam, T, p, k):
        '''Creates a network of n nodes with the following parameters

        n   : Number of nodes
        lam : lam[i] is the arrival rate at station node i
        T   : T[i][j] is the average travel time from node i to node j
        p   : p[i][j] is the routing probability from node i to node j
        k   : k[i] is the number of vehicles station i starts with
        '''
        assert len(k) == len(p) == len(T) == len(lam) == n, 'Seq lengths incorrect!'
        for a, b in zip(T, p):
            assert len(a) == len(b) == n, 'Sub-seq lengths incorrect!'

        self.graph = {}
        self.t = 0
        for i in range(n):
            r = {}
            for j in range(n):
                if i != j:
                    rn_name = from_to(i, j)
                    self.add_node(RoadNode(rn_name, 0, T[i][j], j))
                    r[rn_name] = p[i][j]
            self.add_node(StationNode(i, k[i], lam[i], r))
        self.history = []

    def add_node(self, node):
        self.graph[node.id] = node

    def add_attack(self, i, psi, alpha):
        ''' Adds attack to node i with arrival rate psi and routing probability
        alpha

        psi  : A number representing the arrival rate of attackers
        alpha: A dictionary where alpha[j] is the attackers' routing probability
               rate from node i to node j
        '''
        node = self.graph[i]
        assert isinstance(node, StationNode), 'Can only attack stations'
        newlam = node.mu + psi
        for j in node.r:
            node.r[j] = (alpha[j] * psi +  node.mu * node.r[j]) / newlam
        node.mu = newlam

    def add_single_attack(self, psi, i, j):
        ''' Adds attack with alpha_ij = 1'''
        d = defaultdict(int)
        d[from_to(i, j)] = 1
        self.add_attack(i, psi, d)

    def to_matrix(self):
        n = len(self.graph)
        res = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(self.graph[i].r.get(j) or 0)
            res.append(row)
        return res

    def as_networkx(self):
        import networkx as nx
        G = nx.DiGraph()
        for nodeid, node in self.graph.items():
            G.add_node(nodeid, {'label': 'id={}, n={}, mu={}'\
                                .format(node.id, node.n, round(node.service_rate(), 3))})

        for nodeid, node in self.graph.items():
            if isinstance(node, RoadNode):
                G.add_edge(nodeid, node.rid, {'label': 'p=1'})
            else:
                for nodeid_to, prob in node.r.items():
                    G.add_edge(nodeid, nodeid_to,
                               {'label': 'p={}'.format(round(prob, 4))})
        return G

    def write_graphviz(self, filename):
        import networkx as nx
        nx.write_dot(self.as_networkx(), filename)

    def jump(self, save_history=False):
        '''Simulates one jump of a network'''
        # Get total rates
        rates = [node.service_rate() for node in self.graph.values()]
        total_rates = float(sum(rates))
        # Update time
        dt = random.expovariate(total_rates)
        self.t += dt

        # Get node to update
        dist = {nid: node.service_rate()/ total_rates
                for nid, node in self.graph.items()}
        update_node = self.graph[sample(dist)]

        # Update node if it isn't empty
        if update_node.n == 0:
            # Passenger lost
            update_node.lost += 1
            return
        update_node.add(-1)

        # Pick destination
        dest = update_node.route_to()
        self.graph[dest].add(1)

        # Update history
        if save_history:
            self.history.append(self.get_states())
        return self.t

    def get_last_history(self):
        return self.history[-1]

    def get_states(self):
        '''Returns the states of each node sorted by node_ids'''
        return sorted([node.get_state(self.t) for node in self.graph.values()],
                      key=lambda s: s.id)

    def get_counts(self):
        return zip(*[(i, node.n) for i, node in self.graph.items()])

    def get_station_counts(self):
        return zip(*sorted([(nid, node.n, node.lost) for nid, node in self.graph.items()
                                                     if isinstance(node, StationNode)]))

def full_network(n, lam, T, k):
    ''' Same routing probabilities, constant lam and t.
    n   : number of nodes
    lam : arrival rate of customes (same for all nodes)
    T   : service rate (travel time) at each rode node (same for all nodes)
    k   : starting number of cars at each station node, same for all nodes
    '''
    T = [[T if i != j else 0 for i in range(n)] for j in range(n)]
    p = [[1/float(n-1) if i != j else 0 for i in range(n)] for j in range(n)]
    return Network(n, [lam] * n, T, p, [k] * n)

def l_to_r_attack(n, lam, T, k, psi):
    '''A network of nodes, with a linear virtual passenger chain from node i
    to node i+1, with service rate psi.'''
    nw = full_network(n, lam, T, k)
    for i in range(n - 1):
        d = defaultdict(int)
        d[from_to(i, i+1)] = 1
        nw.add_attack(i, psi, d)
    return nw

def r_to_l_attack(n, lam, T, k, psi):
    '''A network of nodes, with a linear virtual passenger chain from node i
    to node i+1, with service rate psi.'''
    nw = full_network(n, lam, T, k)
    for i in range(1, n):
        d = defaultdict(int)
        d[from_to(0, i)] = 1
        nw.add_attack(0, psi, d)
    return nw

def node_attack(n, lam, T, k, psi):
    '''all the attacks are issued on node 0
    '''
    nw = full_network(n, lam, T, k)
    d = defaultdict(int)
    for i in range(n-1):
        d[from_to(0,i+1)] = 1./(n-1)
    nw.add_attack(0, psi, d)
    for i in range(n-1):
        d = defaultdict(int)
        for j in range(n-1):
            if j != i:
                d[from_to(i+1,j+1)] = 1./(n-2)
        nw.add_attack(i+1, 0., d)
    return nw

def attack_from_0(n, lam, T, k, B):
    nw = full_network(n, lam, T, k)
    d = defaultdict(int)
    for i in range(1, n):
        d[from_to(0,i)] = 1./(n-1)
    nw.add_attack(0, B, d)
    return nw

def attack_to_0(n, lam, T, k, B):
    nw = full_network(n, lam, T, k)
    psi = B/float(n-1)
    for i in range(1, n):
        nw.add_single_attack(psi, i, 0)
    return nw

def attack_to_0_1(n, lam, T, k, B):
    nw = full_network(n, lam, T, k)
    psi = B/float(2 * (n-2))
    for i in range(2, n):
        nw.add_single_attack(psi, i, 0)
        nw.add_single_attack(psi, i, 1)
    return nw

def simulate(network, jumps):
    ''' Simulate the network for JUMPS jumps, and returns TIMES, the time after
    each jump. Also returns COUNTS, the counts of the station nodes of the network
    after each jump, and LOSS, the number of passengers lost '''
    counts, times, loss = [], [], []
    for _ in range(jumps):
        times.append(network.jump())
        _, count, lost = network.get_station_counts()
        counts.append(count)
        loss.append(lost)
    return times, counts, loss

if __name__ == '__main__':
    N = linear_network(5, 0.1, 1, 15)
    for _ in range(1000):
        N.jump()
    N.write_graphviz('out.dot')
