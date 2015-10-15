import csv
import cfg as c
import numpy as np
import heapq as hq
import pickle
from collections import defaultdict
from pdb import set_trace as ST
from utils import FeatureCollection, RBins

rbs = RBins(c.LON, c.LAT, c.ROT * np.pi / 180, c.GRID_SIZE, dia_fac=c.DIAF)

def get_xy(string):
    return map(int, string.split('-'))

def get_str(coords):
    x, y = coords
    return '{}-{}'.format(x, y)

def add_coord(coord):
    def f(delta):
        dx, dy = delta
        x, y = get_xy(coord)
        return get_str((x + dx, y + dy))
    return f

def visualize():
    fc = FeatureCollection()
    for row in csv.DictReader(open('data/lambda.csv')):
        x, y = get_xy(row['station'])
        pickups = int(row['pickups'])
    fc.add_polygon(rbs.get_poly(x, y), {'weight': pickups})

def is_adjacent(a, b):
    ax, ay = get_xy(a)
    bx, by = get_xy(b)
    return int((ax - bx, ay - by) in ((1, 0), (0, 1), (-1, 0), (0, -1)))

def is_set_adjacent(sa, sb):
    for a in sa:
        for b in sb:
            if is_adjacent(a, b):
                return 1
    return 0

def argmin(a):
    i, n = min(enumerate(a), key= lambda x: x[1])
    return i

def cluster_stations(to_include):
    to_include = set(to_include)
    stations = [(int(row['pickups']), [row['station']])
                for row in csv.DictReader(open(c.LAMBDA_FILE))
                if row['station'] in to_include]
    while min(stations)[0] < c.CLUSTER_FACTOR:
        least_n, least_set, ai = min([(n, s, i)
                                      for i, (n, s) in enumerate(stations)])

        adj = [(n, s, i) for i, (n, s) in enumerate(stations)
                         if is_set_adjacent(least_set, s) and i != ai]

        an, aset, bi = min(adj)

        stations.pop(max(ai, bi))
        stations.pop(min(ai, bi))
        stations.append((least_n + an, least_set + aset))

    ss = set()
    for n, s in stations:
        ss.update(s)
    assert(len(ss) == len(to_include))

    fc = FeatureCollection()
    for i, (an, aset) in enumerate(stations):
        if len(aset) == 1: continue
        for s in aset:
            fc.add_polygon(rbs.get_poly(*get_xy(s)),
                           {'n': an, 'i': i})
    fc.dump('clusters.geojson')

    new_include = [s[0] for n, s in stations]
    clusters = {s[0]: s for n, s in stations}
    return sorted(new_include), clusters

def filter_data():
    total = []
    to_exclude = set(c.BLACKLIST)

    for row in csv.DictReader(open(c.LAMBDA_FILE)):
        pickups = int(row['pickups'])
        coord = row['station']
        total.append(coord)
        if pickups < c.LAMBDA_CUTOFF:
            to_exclude.add(coord)

    T = defaultdict(int)
    for row in csv.DictReader(open(c.T_FILE)):
        pickup, dropoff = row['pickup'], row['dropoff']
        if pickup in to_exclude or dropoff in to_exclude:
            continue
        T[pickup] += 1

    for origin, dests in T.items():
        if dests < c.DEST_CUTOFF:
            to_exclude.add(origin)

    to_exclude = to_exclude.difference(c.WHITELIST)
    return sorted([s for s in total if s not in to_exclude]), to_exclude

def generate(filename):
    # Canonical order for output matrices
    to_include, to_exclude = filter_data()

    to_include, clusters = cluster_stations(to_include)

    inv_clusters = {s: s0 for s0, si in clusters.items() for s in si}

    # Final matrix to generate
    mat = {'stations': np.array(to_include),
           'clusters': clusters}

    # Generate lambda
    lam = {}
    for row in csv.DictReader(open(c.LAMBDA_FILE)):
        pickups = int(row['pickups'])
        coord = row['station']
        lam[coord] = pickups / float(c.TOTAL_TIME)

    mat['lam'] = np.array([sum(lam[c] for c in clusters[s]) for s in to_include])

    # Generate T_ij and p_ij
    T = defaultdict(dict)
    p = defaultdict(dict)
    for row in csv.DictReader(open(c.T_FILE)):
        pickup, dropoff = row['pickup'], row['dropoff']
        if pickup in to_exclude or dropoff in to_exclude:
            continue
        proj_p, proj_d = inv_clusters[pickup], inv_clusters[dropoff]
        if proj_p == proj_d:
            continue
        T[proj_p].setdefault(proj_d, []).append(float(row['trip_time_mean']))
        p[proj_p][proj_d] = int(row['counts']) \
                            + p[proj_p].setdefault(proj_d, 0)

    for pu in T.keys():
        for do in T[pu].keys():
            T[pu][do] = np.mean(T[pu][do]) / c.TRIP_TIME_SCALE

    # Convert p_ij into distribution and apply smoothing
    alpha = c.SMOOTHING_FACTOR
    pmat = []
    for origin in to_include:
        dests = p[origin]
        total = sum(dests.values()) - dests.get(origin, 0)
        row = []
        for s in to_include:
            if s == origin:
                # Remove diagonal
                row.append(0)
            else:
                row.append((dests.get(s, 0) + alpha)/
                           float(total + (len(to_include)-1) * alpha))
        pmat.append(row)

    for i, row in enumerate(pmat):
        assert np.isclose(sum(row), 1, 1e-8), 'row{}: {} is not 1!'.format(i, sum(row))

    mat['p'] = np.array(pmat)
    assert mat['p'].shape == (len(to_include), len(to_include)), \
      'Dimensions of p mismatch!'

    # fc = FeatureCollection()
    # for origin, dests in T.items():
    #     fc.add_polygon(rbs.get_poly(*get_xy(origin)),
    #                    {'dests': len(dests)})
    # fc.dump('tmp1.geojson')

    # Convert T_ij into matrix
    failed = 0
    tmat = []
    maxT = max(max(dests.values()) for dests in T.values())
    for origin, dests in T.items():
        row = []
        for s in to_include:
            if s not in dests:
                surrounding = [dests[new_coord]
                               for new_coord in map(add_coord(s), c.DELTAS)
                               if new_coord in dests]
                if len(surrounding) == 0:
                    row.append(maxT)
                    failed += 1
                else:
                    row.append(np.mean(surrounding))
            else:
                row.append(dests[s])
        tmat.append(row)

    mat['T'] = np.array(tmat)
    assert mat['T'].shape == (len(to_include), len(to_include)), \
      'Dimensions of T mismatch!'

    print 'T generation: total={}, failed to match={}, maxT={}'\
      .format( len(tmat) ** 2, failed, maxT)

    # Generate adjacency matrix
    adj = np.array([[int(is_set_adjacent(clusters[o], clusters[d]) and (o != d))
                     for o in to_include]
                    for d in to_include])
    assert adj.shape == (len(to_include), len(to_include)), \
      'Dimensions of adjacency mismatch!'
    mat['adj'] = adj

    pickle.dump(mat, open(filename, 'wb'))
    print 'Saved as: ' + filename


if __name__ == '__main__':
    generate('queueing_params.pkl')
