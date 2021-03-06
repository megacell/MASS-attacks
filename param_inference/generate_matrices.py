import csv
import cfg as c
import numpy as np
import scipy.io as sio
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

    # Final matrix to generate
    mat = {'stations': np.array(to_include)}

    # Generate lambda
    lam = {}
    for row in csv.DictReader(open(c.LAMBDA_FILE)):
        pickups = int(row['pickups'])
        coord = row['station']
        lam[coord] = pickups / float(c.TOTAL_TIME)

    mat['lam'] = np.array([lam[s] for s in to_include])

    # Generate T_ij and p_ij
    T = defaultdict(dict)
    p = defaultdict(dict)
    for row in csv.DictReader(open(c.T_FILE)):
        pickup, dropoff = row['pickup'], row['dropoff']
        if pickup in to_exclude or dropoff in to_exclude:
            continue
        T[pickup][dropoff] = float(row['trip_time_mean']) / c.TRIP_TIME_SCALE
        p[pickup][dropoff] = int(row['counts'])

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

    adj = np.array([[is_adjacent(o, d) for o in to_include] for d in to_include])
    mat['adj'] = adj

    sio.savemat(filename, mat)
    print 'Saved as: ' + filename


if __name__ == '__main__':
    generate('queueing_params.mat')
