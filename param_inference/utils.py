import json
import copy
import numpy as np
from collections import defaultdict, namedtuple

def make_rot_matrix(theta):
    return np.array([[np.cos(theta), - np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def make_dia_matrix(dia_fac):
    return np.array([[1, 0], [dia_fac, 1]])

class RBins:
    def __init__(self, x0, y0, theta, binsize, dia_fac=0):
        self.x0 = x0
        self.y0 = y0
        self.binsize = float(binsize)
        self.bins = defaultdict(list)

        dia_fac = float(dia_fac) # Disalation factor
        self.trf_mat = make_dia_matrix(dia_fac).dot(make_rot_matrix(theta))
        self.inv_mat = make_rot_matrix(-theta).dot(make_dia_matrix(-dia_fac))

    def trans(self, x, y):
        xp, yp = self.trf_mat.dot(np.array([x - self.x0, y - self.y0]))
        return xp, yp

    def inv_trans(self, x, y):
        xp, yp = self.inv_mat.dot(np.array([x, y]))
        return xp + self.x0, yp + self.y0

    def add(self, x, y, item):
        self.bins[self.get_bin(x, y)].append(item)

    def get_bin(self, x, y):
        xp, yp = self.trans(x, y)
        nx = int(xp / self.binsize)
        ny = int(yp / self.binsize)
        return nx, ny

    def get_coord(self, nx, ny):
        '''Lower left coord'''
        return self.inv_trans(nx * self.binsize, ny * self.binsize)

    def get_poly(self, nx, ny):
        c = self.get_coord
        return [c(nx, ny), c(nx + 1, ny), c(nx + 1, ny + 1), c(nx, ny + 1)]

    def clear_bins(self):
        self.bins = defaultdict(list)

class FeatureCollection:
    def __init__(self):
        self.features = []

    def __str__(self):
        json.dumps(self.export())

    def export(self):
        return {'type': 'FeatureCollection',
                'features': self.features}

    def dump(self, file_name):
        json.dump(self.export(), open(file_name, 'wb'))

    def add(self, geom, props):
        try:
            geom = json.loads(geom)
        except:
            pass
        self.features.append({'type': 'Feature',
                              'geometry': geom,
                              'properties': props})

    def add_polygon(self, points, props=None):
        self.add({'type' : 'Polygon',
                  'coordinates': [points + [points[0]]]}, props or {})

    def deepcopy(self):
        fc = FeatureCollection()
        fc.features = copy.deepcopy(self.features)
        return fc
