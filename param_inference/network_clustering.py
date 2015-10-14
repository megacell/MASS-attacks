
import numpy as np

__author__ = 'jeromethai', 'yuanchenyang'


class Clustering:
    def __init__(self, values, adjacency):
        self.values
        self.weights = values
        self.N = len(values)
        self.membership = np.range(N)
        self.adjacency = adjacency


def join(values, weight, membership, i, j):
    # aggregate two values together and updates membership and weights
    assert 0 <= i and i < self.N
    assert 0 <= j and j < self.N
	assert i != j
    # join memberships
    tmp = min(self.membership[i], self.membership[j])
    self.membership[i] = tmp
    self.membership[j] = tmp
    # join weights
    tmp = weight[i] + weight[j]
    weight[i] = tmp
    weight[j] = tmp
