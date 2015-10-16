
import numpy as np

__author__ = 'jeromethai'


def generate_uniform(n=3):
    # generate small uniform network attributes
    # rates = 1 everywhere
    # routing is uniform
    # travel_times = 10 everywhere
    rates = 1. * np.ones((n,))
    routing = .5 * np.ones((n,n))
    routing[range(n), range(n)] = 0.0
    travel_times = 10. * np.ones((n,n))
    travel_times[range(n), range(n)] = 0.0
    return rates, routing, travel_times


def generate_asymmetric():
    # generate small asymmetric network attributes
    # rates = 1 everywhere
    # routing = [ 0,  0,  1]
    #           [ 0,  0,  1]
    #           [.5, .5,  0]
    rates = 1. * np.ones((3,))
    routing = np.zeros((3,3))
    routing[0,2] = 1.
    routing[1,2] = 1.
    routing[2,0] = .5
    routing[2,1] = .5
    travel_times = 10. * np.ones((3,3))
    travel_times[range(3), range(3)] = 0.0
    return rates, routing, travel_times


def r_2_pi(routing, eps=1e-8):
    eps = 1e-6 # fix this
    # compute the stationary distribution given a routing matrix
    eigenvalues, eigenvectors = np.linalg.eig(routing.transpose())
    assert abs(np.max(eigenvalues)-1.0) < eps, 'max eigenvalue is {}'.format(np.max(eigenvalues))
    index = np.argwhere(abs(eigenvalues - 1.0) < eps)[0][0]
    pi = np.real(eigenvectors[:, index])
    return pi / np.sum(pi)


def pi_2_a(throughputs, rates):
    # computes the availabilities 'a' from the throughputs 'pi'
    a = np.divide(throughputs, rates)
    return a / np.max(a)


def is_equal(a, b, eps=1e-8):
    # check if numpy arrays a and b are check_equal
    res = np.sum(abs(a - b)) < eps
    if not res:
        print 'Not equal: ', a, b
    return res


def ball1_projection(v, z=1):
    v[v<0.0]=0.0
    if np.sum(v) <= z: return v
    return simplex_projection(v, z)


def renormalize_matrix(A):
    N = A.shape[0]
    tmp = np.divide(np.ones((N,)), np.sum(A, axis=1))
    return np.dot(np.diag(tmp), A)


def simplex_projection(v, z=1):
    ''' Projects vector v of dimension n onto the n-dimensional simplex

    Taken from: Efficient projections onto the l1 ball for learning in higher
                dimensions, Duchi et.al.
    '''
    v[v<0.0]=0.0
    n = len(v)
    mu = sorted(v, reverse=True)
    musum = np.cumsum(mu)

    rho = max([j for j in range(n) if mu[j] - 1.0/(j+1) * (musum[j] - z) > 0])
    theta = 1.0/(rho + 1) * (sum(mu[i] for i in range(rho+1)) - z)

    w = [max(vi - theta, 0) for vi in v]
    return np.array(w)


def norm_adjacencies(mat):
    # Rescale each non-zero entry in the adjacency matrix to 1 and put zeros
    # on the diagonal
    x, y = mat.shape
    assert x == y, 'Matrix is not square!'
    assert np.min(mat) >= 0, 'Matrix has negative entries!'
    mat[mat > 0] = 1
    mat[range(x), range(y)] = 0.0
