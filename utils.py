
import numpy as np

__author__ = 'jeromethai'


def generate_uniform():
    # generate small uniform network attributes
    # rates = 1 everywhere
    # routing is uniform
    # travel_times = 10 everywhere
    rates = 1. * np.ones((3,))
    routing = .5 * np.ones((3,3))
    routing[range(3), range(3)] = 0.0
    travel_times = 10. * np.ones((3,3))
    travel_times[range(3), range(3)] = 0.0
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


def r_2_pi(routing, eps=10e-8):
    # compute the stationary distribution given a routing matrix
    eigenvalues, eigenvectors = np.linalg.eig(routing.transpose())
    index = np.argwhere(abs(eigenvalues - 1.0) < eps)[0][0]
    pi = np.real(eigenvectors[:, index])
    return pi / np.sum(pi)


def pi_2_a(throughputs, rates):
    # computes the availabilities 'a' from the throughputs 'pi'
    a = np.divide(throughputs, rates)
    return a / np.max(a)


def is_equal(a, b, eps=10e-8):
    # check if numpy arrays a and b are check_equal
    return np.sum(abs(a - b)) < eps