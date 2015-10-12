'''
Attack routing solver:
Optimizing for the Optimal Attack problem with the attack routing probabilities fixed
'''


import numpy as np
from utils import is_equal, pi_2_a, r_2_pi, simplex_projection


__author__ = 'jeromethai'


class AttackRateSolver:
    def __init__(self, network, attack_routing, k, nu_init, eps=10e-8):
        # Class for the Attack Rate Solver
        self.network = network
        self.kappa = attack_routing # the attack routing is fixed
        self.phi = self.network.rates # rates before the attacks
        self.delta = network.routing # routing prob. before attacks
        self.k = k # availability at station k is set to 1
        self.nu = nu_init # nu_init is the initial rate of attacks
        self.eps = eps
        self.N = network.size
        self.w = network.weights # weights for the availabilities in the obj
        self.w_less_k = np.delete(network.weights, k) # weight without k-th entry
        self.b = network.budget
        # objects specific to the gradient descent algorithm
        self.iter = -1 # iteration number
        self.max_iters = 1000
        self.a = None
        self.obj_values = [] # ojective values
        self.check()


    def init_solver(self):
        obj, a = self.objective(self.nu)
        self.update(self.nu, obj, a)


    def check(self):
        # check if the attack_routing 'kappa' is valid
        assert self.kappa.shape[0] == self.N, 'attack_routing is not network size on 0 axis'
        assert self.kappa.shape[1] == self.N, 'attack_routing is not network size on 1 axis'
        assert is_equal(np.sum(self.kappa, axis=1), 1.0, self.eps), \
            'attack_routing not stochastic'
        self.check_nu(self.nu)


    def check_nu(self, nu):
        # check if the attack_rates 'nu'
        assert nu.shape[0] == self.N, 'attack rates is not network size'
        assert np.min(nu) >= 0.0, 'negative attack rates'


    def update(self, nu, obj, a):
        # update with the current nu, obj, and availabilities 'a'
        self.nu = nu
        self.obj_values.append(obj)
        self.a = a
        self.iter = self.iter + 1


    def objective(self, nu):
        # compute the availabilities and the objective at nu
        self.check_nu(nu)
        # compute the combined routing matrix and the combined rate
        lam = self.phi + nu
        tmp = np.dot(np.diag(nu), self.kappa) + np.dot(np.diag(self.phi), self.delta)
        inverse_lam = np.divide(np.ones(self.N,), lam)
        r = np.dot(np.diag(inverse_lam), tmp)
        # compute the availabilities
        pi = r_2_pi(r)
        a = pi_2_a(pi, lam)
        # compute the objective
        obj = np.sum(np.multiply(self.w, a))
        return obj, a


    def gradient(self):
        # compute the gradient of the objective with respect to the attack rates
        # at the current state
        jacobian = []
        obj, a = self.objective(self.nu)
        # compute A
        M = np.dot(np.diag(self.nu), self.kappa) + np.dot(np.diag(self.phi), self.delta)
        A = np.diag(self.phi + self.nu) - M.transpose()
        # remove k-th row and k-th column because a_k is set to 1
        A = np.delete(np.delete(A, self.k, 0), self.k, 1)
        # compute b
        for i in range(self.N):
            b = a[i] * self.kappa[i,:]
            b[i] = b[i] - a[i]
            # remove k-th entry because a_k is set to 1 and solve the equations
            jacobian.append(np.linalg.solve(A, np.delete(b, self.k)))
        return np.dot(np.array(jacobian), self.w_less_k)


    def make_stop(self, max_iter=100):
        return lambda: self.iter > max_iter


    def make_sqrt_step(self, alpha=0.5, beta=1.0):
        return lambda: alpha / np.sqrt(self.iter + beta)


    def solve(self, step, stop):
        # solves using gradient descent
        self.init_solver()
        for i in range(self.max_iters):
            g = self.gradient()
            nu = simplex_projection(self.nu - step() * g, self.b)
            obj, a = self.objective(nu)
            self.update(nu, obj, a)
            if stop(): break
        return self.nu



