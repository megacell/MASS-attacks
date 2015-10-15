'''
Attack routing solver:
Optimizing for the Optimal Attack problem with the attack routing probabilities fixed
'''


import numpy as np
from utils import is_equal, pi_2_a, r_2_pi, simplex_projection, ball1_projection


__author__ = 'jeromethai'


class AttackRateSolver:
    def __init__(self, network, attack_routing, k, nu_init, omega=0.0, \
                    ridge=0.0, eps=1e-8):
        # Class for the Attack Rate Solver
        self.network = network
        self.kappa = attack_routing # the attack routing is fixed
        self.phi = network.rates # rates before the attacks
        self.delta = network.routing # routing prob. before attacks
        self.k = k # availability at station k is set to 1
        self.nu = nu_init # nu_init is the initial rate of 
        self.nu_less_k = np.delete(nu_init, k)
        self.omega = omega
        if isinstance(ridge, int) or isinstance(ridge, float): 
            ridge = ridge * np.ones((network.size,))
        self.ridge = ridge
        self.eps = eps
        self.N = network.size
        self.w = network.weights # weights for the availabilities in the obj
        self.w_less_k = np.delete(network.weights, k) # weight without k-th entry
        self.b = network.budget
        # objects specific to the gradient descent algorithm
        self.iter = -1 # iteration number
        self.max_iters = 100
        self.a = None # availabilities
        self.obj_values = [] # ojective values
        self.check()


    def init_solver(self):
        obj, a = self.objective(self.nu)
        self.update(self.nu, obj, a)        


    def check(self):
        # check if the attack_routing 'kappa' is valid
        assert self.kappa.shape[0] == self.N, \
            'attack_routing has {} rows but N = {}'.format(self.kappa.shape[0], self.N)
        assert self.kappa.shape[1] == self.N, \
            'attack_routing has {} rows but N = {}'.format(self.kappa.shape[1], self.N)
        assert np.min(self.kappa) >= 0.0
        assert is_equal(np.sum(self.kappa, axis=1), 1.0, self.eps), \
            'attack_routing not stochastic {}'.format(np.sum(self.kappa, axis=1))
        self.check_nu(self.nu)
        assert np.min(self.ridge) >= 0.0



    def check_nu(self, nu):
        # check if the attack_rates 'nu'
        assert nu.shape[0] == self.N, 'attack rates is not network size'
        assert np.min(nu) >= 0.0, 'negative attack rates'


    def update(self, nu, obj, a):
        # update with the current nu, obj, and availabilities 'a'
        self.nu = nu
        self.nu_less_k = np.delete(nu, self.k)
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
        g = np.dot(np.array(jacobian), self.w_less_k - self.omega * self.nu_less_k)
        g = g - self.omega * a + np.multiply(self.ridge, self.nu)
        return g


    def make_stop(self, max_iter=100, min_progress=1e-5):
        o = self.obj_values
        return lambda: (self.iter > max_iter) or \
                       (len(o) > 1 and abs(o[-1]-o[-2]) < min_progress)


    def make_sqrt_step(self, alpha=0.5, beta=1.0):
        return lambda: alpha / np.sqrt(self.iter + beta)


    def line_search(self, t, g):
        # do line search
        nu = ball1_projection(self.nu - t * g, self.b)
        obj, a = self.objective(nu)
        obj_values = self.obj_values
        #import pdb; pdb.set_trace()
        while obj >= self.obj_values[-1] and t > 1e-6:
            t = t / 2.
            nu = ball1_projection(self.nu - t * g, self.b)
            obj, a = self.objective(nu)
        return t



    def solve(self, step, stop):
        # solves using gradient descent
        self.init_solver()
        for i in range(self.max_iters):
            g = self.gradient()
            t = self.line_search(step(), g)
            nu = ball1_projection(self.nu - t * g, self.b)
            # nu = simplex_projection(self.nu - step() * g, self.b)
            obj, a = self.objective(nu)
            if stop() or t <= 1e-3: break
            self.update(nu, obj, a)
            print 'iter: ', i
            print 'obj: ', obj
        return {'attack_rates': self.nu, 'obj_values': self.obj_values}



