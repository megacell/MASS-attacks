'''
Attack routing solver:
Optimizing for the Optimal Attack problem with the rate of attacks 'nu' fixed
'''


import cplex_interface
from cvxopt import matrix, spmatrix, sparse, spdiag, solvers
import numpy as np

__author__ = 'jeromethai'


class AttackRoutingSolver:
    # class for computing the best attacks with the rate of attacks fixed
    def __init__(self, network, attack_rates, k, eps=1e-8, cplex=True):
        self.network = network
        self.nu = attack_rates # fixed attack rate
        self.phi = self.network.rates # rates before the attacks
        self.delta = network.routing # routing prob. before attacks
        self.k = k # a_k is set to 1
        self.eps = eps
        self.cplex = cplex
        self.N = network.size
        self.w = network.weights # weights for the availabilities in the obj
        self.adj = network.adjacency
        self.check()


    def check(self):
        # check if the attack_rates are valid
        assert self.nu.shape[0] == self.N, 'attack rates do not match network size'
        assert np.min(self.nu) >= 0.0, 'attack rates are negative'


    def solve(self):
        # solver for the attack routing
        # solver = self.cplex_solver if self.cplex else self.cvxopt_solver
        #flow = solver()
        flow = self.cplex_solver()
        return self.flow_to_availabilities_routing(flow)

    # deprecated
    def cvxopt_solver(self):
        c = matrix(np.repeat(self.w, self.N))
        b, A = self.constraints()
        # lp solver here
        x = np.squeeze(solvers.lp(c,A,b)['x'])
        return x.reshape((self.N, self.N))


    def cplex_solver(self):
        open('tmp.lp', 'w').write(self.to_cplex_lp_file())
        variables, sols = cplex_interface.solve_from_file('tmp.lp', 'o')
        # reconstruct flow matrix from 'sols' returned by cplex solver
        non_zeros = np.where(sols)
        flow = np.zeros((self.N, self.N))
        for i in non_zeros[0]:
            if variables[i][0] == 'a': continue
            a,b = [int(j) for j in variables[i][2:].split('_')]
            flow[a,b] = sols[i]
        return flow


    def constraints(self):
        # construct the constraints for the attack routing problem
        N = self.N
        u = np.tile(range(N), N)
        v = np.repeat(range(N),N)
        w = np.array(range(N*N))
        # build constraint matrix
        A1 = spmatrix(np.repeat(self.nu, N), u, w, (N, N*N))
        A2 = -spmatrix(np.repeat(self.nu + self.phi, N), v, w, (N, N*N))

        I = np.array(range(N))
        J = I + np.array(range(N)) * N
        A3 = spmatrix(self.phi, I, J, (N, N*N))

        tmp = np.dot(np.diag(self.phi), self.delta).transpose()
        A4 = matrix(np.repeat(tmp, N, axis=1))
        A5 = -spmatrix(tmp.flatten(), v, np.tile(J, N), (N, N*N))

        A6 = A1 + A2 + A3 + A4 + A5

        I = np.array([0]*(N-1))
        J = np.array(range(self.k)+range((self.k+1),N)) + N * self.k
        A7 = spmatrix(1., I, J, (1, N*N))
        A = sparse([[A6, -A6, A7, -A7, -spdiag([1.]*(N*N))]])

        tmp = np.zeros(2*N + 2 + N*N)
        tmp[2*N] = 1.
        tmp[2*N + 1] = -1.
        b = matrix(tmp)
        return b, A


    def flow_to_availabilities_routing(self, flow):
        # convert the flow solution of the min cost flow problem back to
        # availabilities and routing probabilities
        flow[range(self.N), range(self.N)] = 0.0
        # availabilities
        avail = np.sum(flow, 1)
        assert np.sum(avail > self.eps) == self.N
        # routing probabilities for the attacks
        tmp = np.divide(np.ones((self.N,)), avail)
        opt_routing = np.dot(np.diag(tmp), flow)
        return avail, opt_routing


    def to_cplex_lp_file(self):
        # generate input file for CPLEX solver
        # http://lpsolve.sourceforge.net/5.5/CPLEX-format.htm
        lam = self.phi + self.nu
        N = self.N
        tmp = np.dot(np.diag(self.phi), self.delta).transpose()


        obj = ' + '.join(['{} a_{}'.format(self.w[i], i)
                          for i in range(self.k) + range(self.k+1, N)])


        # equality constraints
        cst = []
        for i in range(self.k) + range(self.k + 1, N):
            eqn = []
            for j in range(i) + range(i+1, N):
                if self.adj[i,j] == 0.: continue
                eqn.append('{0} y_{3}_{2} - {1} y_{2}_{3}'\
                            .format(self.nu[j], lam[i], i, j))
                if j != self.k:
                    eqn.append('{} a_{}'.format(tmp[i,j], j))
            cst.append('{} = - {}'.format(' + '.join(eqn), tmp[i,self.k]))

        # constraints on the a_i
        for i in range(N):
            eqn = ' + '.join(['y_{}_{}'.format(i,j)
                              for j in range(i) + range(i+1, N) 
                              if self.adj[i,j] == 1.])
            end = '= 1.0' if i == self.k else '- a_{} = 0.0'
            cst.append(eqn + end.format(i))
        cst = '\n  '.join(cst)

        # bounds
        bnd = '\n  '.join(['0 <= y_{}_{}'.format(i,j)
                           for i in range(N)
                           for j in range(i) + range(i+1, N)
                           if self.adj[i,j]== 1.0])

        return cplex_interface.template.format(obj, cst, bnd)
