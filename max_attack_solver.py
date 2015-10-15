'''
Min attack solver: 
Optimizing for the Optimal Attack problem with the availabilities fixed
'''

import cplex_interface
import min_cost_flow_solver as mcf
import numpy as np

__author__ = 'jeromethai'

class MaxAttackSolver:
    # class for computing the min attacks with the availabilities fixed
    def __init__(self, network, target_availabilities, ridge=0.0, full_adj=True, \
                        eps=1e-8):
        self.network = network
        self.a = target_availabilities # availabilities are fixed
        self.phi = network.rates # rates before the attacks
        self.delta = network.routing # routing prob. before attacks
        self.eps = eps
        self.N = network.size
        self.ridge = ridge
        self.full_adj = full_adj
        self.adj = network.full_adjacency if full_adj else network.adjacency
        self.b = network.budget
        self.sources = None
        self.inverse_a = None
        self.check()


    def check(self):
        # check that the target is the right size and positive
        assert self.a.shape[0] == self.N, 'availabilities do not match network size'
        assert np.max(self.a) == 1.0, 'availabilities not <= 1.0'
        assert np.min(self.a) >= self.eps, 'availabilities not positive'


    def solver_init(self):
        print 'initialize paramaters of the LP ...'
        # produce the casting into a min-cost flow problem
        # compute source terms
        pickup_rates = np.multiply(self.phi, self.a)
        sources = pickup_rates - np.dot(self.delta.transpose(), pickup_rates)
        assert abs(np.sum(sources)) < self.eps
        self.sources = sources
        self.inverse_a = np.divide(np.ones((self.N,)), self.a)


    def flow_to_rates_routing(self, flow, previous_routing=None):
        # convert the flow solution of the min cost flow problem
        # back into rates
        # makes sure that the diagonal is equal to zero
        N = self.N
        flow[range(N), range(N)] = 0.0
        opt_rates = np.divide(np.sum(flow, 1), self.a)
        # convert the flow into routing
        tmp = np.multiply(opt_rates, self.a)
        zeroes = np.where(tmp < self.eps)[0]
        tmp[zeroes] = np.sum(self.adj[zeroes,:], axis=1)
        #adj = self.adj
        flow[zeroes, :] = self.adj[zeroes,:]
        flow[range(N), range(N)] = 0.0
        inverse_tmp = np.divide(np.ones((N,)), tmp)
        opt_routing = np.dot(np.diag(inverse_tmp), flow)
        opt_rates[opt_rates < 0.0] = 0.0
        opt_routing[opt_routing < 0.0] = 0.
        #import pdb; pdb.set_trace()
        return opt_rates, opt_routing


    def solve(self):
        # initialize the parameters for the min-cost-flow problem
        # it returns that optimal rates and routing probabilities
        self.solver_init() # initialize the sources terms
        # runs the min-cost-flow problem
        flow = self.cplex_solver()
        return self.flow_to_rates_routing(flow)


    def cplex_solver(self):
        open('tmp.lp', 'w').write(self.to_cplex_lp_file())
        "solver to be feeded to CPLEX for max_attack"
        # print self.to_cplex_lp_file()
        variables, sols = cplex_interface.solve_from_file('tmp.lp', 'o')
        non_zeros = np.where(sols)
        flow = np.zeros((self.N,self.N))
        for i in non_zeros[0]:
            a,b = [int(j) for j in variables[i][2:].split('_')]
            flow[a,b] = sols[i]
        return flow


    def to_cplex_lp_file(self):
        # generate input file for CPLEX solver
        # http://lpsolve.sourceforge.net/5.5/CPLEX-format.htm
        N = self.N
        # Objective
        obj = ' '.join(['- x_{}_{}'.format(i, j)
                       for i in range(N)
                       for j in range(N)
                       if self.adj[i,j]==1.])

        # equality constraints
        cst1 = '\n  '.join(['+'.join(['x_{}_{} - x_{}_{}'.format(j, i, i, j)
                                     for j in range(i) + range(i+1, N)
                                     if self.adj[i,j]==1.])
                            + '= {}'.format(self.sources[i])
                           for i in range(N)])
        cst2 = '+'.join(['{} x_{}_{}'.format(self.inverse_a[i], i, j)
                       for i in range(N)
                       for j in range(N)
                       if self.adj[i,j]==1.])
        cst = cst1 + '\n  ' + cst2 + ' <= {}'.format(self.b)
        # bounds
        bnd = '\n  '.join(['0 <= x_{}_{}'.format(i,j)
                           for i in range(N)
                           for j in range(N)
                           if self.adj[i,j]==1.])

        return cplex_interface.template.format(obj, cst, bnd)

