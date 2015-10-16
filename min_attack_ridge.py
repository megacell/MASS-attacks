'''
Min attack solver:
Optimizing for the REGULARIZED Optimal Attack problem with the availabilities fixed
'''

import numpy as np
import cplex_interface


__author__ = 'jeromethai'

class MinAttackRidge:
    # class for computing the min attacks with the availabilities fixed
    # and with an additional regularization term
    def __init__(self, network, target, cost, ridge, full_adj=True, eps=1e-8):
        if isinstance(ridge, int) or isinstance(ridge, float): 
            ridge = ridge * np.ones((network.size,))
        if isinstance(cost, int) or isinstance(cost, float): 
            cost = cost * np.ones((network.size,))
        self.network = network
        self.a = target
        self.cost = cost
        self.phi = network.rates # rates before the attacks
        self.delta = network.routing # routing prob. before attacks
        self.N = network.size
        self.full_adj = full_adj
        self.adj = network.full_adjacency if full_adj else network.adjacency
        self.ridge = ridge
        self.source = None
        self.eps = eps
        self.b = network.budget
        self.coef_ridge = None
        self.coef_cost = None
        self.check()


    def check(self):
        # check that the target is the right size and positive
        assert self.a.shape[0] == self.N, 'availabilities do not match network size'
        assert np.max(self.a) == 1.0, 'availabilities not <= 1.0'
        assert np.min(self.a) >= self.eps, 'availabilities not positive'
        # check costs
        assert self.cost.shape[0] == self.N, 'costs do not match network size'
        assert np.min(self.cost) >= 0.0, 'costs negative'
        # check ridge
        assert self.ridge.shape[0] == self.N, 'costs do not match network size'
        assert np.min(self.ridge) >= 0.0, 'ridge negative'
        # check that ridge and costs are not zeros at the same time
        assert np.min(self.ridge) > 0.0 or np.min(self.cost) > 0.0


    def solver_init(self):
        print 'initialize paramaters of the LP ...'
        # produce the casting into a min-cost flow problem
        # compute source terms
        pickup_rates = np.multiply(self.phi, self.a)
        self.source = pickup_rates - np.dot(self.delta.transpose(), pickup_rates)
        assert abs(np.sum(self.source)) < self.eps
        self.inverse_a = np.divide(np.ones((self.N,)), self.a)
        self.coef_ridge = np.divide(self.ridge, np.square(self.a))
        self.coef_cost = np.divide(self.cost, self.a)



    def flow_to_rates_routing(self, flow, previous_routing=None):
        # convert the flow solution of the min cost flow problem
        # back into rates
        # makes sure that the diagonal is equal to zero
        N = self.N
        flow[range(N), range(N)] = 0.0
        opt_rates = np.divide(np.sum(flow, axis=1), self.a)
        # convert the flow into routing
        tmp = np.multiply(opt_rates, self.a)
        # get zeroes in the vector a * nu
        zeroes = np.where(tmp < self.eps)[0]
        # a[i]*nu[i]=0 -> update to row[i] to 1/deg
        tmp[zeroes] = np.sum(self.adj[zeroes,:], axis=1)
        flow[zeroes, :] = self.adj[zeroes,:]
        flow[range(N), range(N)] = 0.0
        inverse_tmp = np.divide(np.ones((N,)), tmp)
        opt_routing = np.dot(np.diag(inverse_tmp), flow)
        opt_rates[opt_rates < 0.0] = 0.0
        opt_routing[opt_routing < 0.0] = 0.0
        return opt_rates, opt_routing


    def solve(self):
        self.solver_init() # initialize the sources and cost terms
        # runs the min-cost-flow problem
        flow = self.cplex_solver()
        return self.flow_to_rates_routing(flow)


    def cplex_solver(self):
        open('tmp.lp', 'w').write(self.to_cplex_qp_file())
        "solver to be feeded to CPLEX for max_attack"
        variables, sols = cplex_interface.solve_from_file('tmp.lp', 'o')
        non_zeros = np.where(sols)
        flow = np.zeros((self.N,self.N))
        for i in non_zeros[0]:
            a,b = [int(j) for j in variables[i][2:].split('_')]
            flow[a,b] = sols[i]
        # avail = self.a
        # import pdb; pdb.set_trace()
        return flow


    def to_cplex_qp_file(self):
        N = self.N
        # objective
        obj1 = ' + '.join(['{} x_{}_{}'.format(self.coef_cost[i], i, j)
                            for i in range(N)
                            for j in range(i) + range(i+1,N)
                            if self.adj[i,j] > 0.0])
        # obj2 = ' + '.join(['{} x_{}_{} * x_{}_{}'.format(self.coef_ridge[i], i, j, i, k)
        #                     for i in range(N)
        #                     for j in range(i) + range(i+1,N)
        #                     for k in range(i) + range(i+1,N)
        #                     if self.adj[i,j] > 0.0 and self.adj[i,k] > 0.0])
        # obj = '{} + [ {} ] / 2'.format(obj1, obj2)
        obj = obj1
        # equality constraints
        cst1 = '\n  '.join(['+'.join(['x_{}_{} - x_{}_{}'.format(j, i, i, j)
                             for j in range(i) + range(i+1, N)
                             if self.adj[i,j] > 0.0])
                    + '= {}'.format(self.source[i])
                   for i in range(N)])
        # budget constraint
        cst2 = ' + '.join(['{} x_{}_{}'.format(self.inverse_a[i], i, j)
                           for i in range(N)
                           for j in range(i) + range(i+1, N)
                           if self.adj[i,j] > 0.0])
        cst = cst1 + '\n  ' + cst2 + ' <= {}'.format(self.b)
        # bounds
        bnd = '\n  '.join(['0 <= x_{}_{}'.format(i,j)
                           for i in range(N)
                           for j in range(i) + range(i+1, N)
                           if self.adj[i,j] > 0.0])
        return cplex_interface.template.format(obj, cst, bnd)




    def to_cplex_lp_file_deprecated(self):
        N = self.N
        # Objective
        obj1 = ' + '.join(['{} n_{}'.format(self.cost[i], i) for i in range(N)])
        obj2 = ' + '.join(['{} n_{} ^2'.format(self.ridge[i], i) for i in range(N)])
        if np.min(self.ridge) == 0.0:
            obj = obj1
        elif np.min(self.cost) == 0.0:
            obj = '[ {} ] / 2'.format(obj2)
        else:
            obj = '{} + [ {} ] / 2'.format(obj1, obj2)
        # flow constraints 
        cst1 = '\n  '.join(['+'.join(['x_{}_{} - x_{}_{}'.format(j, i, i, j)
                                     for j in range(i) + range(i+1, N)
                                     if self.adj[i,j]==1.])
                            + '= {}'.format(self.source[i])
                           for i in range(N)])
        # budget constraint
        bdg = ' + '.join(['n_{}'.format(i) for i in range(N)]) + ' <= {}'.format(self.b)
        # sum_j x_ij = a_i n_i
        cst2 = []
        for i in range(N):
            eqn = ' + '.join(['x_{}_{}'.format(i,j)
                                for j in range(i) + range(i+1, N)
                                if self.adj[i,j] == 1.])
            end = ' - {} n_{} = 0.0'.format(self.a[i], i)
            cst2.append(eqn + end)
        cst = cst1 + '\n  ' + '\n  '.join(cst2) + '\n  ' + bdg
        # bounds
        bnd1 = '\n  '.join(['0 <= x_{}_{}'.format(i,j)
                   for i in range(N)
                   for j in range(N)
                   if self.adj[i,j]==1.])
        bnd2 = '\n '.join(['0 <= n_{}'.format(i) for i in range(N)])
        bnd = bnd1 + '\n  ' + bnd2
        return cplex_interface.template.format(obj, cst, bnd)



