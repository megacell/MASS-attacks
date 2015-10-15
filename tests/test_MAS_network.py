'''Tests for the Mobility-As-a-Service (MAS) class
'''

import unittest
import MAS_network as MAS
import numpy as np
from utils import generate_uniform, generate_asymmetric, is_equal
import pickle as pkl
import os.path

__author__ = 'jeromethai'


MAT_FILE = 'data/queueing_params.pkl'

class TestMasNetwork(unittest.TestCase):



    def test_MAS_network(self):
        # test valid network
        network = MAS.Network(*generate_uniform())
        network.check()


    def test_ill_defined_MAS_network(self):
        # test network with a rate too small
        rates, routing, travel_times = generate_uniform()
        rates[0] = 0.
        network = MAS.Network(rates, routing, travel_times)
        try:
            network.check()
            self.assertTrue(False)
        except AssertionError as e:
            self.assertEqual(e.args[0], 'rates too small')


    def test_throughputs(self):
        # compute and check throughputs for uniform network
        network = MAS.Network(*generate_uniform())
        # check throughputs
        self.assertTrue(is_equal(network.throughputs(), (1./3) * np.ones((3,))))
        # check availabilities
        self.assertTrue(is_equal(network.availabilities(), np.ones((3,))))


    def test_throughputs_2(self):
        # compute and check throughputs for asymmetric network
        network = MAS.Network(*generate_asymmetric())
        self.assertTrue(is_equal(network.throughputs(), np.array([.25, .25, .5])))


    def test_mean_travel_time(self):
        network = MAS.Network(*generate_uniform())
        self.assertTrue(network.mean_travel_time == 20./3)


    def test_availabilities(self):
        # compute and check availabilities for asymmetric network
        rates, routing, travel_times = generate_asymmetric()
        rates[0] = 2.
        network = MAS.Network(rates, routing, travel_times)
        self.assertTrue(is_equal(network.availabilities(), np.array([.25, .5, 1.])))


    def test_balance(self):
        # test if the balance strategy effectively balance the network
        network = MAS.Network(*generate_asymmetric())
        tmp = np.array([.5, .5, 1.])
        self.assertTrue(is_equal(network.new_availabilities(), tmp))
        opt_rates, opt_routing = network.balance()
        tmp = np.array([1., 1., 1.])
        self.assertTrue(is_equal(network.new_availabilities(), tmp))


    def test_min_attack(self):
        # test if the attacks affectively achieve the target availabilities
        target = np.array([.25, .5, 1.])
        network = MAS.Network(*generate_asymmetric())
        opt_rates, opt_routing = network.min_attack(target)
        self.assertTrue(is_equal(network.new_availabilities(), target))


    def test_attack_routing(self):
        # test if the routing of attacks works given fixed attack rates
        network = MAS.Network(*generate_uniform())
        # attack rates are fixed
        attack_rates = np.array([1., 1., 1.])
        # find routing minimizing the weighted sum of availabilities
        # fix the availability at station 2 to be equal to 1
        k = 2
        # get the availabilities 'a' and routing that led to 'a'
        a, routing = network.opt_attack_routing(attack_rates, k)
        self.assertTrue(is_equal(a, network.new_availabilities(), 1e-7))
        self.assertTrue(abs(np.sum(a) - 5./3))


    def test_attack_routing_2(self):
        # test if the routing of attacks works given fixed attack rates
        network = MAS.Network(*generate_uniform())
        # attack rates are fixed
        attack_rates = np.array([1., 1., 0.])
        # find routing minimizing the weighted sum of availabilities
        # fix the availability at station 2 to be equal to 1
        k = 2
        # get the availabilities 'a' and routing that led to 'a'
        a, routing = network.opt_attack_routing(attack_rates, k)
        self.assertTrue(is_equal(a, network.new_availabilities(), 1e-7))
        self.assertTrue(abs(np.sum(a) - 5./3))


    def test_load_network(self):
        network = MAS.load_network(MAT_FILE)
        a = network.new_availabilities()
        #network.balance()
        #print network.new_availabilities()


    def test_cplex_balance_small_network(self):
        network = MAS.Network(*generate_asymmetric())
        tmp = np.array([.5, .5, 1.])
        self.assertTrue(is_equal(network.new_availabilities(), tmp))
        network.balance(cplex=True)
        self.assertTrue(is_equal(network.new_availabilities(), np.ones((3,))))


    def test_cplex_balance_full_network(self):
        # loading the full network
        network = MAS.load_network(MAT_FILE)
        network.balance(cplex=True)
        self.assertTrue(is_equal(network.new_availabilities(), np.ones((network.size,))))


    def test_cplex_attack_routing_small_network(self):
        # trying with CPLEX, see test_attack_routing_2() above for more details
        network = MAS.Network(*generate_uniform())
        attack_rates = np.array([1., 1., 0.])
        k = 2
        a, routing = network.opt_attack_routing(attack_rates, k, cplex=True)
        self.assertTrue(is_equal(a, network.new_availabilities()))
        self.assertTrue(abs(np.sum(a) - 5./3))


    def test_cplex_attack_routing_full_network(self):
        network = MAS.load_network(MAT_FILE)
        k = np.where(network.new_availabilities() - 1. == 0.0)[0][0]
        print 'availabilities before attacks', np.sum(network.new_availabilities())
        attack_rates = 5. * np.ones((network.size,))
        a, routing = network.opt_attack_routing(attack_rates, k, cplex=True)
        self.assertTrue(is_equal(a, network.new_availabilities()))
        print 'availabilities after attacks', np.sum(network.new_availabilities())


    # def test_opt_attack_rate(self):
    #     network = MAS.Network(*generate_asymmetric())
    #     attack_routing = np.array([[0., 0., 1.],[.5, 0., .5],[.5, .5, 0.]])
    #     nu_init = np.array([1., 0., 0.])
    #     k = 2
    #     network.opt_attack_rate(attack_routing, k, nu_init)
    #     print network.new_availabilities()


    def test_max_attack(self):
        network = MAS.Network(*generate_asymmetric())
        target = np.array([.25, .25, 1.])
        network.budget = 10.0
        network.max_attack(target / np.max(target))
        print np.sum(network.attack_rates)
        print np.sum(abs(network.new_availabilities() - target))
        print np.sum(np.multiply(network.attack_rates, target))


    # def test_max_attack(self):
    #     network = MAS.load_network(MAT_FILE)
    #     network.budget = 200.0
    #     target = np.random.rand(network.size,)
    #     network.max_attack(target / np.max(target))
    #     print np.sum(network.attack_rates)
    #     print np.sum(abs(network.new_availabilities(), target))
    #     print np.sum(np.multiply(network.attack_rates, target))



    # def test_opt_attack_rate_full_network(self):
    #     network = MAS.load_network(MAT_FILE)
    #     network.budget = network.size * 5.
    #     k = np.where(network.new_availabilities() - 1. == 0.0)[0][0]
    #     print k

    #     pklfile = 'data/attack_strategy.pkl'
    #     if not os.path.isfile(pklfile):
    #         attack_rates = 5. * np.ones((network.size,))
    #         a, routing = network.opt_attack_routing(attack_rates, k, cplex=True)
    #         pkl.dump({'availabilities': a,
    #                   'attack_routing': routing,
    #                   'attack_rates':attack_rates},
    #                  open(pklfile, 'wb'))
    #     attack = pkl.load(open(pklfile))
    #     print 'availabilities before optmization', np.sum(attack['availabilities'])
    #     attack_routing = attack['attack_routing']
    #     nu_init = attack['attack_rates']
    #     network.opt_attack_rate(attack_routing, k, nu_init, alpha=10., beta=1., max_iters=10)
    #     print
    #     print 'availabilities after optmization', np.sum(network.new_availabilities())
    #     print np.max(network.new_availabilities())
    #     print np.where(network.new_availabilities() - 1. == 0.0)[0][0]


    # def test_set_weights_to_min_time_usage(self):
    #     network = MAS.load_network(MAT_FILE)
    #     network.set_weights_to_min_time_usage()


    def test_single_destination_attack(self):
        network = MAS.Network(*generate_asymmetric())
        network.single_destination_attack(2)
        self.assertTrue(is_equal(network.new_availabilities(), np.array([1./3, 1./3, 1.])))


    def test_load_full_network_with_adjacency(self):
        network = MAS.load_network(MAT_FILE)
        self.assertTrue(np.sum(network.adjacency) / (network.size * network.size) < 4.)


    def test_adjacencies(self):
        network = MAS.Network(*generate_uniform(4))
        reachable_1 = np.array([[0, 1, 1, 0],
                                [1, 0, 0, 1],
                                [1, 0, 0, 1],
                                [0, 1, 1, 0]])
        reachable_2 = np.array([[0, 1, 1, 1],
                                [1, 0, 1, 1],
                                [1, 1, 0, 1],
                                [1, 1, 1, 0]])
        network.adjacency_1 = reachable_1
        self.assertTrue(is_equal(reachable_1.flatten(),
                                 network.get_adjacencies(1).flatten()))
        self.assertTrue(is_equal(reachable_2.flatten(),
                                 network.get_adjacencies(2).flatten()))


    def test_sparsify_routing(self):
        nw = MAS.load_network(MAT_FILE)
        nw.balance()
        nw.combine()
        print np.sum(nw.routing > 0.0)
        self.assertTrue(np.sum(np.sum(nw.routing, axis=1)) == nw.size)
        nw.sparsify_routing(0.8)
        print np.sum(nw.routing > 0.0)
        self.assertTrue(np.sum(np.sum(nw.routing, axis=1)) == nw.size)



if __name__ == '__main__':
    unittest.main()
