'''Tests for the Mobility-As-a-Service (MAS) class
'''

import unittest
import MAS_network as MAS
import numpy as np
from utils import generate_uniform, generate_asymmetric, is_equal

__author__ = 'jeromethai'

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
        self.assertTrue(is_equal(a, network.new_availabilities()))
        self.assertTrue(abs(np.sum(a) - 7./3))


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
        self.assertTrue(is_equal(a, network.new_availabilities()))
        self.assertTrue(abs(np.sum(a) - 5./3))


    def test_load_network(self):
        network = MAS.load_network('data/queueing_params.mat')
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
        network = MAS.load_network('data/queueing_params.mat')
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


    # the following test is a bit slow, but should work!

    # def test_cplex_attack_routing_full_network(self):
    #     network = MAS.load_network('data/queueing_params.mat')
    #     k = np.where(network.new_availabilities() - 1. == 0.0)[0][0]
    #     print 'availabilities before attacks', np.sum(network.new_availabilities())
    #     attack_rates = 5. * np.ones((network.size,))
    #     a, routing = network.opt_attack_routing(attack_rates, k, cplex=True)
    #     self.assertTrue(is_equal(a, network.new_availabilities()))
    #     print 'availabilities after attacks', np.sum(network.new_availabilities())




if __name__ == '__main__':
    unittest.main()
