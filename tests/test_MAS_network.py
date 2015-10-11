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
        rates, routing, travel_times = generate_uniform()
        network = MAS.Network(rates, routing, travel_times)
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
        rates, routing, travel_times = generate_uniform()
        network = MAS.Network(rates, routing, travel_times)
        # check throughputs
        self.assertTrue(is_equal(network.throughputs(), (1./3) * np.ones((3,))))
        # check availabilities
        self.assertTrue(is_equal(network.availabilities(), np.ones((3,))))


    def test_throughputs_2(self):
        # compute and check throughputs for asymmetric network
        rates, routing, travel_times = generate_asymmetric()
        network = MAS.Network(rates, routing, travel_times)
        self.assertTrue(is_equal(network.throughputs(), np.array([.25, .25, .5])))   


    def test_mean_travel_time(self):
        rates, routing, travel_times = generate_uniform()
        network = MAS.Network(rates, routing, travel_times)
        self.assertTrue(network.mean_travel_time == 20./3)


    def test_availabilities(self):
        # compute and check availabilities for asymmetric network
        rates, routing, travel_times = generate_asymmetric()
        rates[0] = 2.
        network = MAS.Network(rates, routing, travel_times)
        self.assertTrue(is_equal(network.availabilities(), np.array([.25, .5, 1.])))   


    def test_balance(self):
        # test if the balance strategy effectively balance the network
        rates, routing, travel_times = generate_asymmetric()
        network = MAS.Network(rates, routing, travel_times)
        tmp = np.array([.5, .5, 1.])
        self.assertTrue(is_equal(network.new_availabilities(), tmp))
        opt_rates, opt_routing = network.balance()
        tmp = np.array([1., 1., 1.])
        self.assertTrue(is_equal(network.new_availabilities(), tmp))


    def test_min_attack(self):
        # test if the attacks affectively achieve the target availabilities
        target = np.array([.25, .5, 1.])
        rates, routing, travel_times = generate_asymmetric()
        network = MAS.Network(rates, routing, travel_times)
        opt_rates, opt_routing = network.min_attack(target)
        self.assertTrue(is_equal(network.new_availabilities(), target))


    def test_routing_attack(self):
        # test if the routing of attacks works given fixed attack rates
        rates, routing, travel_times = generate_uniform()
        network = MAS.Network(rates, routing, travel_times)
        # attack rates are fixed
        attack_rates = np.array([1., 1., 1.])
        # find routing minimizing the weighted sum of availabilities
        # fix the availability at station 2 to be equal to 1
        k = 2 
        # get the availabilities 'a' and routing that led to 'a'
        a, routing = network.opt_attack_routing(attack_rates, k)
        self.assertTrue(is_equal(a, np.array([2./3, 2./3, 1.])))
        self.assertTrue(is_equal(a, network.new_availabilities()))


    def test_load_network(self):
        network = MAS.load_network('data/queueing_params.mat')
        a = network.new_availabilities()
        #network.balance()
        #print network.new_availabilities()




if __name__ == '__main__':
    unittest.main()
