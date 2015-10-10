'''Tests for the Mobility-As-a-Service (MAS) class
'''

import unittest
import MAS_network as MAS
import numpy as np
from utils import generate_uniform, generate_asymmetric

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
        eps = 10e-8
        rates, routing, travel_times = generate_uniform()
        network = MAS.Network(rates, routing, travel_times)
        # check throughputs
        delta = network.throughputs() - (1./3) * np.ones((3,))
        self.assertTrue(np.sum(abs(delta)) < eps)
        # check availabilities
        delta = network.availabilities() - np.ones((3,))
        self.assertTrue(np.sum(abs(delta)) < eps)


    def test_throughputs_2(self):
        # compute and check throughputs for asymmetric network
        eps = 10e-8
        rates, routing, travel_times = generate_asymmetric()
        network = MAS.Network(rates, routing, travel_times)
        delta = network.throughputs() - np.array([.25, .25, .5])
        self.assertTrue(np.sum(abs(delta)) < eps)   


    def test_mean_travel_time(self):
        rates, routing, travel_times = generate_uniform()
        network = MAS.Network(rates, routing, travel_times)
        self.assertTrue(network.mean_travel_time == 20./3)


    def test_availabilities(self):
        # compute and check availabilities for asymmetric network
        eps = 10e-8
        rates, routing, travel_times = generate_asymmetric()
        rates[0] = 2.
        network = MAS.Network(rates, routing, travel_times)
        delta = network.availabilities() - np.array([.25, .5, 1.])
        self.assertTrue(np.sum(abs(delta)) < eps)   


    def test_balance(self):
        eps = 10e-8
        rates, routing, travel_times = generate_asymmetric()
        network = MAS.Network(rates, routing, travel_times)
        delta = abs(network.new_availabilities() - np.array([.5, .5, 1.]))
        self.assertTrue(np.sum(delta) < eps)
        opt_rates, opt_routing = network.balance()
        network.update(opt_rates, opt_routing)
        delta = abs(network.new_availabilities() - np.array([1., 1., 1.]))
        self.assertTrue(np.sum(delta) < eps)


    def test_min_attack(self):
        eps = 10e-8
        target = np.array([.25, .5, 1.])
        rates, routing, travel_times = generate_asymmetric()
        network = MAS.Network(rates, routing, travel_times)
        opt_rates, opt_routing = network.min_attack(target)
        network.update(opt_rates, opt_routing)
        delta = abs(network.new_availabilities() - target)
        self.assertTrue(np.sum(delta) < eps)


if __name__ == '__main__':
    unittest.main()
